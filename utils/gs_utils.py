import torch
import gin 
import math
import numpy as np
import os, cv2
from collections import OrderedDict
from plyfile import PlyData, PlyElement
import json
from argparse import Namespace
import torch_scatter
from gsplat.rendering import rasterization
BLOCK_WIDTH = 16 

C0 = 0.28209479177387814


def make_grid(imgs, nrow=3, ncols=3):
    img_h, img_w = imgs[0].shape[:2]
    if imgs[0].ndim == 3:
        grid = np.zeros((img_h * nrow, img_w * ncols, 3), dtype=np.uint8)
    else:
        grid = np.zeros((img_h * nrow, img_w * ncols), dtype=np.uint8)
    for i in range(nrow):
        for j in range(ncols):
            if i * ncols + j >= len(imgs):
                break
            grid[i * img_h : (i + 1) * img_h, j * img_w : (j + 1) * img_w] = imgs[i * ncols + j]
    return grid


def sanitize_for_filename(value):
    return str(value).replace("/", "_").replace("\\", "_")


def copy_gt_attributes(gs, target_gs, attribute_keys):
    for key in attribute_keys:
        if key in gs and key in target_gs:
            gs[key] = target_gs[key].to(device=gs[key].device, dtype=gs[key].dtype).clone()
    return gs


def scale_means_origin(gs, scale):
    scaled_gs = {key: value.clone() for key, value in gs.items()}
    if "means" in scaled_gs:
        scaled_gs["means"] = scaled_gs["means"] * float(scale)
    return scaled_gs


def unscale_means_origin(gs, scale):
    unscaled_gs = {key: value.clone() for key, value in gs.items()}
    if "means" in unscaled_gs:
        unscaled_gs["means"] = unscaled_gs["means"] / float(scale)
    return unscaled_gs


def SH2RGB(sh):
    return sh * C0 + 0.5
def RGB2SH(rgb):
    return (rgb - 0.5) / C0

def _to_python_scalar(value):
    return value.item() if torch.is_tensor(value) else value


def _camera_to_viewmats(camera_to_worlds, device, dtype):
    """Convert OpenGL/Blender camera-to-world matrices to gsplat world-to-camera matrices."""
    if camera_to_worlds.ndim == 2:
        camera_to_worlds = camera_to_worlds.unsqueeze(0)
    camera_to_worlds = camera_to_worlds.to(device=device, dtype=dtype)

    rotations = camera_to_worlds[:, :3, :3]
    translations = camera_to_worlds[:, :3, 3:4]
    axis_flip = torch.diag(
        torch.tensor([1, -1, -1], device=device, dtype=dtype)
    )
    rotations = rotations @ axis_flip

    rotations_inv = rotations.transpose(-1, -2)
    translations_inv = -rotations_inv @ translations
    viewmats = torch.eye(4, device=device, dtype=dtype).expand(
        camera_to_worlds.shape[0], -1, -1
    ).clone()
    viewmats[:, :3, :3] = rotations_inv
    viewmats[:, :3, 3:4] = translations_inv
    return viewmats


def _prepare_render_inputs(gs_params):
    # gsplat's CUDA kernels require float32 rather than float16 inputs.
    gs_params = {
        key: value.float() if value.dtype == torch.half else value
        for key, value in gs_params.items()
    }
    means = gs_params['means']
    scales = torch.exp(gs_params['scales'])
    quats = gs_params['quats']

    if 'opacities' in gs_params:
        opacities = torch.sigmoid(gs_params['opacities'])
    elif 'opacities_sigmoid' in gs_params:
        opacities = gs_params['opacities_sigmoid']
    else:
        raise ValueError("No opacities found in gs_params")
    opacities = opacities.reshape(-1)
    if opacities.shape[0] != means.shape[0]:
        raise ValueError(
            f"Expected one opacity per Gaussian, got {opacities.shape[0]} "
            f"opacities for {means.shape[0]} Gaussians"
        )

    features_rest = gs_params.get('features_rest')
    if features_rest is not None and features_rest.shape[1] > 0:
        colors = torch.cat(
            [gs_params['features_dc'].unsqueeze(1), features_rest], dim=1
        )
        sh_degree = int(math.sqrt(colors.shape[1]) - 1)
    else:
        colors = torch.sigmoid(gs_params['features_dc'])
        sh_degree = None

    return means, quats, scales, opacities, colors, sh_degree


def _rasterize_gaussians(
    gs_params,
    camera_to_worlds,
    cx,
    cy,
    fx,
    fy,
    width,
    height,
    background_color,
):
    means, quats, scales, opacities, colors, sh_degree = _prepare_render_inputs(
        gs_params
    )
    viewmats = _camera_to_viewmats(
        camera_to_worlds, device=means.device, dtype=means.dtype
    )
    camera_count = viewmats.shape[0]

    K = torch.tensor(
        [
            [_to_python_scalar(fx), 0.0, _to_python_scalar(cx)],
            [0.0, _to_python_scalar(fy), _to_python_scalar(cy)],
            [0.0, 0.0, 1.0],
        ],
        device=means.device,
        dtype=means.dtype,
    )
    Ks = K.unsqueeze(0).expand(camera_count, -1, -1).contiguous()
    H, W = int(_to_python_scalar(height)), int(_to_python_scalar(width))

    render_colors, render_alphas, _ = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=W,
        height=H,
        sh_degree=sh_degree,
        packed=True,
        tile_size=BLOCK_WIDTH,
        render_mode="RGB",
        rasterize_mode="classic",
    )

    # gsplat 1.5.3 does not handle per-camera backgrounds correctly in packed
    # mode, so composite the shared background from the returned alpha instead.
    background = torch.as_tensor(
        background_color, device=means.device, dtype=render_colors.dtype
    ).reshape(-1)
    if background.shape[0] != render_colors.shape[-1]:
        raise ValueError(
            f"Expected a {render_colors.shape[-1]}-channel background, "
            f"got shape {tuple(background.shape)}"
        )
    render_colors = render_colors + (1.0 - render_alphas) * background.view(
        1, 1, 1, -1
    )
    render_colors = torch.clamp(render_colors, max=1.0)
    return render_colors, render_alphas


def rasterize_gaussians_to_multiimgs(gs_params, cameras, batched=False):
    """Render several cameras, optionally using gsplat's native camera batching."""
    camera_to_worlds = cameras['camera_to_worlds']
    if batched:
        rgbs, alphas = _rasterize_gaussians(
            gs_params,
            camera_to_worlds,
            cx=cameras['cx'],
            cy=cameras['cy'],
            fx=cameras['fx'],
            fy=cameras['fy'],
            width=cameras['width'],
            height=cameras['height'],
            background_color=cameras['background_color'],
        )
        return list(rgbs.unbind(0)), list(alphas.unbind(0))

    rgbs, alphas = [], []
    for camera_to_world in camera_to_worlds:
        rgb, alpha = rasterize_gaussians_to_singleimg(gs_params, camera_to_world, **cameras)
        rgbs.append(rgb)
        alphas.append(alpha)
    return rgbs, alphas


def rasterize_gaussians_to_singleimg(gs_params, camera_to_world, cx, cy, fx, fy, width, height, background_color, **kwargs):
    rgbs, alphas = _rasterize_gaussians(
        gs_params,
        camera_to_world,
        cx=cx,
        cy=cy,
        fx=fx,
        fy=fy,
        width=width,
        height=height,
        background_color=background_color,
    )
    return rgbs[0], alphas[0]

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))


def _camera_to_homogeneous(c2w_opengl):
    camera = c2w_opengl.detach().cpu().numpy()
    if camera.shape == (4, 4):
        return camera.copy()
    if camera.shape == (3, 4):
        homogeneous = np.eye(4, dtype=camera.dtype)
        homogeneous[:3, :4] = camera
        return homogeneous
    raise ValueError(
        "camera_to_worlds entries must have shape (3, 4) or (4, 4), "
        f"got {camera.shape}"
    )


def prepare_viewer(cameras, dirname, sh_degree):    #1. cfg_args
    cfg_dict = {}
    cfg_dict['source_path'] = '' # It does not matter
    cfg_dict['sh_degree'] = sh_degree
    cfg_dict['white_background'] = False
    with open(dirname+'/cfg_args', 'w') as f:
        f.write(str(Namespace(**cfg_dict)))
    #2. Camera pose
    cameras_towrite= []
    for i, c2w_opengl in enumerate(cameras['camera_to_worlds']):
        cam = {'id':i, 'img_name':f'img_{i}.png',
               'width': cameras['width'].item(),
                'height': cameras['height'].item(),
                'fx': cameras['fx'].item(),
                'fy': cameras['fy'].item(),
                'FovX': None, 'FovY': None,
                'position': None, 'rotation': None}
        cam['FovX'] = focal2fov(cam['fx'], cam['width'])
        cam['FovY'] = focal2fov(cam['fy'], cam['height'])
        c2w_colmap_4x4 = _camera_to_homogeneous(c2w_opengl)
        c2w_colmap_4x4[:3,1:3]*=-1 #flip y and z
        w2c = np.linalg.inv(c2w_colmap_4x4)
        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]
        Rt = np.zeros((4, 4))
        Rt[:3, :3] = R.transpose()
        Rt[:3, 3] = T 
        Rt[3, 3] = 1.0 

        W2C = np.linalg.inv(Rt) 
        pos = W2C[:3, 3] 
        rot = W2C[:3, :3] 
        serializable_array_2d = [x.tolist() for x in rot]
        cam['position'] = pos.tolist()
        cam['rotation'] = serializable_array_2d
        cameras_towrite.append(cam)
    with open(dirname+'/cameras.json', 'w') as f:
        json.dump(cameras_towrite, f)


def export_ply_forviewer(gs_params, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    count = 0
    map_to_tensors = OrderedDict()

    with torch.no_grad():
        positions = gs_params['means'].cpu().numpy()
        count = positions.shape[0]
        n = count
        map_to_tensors["x"] = positions[:, 0]
        map_to_tensors["y"] = positions[:, 1]
        map_to_tensors["z"] = positions[:, 2]
        map_to_tensors["nx"] = np.zeros(n, dtype=np.float32)
        map_to_tensors["ny"] = np.zeros(n, dtype=np.float32)
        map_to_tensors["nz"] = np.zeros(n, dtype=np.float32)


        if 'features_rest' in gs_params and gs_params['features_rest'].shape[1]!=0:
            shs_0 = gs_params['features_dc'].contiguous().cpu().numpy() #N,3
            for i in range(shs_0.shape[1]):
                map_to_tensors[f"f_dc_{i}"] = shs_0[:, i, None]
            # transpose(1, 2) was needed to match the sh order in Inria version
            shs_rest = gs_params['features_rest'].transpose(1, 2).contiguous().cpu().numpy()
            shs_rest = shs_rest.reshape((n, -1))
            for i in range(shs_rest.shape[-1]):
                map_to_tensors[f"f_rest_{i}"] = shs_rest[:, i, None]
        else:
            #convert logit(color) to features_dc
            color = torch.sigmoid(gs_params['features_dc'])
            shs_0 = RGB2SH(color).cpu().numpy()
            for i in range(shs_0.shape[1]):
                map_to_tensors[f"f_dc_{i}"] = shs_0[:, i, None]

        map_to_tensors["opacity"] = gs_params['opacities'].data.cpu().numpy()
        scales =  gs_params['scales'].data.cpu().numpy()
        for i in range(3):
            map_to_tensors[f"scale_{i}"] = scales[:, i, None]

        quats = gs_params['quats'].data.cpu().numpy()
        for i in range(4):
            map_to_tensors[f"rot_{i}"] = quats[:, i, None]


    write_ply_v2(str(filename), map_to_tensors)

def load_ply_forviewer(filename):
    """Load Gaussian parameters written by export_ply_forviewer."""
    vertex = PlyData.read(str(filename))["vertex"]
    names = vertex.data.dtype.names or []
    rest_names = sorted([name for name in names if name.startswith("f_rest_")], key=lambda name: int(name[7:]))
    means = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=-1)
    features_dc = np.stack([vertex[f"f_dc_{index}"] for index in range(3)], axis=-1)
    if rest_names:
        features_rest = np.stack([vertex[name] for name in rest_names], axis=-1)
        features_rest = features_rest.reshape(means.shape[0], 3, -1).transpose(0, 2, 1)
    else:
        features_rest = np.zeros((means.shape[0], 0, 3), dtype=np.float32)
    return {
        "means": torch.from_numpy(means.astype(np.float32)),
        "features_dc": torch.from_numpy(features_dc.astype(np.float32)),
        "features_rest": torch.from_numpy(features_rest.astype(np.float32)),
        "opacities": torch.from_numpy(np.asarray(vertex["opacity"], dtype=np.float32).reshape(-1, 1)),
        "scales": torch.from_numpy(np.stack([vertex[f"scale_{index}"] for index in range(3)], axis=-1).astype(np.float32)),
        "quats": torch.from_numpy(np.stack([vertex[f"rot_{index}"] for index in range(4)], axis=-1).astype(np.float32)),
    }
def write_ply_v2(path, map_to_tensors):
    '''
    from Inria's 3DGS implementation
    Save 3DGS for their viewer
    '''
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    # All channels except the 3 DC
    all_keys = list(map_to_tensors.keys())
    f_dc = []
    for key in all_keys:
        if key.startswith('f_dc_'):
            l.append(key)
            f_dc.append(map_to_tensors[key])
    f_dc = np.concatenate(f_dc, axis=1) # N, 3


    f_rest = []
    for key in all_keys:
        if key.startswith('f_rest_'):
            l.append(key)
            f_rest.append(map_to_tensors[key]) # (N, 1)
    if len(f_rest) > 0:
        f_rest = np.concatenate(f_rest, axis=1) # (N,D)
    else:
        f_rest = np.zeros((f_dc.shape[0], 0))


    l.append('opacity')
    opacities = map_to_tensors['opacity']


    scale = []
    for key in all_keys:
        if key.startswith('scale_'):
            l.append(key)
            scale.append(map_to_tensors[key])
    scale = np.concatenate(scale, axis=1) # (N, 3)


    rotation = []
    for key in all_keys:
        if key.startswith('rot_'):
            l.append(key)
            rotation.append(map_to_tensors[key])
    rotation = np.concatenate(rotation, axis=1) # (N, 4)


    dtype_full = [(attribute, 'f4') for attribute in l]
    N = map_to_tensors['x'].shape[0]
    elements = np.empty(N, dtype=dtype_full)
    xyz = np.stack([map_to_tensors['x'], map_to_tensors['y'], map_to_tensors['z']], axis=1)
    normals = np.zeros_like(xyz)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)


def clone_gaussians(gs):
    return {key: value.clone() for key, value in gs.items()}


def convert_gaussian_frame(gs, source_scaler, target_scaler):
    means = gs["means"]
    source_scale = torch.as_tensor(
        source_scaler.scale_, device=means.device, dtype=means.dtype
    )
    source_translation = torch.as_tensor(
        source_scaler.trans_, device=means.device, dtype=means.dtype
    )
    target_scale = torch.as_tensor(
        target_scaler.scale_, device=means.device, dtype=means.dtype
    )
    target_translation = torch.as_tensor(
        target_scaler.trans_, device=means.device, dtype=means.dtype
    )

    raw_means = (means - source_translation) / source_scale
    converted = {"means": raw_means * target_scale + target_translation}
    for key, value in gs.items():
        if key == "means":
            continue
        if key == "scales":
            converted[key] = value - torch.log(source_scale) + torch.log(target_scale)
        else:
            converted[key] = value.clone()
    return converted
