"""Shared Viser camera, point color, and GSplat background rendering utilities."""

import threading
import time
import traceback

import numpy as np
import torch

C0 = 0.28209479177387814


def _rgb_from_gaussians(gaussians, indices):
    features_dc = gaussians["features_dc"][indices]
    if gaussians["features_rest"].shape[1] == 0:
        rgb = torch.sigmoid(features_dc)
    else:
        rgb = torch.clamp(features_dc * C0 + 0.5, 0, 1)
    return rgb.mul(255).round().byte().numpy()


def _quat_wxyz_to_matrix(quaternion):
    w, x, y, z = quaternion
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ], dtype=np.float32)


def _camera_to_viewmat(camera):
    rotation_world_camera = _quat_wxyz_to_matrix(np.asarray(camera.wxyz, dtype=np.float32))
    translation_world_camera = np.asarray(camera.position, dtype=np.float32).reshape(3, 1)
    viewmat = np.eye(4, dtype=np.float32)
    viewmat[:3, :3] = rotation_world_camera.T
    viewmat[:3, 3:4] = -rotation_world_camera.T @ translation_world_camera
    return viewmat


def _camera_intrinsics(camera, height):
    height = int(height)
    width = max(1, int(round(height * float(camera.aspect))))
    focal = 0.5 * height / np.tan(0.5 * float(camera.fov))
    return np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], dtype=np.float32), width


def _render_gsplat_for_camera(camera, gaussians, render_cfg):
    from gsplat.rendering import rasterization
    from utils.gs_utils import _prepare_render_inputs

    device = torch.device(render_cfg["device"])
    current = {key: value.to(device) for key, value in gaussians.items()}
    means, quats, scales, opacities, colors, sh_degree = _prepare_render_inputs(current)
    intrinsics, width = _camera_intrinsics(camera, render_cfg["height"])
    viewmats = torch.from_numpy(_camera_to_viewmat(camera)).to(device).unsqueeze(0).contiguous()
    Ks = torch.from_numpy(intrinsics).to(device).unsqueeze(0).contiguous()
    with torch.no_grad():
        rgb, alpha, _ = rasterization(
            means=means, quats=quats, scales=scales, opacities=opacities, colors=colors,
            viewmats=viewmats, Ks=Ks, width=width, height=render_cfg["height"],
            near_plane=render_cfg["near_plane"], far_plane=render_cfg["far_plane"],
            radius_clip=render_cfg["radius_clip"], eps2d=render_cfg["eps2d"],
            sh_degree=sh_degree, packed=True, render_mode="RGB",
            rasterize_mode=render_cfg["rasterize_mode"], camera_model="pinhole",
        )
    background = torch.as_tensor(render_cfg["background"], device=device, dtype=rgb.dtype).view(1, 1, 1, 3)
    return (rgb + (1 - alpha) * background)[0].clamp(0, 1).mul(255).byte().cpu().numpy()


class GsplatBackgroundRenderer:
    """Render the current state from native Viser cameras, one GPU job at a time."""

    def __init__(self, server, gaussians, state, render_cfg):
        self.server, self.gaussians, self.state, self.render_cfg = server, gaussians, state, render_cfg
        self.lock = threading.Lock()
        self.last_render_time = {}

    def update_gaussians(self, gaussians):
        with self.lock:
            self.gaussians = gaussians

    def render_client(self, client, force=False):
        if self.state["view_mode"] != "gsplat":
            return
        if getattr(client.camera._state, "update_timestamp", 0.0) == 0.0:
            return
        client_id = id(client)
        now = time.monotonic()
        if not force and now - self.last_render_time.get(client_id, 0.0) < self.render_cfg["interval"]:
            return
        self.last_render_time[client_id] = now
        try:
            with self.lock:
                image = _render_gsplat_for_camera(client.camera, self.gaussians, self.render_cfg)
            client.set_background_image(image, format="jpeg", jpeg_quality=85)
        except Exception:
            print("GSplat background render failed:", flush=True)
            traceback.print_exc()

    def render_all(self):
        for client in self.server.get_clients().values():
            self.render_client(client, force=True)

    def clear_all(self):
        color = np.round(np.asarray(self.render_cfg["background"]) * 255).astype(np.uint8)
        image = np.broadcast_to(color, (2, 2, 3)).copy()
        for client in self.server.get_clients().values():
            client.set_background_image(image, format="png")


def _server_component(server, modern_name):
    return getattr(server, modern_name, server)


def _server_method(component, server, modern_name, legacy_name):
    method = getattr(component, modern_name, None) or getattr(server, legacy_name, None)
    if method is None:
        raise AttributeError(f"Installed Viser exposes neither {modern_name} nor {legacy_name}")
    return method


def _validate_gsplat_available(device):
    try:
        import gsplat  # noqa: F401
    except ImportError as error:
        raise RuntimeError("GSplat background mode requires gsplat") from error
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("GSplat background mode requested CUDA, but CUDA is unavailable")


