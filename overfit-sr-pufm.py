import json
import os
import random

import cv2
import gin
import numpy as np
import torch
import torch.nn.functional as F
from absl import app, flags
from tqdm import tqdm

from dataset.GS_multi import SplatFactoMultiLevelDataset
from models.feature_flow_predictor import GSFlowPredictor
from models.feature_predictor import FeaturePredictor  # Registers legacy gin keys used by GS_multi.
from utils import gpu_utils, gs_utils
from utils.log_utils import ProcessSafeLogger
from utils.metrics import MetricComputer
from utils.optimizers import build_optimizer, build_scheduler
from utils.sr_densify_utils import build_densified_input_gs


flags.DEFINE_string("output_dir", "output_overfit_pufm", "Output directory")
flags.DEFINE_string("eval_subdir", "eval_final", "Eval subdirectory")
flags.DEFINE_string("scene_name", "", "Scene name to overfit")
flags.DEFINE_boolean("compare_with_input", False, "Compare with input 3DGS")
flags.DEFINE_boolean("save_viewer", True, "Save viewer point clouds")
flags.DEFINE_boolean("save_residuals", True, "Save residual tensors and stats")
flags.DEFINE_integer("input_factor", 4, "Low-resolution GS factor used as densification source")
flags.DEFINE_integer("target_factor", 2, "High-resolution GS/image factor used as flow target")
flags.DEFINE_enum("alignment", "emd", ["emd", "nearest"], "Interpolated-to-target alignment method")
flags.DEFINE_enum(
    "attribute_init",
    "aligned",
    ["aligned", "3dgs"],
    "How to initialize non-position GS attributes after high-res positions are fixed",
)
flags.DEFINE_float("emd_eps", 0.01, "Auction EMD epsilon")
flags.DEFINE_integer("emd_iters", 100, "Auction EMD iterations")
flags.DEFINE_boolean("gt_features_dc", False, "Initialize features_dc from GT high-res GS")
flags.DEFINE_boolean("gt_features_rest", False, "Initialize features_rest from GT high-res GS")
flags.DEFINE_boolean("gt_opacities", False, "Initialize opacities from GT high-res GS")
flags.DEFINE_boolean("gt_scales", False, "Initialize scales from GT high-res GS")
flags.DEFINE_boolean("gt_quats", False, "Initialize quats from GT high-res GS")
flags.DEFINE_integer("flow_steps", None, "Euler sampling steps; overrides gin flow_matching.flow_steps")
flags.DEFINE_string("flow_space", None, "Flow space: render, bounded, or raw_scale_opacity")
flags.DEFINE_float("flow_noise_std", None, "Stddev of Gaussian noise added to x0 in flow-space")
flags.DEFINE_multi_string("gin_file", None, "List of paths to the config files.")
flags.DEFINE_multi_string("gin_param", "", "Newline separated list of Gin parameter bindings.")

FLAGS = flags.FLAGS
FLOW_SPACES = {"render", "bounded", "raw_scale_opacity"}
FLOW_KEYS = ["means", "features_dc", "features_rest", "opacities", "scales", "quats"]
EPS = 1e-6


@gin.configurable
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@gin.configurable
def training(
    output_dir=None,
    total_steps=gin.REQUIRED,
    pretrain_steps=gin.REQUIRED,
    eval_interval=gin.REQUIRED,
    log_interval=gin.REQUIRED,
    save_interval=gin.REQUIRED,
    log_image_interval=gin.REQUIRED,
    grad_clip_norm=gin.REQUIRED,
    resume_from_step=0,
    enable_amp=False,
    empty_cache_fre=-1,
):
    return {
        "output_dir": output_dir,
        "total_steps": total_steps,
        "pretrain_steps": pretrain_steps,
        "eval_interval": eval_interval,
        "log_interval": log_interval,
        "save_interval": save_interval,
        "log_image_interval": log_image_interval,
        "grad_clip_norm": grad_clip_norm,
        "resume_from_step": resume_from_step,
        "enable_amp": enable_amp,
        "empty_cache_fre": empty_cache_fre,
    }


@gin.configurable
def flow_matching(
    flow_steps=5,
    flow_space="render",
    flow_noise_std=0.0,
    loss_weights=None,
    flow_loss_weight=1.0,
    render_loss_weight=1.0,
    flow_loss_grad_keys=None,
):
    if loss_weights is None:
        loss_weights = {key: 1.0 for key in FLOW_KEYS}
    if flow_loss_grad_keys is None:
        flow_loss_grad_keys = ["means"]
    return {
        "flow_steps": flow_steps,
        "flow_space": flow_space,
        "flow_noise_std": flow_noise_std,
        "loss_weights": loss_weights,
        "flow_loss_weight": flow_loss_weight,
        "render_loss_weight": render_loss_weight,
        "flow_loss_grad_keys": list(flow_loss_grad_keys),
    }


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


def _to_cpu(data):
    if torch.is_tensor(data):
        return data.detach().cpu()
    if isinstance(data, dict):
        return {k: _to_cpu(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_to_cpu(v) for v in data]
    if isinstance(data, tuple):
        return tuple(_to_cpu(v) for v in data)
    return data


def _sanitize_for_filename(value):
    return str(value).replace("/", "_").replace("\\", "_")


def _clone_gs(gs):
    return {key: value.clone() for key, value in gs.items()}


def _clamp_unit(value):
    return value.clamp(EPS, 1.0 - EPS)


def _logit(value):
    return torch.logit(_clamp_unit(value))


def raw_to_flow_gs(gs, flow_space):
    flow = {}
    for key, value in gs.items():
        if key == "scales":
            flow[key] = torch.exp(value)
        elif key == "opacities":
            flow[key] = torch.sigmoid(value)
        elif key == "quats" and flow_space in ["render", "bounded"]:
            flow[key] = F.normalize(value, dim=-1)
        elif key in ["features_dc", "features_rest"] and flow_space == "bounded":
            flow[key] = torch.sigmoid(value)
        else:
            flow[key] = value.clone()
    return flow


def flow_to_raw_gs(flow, flow_space):
    raw = {}
    for key, value in flow.items():
        if key == "scales":
            raw[key] = torch.log(torch.clamp_min(value, EPS))
        elif key == "opacities":
            raw[key] = _logit(value)
        elif key == "quats" and flow_space in ["render", "bounded"]:
            raw[key] = F.normalize(value, dim=-1)
        elif key in ["features_dc", "features_rest"] and flow_space == "bounded":
            raw[key] = _logit(value)
        else:
            raw[key] = value.clone()
    return raw


def flow_state_like(lhs, rhs, alpha):
    return {key: alpha * rhs[key] + (1.0 - alpha) * lhs[key] for key in lhs.keys() if key in rhs}


def add_flow_noise(flow_gs, std):
    if std <= 0:
        return _clone_gs(flow_gs)
    return {key: value + torch.randn_like(value) * std for key, value in flow_gs.items()}


def flow_loss(pred_vel, target_vel, loss_weights, grad_keys):
    losses = {}
    total = None
    grad_keys = set(grad_keys)
    for key in FLOW_KEYS:
        if key not in pred_vel or key not in target_vel:
            continue
        loss = (pred_vel[key] - target_vel[key]).pow(2).mean()
        losses[key] = loss
        weighted_loss = loss if key in grad_keys else loss.detach()
        weighted = float(loss_weights.get(key, 1.0)) * weighted_loss
        total = weighted if total is None else total + weighted
    if total is None:
        raise ValueError("No overlapping flow keys between prediction and target")
    return total, losses


def extrapolate_target_flow_gs(query_flow_gs, pred_vel, t):
    remaining = (1.0 - t).view(1, 1)
    target_hat = {}
    for key, value in query_flow_gs.items():
        if key in pred_vel:
            target_hat[key] = value + remaining * pred_vel[key]
        else:
            target_hat[key] = value.clone()
    return target_hat


def render_l1_loss(gs_raw, cameras, images):
    pred_imgs, _ = gs_utils.rasterize_gaussians_to_multiimgs(gs_raw, cameras)
    loss = 0.0
    for pred_img, gt_img in zip(pred_imgs, images):
        gt_rgb = gt_img[..., :3]
        loss = loss + (pred_img - gt_rgb).abs().mean()
    return loss / max(1, len(pred_imgs)), pred_imgs


def sample_flow_model(model, source_flow_gs, scene_idx, flow_steps, flow_space):
    if flow_steps <= 0:
        raise ValueError("flow_steps must be positive")
    model.eval()
    state = _clone_gs(source_flow_gs)
    device = state["means"].device
    with torch.no_grad():
        for step in range(flow_steps):
            t_value = torch.full((1,), float(step) / float(flow_steps), device=device)
            pred_vel = model(batch_flow_gs=[state], batch_scene_idx=[scene_idx], t=t_value)[0]
            for key in state.keys():
                if key in pred_vel:
                    state[key] = state[key] + pred_vel[key] / float(flow_steps)
    return flow_to_raw_gs(state, flow_space)


def _build_dataset():
    with gin.config_scope("train_dataset"):
        return SplatFactoMultiLevelDataset()


def _scene_name_from_dataset(dataset, idx):
    return dataset.folders[idx]["scene_name"]


def _find_scene_index(dataset, scene_name):
    if scene_name == "":
        return 0
    for idx in range(len(dataset.folders)):
        if _scene_name_from_dataset(dataset, idx) == scene_name:
            return idx
    return 0


def _build_split_payload(dataset, scene_idx, scene_name, factor_entry, split):
    meta = factor_entry["meta"]
    imgs_path = factor_entry["imgs_path"]
    imgs_name = factor_entry["imgs_name"]

    if dataset.background_color == "random":
        background = torch.rand(3)
    else:
        background = torch.tensor(dataset.background_color, dtype=torch.float32) / 255.0

    total_num = len(meta["camera_to_worlds"])
    if split in ["train", "test"]:
        cam_ids = np.arange(total_num)
    else:
        raise ValueError(f"Unsupported split: {split}")

    images = [dataset.read_image(imgs_path[i], background=background) for i in cam_ids]
    images_name = [imgs_name[i] for i in cam_ids]
    camera_to_worlds = meta["camera_to_worlds"][cam_ids]

    cameras = {
        "camera_to_worlds": torch.as_tensor(camera_to_worlds).float(),
        "fx": torch.as_tensor(meta["fx"]).float(),
        "fy": torch.as_tensor(meta["fy"]).float(),
        "cx": torch.as_tensor(meta["cx"]).float(),
        "cy": torch.as_tensor(meta["cy"]).float(),
        "width": torch.as_tensor(meta["width"]).float(),
        "height": torch.as_tensor(meta["height"]).float(),
        "background_color": background,
    }

    return {
        "gs_params": factor_entry["gs_params"],
        "images": images,
        "images_name": images_name,
        "cameras": cameras,
        "scene_idx": scene_idx,
        "scene_name": scene_name,
    }


def evaluate_single_scene(
    model,
    input_gs,
    source_flow_gs,
    scene_idx,
    scene_name,
    eval_images,
    eval_cameras,
    image_names,
    output_dir,
    flow_steps,
    flow_space,
    eval_chunk_size=None,
    gt_gs=None,
    compare_with_input=False,
    save_viewer=True,
    save_residuals=True,
    output_gt=True,
):
    metric_computer = MetricComputer()
    metric_computer_input = MetricComputer() if compare_with_input else None
    device = next(model.parameters()).device
    num_views = len(eval_images)
    if num_views == 0:
        raise ValueError("Evaluation payload has zero views")

    if eval_chunk_size is None or eval_chunk_size <= 0:
        eval_chunk_size = num_views
    eval_chunk_size = min(eval_chunk_size, num_views)

    os.makedirs(output_dir, exist_ok=True)
    residual_dir = None
    if save_residuals:
        residual_dir = os.path.join(output_dir, "residuals")
        os.makedirs(residual_dir, exist_ok=True)

    pred_single_dir = os.path.join(output_dir, f"pred/{scene_name}")
    os.makedirs(pred_single_dir, exist_ok=True)

    compare_dir = None
    if compare_with_input:
        compare_dir = os.path.join(output_dir, f"compare/{scene_name}")
        os.makedirs(compare_dir, exist_ok=True)

    with torch.no_grad():
        out_gs = sample_flow_model(model, source_flow_gs, scene_idx, flow_steps, flow_space)
        pred_preview = []
        gt_preview = []

        for start in range(0, num_views, eval_chunk_size):
            end = min(start + eval_chunk_size, num_views)
            chunk_name = f"{scene_idx}_{start:06d}"

            chunk_images = gpu_utils.move_to_device(eval_images[start:end], device)
            chunk_cameras = {
                key: (value[start:end] if key == "camera_to_worlds" else value)
                for key, value in eval_cameras.items()
            }
            chunk_cameras = gpu_utils.move_to_device(chunk_cameras, device)

            pred_imgs, _ = gs_utils.rasterize_gaussians_to_multiimgs(out_gs, chunk_cameras)
            pred_imgs = torch.stack(pred_imgs, dim=0)
            gt_imgs = torch.stack(chunk_images, dim=0)

            if gt_imgs.shape[-1] == 4:
                masks = gt_imgs[..., 3].unsqueeze(-1)
                pred_imgs = pred_imgs * masks
                gt_imgs = (gt_imgs[..., :3] * 255).to(torch.uint8)
                pred_imgs = (pred_imgs * 255).to(torch.uint8)
            else:
                masks = None
                gt_imgs = (gt_imgs * 255).to(torch.uint8)
                pred_imgs = (pred_imgs * 255).to(torch.uint8)

            preview_slots = 9 - len(pred_preview)
            if preview_slots > 0:
                pred_preview.extend([im.cpu().numpy().astype(np.uint8) for im in pred_imgs[:preview_slots]])
                if output_gt:
                    gt_preview.extend([im.cpu().numpy().astype(np.uint8) for im in gt_imgs[:preview_slots]])

            metric_computer.update(pred_imgs, gt_imgs, name=chunk_name)

            for name, pred_img in zip(image_names[start:end], pred_imgs):
                pred_img = pred_img.cpu().numpy().astype(np.uint8)
                cv2.imwrite(os.path.join(pred_single_dir, name), pred_img[:, :, ::-1])

            if compare_with_input:
                input_imgs, _ = gs_utils.rasterize_gaussians_to_multiimgs(input_gs, chunk_cameras)
                input_imgs = torch.stack(input_imgs, dim=0)
                if masks is not None:
                    input_imgs = input_imgs * masks
                    input_imgs = (input_imgs * 255).to(torch.uint8)
                else:
                    input_imgs = (input_imgs * 255).to(torch.uint8)
                metric_computer_input.update(input_imgs, gt_imgs, name=chunk_name)

                for global_idx, (gt_img, input_img, pred_img) in enumerate(
                    zip(gt_imgs, input_imgs, pred_imgs), start=start
                ):
                    gt_img = gt_img.cpu().numpy().astype(np.uint8)
                    input_img = input_img.cpu().numpy().astype(np.uint8)
                    pred_img = pred_img.cpu().numpy().astype(np.uint8)
                    cmp_img = np.concatenate([gt_img, input_img, pred_img], axis=1)
                    cv2.imwrite(os.path.join(compare_dir, f"{global_idx:04d}.png"), cmp_img[:, :, ::-1])

        if len(pred_preview) > 0:
            pred_grid = cv2.cvtColor(make_grid(pred_preview), cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(output_dir, f"scene{scene_idx}_pred.png"), pred_grid)

        if output_gt and len(gt_preview) > 0:
            gt_grid = cv2.cvtColor(make_grid(gt_preview), cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(output_dir, f"scene{scene_idx}_gt.png"), gt_grid)

        if save_viewer:
            viewerdir = os.path.join(output_dir, f"viewer/{scene_name}")
            os.makedirs(viewerdir, exist_ok=True)
            gs_utils.prepare_viewer(eval_cameras, viewerdir, model.sh_degree)
            gs_utils.export_ply_forviewer(input_gs, os.path.join(viewerdir, "point_cloud/input.ply"))
            gs_utils.export_ply_forviewer(out_gs, os.path.join(viewerdir, "point_cloud/output.ply"))
            if gt_gs is not None:
                gs_utils.export_ply_forviewer(gt_gs, os.path.join(viewerdir, "point_cloud/gt.ply"))

        if save_residuals:
            residual_keys = sorted([key for key in out_gs.keys() if key in input_gs])
            residuals = {}
            residual_stats = {}
            for key in residual_keys:
                residual = out_gs[key] - input_gs[key]
                residuals[key] = residual
                residual_stats[key] = {
                    "mean": float(residual.mean().item()),
                    "abs_mean": float(residual.abs().mean().item()),
                }

            scene_stem = f"{int(scene_idx)}_{_sanitize_for_filename(scene_name)}"
            pt_payload = {
                "scene_idx": int(scene_idx),
                "scene_name": scene_name,
                "residual_type": "sampled_out_minus_input",
                "flow_steps": int(flow_steps),
                "flow_space": flow_space,
                "residual_keys": residual_keys,
                "residuals": _to_cpu(residuals),
                "input_gs": _to_cpu(input_gs),
                "output_gs": _to_cpu(out_gs),
                "target_gs": _to_cpu(gt_gs) if gt_gs is not None else None,
                "cameras": _to_cpu(eval_cameras),
            }
            torch.save(pt_payload, os.path.join(residual_dir, f"{scene_stem}.pt"))

            stats_payload = {
                "scene_idx": int(scene_idx),
                "scene_name": scene_name,
                "num_gaussians": int(input_gs["means"].shape[0]),
                "residual_type": "sampled_out_minus_input",
                "flow_steps": int(flow_steps),
                "flow_space": flow_space,
                "residual_keys": residual_keys,
                "residual_stats": residual_stats,
            }
            with open(os.path.join(residual_dir, f"{scene_stem}.json"), "w") as f:
                json.dump(stats_payload, f, indent=2)

    metrics = metric_computer.finalize()
    metric_computer.write_to_file(os.path.join(output_dir, "metrics.json"))

    if compare_with_input:
        metrics_input = metric_computer_input.finalize()
        metric_computer_input.write_to_file(os.path.join(output_dir, "metrics_input.json"))
    else:
        metrics_input = {}

    model.train()
    return metrics, metrics_input


def _resolve_flow_cfg():
    cfg = flow_matching()
    if FLAGS.flow_steps is not None:
        cfg["flow_steps"] = FLAGS.flow_steps
    if FLAGS.flow_space is not None:
        cfg["flow_space"] = FLAGS.flow_space
    if FLAGS.flow_noise_std is not None:
        cfg["flow_noise_std"] = FLAGS.flow_noise_std
    if cfg["flow_space"] not in FLOW_SPACES:
        raise ValueError(f"Unsupported flow_space={cfg['flow_space']}; expected one of {sorted(FLOW_SPACES)}")
    return cfg


def main(argv):
    del argv
    os.makedirs(FLAGS.output_dir, exist_ok=True)

    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)
    train_cfg = training(output_dir=FLAGS.output_dir)
    flow_cfg = _resolve_flow_cfg()
    set_seed()

    with open(os.path.join(FLAGS.output_dir, "config.gin"), "w") as f:
        f.writelines(gin.operative_config_str())

    logger = ProcessSafeLogger(os.path.join(FLAGS.output_dir, "overfit.log")).get_logger()
    device = torch.device("cuda")

    dataset = _build_dataset()
    scene_idx = _find_scene_index(dataset, FLAGS.scene_name)
    scene = dataset.load_scene(scene_idx)
    input_factor_entry = scene["factor_data"][FLAGS.input_factor]
    target_factor_entry = scene["factor_data"][FLAGS.target_factor]

    train_payload = _build_split_payload(dataset, scene["idx"], scene["scene_name"], target_factor_entry, split="train")
    eval_payload = _build_split_payload(dataset, scene["idx"], scene["scene_name"], target_factor_entry, split="test")

    gt_attribute_keys = []
    for flag_name, key in [
        ("gt_features_dc", "features_dc"),
        ("gt_features_rest", "features_rest"),
        ("gt_opacities", "opacities"),
        ("gt_scales", "scales"),
        ("gt_quats", "quats"),
    ]:
        if getattr(FLAGS, flag_name):
            gt_attribute_keys.append(key)

    input_gs_raw = build_densified_input_gs(
        input_factor_entry=input_factor_entry,
        target_factor_entry=target_factor_entry,
        alignment=FLAGS.alignment,
        attribute_init=FLAGS.attribute_init,
        emd_eps=FLAGS.emd_eps,
        emd_iters=FLAGS.emd_iters,
        device=device,
        output_dir=FLAGS.output_dir,
        gt_attribute_keys=gt_attribute_keys,
    )
    target_gs_raw = gpu_utils.move_to_device(target_factor_entry["gs_params"], device)
    source_flow_gs = raw_to_flow_gs(input_gs_raw, flow_cfg["flow_space"])
    target_flow_gs = raw_to_flow_gs(target_gs_raw, flow_cfg["flow_space"])
    batch_scene_idx = [scene["idx"]]

    eval_images = eval_payload["images"]
    eval_cameras = eval_payload["cameras"]
    eval_chunk_size = dataset.image_per_scene if dataset.image_per_scene is not None else len(eval_images)
    if eval_chunk_size <= 0:
        eval_chunk_size = len(eval_images)

    model = GSFlowPredictor().to(device)
    if model.resume_ckpt is not None:
        model.load_state_dict(torch.load(model.resume_ckpt, map_location="cpu"))
    model.train()

    with gin.config_scope("train2D"):
        optimizer = build_optimizer(model)
        scheduler = build_scheduler(optimizer)

    total_steps = train_cfg["total_steps"]
    log_interval = train_cfg["log_interval"]
    log_image_interval = train_cfg["log_image_interval"]
    save_interval = train_cfg["save_interval"]
    eval_interval = train_cfg["eval_interval"]
    grad_clip_norm = train_cfg["grad_clip_norm"]
    resume_from_step = train_cfg["resume_from_step"]
    enable_amp = train_cfg["enable_amp"]
    empty_cache_fre = train_cfg["empty_cache_fre"]

    scaler = torch.cuda.amp.GradScaler(enabled=enable_amp)

    print(
        f"PUFM flow scene={scene['scene_name']} idx={scene['idx']} "
        f"train_views={len(train_payload['images'])} eval_views={len(eval_payload['images'])} "
        f"input_gaussians={input_factor_entry['gs_params']['means'].shape[0]} "
        f"densified_gaussians={input_gs_raw['means'].shape[0]} "
        f"target_gaussians={target_gs_raw['means'].shape[0]} "
        f"input_factor={FLAGS.input_factor} target_factor={FLAGS.target_factor} "
        f"alignment={FLAGS.alignment} attribute_init={FLAGS.attribute_init} "
        f"gt_attributes={','.join(gt_attribute_keys) if gt_attribute_keys else 'none'} "
        f"flow_space={flow_cfg['flow_space']} flow_steps={flow_cfg['flow_steps']} "
        f"flow_noise_std={flow_cfg['flow_noise_std']} "
        f"flow_loss_weight={flow_cfg['flow_loss_weight']} "
        f"render_loss_weight={flow_cfg['render_loss_weight']} "
        f"flow_loss_grad_keys={','.join(flow_cfg['flow_loss_grad_keys'])}"
    )

    os.makedirs(os.path.join(FLAGS.output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(FLAGS.output_dir, "checkpoints"), exist_ok=True)

    train_images_device = gpu_utils.move_to_device(train_payload["images"], device)
    train_cameras_device = gpu_utils.move_to_device(train_payload["cameras"], device)
    gt_imgs_uint8 = [(img[..., :3] * 255).detach().cpu().numpy().astype(np.uint8) for img in train_images_device]
    if len(gt_imgs_uint8) > 0:
        gt_grid = cv2.cvtColor(make_grid(gt_imgs_uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(FLAGS.output_dir, "train", "00000000_gt.png"), gt_grid)

    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(range(resume_from_step, total_steps))
    for step in pbar:
        t = torch.rand(1, device=device)
        t = 1.0 - torch.cos(t * torch.pi / 2.0)
        alpha = t.view(1, 1)
        noisy_source_flow = add_flow_noise(source_flow_gs, float(flow_cfg["flow_noise_std"]))
        query_flow_gs = flow_state_like(noisy_source_flow, target_flow_gs, alpha)
        target_vel = {key: target_flow_gs[key] - noisy_source_flow[key] for key in query_flow_gs.keys()}

        with torch.cuda.amp.autocast(enabled=enable_amp):
            pred_vel = model(batch_flow_gs=[query_flow_gs], batch_scene_idx=batch_scene_idx, t=t)[0]
            flow_total_loss, attr_losses = flow_loss(
                pred_vel, target_vel, flow_cfg["loss_weights"], flow_cfg["flow_loss_grad_keys"]
            )
            target_hat_flow_gs = extrapolate_target_flow_gs(query_flow_gs, pred_vel, t)
            target_hat_raw_gs = flow_to_raw_gs(target_hat_flow_gs, flow_cfg["flow_space"])
            render_loss, pred_imgs_for_log = render_l1_loss(target_hat_raw_gs, train_cameras_device, train_images_device)
            total_loss = (
                float(flow_cfg["flow_loss_weight"]) * flow_total_loss
                + float(flow_cfg["render_loss_weight"]) * render_loss
            )

        if enable_amp:
            scaler.scale(total_loss).backward()
            if grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        postfix = {
            "loss": f"{total_loss.item():.4f}",
            "flow": f"{flow_total_loss.item():.4f}",
            "render": f"{render_loss.item():.4f}",
            "t": f"{t.item():.3f}",
            "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
        }
        for key in ["means", "features_dc", "features_rest", "opacities", "scales", "quats"]:
            if key in attr_losses:
                postfix[key] = f"{attr_losses[key].item():.3e}"
        pbar.set_postfix(postfix)

        if empty_cache_fre > 0 and (step + 1) % empty_cache_fre == 0:
            torch.cuda.empty_cache()

        if step % log_interval == 0:
            attr_str = " ".join([f"{key}={value.item():.6f}" for key, value in attr_losses.items()])
            logger.info(
                f"step={step} total={total_loss.item():.6f} "
                f"flow={flow_total_loss.item():.6f} render={render_loss.item():.6f} "
                f"flow_w={float(flow_cfg['flow_loss_weight']):.6f} "
                f"render_w={float(flow_cfg['render_loss_weight']):.6f} "
                f"fm_grad={','.join(flow_cfg['flow_loss_grad_keys'])} t={t.item():.6f} "
                f"lr={optimizer.param_groups[0]['lr']:.8f} {attr_str}"
            )

        if step % log_image_interval == 0:
            with torch.no_grad():
                train_out_gs = sample_flow_model(
                    model, source_flow_gs, scene["idx"], int(flow_cfg["flow_steps"]), flow_cfg["flow_space"]
                )
                pred_imgs, _ = gs_utils.rasterize_gaussians_to_multiimgs(train_out_gs, train_cameras_device)
                pred_imgs_uint8 = [(im * 255).detach().cpu().numpy().astype(np.uint8) for im in pred_imgs[:9]]
                if len(pred_imgs_uint8) > 0:
                    pred_grid = cv2.cvtColor(make_grid(pred_imgs_uint8), cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(FLAGS.output_dir, "train", f"{step:08d}_pred.png"), pred_grid)

        if step % eval_interval == 0:
            eval_dir = os.path.join(FLAGS.output_dir, "eval", f"{step:08d}")
            metrics, metrics_input = evaluate_single_scene(
                model=model,
                input_gs=input_gs_raw,
                source_flow_gs=source_flow_gs,
                gt_gs=target_gs_raw,
                scene_idx=eval_payload["scene_idx"],
                scene_name=eval_payload["scene_name"],
                eval_images=eval_images,
                eval_cameras=eval_cameras,
                image_names=eval_payload["images_name"],
                output_dir=eval_dir,
                flow_steps=int(flow_cfg["flow_steps"]),
                flow_space=flow_cfg["flow_space"],
                eval_chunk_size=eval_chunk_size,
                compare_with_input=FLAGS.compare_with_input,
                save_viewer=FLAGS.save_viewer,
                save_residuals=FLAGS.save_residuals,
                output_gt=(step == 0),
            )
            metric_str = " ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            logger.info(f"Eval step {step}: {metric_str}")
            if FLAGS.compare_with_input:
                metric_str = " ".join([f"{k}: {v:.4f}" for k, v in metrics_input.items()])
                logger.info(f"Eval input step {step}: {metric_str}")

        if (step + 1) % save_interval == 0:
            torch.save(model.state_dict(), os.path.join(FLAGS.output_dir, "checkpoints", f"model_{step:08d}.pth"))

    torch.save(model.state_dict(), os.path.join(FLAGS.output_dir, "checkpoints", "model_last.pth"))

    final_eval_dir = os.path.join(FLAGS.output_dir, FLAGS.eval_subdir)
    metrics, metrics_input = evaluate_single_scene(
        model=model,
        input_gs=input_gs_raw,
        source_flow_gs=source_flow_gs,
        gt_gs=target_gs_raw,
        scene_idx=eval_payload["scene_idx"],
        scene_name=eval_payload["scene_name"],
        eval_images=eval_images,
        eval_cameras=eval_cameras,
        image_names=eval_payload["images_name"],
        output_dir=final_eval_dir,
        flow_steps=int(flow_cfg["flow_steps"]),
        flow_space=flow_cfg["flow_space"],
        eval_chunk_size=eval_chunk_size,
        compare_with_input=FLAGS.compare_with_input,
        save_viewer=FLAGS.save_viewer,
        save_residuals=FLAGS.save_residuals,
        output_gt=True,
    )

    metric_str = " ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    logger.info(f"Final eval: {metric_str}")
    if FLAGS.compare_with_input:
        metric_str = " ".join([f"{k}: {v:.4f}" for k, v in metrics_input.items()])
        logger.info(f"Final eval input: {metric_str}")


if __name__ == "__main__":
    app.run(main)
