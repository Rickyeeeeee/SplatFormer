import hashlib
import json
import os
import random
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

import cv2
import gin
import numpy as np
import torch
import torch.distributed as dist
from absl import app, flags
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None

from dataset.GS_SR import SplatFactoSRDataset
from models.feature_flow_predictor import GSFlowPredictor
from models.feature_predictor import FeaturePredictor  # Registers legacy Gin keys.
from sr import flow
from sr.alignment import prepare_alignment
from utils import gpu_utils, gs_utils, loss_utils
from utils.gpu_utils import seed_everything
from utils.gs_utils import make_grid
from utils.log_utils import ProcessSafeLogger
from utils.loss_utils import SUPPORTED_GS_KEYS
from utils.metrics import MetricComputer
from utils.optimizers import build_optimizer, build_scheduler


flags.DEFINE_string("output_dir", "output_sr_gsfm", "Output directory")
flags.DEFINE_integer("batch_size", 1, "Scene samples per GPU per optimizer update")
flags.DEFINE_integer("grad_accum_steps", 1, "Forward/backward passes splitting each GPU's batch")
flags.register_validator("batch_size", lambda value: value >= 1, message="batch_size must be at least 1")
flags.register_multi_flags_validator(
    ["batch_size", "grad_accum_steps"],
    lambda values: 1 <= values["grad_accum_steps"] <= values["batch_size"],
    message="grad_accum_steps must be between 1 and batch_size",
)
flags.DEFINE_integer("num_workers", 2, "CPU DataLoader workers per GPU; 0 loads synchronously")
flags.DEFINE_integer("prefetch_factor", 2, "Microbatches prefetched per worker")
flags.DEFINE_boolean("pin_memory", True, "Pin training tensors for asynchronous GPU transfer")
flags.register_validator("num_workers", lambda value: value >= 0, message="num_workers must be nonnegative")
flags.register_validator("prefetch_factor", lambda value: value >= 1, message="prefetch_factor must be positive")
flags.DEFINE_boolean("only_eval", False, "Only run evaluation")
flags.DEFINE_boolean("save_residuals", False, "Accepted for trainer compatibility")
flags.DEFINE_boolean("use_wandb", True, "Log metrics to Weights & Biases")
flags.DEFINE_string("wandb_project", "3dgs-super-resolution", "Weights & Biases project")
flags.DEFINE_string("wandb_dir", None, "Weights & Biases output directory")
flags.DEFINE_string("wandb_name", None, "Weights & Biases run name")
flags.DEFINE_string("eval_subdir", "eval_final", "Eval subdirectory")
flags.DEFINE_integer("eval_seed", 0, "Seed for fixed evaluation scenes, times, noise, and views")
flags.DEFINE_boolean("compare_with_input", False, "Compare with input 3DGS")
flags.DEFINE_boolean("save_viewer", True, "Save viewer point clouds")
flags.DEFINE_enum("alignment", "emd", ["emd", "random", "fit_lr_to_hr", "fit_hr_to_lr"], "Matching modes")
flags.DEFINE_enum("attribute_init", "aligned", ["aligned", "3dgs"], "attribute initialization for emd and random.",)
flags.DEFINE_float("emd_eps", 0.01, "Auction EMD epsilon")
flags.DEFINE_integer("emd_iters", 100, "Auction EMD iterations")
flags.DEFINE_multi_string("gin_file", None, "List of paths to the config files.")
flags.DEFINE_multi_string("gin_param", "", "Newline separated list of Gin parameter bindings.")

FLAGS = flags.FLAGS
EVAL_FLOW_STEPS = [10]
MEANS_LOSS_REDUCTION = "mean"  # Set to "sum" to match PUFM-style summed point loss.

torch.set_num_threads(2)

@gin.configurable
def set_seed(seed, rank=0):
    seed_everything(seed + rank)
    return seed + rank


@gin.configurable
def flow_matching(
    gs_statistics_path="/project/ricky/splatformer-sr-data/gs_statistics.json",
    flow_steps=5,
    flow_noise_std=0.0,
    flow_t_eps=1e-4,
    loss_type="velocity",
    velocity_variance_floor=1e-8,
):
    if loss_type not in ("velocity", "x1"):
        raise ValueError(
            f"Unsupported flow_matching.loss_type={loss_type!r}; expected 'velocity' or 'x1'"
        )
    if float(velocity_variance_floor) <= 0.0:
        raise ValueError(
            "flow_matching.velocity_variance_floor must be positive, "
            f"got {velocity_variance_floor}"
        )
    return {
        "gs_statistics_path": gs_statistics_path,
        "flow_steps": flow_steps,
        "flow_noise_std": flow_noise_std,
        "flow_t_eps": flow_t_eps,
        "loss_type": loss_type,
        "velocity_variance_floor": float(velocity_variance_floor),
    }

@gin.configurable
def loss_mixing(schedule="linear"):
    valid_schedules = ("linear", "free-range-gs", "fm-only")
    if schedule not in valid_schedules:
        raise ValueError(
            f"Unsupported loss_mixing.schedule={schedule!r}; "
            f"expected one of {valid_schedules}"
        )
    return {"schedule": schedule}


@gin.configurable
def feature_mse_loss(loss_weights=None, quat_direct_mse=False):
    default_weights = {key: 1.0 for key in SUPPORTED_GS_KEYS}
    if loss_weights is None:
        loss_weights = {}
    unknown_keys = sorted(set(loss_weights) - set(SUPPORTED_GS_KEYS))
    if unknown_keys:
        raise ValueError(
            f"Unsupported feature_mse_loss.loss_weights keys: {unknown_keys}. "
            f"Supported keys are: {SUPPORTED_GS_KEYS}"
        )
    default_weights.update(
        {key: float(value) for key, value in loss_weights.items()}
    )
    return {
        "loss_weights": default_weights,
        "quat_direct_mse": quat_direct_mse,
    }


def compute_all_feature_mse_loss(
    out_gs,
    target_gs,
    loss_weights,
    quat_direct_mse,
):
    missing_output = [
        key for key in SUPPORTED_GS_KEYS if key not in out_gs
    ]
    missing_target = [
        key for key in SUPPORTED_GS_KEYS if key not in target_gs
    ]
    if missing_output:
        raise ValueError(
            f"GSFlowPredictor must output every Gaussian attribute; "
            f"missing {missing_output}"
        )
    if missing_target:
        raise ValueError(
            f"All-attribute flow target is missing {missing_target}"
        )

    losses = {}
    weighted_losses = {}
    total_loss = None
    for key in SUPPORTED_GS_KEYS:
        pred = out_gs[key]
        target = target_gs[key].to(
            device=pred.device, dtype=pred.dtype
        )
        if pred.shape != target.shape:
            raise ValueError(
                f"Shape mismatch for Gaussian attribute {key!r}: "
                f"output {tuple(pred.shape)} vs target {tuple(target.shape)}"
            )
        loss = loss_utils.feature_loss_value(
            key,
            pred,
            target,
            post_activate_loss=True,
            quat_direct_mse=quat_direct_mse,
            means_loss_reduction=MEANS_LOSS_REDUCTION,
        )
        weighted_loss = float(loss_weights[key]) * loss
        losses[key] = loss
        weighted_losses[key] = weighted_loss
        total_loss = (
            weighted_loss if total_loss is None else total_loss + weighted_loss
        )

    if torch.is_grad_enabled() and not total_loss.requires_grad:
        raise ValueError(
            "GSFlowPredictor outputs do not receive gradients for the "
            "all-attribute x1 loss"
        )
    return total_loss, losses, weighted_losses


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
    eval_chunk_size=None,
    gt_gs=None,
    compare_with_input=False,
    save_viewer=True,
    output_gt=True,
):
    was_training = model.training
    metric_computer = MetricComputer()
    metric_computer_input = MetricComputer() if compare_with_input else None
    device = next(model.parameters()).device
    num_views = len(eval_images)
    if num_views == 0:
        raise ValueError("Evaluation has zero views")

    if eval_chunk_size is None or eval_chunk_size <= 0:
        eval_chunk_size = num_views
    eval_chunk_size = min(eval_chunk_size, num_views)

    os.makedirs(output_dir, exist_ok=True)

    pred_single_dir = os.path.join(output_dir, f"pred/{scene_name}")
    os.makedirs(pred_single_dir, exist_ok=True)

    compare_dir = None
    if compare_with_input:
        compare_dir = os.path.join(output_dir, f"compare/{scene_name}")
        os.makedirs(compare_dir, exist_ok=True)

    with torch.no_grad():
        out_gs = flow.sample_flow_model(model, source_flow_gs, scene_idx, flow_steps)
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

    metrics = metric_computer.finalize()
    metric_computer.write_to_file(os.path.join(output_dir, "metrics.json"))

    if compare_with_input:
        metrics_input = metric_computer_input.finalize()
        metric_computer_input.write_to_file(os.path.join(output_dir, "metrics_input.json"))
    else:
        metrics_input = {}

    model.train(was_training)
    return metrics, metrics_input



def init_wandb(output_dir):
    if not FLAGS.use_wandb or (dist.is_initialized() and dist.get_rank() != 0):
        return None
    if wandb is None:
        raise ImportError("wandb is not installed")
    return wandb.init(project=FLAGS.wandb_project, dir=FLAGS.wandb_dir, name=FLAGS.wandb_name)


def wandb_log(values, step=None):
    if dist.is_initialized() and dist.get_rank() != 0:
        return
    if wandb is not None and wandb.run is not None:
        wandb.log(values, step=step)


def build_dataset(scope, alignment=None, **kwargs):
    return SplatFactoSRDataset.from_gin_scope(scope, alignment=alignment, **kwargs)


class SRSceneDataset(torch.utils.data.Dataset):
    """Load explicitly assigned scenes without invoking the iterable dataset's sharding."""

    def __init__(self, dataset):
        self.dataset = dataset
        self.gin_config = gin.config_str()

    def __len__(self):
        return len(self.dataset.folders)

    def __getitem__(self, scene_idx):
        try:
            return self.dataset.load_scene(scene_idx, sample_views=True, fit_alignment=self.dataset.alignment)
        except Exception as error:
            raise RuntimeError(f"Failed to load scene {scene_idx}: {self.dataset.folders[scene_idx]}") from error


class SceneMicrobatchSampler(torch.utils.data.Sampler):
    """Generate rank-local microbatches continuously, including across epoch boundaries."""

    def __init__(self, scene_count, microbatch_sizes, rank=0, world_size=1, split_across_gpus=True, seed=0):
        if scene_count < 1:
            raise ValueError("Training dataset contains no scenes")
        if not microbatch_sizes or min(microbatch_sizes) < 1:
            raise ValueError("Microbatch sizes must be positive")
        if not 0 <= rank < world_size:
            raise ValueError("Invalid rank/world_size")
        self.scene_count = scene_count
        self.microbatch_sizes = microbatch_sizes
        self.rank = rank
        self.world_size = world_size
        self.split_across_gpus = split_across_gpus
        self.seed = seed

    def __iter__(self):
        epoch = 0
        remaining = iter(())
        rng = random.Random(self.seed)
        while True:
            for size in self.microbatch_sizes:
                batch = []
                while len(batch) < size:
                    scene_idx = next(remaining, None)
                    if scene_idx is None:
                        if self.split_across_gpus:
                            order = np.random.RandomState(epoch).permutation(self.scene_count)
                            per_rank = (self.scene_count + self.world_size - 1) // self.world_size
                            order = np.resize(order, per_rank * self.world_size)
                            order = order[self.rank * per_rank:(self.rank + 1) * per_rank].tolist()
                        else:
                            order = list(range(self.scene_count))
                            rng.shuffle(order)
                        remaining = iter(order)
                        epoch += 1
                        scene_idx = next(remaining)
                    batch.append(scene_idx)
                yield batch


def collate_scenes(scenes):
    # Scenes have variable Gaussian counts, so retain a list instead of stacking.
    return scenes


def seed_loader_worker(worker_id):
    del worker_id
    # Spawned workers need bindings used inside load_scene, such as MinMaxScaler.
    with gin.unlock_config():
        gin.parse_config(torch.utils.data.get_worker_info().dataset.gin_config)
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)


def build_train_loader(dataset, microbatch_sizes, seed, rank=0, world_size=1):
    sampler = SceneMicrobatchSampler(len(dataset.folders), microbatch_sizes, rank, world_size, dataset.split_across_gpus, seed)
    worker_options = {}
    if FLAGS.num_workers > 0:
        worker_options = dict(multiprocessing_context="spawn", persistent_workers=True, prefetch_factor=FLAGS.prefetch_factor)
    return torch.utils.data.DataLoader(
        SRSceneDataset(dataset), batch_sampler=sampler, collate_fn=collate_scenes,
        num_workers=FLAGS.num_workers, pin_memory=FLAGS.pin_memory, worker_init_fn=seed_loader_worker,
        generator=torch.Generator().manual_seed(seed), **worker_options,
    )


@contextmanager
def training_microbatches(loader):
    iterator = iter(loader)
    try:
        yield iterator
    finally:
        if loader.num_workers > 0:
            # DataLoader has no public close API for persistent worker iterators.
            iterator._shutdown_workers()


def move_training_data(data, device, non_blocking=False):
    if torch.is_tensor(data):
        return data.to(device, non_blocking=non_blocking)
    if isinstance(data, dict):
        return {key: move_training_data(value, device, non_blocking) for key, value in data.items()}
    if isinstance(data, (list, tuple)):
        return type(data)(move_training_data(value, device, non_blocking) for value in data)
    return data


def build_gaussian_pair(dataset, scene, device, non_blocking=False):
    input_data = scene["data"][dataset.src_resolution]
    target_data = scene["data"][dataset.tgt_resolution]
    # Select the fitted target before transferring to avoid an unused HR allocation.
    target_params = scene["fit_lr_to_hr"]["tgt_gs"] if FLAGS.alignment == "fit_lr_to_hr" else target_data["gs_params"]
    target_gs = move_training_data(target_params, device, non_blocking)
    if FLAGS.alignment == "fit_lr_to_hr":
        source_gs = move_training_data(input_data["gs_params"], device, non_blocking)
    elif FLAGS.alignment == "fit_hr_to_lr":
        source_gs = move_training_data(scene[FLAGS.alignment]["tgt_gs"], device, non_blocking)
    else:
        source_gs, target_gs, _ = prepare_alignment(
            dataset=dataset, scene=scene, input_resolution_entry=input_data,
            target_resolution_entry=target_data, target_images=None,
            target_cameras=None, target_gs=target_gs, output_dir="", logger=None,
            device=device, eval_chunk_size=0, alignment=FLAGS.alignment,
            attribute_init=FLAGS.attribute_init, emd_eps=FLAGS.emd_eps,
            emd_iters=FLAGS.emd_iters, input_resolution=dataset.src_resolution,
            target_resolution=dataset.tgt_resolution, write_artifacts=False,
        )
    return target_data, source_gs, target_gs


def select_train_eval_scenes(dataset, output_dir):
    """Persist scene names so evaluation survives changes in manifest ordering."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    payload = [None]
    if rank == 0:
        try:
            path = Path(output_dir) / "train_eval_scenes.json"
            available = sorted({scene["scene_name"] for scene in dataset.folders})
            if path.exists():
                selection = json.loads(path.read_text())
            else:
                selection = {"seed": FLAGS.eval_seed, "scene_names": random.Random(FLAGS.eval_seed).sample(available, min(5, len(available)))}
            names = selection["scene_names"]
            missing = sorted(set(names) - set(available))
            if missing:
                raise ValueError(f"Saved training evaluation scenes are unavailable: {missing}")
            if len(names) != min(5, len(available)) or len(set(names)) != len(names):
                raise ValueError("Training evaluation selection must contain five unique scenes (or all available scenes if fewer)")
            if not path.exists():
                path.write_text(json.dumps(selection, indent=2))
            payload[0] = {"scene_names": names}
        except (OSError, ValueError, KeyError, TypeError) as error:
            payload[0] = {"error": str(error)}
    if dist.is_initialized():
        dist.broadcast_object_list(payload, src=0)
    if "error" in payload[0]:
        raise ValueError(payload[0]["error"])
    return [dataset.scene_index(name) for name in payload[0]["scene_names"]]


@contextmanager
def evaluation_rng(device, split, scene_name):
    """Keep scene sampling independent of rank, evaluation order, and training RNG."""
    seed = int.from_bytes(hashlib.sha256(f"{FLAGS.eval_seed}:{split}:{scene_name}".encode()).digest()[:4], "little")
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = [device.index] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(seed)
            np.random.seed(seed)
            torch.random.default_generator.manual_seed(seed)
            if devices:
                torch.cuda.default_generators[device.index].manual_seed(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def evaluate_dataset(model, dataset, output_dir, *, scene_indices=None, split="test",
                     flow_config=None, mix_config=None, mse_config=None, config=None,
                     velocity_variances=None, lpips_function=None, loss_view_count=None):
    # Uneven evaluation shards must not run forwards through DDP.
    model = model.module if isinstance(model, DDP) else model
    was_training = model.training
    model.eval()
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    device = next(model.parameters()).device
    indices = list(range(len(dataset.folders))) if scene_indices is None else list(scene_indices)
    results = {flow_steps: [] for flow_steps in EVAL_FLOW_STEPS}
    losses = []
    try:
        os.makedirs(output_dir, exist_ok=True)
        if dist.is_initialized():
            for buffer in model.buffers():
                dist.broadcast(buffer, src=0)
        with torch.no_grad():
            for scene_idx in tqdm(indices[rank::world_size], desc=f"Evaluating {split}", disable=rank != 0):
                scene_name = dataset.folders[scene_idx]["scene_name"]
                with evaluation_rng(device, split, scene_name):
                    scene = dataset.load_scene(scene_idx, sample_views=False, fit_alignment=FLAGS.alignment)
                    target_data, source_gs, target_gs = build_gaussian_pair(dataset, scene, device)
                    eval_chunk_size = dataset.image_per_scene or len(target_data["images"])
                    if eval_chunk_size <= 0:
                        eval_chunk_size = len(target_data["images"])
                    if flow_config is not None:
                        sample = prepare_scene_sample(source_gs, target_gs, target_data, scene_idx, loss_view_count,
                                                      device, flow_config, mix_config)
                        loss, statistics = compute_microbatch_loss(
                            model, [sample], flow_config, mix_config, mse_config, config, velocity_variances, lpips_function,
                        )
                        losses.append((scene_idx, scene_name, {key: value.item() for key, value in statistics.items()}))
                        del sample, loss, statistics
                    for flow_steps in EVAL_FLOW_STEPS:
                        metrics, _ = evaluate_single_scene(
                            model=model, input_gs=source_gs, source_flow_gs=source_gs,
                            scene_idx=scene["scene_idx"], scene_name=scene["scene_name"],
                            eval_images=target_data["images"], eval_cameras=target_data["cameras"],
                            image_names=target_data["images_name"],
                            output_dir=os.path.join(output_dir, f"flow_steps_{flow_steps:02d}", scene["scene_name"]),
                            flow_steps=flow_steps, eval_chunk_size=eval_chunk_size,
                            gt_gs=target_gs, compare_with_input=FLAGS.compare_with_input,
                            save_viewer=FLAGS.save_viewer, output_gt=True,
                        )
                        results[flow_steps].append((scene_idx, metrics))
                    del scene, target_data, source_gs, target_gs
        if dist.is_initialized():
            gathered = [None for _ in range(world_size)]
            dist.all_gather_object(gathered, (results, losses))
            results = {flow_steps: [item for metrics, _ in gathered for item in metrics[flow_steps]] for flow_steps in EVAL_FLOW_STEPS}
            losses = [item for _, scene_losses in gathered for item in scene_losses]
        results = {flow_steps: [metrics for _, metrics in sorted(values)] for flow_steps, values in results.items()}
        reduced = {
            flow_steps: {key: float(np.mean([metrics[key] for metrics in values])) for key in values[0]}
            for flow_steps, values in results.items() if values
        }
        losses.sort(key=lambda item: item[0])
        reduced_losses = {key: float(np.mean([values[key] for _, _, values in losses])) for key in losses[0][2]} if losses else {}
        if rank == 0:
            with open(os.path.join(output_dir, "metrics.json"), "w") as output_file:
                json.dump(reduced, output_file, indent=2)
            if flow_config is not None:
                with open(os.path.join(output_dir, "losses.json"), "w") as output_file:
                    json.dump({"mean": reduced_losses, "scenes": [{"scene_idx": index, "scene_name": name, **values} for index, name, values in losses]}, output_file, indent=2)
        return reduced, reduced_losses
    finally:
        model.train(was_training)


def evaluate_sets(model, test_dataset, train_eval_dataset, train_eval_indices, output_dir, step,
                  flow_config, mix_config, mse_config, config, velocity_variances, lpips_function):
    values = {}
    for dataset, indices, split, directory, prefix in (
        (test_dataset, None, "test", output_dir, "eval"),
        (train_eval_dataset, train_eval_indices, "train", os.path.join(output_dir, "train"), "eval_train"),
    ):
        metrics, losses = evaluate_dataset(
            model, dataset, directory, scene_indices=indices, split=split,
            flow_config=flow_config, mix_config=mix_config, mse_config=mse_config, config=config,
            velocity_variances=velocity_variances, lpips_function=lpips_function,
            loss_view_count=train_eval_dataset.image_per_scene,
        )
        for flow_steps, step_metrics in metrics.items():
            values.update({f"{prefix}/{flow_steps}/{key}": value for key, value in step_metrics.items()})
        values.update({f"{prefix}/loss/{key}": value for key, value in losses.items()})
    wandb_log(values, step=step)


def prepare_scene_sample(source_flow_gs, target_flow_gs, target_data, scene_idx, view_count,
                         device, flow_config, mix_config, non_blocking=False):
    """Sample a training objective from an already transferred, aligned pair."""
    time_value = torch.empty(1, device=device).uniform_(float(flow_config["flow_t_eps"]), 1.0 - float(flow_config["flow_t_eps"]))
    train_images, train_cameras = None, None
    if mix_config["schedule"] != "fm-only":
        view_count = min(view_count or len(target_data["images"]), len(target_data["images"]))
        indices = np.random.permutation(len(target_data["images"]))[:view_count]
        train_images = move_training_data([target_data["images"][index] for index in indices], device, non_blocking)
        train_cameras = dict(target_data["cameras"])
        train_cameras["camera_to_worlds"] = target_data["cameras"]["camera_to_worlds"][indices]
        train_cameras = move_training_data(train_cameras, device, non_blocking)
    query_gs, flow_noise, gamma, gamma_dot = flow.sample_stochastic_interpolant(source_flow_gs, target_flow_gs, time_value, float(flow_config["flow_noise_std"]))
    return {
        "source": source_flow_gs, "target": target_flow_gs, "query": query_gs,
        "noise": flow_noise, "gamma": gamma, "gamma_dot": gamma_dot, "time": time_value,
        "images": train_images, "cameras": train_cameras, "scene_idx": scene_idx,
    }


def prepare_microbatch(dataset, scenes, device, flow_config, mix_config, non_blocking=False):
    """Transfer and align the current microbatch, then sample interpolants on the GPU."""
    samples = []
    for scene in scenes:
        target_data, source_flow_gs, target_flow_gs = build_gaussian_pair(dataset, scene, device, non_blocking)
        samples.append(prepare_scene_sample(source_flow_gs, target_flow_gs, target_data, scene["scene_idx"],
                                            dataset.image_per_scene, device, flow_config, mix_config, non_blocking))
    return samples


def compute_microbatch_loss(model, samples, flow_config, mix_config, mse_config,
                            config, velocity_variances, lpips_function):
    """Return summed scene losses and detached device statistics."""
    model_module = model.module if isinstance(model, DDP) else model
    needs_target_images = mix_config["schedule"] != "fm-only"
    summed_loss = None
    statistics = {}
    with torch.cuda.amp.autocast(enabled=config["enable_amp"]):
        predictions = model(
            batch_flow_gs=[sample["query"] for sample in samples],
            batch_scene_idx=[sample["scene_idx"] for sample in samples],
            batch_reference_means=[sample["source"]["means"] for sample in samples],
            t=torch.cat([sample["time"] for sample in samples]),
        )
        for sample, predicted_velocity in zip(samples, predictions):
            source_flow_gs, target_flow_gs = sample["source"], sample["target"]
            query_gs, flow_noise = sample["query"], sample["noise"]
            gamma, gamma_dot, time_value = sample["gamma"], sample["gamma_dot"], sample["time"]
            predicted_x1 = flow.predict_x1_from_velocity(model_module, source_flow_gs, query_gs, predicted_velocity, flow_noise, gamma, gamma_dot, time_value)
            if flow_config["loss_type"] == "velocity":
                flow_loss, attribute_losses, weighted_attribute_losses = flow.compute_variance_normalized_velocity_loss(
                    pred_vel=predicted_velocity, source_flow_gs=source_flow_gs,
                    target_flow_gs=target_flow_gs, flow_noise=flow_noise,
                    gamma_dot=gamma_dot, velocity_variances=velocity_variances,
                    loss_weights={key: 1.0 for key in SUPPORTED_GS_KEYS},
                )
            else:
                flow_loss, attribute_losses, weighted_attribute_losses = compute_all_feature_mse_loss(predicted_x1, target_flow_gs, mse_config["loss_weights"], mse_config["quat_direct_mse"])
            if needs_target_images:
                train_images = sample["images"]
                predicted_images, _ = gs_utils.rasterize_gaussians_to_multiimgs(predicted_x1, sample["cameras"])
                render_l1 = sum((prediction - target[..., :3]).abs().mean() for prediction, target in zip(predicted_images, train_images)) / len(predicted_images)
                render_lpips = render_l1.new_zeros(()) if lpips_function is None else sum(lpips_function(prediction.unsqueeze(0), target[..., :3].unsqueeze(0)).mean() for prediction, target in zip(predicted_images, train_images)) / len(predicted_images)
                render_loss = config["image_l1_loss_weight"] * render_l1 + config["lpips_loss_weight"] * render_lpips
            else:
                render_loss = flow_loss.new_zeros(())
            flow_weight, render_weight = flow.loss_mix_weights(time_value, mix_config["schedule"])
            total_loss = flow_weight * flow_loss + render_weight * render_loss
            summed_loss = total_loss if summed_loss is None else summed_loss + total_loss
            values = {"total_loss": total_loss, "flow_loss": flow_loss, "render_loss": render_loss, "time": time_value}
            values.update({f"{key}_loss": value for key, value in attribute_losses.items()})
            values.update({f"{key}_weighted": value for key, value in weighted_attribute_losses.items()})
            for key, value in values.items():
                statistics[key] = statistics.get(key, 0.0) + value.detach().reshape(())
    return summed_loss, statistics


@gin.configurable("training")
def training_config(
    output_dir=None, total_steps=gin.REQUIRED, eval_interval=gin.REQUIRED,
    log_interval=gin.REQUIRED, save_interval=gin.REQUIRED,
    log_image_interval=gin.REQUIRED, grad_clip_norm=gin.REQUIRED,
    image_l1_loss_weight=1.0, lpips_loss_weight=0.0, resume_from_step=0,
    enable_amp=False, empty_cache_fre=-1,
):
    return locals()


def training():
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if distributed:
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank() if distributed else 0
    world_size = dist.get_world_size() if distributed else 1
    os.makedirs(FLAGS.output_dir, exist_ok=True)
    config = training_config(output_dir=FLAGS.output_dir)
    flow_config = flow_matching()
    mix_config = loss_mixing()
    mse_config = feature_mse_loss()
    needs_target_images = mix_config["schedule"] != "fm-only"
    loader_seed = set_seed(rank=rank)
    logger = ProcessSafeLogger(os.path.join(FLAGS.output_dir, "train.log")).get_logger() if rank == 0 else None
    init_wandb(FLAGS.output_dir)
    train_dataset = build_dataset(
        "train_dataset", alignment=FLAGS.alignment, load_src_gs=True,
        load_tgt_gs=True, load_src_images=False,
        load_tgt_images=needs_target_images,
    )
    test_dataset = build_dataset(
        "test_dataset", load_src_gs=True, load_tgt_gs=True,
        load_src_images=True, load_tgt_images=True,
    )
    train_eval_dataset = build_dataset(
        "train_dataset", alignment=FLAGS.alignment, load_src_gs=True,
        load_tgt_gs=True, load_src_images=False, load_tgt_images=True,
    )
    train_eval_indices = select_train_eval_scenes(train_eval_dataset, FLAGS.output_dir)
    model = GSFlowPredictor()
    if distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if model.resume_ckpt is not None:
        model.load_state_dict(torch.load(model.resume_ckpt, map_location="cpu"))
    model = model.to(device)
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
    model_module = model.module if isinstance(model, DDP) else model
    model.train(not FLAGS.only_eval)
    with gin.config_scope("train2D"):
        optimizer = build_optimizer(model_module)
        scheduler = build_scheduler(optimizer)
    velocity_variances = None
    variance_report = {}
    if flow_config["loss_type"] == "velocity":
        raw_variances, velocity_variances = flow.load_aggregate_velocity_variances(
            flow_config["gs_statistics_path"], flow_config["velocity_variance_floor"],
            device=device, dtype=next(model.parameters()).dtype,
        )
        variance_report = {
            key: {"variance": raw_variances[key].detach().cpu().tolist(),
                  "effective_variance": velocity_variances[key].detach().cpu().tolist()}
            for key in SUPPORTED_GS_KEYS
        }
    batch_size = FLAGS.batch_size
    quotient, remainder = divmod(batch_size, FLAGS.grad_accum_steps)
    microbatch_sizes = [quotient + (index < remainder) for index in range(FLAGS.grad_accum_steps)]
    brief = (
        f"world_size={world_size} batch_size={batch_size} global_batch_size={batch_size * world_size} "
        f"grad_accum_steps={FLAGS.grad_accum_steps} microbatch_sizes={microbatch_sizes}\n"
        f"num_workers={FLAGS.num_workers} prefetch_factor={FLAGS.prefetch_factor if FLAGS.num_workers else 0} pin_memory={FLAGS.pin_memory}\n"
        f"Train SR GSFM scenes={len(train_dataset.folders)} test_scenes={len(test_dataset.folders)}\n"
        f"alignment={FLAGS.alignment} mix_schedule={mix_config['schedule']} target_training_images={needs_target_images}\n"
        f"flow_config={flow_config}\nvelocity_variances={json.dumps(variance_report)}"
    )
    if rank == 0:
        print(brief)
        logger.info(brief)
        with open(os.path.join(FLAGS.output_dir, "config.gin"), "w") as output_file:
            output_file.write(gin.operative_config_str())
        os.makedirs(os.path.join(FLAGS.output_dir, "checkpoints"), exist_ok=True)
    if distributed:
        dist.barrier()
    scaler = torch.cuda.amp.GradScaler(enabled=config["enable_amp"])
    lpips_function = loss_utils.lpips_loss_fn() if needs_target_images and config["lpips_loss_weight"] > 0 else None
    if not FLAGS.only_eval:
        optimizer.zero_grad(set_to_none=True)
        loader = build_train_loader(train_dataset, microbatch_sizes, loader_seed, rank, world_size)
        with training_microbatches(loader) as train_iter:
            progress = tqdm(range(config["resume_from_step"], config["total_steps"]), desc="Training", disable=rank != 0)
            timing_start = None
            timed_steps = 0
            data_wait = 0.0
            for train_step in progress:
                if timing_start is None and train_step - config["resume_from_step"] >= 10:
                    torch.cuda.synchronize(device)
                    timing_start = time.perf_counter()
                    timed_steps = 0
                    data_wait = 0.0
                if timing_start is not None:
                    timed_steps += 1
                batch_statistics = {}
                scene_identities = []
                for microbatch_index in range(len(microbatch_sizes)):
                    wait_start = time.perf_counter()
                    scenes = next(train_iter)
                    data_wait += time.perf_counter() - wait_start
                    scene_identities.extend((scene["scene_name"], scene["scene_idx"]) for scene in scenes)
                    # Synchronize accumulated gradients only on the last backward pass.
                    sync_context = model.no_sync() if distributed and microbatch_index < len(microbatch_sizes) - 1 else nullcontext()
                    with sync_context:
                        samples = prepare_microbatch(train_dataset, scenes, device, flow_config, mix_config, FLAGS.pin_memory)
                        microbatch_loss, statistics = compute_microbatch_loss(
                            model, samples, flow_config, mix_config, mse_config, config, velocity_variances, lpips_function,
                        )
                        microbatch_loss = microbatch_loss / batch_size
                        if config["enable_amp"]:
                            scaler.scale(microbatch_loss).backward()
                        else:
                            microbatch_loss.backward()
                    del microbatch_loss, samples, scenes
                    for key, value in statistics.items():
                        batch_statistics[key] = batch_statistics.get(key, 0.0) + value / batch_size

                optimizer_stepped = True
                if config["enable_amp"]:
                    previous_scale = scaler.get_scale()
                    if config["grad_clip_norm"] > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip_norm"])
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer_stepped = scaler.get_scale() >= previous_scale
                else:
                    if config["grad_clip_norm"] > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip_norm"])
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if optimizer_stepped:
                    scheduler.step()
                if train_step % config["log_interval"] == 0:
                    keys = sorted(batch_statistics)
                    reduced = torch.stack([batch_statistics[key] for key in keys]).to(dtype=torch.float64)
                    if distributed:
                        dist.all_reduce(reduced)
                        reduced /= world_size
                    values = {f"train/{key}": value for key, value in zip(keys, reduced.tolist())}
                    if timed_steps:
                        elapsed = time.perf_counter() - timing_start
                        timings = torch.tensor([elapsed, data_wait], device=device, dtype=torch.float64)
                        if distributed:
                            dist.all_reduce(timings, op=dist.ReduceOp.MAX)
                        elapsed, wait = timings.tolist()
                        values.update({"train/seconds_per_update": elapsed / timed_steps,
                                       "train/scenes_per_second": timed_steps * batch_size * world_size / elapsed,
                                       "train/data_wait_seconds": wait / timed_steps})
                    if rank == 0:
                        progress.set_postfix(loss=f"{values['train/total_loss']:.4f}", scene=scene_identities[0][0], lr=f"{optimizer.param_groups[0]['lr']:.2e}")
                        values.update({
                            "train/lr": optimizer.param_groups[0]["lr"], "train/scene_idx": scene_identities[0][1],
                            "train/batch_size": batch_size, "train/global_batch_size": batch_size * world_size,
                            "train/grad_accum_steps": FLAGS.grad_accum_steps,
                        })
                        wandb_log(values, step=train_step)
                        logger.info("step=%d scenes=%s total=%.6f flow=%.6f render=%.6f", train_step, scene_identities, values["train/total_loss"], values["train/flow_loss"], values["train/render_loss"])
                        if timed_steps:
                            logger.info("seconds_per_update=%.4f scenes_per_second=%.4f data_wait_seconds=%.4f", values["train/seconds_per_update"], values["train/scenes_per_second"], values["train/data_wait_seconds"])
                    timing_start = None
                if config["empty_cache_fre"] > 0 and (train_step + 1) % config["empty_cache_fre"] == 0:
                    torch.cuda.empty_cache()
                final_step = train_step == config["total_steps"] - 1
                if train_step % config["eval_interval"] == 0 or final_step:
                    eval_dir = os.path.join(FLAGS.output_dir, FLAGS.eval_subdir if final_step else "eval", "" if final_step else f"{train_step:08d}")
                    evaluate_sets(model, test_dataset, train_eval_dataset, train_eval_indices, eval_dir, train_step,
                                  flow_config, mix_config, mse_config, config, velocity_variances, lpips_function)
                    if distributed:
                        dist.barrier()
                    model.train()
                    timing_start = None
                if (train_step + 1) % config["save_interval"] == 0:
                    if rank == 0:
                        torch.save(model_module.state_dict(), os.path.join(FLAGS.output_dir, "checkpoints", f"model_{train_step:08d}.pth"))
                    if distributed:
                        dist.barrier()
                    timing_start = None
    if FLAGS.only_eval:
        evaluate_sets(model, test_dataset, train_eval_dataset, train_eval_indices,
                      os.path.join(FLAGS.output_dir, FLAGS.eval_subdir), config["resume_from_step"],
                      flow_config, mix_config, mse_config, config, velocity_variances, lpips_function)
    if rank == 0:
        torch.save(model_module.state_dict(), os.path.join(FLAGS.output_dir, "checkpoints", "model_last.pth"))
    if distributed:
        dist.barrier()


def main(argv):
    del argv
    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)
    try:
        training()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        if wandb is not None and wandb.run is not None:
            wandb.run.finish()


if __name__ == "__main__":
    app.run(main)
