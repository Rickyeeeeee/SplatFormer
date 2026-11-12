"""Distributed unconditional flow matching for Gaussian DiT."""

import json
import math
import os
from contextlib import nullcontext
from pathlib import Path

import cv2
import gin
import torch
import torch.distributed as dist
from absl import app, flags
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None

from dataset import gs_io
from dataset.GS_SR import SplatFactoSRDataset
from dataset.scene_loader import build_train_loader, training_microbatches
from models.gaussian_dit_predictor import GaussianDiTPredictor
from sr import noise_to_gs
from sr.noise_to_gs import flow_matching
from utils import gpu_utils, gs_utils, noise_to_gs_training
from utils.data_augmentation import (rotate_gaussians, sample_uniform_rotation_quaternion,
                                     sample_uniform_z_rotation_quaternion, validate_rotation_max_degrees)
from utils.gpu_utils import seed_everything
from utils.gpu_scene_cache import build_gpu_scene_cache
from utils.ema import create_ema_model, update_ema_model
from utils.gs_normalization import GaussianStandardizer, STATISTIC_KEYS
from utils.log_utils import ProcessSafeLogger
from utils.noise_to_gs_training import (epoch_training_steps, evenly_spaced_camera_indices, evaluation_seed,
                                        mean_feature_losses, prepare_fixed_count_target, select_available_scenes)
from utils.optimizers import build_optimizer, build_scheduler


flags.DEFINE_string("output_dir", "output_train_noise_to_gs", "Run output directory")
flags.DEFINE_string("train_splat_dir", None, "Direct training gsplat scene directory")
flags.DEFINE_string("test_splat_dir", None, "Direct test gsplat scene directory")
flags.DEFINE_string("eval_scene_name", None, "Test scene supplying evaluation cameras; defaults to first selected training scene")
flags.DEFINE_string("init_checkpoint", None, "Optional model weights for a warm start")
flags.DEFINE_integer("train_scene_count", -1, "Train on the first N listed scenes; -1 uses all available scenes")
flags.DEFINE_integer("epochs", None, "Training epochs; overrides the configured total steps")
flags.DEFINE_integer("eval_sample_count", 16, "Fixed number of noise samples per evaluation")
flags.DEFINE_integer("gs_count", 1000, "Gaussians in every training target and generated scene")
flags.DEFINE_integer("batch_size", 4, "Scenes per GPU per optimizer update")
flags.DEFINE_integer("grad_accum_steps", 4, "Microbatches per GPU optimizer update")
flags.DEFINE_integer("num_workers", 2, "Scene loader workers per GPU")
flags.DEFINE_integer("prefetch_factor", 2, "Microbatches prefetched per worker")
flags.DEFINE_boolean("pin_memory", True, "Pin loaded CPU tensors")
flags.DEFINE_boolean("cache_dataset_on_gpu", False, "Cache all selected training scenes on each GPU")
flags.DEFINE_float("gpu_cache_headroom_gib", 4.0, "Minimum free GPU memory after caching, in GiB")
flags.DEFINE_boolean("use_wandb", True, "Log training and previews to W&B")
flags.DEFINE_string("wandb_project", "noise-to-gs", "W&B project")
flags.DEFINE_string("wandb_name", None, "W&B run name")
flags.DEFINE_multi_string("gin_file", None, "Gin configuration files")
flags.DEFINE_multi_string("gin_param", "", "Gin parameter bindings")
flags.register_multi_flags_validator(
    ["batch_size", "grad_accum_steps"],
    lambda values: 1 <= values["grad_accum_steps"] <= values["batch_size"],
    message="grad_accum_steps must be between 1 and batch_size",
)
FLAGS = flags.FLAGS
EVAL_VIEW_COUNT = 8
METRIC_VIEW_BATCH_SIZE = 8


@gin.configurable
def set_seed(seed=42, rank=0):
    seed_everything(seed + rank)
    return seed


@gin.configurable
def training_augmentation(random_rotate=False, rotation_mode="gravity_consistent", rotation_max_degrees=None,
                          rotation_pivot=(0.5, 0.5, 0.5)):
    if rotation_mode not in ("full", "gravity_consistent"):
        raise ValueError("rotation_mode must be 'full' or 'gravity_consistent'")
    validate_rotation_max_degrees(rotation_max_degrees)
    pivot = tuple(float(value) for value in rotation_pivot)
    if len(pivot) != 3 or any(not math.isfinite(value) for value in pivot):
        raise ValueError("rotation_pivot must have three finite values")
    return {"random_rotate": bool(random_rotate), "rotation_mode": rotation_mode,
            "rotation_max_degrees": rotation_max_degrees, "rotation_pivot": pivot}


@gin.configurable("training")
def training_config(total_steps=100000, eval_interval=5000, save_interval=5000, log_interval=20,
                    grad_clip_norm=2.0, enable_amp=True, ema_enabled=False, ema_decay=0.9999):
    if min(total_steps, eval_interval, save_interval, log_interval) < 1:
        raise ValueError("Training and reporting intervals must be positive")
    return locals()


def load_evaluation_cameras(dataset, scene_name, splat_dir, device, load_images=False):
    """Use reference GS to place evaluation cameras in the shared unit frame."""
    entry = dataset.folders[dataset.scene_index(scene_name)]
    entry["resolution_paths"][dataset.tgt_resolution]["gsplat_dir"] = str(Path(splat_dir) / scene_name)
    gaussians, _ = gs_io.load_gsplat(entry["resolution_paths"][dataset.tgt_resolution]["gsplat_dir"])
    selected = dataset.processor.selection_mask(gaussians)
    _, coordinate_scaler = dataset.processor.normalize({key: value[selected] for key, value in gaussians.items()})
    camera_data = dataset.load_camera_meta(entry, dataset.tgt_resolution, coordinate_scaler)
    indices = (list(range(len(camera_data["imgs_name"]))) if load_images
               else evenly_spaced_camera_indices(len(camera_data["imgs_name"]), EVAL_VIEW_COUNT))
    cameras = dict(camera_data["meta"])
    cameras["camera_to_worlds"] = cameras["camera_to_worlds"][indices]
    cameras["background_color"] = dataset.build_background()
    cameras = {key: value.to(device) if torch.is_tensor(value) else value for key, value in cameras.items()}
    names = [camera_data["imgs_name"][index] for index in indices]
    if load_images:
        images, _, _ = dataset.load_resolution_views(camera_data, camera_ids=indices, background=dataset.build_background())
        return cameras, indices, names, torch.stack(images)
    return cameras, indices, names


def evaluate(model, standardizer, cameras, camera_indices, camera_names, flow_cfg, step, rank, world_size, variant=None, metric_context=None):
    """Each rank writes disjoint fixed-seed samples; no target images enter sampling."""
    was_training = model.training
    model.eval()
    directory = Path(FLAGS.output_dir) / "eval" / f"{step:08d}"
    if variant == "model":
        directory /= "model"
    if rank == 0:
        directory.mkdir(parents=True, exist_ok=True)
        metadata = {"step": step, "weights": variant or "model", "reference_scene": FLAGS.eval_scene_name, "camera_indices": camera_indices,
                    "camera_names": camera_names, "gs_count": FLAGS.gs_count,
                    "statistics_path": flow_cfg["gs_statistics_path"],
                    "flow_steps": flow_cfg["flow_steps"],
                    "sample_seeds": [evaluation_seed(flow_cfg["eval_noise_seed"], slot)
                                     for slot in range(FLAGS.eval_sample_count)]}
        (directory / "samples.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if world_size > 1:
        dist.barrier()
    metric_computer = None
    if metric_context is not None and rank < FLAGS.eval_sample_count:
        from utils.metrics import MetricComputer
        with torch.cuda.device(next(model.parameters()).device):
            metric_computer = MetricComputer()
    with torch.no_grad(), torch.random.fork_rng(devices=[next(model.parameters()).device.index]):
        for slot in range(rank, FLAGS.eval_sample_count, world_size):
            slot_dir = directory / f"sample_{slot:02d}"
            slot_dir.mkdir(parents=True, exist_ok=True)
            gaussians = noise_to_gs.sample_flow_model(
                model, FLAGS.gs_count, 0, flow_cfg["flow_steps"], standardizer,
                noise_seed=evaluation_seed(flow_cfg["eval_noise_seed"], slot), t_eps=flow_cfg["flow_t_eps"],
                coupling=flow_cfg["coupling"], model_output_type=flow_cfg["model_output_type"]
            )
            sh_degree_zero = model.sh_degree == 0
            gs_utils.export_ply_forviewer(gaussians, slot_dir / "output.ply", sh_degree_zero=sh_degree_zero)
            images, _ = gs_utils.rasterize_gaussians_to_multiimgs(
                gaussians, cameras, batched=True, sh_degree_zero=sh_degree_zero
            )
            if metric_computer is not None:
                metric_computer.results = {key: [] for key in metric_computer.metrics}
                metric_computer.results_dict = {}
                for start in range(0, len(metric_context["images"]), METRIC_VIEW_BATCH_SIZE):
                    end = min(start + METRIC_VIEW_BATCH_SIZE, len(metric_context["images"]))
                    metric_cameras = dict(metric_context["cameras"])
                    metric_cameras["camera_to_worlds"] = metric_cameras["camera_to_worlds"][start:end]
                    metric_images, _ = gs_utils.rasterize_gaussians_to_multiimgs(
                        gaussians, metric_cameras, batched=True, sh_degree_zero=sh_degree_zero
                    )
                    predicted = torch.stack(metric_images).clamp(0, 1)
                    target = metric_context["images"][start:end].to(predicted.device)
                    if target.shape[-1] == 4:
                        predicted = predicted * target[..., 3:4]
                    metric_computer.update(predicted, target[..., :3].clamp(0, 1), str(start))
                report = {"reference_scene": metric_context["scene"], "camera_names": metric_context["names"],
                          "prediction": metric_computer.finalize(), "gsplat_baseline": metric_context["baseline"]}
                (slot_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
            for view, image in enumerate(images):
                rgb = (image.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
                cv2.imwrite(str(slot_dir / f"view_{view:02d}.png"), rgb[:, :, ::-1])
            del gaussians, images
    if world_size > 1:
        dist.barrier()
    metric_summary = None
    if rank == 0 and metric_context is not None:
        reports = [json.loads((directory / f"sample_{slot:02d}" / "metrics.json").read_text())
                   for slot in range(FLAGS.eval_sample_count)]
        mean = {key: sum(report["prediction"][key] for report in reports) / len(reports)
                for key in ("psnr", "ssim", "lpips")}
        metric_summary = {"reference_scene": metric_context["scene"], "camera_names": metric_context["names"],
                          "prediction_mean": mean, "gsplat_baseline": metric_context["baseline"]}
        (directory / "metrics.json").write_text(json.dumps(metric_summary, indent=2) + "\n")
        print(f"Evaluation step={step} weights={variant or 'model'} metrics={mean} "
              f"gsplat_baseline={metric_context['baseline']['metrics']}", flush=True)
    if rank == 0 and wandb is not None and wandb.run is not None:
        first_views = [cv2.imread(str(directory / f"sample_{slot:02d}" / "view_00.png"))[:, :, ::-1]
                       for slot in range(FLAGS.eval_sample_count)]
        columns = min(4, FLAGS.eval_sample_count)
        rows = (FLAGS.eval_sample_count + columns - 1) // columns
        metric = "eval/samples_model" if variant == "model" else "eval/samples"
        values = {metric: wandb.Image(gs_utils.make_grid(first_views, nrow=rows, ncols=columns))}
        if metric_summary is not None:
            values.update({f"eval/{variant or 'model'}/{key}": value for key, value in metric_summary["prediction_mean"].items()})
        wandb.log(values, step=step, commit=variant != "ema")
    model.train(was_training)


def training():
    if not FLAGS.train_splat_dir or (FLAGS.eval_scene_name and not FLAGS.test_splat_dir):
        raise ValueError("--train_splat_dir is required; --test_splat_dir is required for a test evaluation scene")
    if FLAGS.gs_count < 1 or FLAGS.num_workers < 0 or FLAGS.prefetch_factor < 1:
        raise ValueError("gs_count and prefetch_factor must be positive; num_workers must be nonnegative")
    if FLAGS.train_scene_count != -1 and FLAGS.train_scene_count < 1:
        raise ValueError("train_scene_count must be -1 or a positive integer")
    if (FLAGS.epochs is not None and FLAGS.epochs < 1) or FLAGS.eval_sample_count < 1:
        raise ValueError("epochs and eval_sample_count must be positive")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank() if distributed else 0
    world_size = dist.get_world_size() if distributed else 1
    base_seed = set_seed(rank=rank)
    config = training_config()
    flow_cfg = flow_matching()
    augmentation_cfg = training_augmentation()
    output_dir = Path(FLAGS.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = ProcessSafeLogger(str(output_dir / "train.log")).get_logger() if rank == 0 else None

    train_dataset = SplatFactoSRDataset.from_gin_scope(
        "train_dataset", load_src_gs=False, load_tgt_gs=True, load_src_images=False,
        load_tgt_images=False, max_gs_num=FLAGS.gs_count
    )
    if train_dataset.src_resolution != train_dataset.tgt_resolution:
        raise ValueError("Unconditional noise-to-GS requires equal source and target resolution")
    skipped = select_available_scenes(train_dataset, FLAGS.train_splat_dir, FLAGS.train_scene_count)
    if FLAGS.epochs is not None:
        config["total_steps"] = epoch_training_steps(FLAGS.epochs, len(train_dataset.folders), FLAGS.batch_size, world_size)
    if FLAGS.eval_scene_name:
        eval_dataset = SplatFactoSRDataset.from_gin_scope(
            "test_dataset", load_src_gs=False, load_tgt_gs=False, load_src_images=False,
            load_tgt_images=False, max_gs_num=FLAGS.gs_count
        )
        if eval_dataset.src_resolution != eval_dataset.tgt_resolution:
            raise ValueError("Unconditional noise-to-GS requires equal source and target resolution")
        eval_splat_dir = FLAGS.test_splat_dir
    else:
        FLAGS.eval_scene_name = train_dataset.folders[0]["scene_name"]
        eval_dataset, eval_splat_dir = train_dataset, FLAGS.train_splat_dir
    cameras, camera_indices, camera_names = load_evaluation_cameras(
        eval_dataset, FLAGS.eval_scene_name, eval_splat_dir, device
    )
    metric_context = None
    if len(train_dataset.folders) == 1:
        metric_scene = train_dataset.folders[0]["scene_name"]
        metric_cameras, _, metric_names, metric_images = load_evaluation_cameras(
            train_dataset, metric_scene, FLAGS.train_splat_dir, device, load_images=True
        )
        gs_dir = Path(FLAGS.train_splat_dir) / metric_scene
        checkpoint = Path(gs_io.latest_gsplat_checkpoint(gs_dir))
        step_number = checkpoint.stem.split("_")[1]
        baseline_path = gs_dir / "stats" / f"val_step{int(step_number):04d}.json"
        baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else None
        metric_context = {"scene": metric_scene, "cameras": metric_cameras, "names": metric_names,
                          "images": metric_images, "baseline": {"path": str(baseline_path), "metrics": baseline,
                                                               "view_scope": "gsplat validation views"}}
        if rank == 0:
            logger.info("single_scene_metrics=%s gsplat_baseline=%s", metric_scene, baseline)
    model = GaussianDiTPredictor()
    noise_to_gs.validate_coupling(model, flow_cfg["coupling"])
    if model.coordinate_mode not in ("morton_rank", "current") or model.max_gaussians != FLAGS.gs_count:
        raise ValueError("Trainer requires unconditional DiT with max_gaussians equal to gs_count")
    if augmentation_cfg["random_rotate"] and model.sh_degree > 1:
        raise ValueError("Random rotation supports Gaussian SH degree 0 or 1")
    if FLAGS.init_checkpoint:
        model.load_state_dict(torch.load(FLAGS.init_checkpoint, map_location="cpu", weights_only=True))
    model = model.to(device)
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
    model_module = model.module if distributed else model
    ema_model = create_ema_model(model_module, config["ema_enabled"], config["ema_decay"])
    with open(flow_cfg["gs_statistics_path"]) as statistics_file:
        statistics_config = json.load(statistics_file).get("config", {})
    if statistics_config.get("dataset_split", "train") != "train":
        raise ValueError("Noise-to-GS training requires training-set Gaussian statistics")
    standardizer = GaussianStandardizer(flow_cfg["gs_statistics_path"], flow_cfg["normalization_variance_floor"],
                                        sh_degree=0 if model_module.sh_degree == 0 else None)
    with gin.config_scope("train2D"):
        optimizer = build_optimizer(model_module)
        scheduler = build_scheduler(optimizer, total_step=config["total_steps"]) if FLAGS.epochs is not None else build_scheduler(optimizer)
    sizes = [FLAGS.batch_size // FLAGS.grad_accum_steps + (index < FLAGS.batch_size % FLAGS.grad_accum_steps)
             for index in range(FLAGS.grad_accum_steps)]
    loader = None
    if FLAGS.cache_dataset_on_gpu:
        loader = build_gpu_scene_cache(train_dataset, sizes, base_seed + rank, rank, world_size,
                                       model_module.sh_degree, device, FLAGS.gpu_cache_headroom_gib, logger,
                                       standardizer=standardizer, coordinate_mode=model_module.coordinate_mode,
                                       preprocessed=not augmentation_cfg["random_rotate"])
    if loader is None:
        loader = build_train_loader(train_dataset, sizes, base_seed + rank, rank, world_size,
                                    num_workers=FLAGS.num_workers, prefetch_factor=FLAGS.prefetch_factor,
                                    pin_memory=FLAGS.pin_memory)
    scaler = torch.cuda.amp.GradScaler(enabled=config["enable_amp"])
    rotation_sampler = (sample_uniform_z_rotation_quaternion if augmentation_cfg["rotation_mode"] == "gravity_consistent"
                        else sample_uniform_rotation_quaternion)
    rotation_generator = torch.Generator(device=device).manual_seed(base_seed + rank)

    if rank == 0:
        (output_dir / "checkpoints").mkdir(exist_ok=True)
        (output_dir / "config.gin").write_text(gin.operative_config_str())
        (output_dir / "normalization_statistics.json").write_text(json.dumps(standardizer.report(), indent=2) + "\n")
        (output_dir / "training_scenes.json").write_text(json.dumps({"eligible": [entry["scene_name"] for entry in train_dataset.folders],
                                                                  "skipped": skipped}, indent=2) + "\n")
        logger.info("world_size=%d batch_size_per_gpu=%d grad_accum_steps=%d train_scenes=%d skipped=%d gs_count=%d",
                    world_size, FLAGS.batch_size, FLAGS.grad_accum_steps, len(train_dataset.folders), len(skipped), FLAGS.gs_count)
        logger.info("epochs=%s train_scenes=%d global_batch_size=%d total_steps=%d eval_sample_count=%d",
                    FLAGS.epochs, len(train_dataset.folders), FLAGS.batch_size * world_size,
                    config["total_steps"], FLAGS.eval_sample_count)
        logger.info("ema_enabled=%s ema_decay=%s", config["ema_enabled"], config["ema_decay"])
        if FLAGS.use_wandb:
            if wandb is None:
                raise ImportError("wandb is not installed")
            wandb.init(project=FLAGS.wandb_project, name=FLAGS.wandb_name or output_dir.name,
                       dir=str(output_dir), config={"gin_config": gin.operative_config_str(),
                                                     "gs_count": FLAGS.gs_count, "world_size": world_size})
    if distributed:
        dist.barrier()
    evaluate(ema_model if ema_model is not None else model_module, standardizer, cameras, camera_indices, camera_names,
             flow_cfg, 0, rank, world_size, variant="ema" if ema_model is not None else None, metric_context=metric_context)
    if ema_model is not None:
        evaluate(model_module, standardizer, cameras, camera_indices, camera_names, flow_cfg, 0, rank, world_size, variant="model", metric_context=metric_context)

    with training_microbatches(loader) as train_iter:
        for step in tqdm(range(1, config["total_steps"] + 1), disable=rank != 0):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss_sum = torch.zeros((), device=device)
            feature_loss_sum = torch.zeros(len(STATISTIC_KEYS), device=device)
            for microbatch_index in range(len(sizes)):
                scenes = next(train_iter)
                targets = []
                for scene in scenes:
                    target = gpu_utils.move_training_data(scene["data"][train_dataset.tgt_resolution]["gs_params"],
                                                          device, non_blocking=FLAGS.pin_memory)
                    if not getattr(loader, "preprocessed", False):
                        if model_module.sh_degree == 0:
                            target["features_rest"] = target["features_rest"][:, :0, :]
                        if augmentation_cfg["random_rotate"]:
                            rotation = rotation_sampler(target["means"].dtype, device, rotation_generator,
                                                        max_degrees=augmentation_cfg["rotation_max_degrees"])
                            target = rotate_gaussians(target, rotation, augmentation_cfg["rotation_pivot"])
                        target = standardizer.encode(prepare_fixed_count_target(target, FLAGS.gs_count,
                                                                               model_module.coordinate_mode))
                    targets.append(target)
                times = [torch.empty((), device=device).uniform_(flow_cfg["flow_t_eps"], 1 - flow_cfg["flow_t_eps"])
                         for _ in targets]
                paths = [noise_to_gs.construct_path(target, time, coupling=flow_cfg["coupling"])
                         for target, time in zip(targets, times)]
                sync = model.no_sync() if distributed and microbatch_index < len(sizes) - 1 else nullcontext()
                with sync, torch.cuda.amp.autocast(enabled=config["enable_amp"]):
                    outputs = model(batch_flow_gs=[query for query, _ in paths],
                                    batch_scene_idx=[0] * len(paths), batch_reference_means=None,
                                    t=torch.stack(times))
                    predictions = [noise_to_gs.convert_prediction(output, query, time, flow_cfg["model_output_type"],
                                                                   flow_cfg["prediction_type"], min_denominator=0.05)
                                   for output, (query, _), time in zip(outputs, paths, times)]
                    loss_targets = targets if flow_cfg["prediction_type"] == "x" else [velocity for _, velocity in paths]
                    if flow_cfg["model_output_type"] == "x" and flow_cfg["prediction_type"] == "v":
                        loss_targets = [noise_to_gs.convert_prediction(target, query, time, "x", "v", min_denominator=0.05)
                                        for target, (query, _), time in zip(targets, paths, times)]
                    loss_parts = [noise_to_gs_training.attribute_mse(prediction, target, flow_cfg["feature_loss_weights"])
                                  for prediction, target in zip(predictions, loss_targets)]
                    summed = sum(parts[0] for parts in loss_parts)
                    loss = summed / FLAGS.batch_size
                if config["enable_amp"]:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                loss_sum += summed.detach()
                if step % config["log_interval"] == 0:
                    feature_loss_sum += sum(torch.stack([parts[1][key].detach() for key in STATISTIC_KEYS])
                                            for parts in loss_parts)
                del scenes, targets, paths, outputs, predictions, loss_targets, loss
            if config["enable_amp"]:
                previous_scale = scaler.get_scale()
                scaler.unscale_(optimizer)
                if config["grad_clip_norm"] > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip_norm"])
                scaler.step(optimizer)
                scaler.update()
                stepped = scaler.get_scale() >= previous_scale
            else:
                if config["grad_clip_norm"] > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip_norm"])
                optimizer.step()
                stepped = True
            if stepped:
                scheduler.step()
                if ema_model is not None:
                    update_ema_model(ema_model, model_module, config["ema_decay"])
            if step % config["log_interval"] == 0:
                if distributed:
                    dist.all_reduce(loss_sum)
                    dist.all_reduce(feature_loss_sum)
                average_loss = (loss_sum / (FLAGS.batch_size * world_size)).item()
                if rank == 0:
                    lr = optimizer.param_groups[0]["lr"]
                    raw, weighted = mean_feature_losses(feature_loss_sum, FLAGS.batch_size * world_size,
                                                        flow_cfg["feature_loss_weights"])
                    logger.info("step=%d loss=%.6f lr=%.8f feature_loss=%s weighted_feature_loss=%s",
                                step, average_loss, lr, raw, weighted)
                    if wandb is not None and wandb.run is not None:
                        metrics = {"train/loss": average_loss, "train/lr": lr}
                        metrics.update({f"train/feature_loss/{key}": value for key, value in raw.items()})
                        metrics.update({f"train/weighted_feature_loss/{key}": value for key, value in weighted.items()})
                        wandb.log(metrics, step=step)
            if step % config["eval_interval"] == 0 or step == config["total_steps"]:
                evaluate(ema_model if ema_model is not None else model_module, standardizer, cameras, camera_indices, camera_names,
                         flow_cfg, step, rank, world_size, variant="ema" if ema_model is not None else None, metric_context=metric_context)
                if ema_model is not None:
                    evaluate(model_module, standardizer, cameras, camera_indices, camera_names,
                             flow_cfg, step, rank, world_size, variant="model", metric_context=metric_context)
            if step % config["save_interval"] == 0 and rank == 0:
                torch.save(model_module.state_dict(), output_dir / "checkpoints" / f"model_{step:08d}.pth")
                if ema_model is not None:
                    torch.save(ema_model.state_dict(), output_dir / "checkpoints" / f"model_ema_{step:08d}.pth")
    if rank == 0:
        torch.save(model_module.state_dict(), output_dir / "checkpoints" / "model_last.pth")
        if ema_model is not None:
            torch.save(ema_model.state_dict(), output_dir / "checkpoints" / "model_ema_last.pth")
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
