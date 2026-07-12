import os
import cv2
import gin
import numpy as np
import torch
from absl import app, flags
from tqdm import tqdm

from dataset.GS_multi import SplatFactoMultiLevelDataset
from models.feature_predictor import FeaturePredictor
from utils import gpu_utils, gs_utils, loss_utils
from utils.gpu_utils import seed_everything
from utils.gs_utils import make_grid
from utils.log_utils import ProcessSafeLogger
from utils.metrics import MetricComputer, psnr
from utils.optimizers import build_optimizer, build_scheduler
from utils.sr_densify_utils import build_densified_input_gs, save_densify_stage_plys


flags.DEFINE_string("output_dir", "output_overfit", "Output directory")
flags.DEFINE_string("eval_subdir", "eval_final", "Eval subdirectory")
flags.DEFINE_string("scene_name", "", "Scene name to overfit")
flags.DEFINE_boolean("compare_with_input", False, "Compare with input 3DGS")
flags.DEFINE_boolean("save_viewer", True, "Save viewer point clouds")
flags.DEFINE_integer("input_factor", 4, "Low-resolution GS factor used as densification source")
flags.DEFINE_integer("target_factor", 2, "High-resolution GS/image factor used as overfit target")
flags.DEFINE_integer(
    "image_batch_size",
    -1,
    "Number of target views rendered per train step. Use <=0 to render all views.",
)
flags.DEFINE_enum(
    "means_source",
    "gt",
    ["gt", "predicted"],
    "Source for stage-2 input means: GT target means or residual output from a means predictor.",
)
flags.DEFINE_enum("alignment", "emd", ["emd", "nearest", "none"], "Interpolated-to-target alignment method")
flags.DEFINE_enum(
    "attribute_init",
    "aligned",
    ["aligned", "3dgs"],
    "How to initialize non-position GS attributes after high-res positions are fixed",
)
flags.DEFINE_float("emd_eps", 0.01, "Auction EMD epsilon")
flags.DEFINE_integer("emd_iters", 100, "Auction EMD iterations")
flags.DEFINE_multi_string("gin_file", None, "List of paths to the config files.")
flags.DEFINE_multi_string("gin_param", "", "Newline separated list of Gin parameter bindings.")

FLAGS = flags.FLAGS

INPUT_FACTOR = 4
TARGET_FACTOR = 2
MEANS_FEATURES = ["means"]
ATTRIBUTE_FEATURES = ["features_dc", "features_rest", "opacities", "scales", "quats"]


@gin.configurable
def set_seed(seed):
    seed_everything(seed)


@gin.configurable
def training(
    output_dir=None,
    total_steps = gin.REQUIRED,
    pretrain_steps = gin.REQUIRED,
    eval_interval = gin.REQUIRED,
    log_interval = gin.REQUIRED,
    save_interval = gin.REQUIRED,
    log_image_interval = gin.REQUIRED,
    grad_clip_norm = gin.REQUIRED,
    image_l1_loss_weight=1.0,
    lpips_loss_weight=0.0,
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
        "image_l1_loss_weight": image_l1_loss_weight,
        "lpips_loss_weight": lpips_loss_weight,
        "resume_from_step": resume_from_step,
        "enable_amp": enable_amp,
        "empty_cache_fre": empty_cache_fre,
    }




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


def _build_split_payload(dataset, scene_idx, scene_name, factor_entry, split, image_batch_size=None):
    meta = factor_entry["meta"]
    imgs_path = factor_entry["imgs_path"]
    imgs_name = factor_entry["imgs_name"]

    if dataset.background_color == "random":
        background = torch.rand(3)
    else:
        background = torch.tensor(dataset.background_color, dtype=torch.float32) / 255.0

    total_num = len(meta["camera_to_worlds"])
    if split == "train":
        if image_batch_size is None or int(image_batch_size) <= 0:
            cam_ids = np.arange(total_num)
        else:
            sample_num = min(int(image_batch_size), total_num)
            cam_ids = np.random.permutation(total_num)[:sample_num]
    elif split == "test":
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



def _clone_gs(gs):
    return {key: value.clone() for key, value in gs.items()}


def _replace_means(gs, means):
    out_gs = _clone_gs(gs)
    out_gs["means"] = means.to(device=gs["means"].device, dtype=gs["means"].dtype)
    return out_gs


def _bind_feature_predictor(output_features, output_features_type="res"):
    with gin.unlock_config():
        gin.bind_parameter("FeaturePredictor.output_features", list(output_features))
        gin.bind_parameter("FeaturePredictor.output_features_type", output_features_type)


def _build_feature_predictor(output_features, device, logger, checkpoint_key):
    _bind_feature_predictor(output_features, output_features_type="res")
    model = FeaturePredictor().to(device)
    _load_resume_checkpoint(model, checkpoint_key, logger)
    model.train()
    return model


def _load_resume_checkpoint(model, checkpoint_key, logger):
    if model.resume_ckpt is None:
        return
    checkpoint = torch.load(model.resume_ckpt, map_location="cpu")
    if isinstance(checkpoint, dict) and checkpoint_key in checkpoint:
        state_dict = checkpoint[checkpoint_key]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    logger.info(f"Loaded {checkpoint_key} checkpoint from {model.resume_ckpt}")


def _stage2_input_gs(input_gs, target_gs, means_model, means_source, scene_idx):
    if means_source == "gt":
        return _replace_means(input_gs, target_gs["means"])
    if means_source == "predicted":
        if means_model is None:
            raise ValueError("means_model is required when --means_source=predicted")
        return means_model(batch_normalized_gs=[input_gs], batch_scene_idx=[scene_idx])[0]
    raise ValueError(f"Unsupported means_source={means_source}")


def _forward_two_stage(
    attribute_model,
    input_gs,
    scene_idx,
    target_gs,
    means_source,
    means_model=None,
    return_stage1=False,
):
    stage1_gs = _stage2_input_gs(input_gs, target_gs, means_model, means_source, scene_idx)
    stage2_gs = attribute_model(batch_normalized_gs=[stage1_gs], batch_scene_idx=[scene_idx])[0]
    if return_stage1:
        return stage2_gs, stage1_gs
    return stage2_gs


def _active_models(attribute_model, means_model):
    models = [attribute_model]
    if means_model is not None:
        models.insert(0, means_model)
    return models


def _trainable_parameters(models):
    return [param for model in models for param in model.parameters() if param.requires_grad]


def _checkpoint_payload(attribute_model, means_model, means_source):
    payload = {
        "means_source": means_source,
        "attribute_model": attribute_model.state_dict(),
    }
    if means_model is not None:
        payload["means_model"] = means_model.state_dict()
    return payload



def evaluate_single_scene(
    attribute_model,
    input_gs,
    scene_idx,
    scene_name,
    eval_images,
    eval_cameras,
    image_names,
    output_dir,
    eval_chunk_size=None,
    gt_gs=None,
    compare_with_input=False,
    save_viewer=True,
    output_gt=True,
    target_gs_for_means=None,
    means_source="gt",
    means_model=None,
):
    if target_gs_for_means is None:
        raise ValueError("target_gs_for_means is required for two-stage evaluation")
    attribute_model.eval()
    if means_model is not None:
        means_model.eval()
    metric_computer = MetricComputer()
    metric_computer_input = MetricComputer() if compare_with_input else None
    device = next(attribute_model.parameters()).device
    num_views = len(eval_images)
    if num_views == 0:
        raise ValueError("Evaluation payload has zero views")

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
        out_gs, stage1_gs = _forward_two_stage(
            attribute_model=attribute_model,
            input_gs=input_gs,
            scene_idx=scene_idx,
            target_gs=target_gs_for_means,
            means_source=means_source,
            means_model=means_model,
            return_stage1=True,
        )

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
            gs_utils.prepare_viewer(eval_cameras, viewerdir, attribute_model.sh_degree)
            gs_utils.export_ply_forviewer(
                gs_params=input_gs,
                filename=os.path.join(viewerdir, "point_cloud/00_input_gs.ply"),
            )
            gs_utils.export_ply_forviewer(
                gs_params=stage1_gs,
                filename=os.path.join(viewerdir, "point_cloud/01_stage1_output_gs.ply"),
            )
            gs_utils.export_ply_forviewer(
                gs_params=out_gs,
                filename=os.path.join(viewerdir, "point_cloud/02_stage2_output_gs.ply"),
            )
            if gt_gs is not None:
                gs_utils.export_ply_forviewer(
                    gs_params=gt_gs,
                    filename=os.path.join(viewerdir, "point_cloud/03_gt_gs.ply"),
                )


    metrics = metric_computer.finalize()
    metric_computer.write_to_file(os.path.join(output_dir, "metrics.json"))

    if compare_with_input:
        metrics_input = metric_computer_input.finalize()
        metric_computer_input.write_to_file(os.path.join(output_dir, "metrics_input.json"))
    else:
        metrics_input = {}

    attribute_model.train()
    if means_model is not None:
        means_model.train()
    return metrics, metrics_input


def main(argv):
    del argv
    os.makedirs(FLAGS.output_dir, exist_ok=True)

    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)
    train_cfg = training(output_dir=FLAGS.output_dir)
    set_seed()

    logger = ProcessSafeLogger(os.path.join(FLAGS.output_dir, "overfit.log")).get_logger()
    device = torch.device("cuda")

    dataset: SplatFactoMultiLevelDataset = _build_dataset()
    scene_idx = _find_scene_index(dataset, FLAGS.scene_name)
    scene = dataset.load_scene(scene_idx)
    input_factor_entry = scene["factor_data"][FLAGS.input_factor]
    target_factor_entry = scene["factor_data"][FLAGS.target_factor]

    train_payload = _build_split_payload(
        dataset,
        scene["idx"],
        scene["scene_name"],
        target_factor_entry,
        split="train",
        image_batch_size=FLAGS.image_batch_size,
    )
    eval_payload = _build_split_payload(
        dataset, scene["idx"], scene["scene_name"], target_factor_entry, split="test"
    )
    if len(eval_payload["images"]) == 0:
        eval_payload = _build_split_payload(
            dataset, scene["idx"], scene["scene_name"], target_factor_entry, split="train"
        )

    densified_input_gs, densify_stage_gs = build_densified_input_gs(
        input_factor_dict=input_factor_entry,
        target_factor_dict=target_factor_entry,
        alignment=FLAGS.alignment,
        attribute_init=FLAGS.attribute_init,
        emd_eps=FLAGS.emd_eps,
        emd_iters=FLAGS.emd_iters,
        device=device,
        return_stages=True,
    )
    save_densify_stage_plys(
        output_dir=FLAGS.output_dir,
        low_res_gs=densify_stage_gs["00_low_res_gs.ply"],
        interpolated_gs=densify_stage_gs["01_interpolated_high_res_gs.ply"],
        gt_high_res_gs=densify_stage_gs["02_gt_high_res_gs.ply"],
        input_high_res_gs=densify_stage_gs["03_input_high_res_gs.ply"],
    )
    target_gs = gpu_utils.move_to_device(target_factor_entry["gs_params"], device)
    batch_gs = gpu_utils.move_to_device([densified_input_gs], device)
    batch_scene_idx = [scene["idx"]]

    eval_images = eval_payload["images"]
    eval_cameras = eval_payload["cameras"]
    eval_chunk_size = dataset.image_per_scene if dataset.image_per_scene is not None else len(eval_images)
    if eval_chunk_size <= 0:
        eval_chunk_size = len(eval_images)

    means_model = None
    if FLAGS.means_source == "predicted":
        means_model = _build_feature_predictor(MEANS_FEATURES, device, logger, "means_model")
    attribute_model = _build_feature_predictor(ATTRIBUTE_FEATURES, device, logger, "attribute_model")

    active_models = _active_models(attribute_model, means_model)
    optimizers = []
    schedulers = []
    with gin.config_scope("train2D"):
        if means_model is not None:
            means_optimizer = build_optimizer(means_model)
            means_scheduler = build_scheduler(means_optimizer)
            optimizers.append(means_optimizer)
            schedulers.append(means_scheduler)
        attribute_optimizer = build_optimizer(attribute_model)
        attribute_scheduler = build_scheduler(attribute_optimizer)
        optimizers.append(attribute_optimizer)
        schedulers.append(attribute_scheduler)

    with open(os.path.join(FLAGS.output_dir, "config.gin"), "w") as f:
        f.writelines(gin.operative_config_str())

    total_steps = train_cfg["total_steps"]
    log_interval = train_cfg["log_interval"]
    log_image_interval = train_cfg["log_image_interval"]
    save_interval = train_cfg["save_interval"]
    eval_interval = train_cfg["eval_interval"]
    grad_clip_norm = train_cfg["grad_clip_norm"]
    resume_from_step = train_cfg["resume_from_step"]
    enable_amp = train_cfg["enable_amp"]
    empty_cache_fre = train_cfg["empty_cache_fre"]
    image_l1_loss_weight = train_cfg["image_l1_loss_weight"]
    lpips_loss_weight = train_cfg["lpips_loss_weight"]

    scaler = torch.cuda.amp.GradScaler(enabled=enable_amp)
    lpips_loss_func = loss_utils.lpips_loss_fn() if lpips_loss_weight > 0 else None

    training_brief = (
        f"Overfit scene={scene['scene_name']} idx={scene['idx']} \n"
        f"train_views={len(train_payload['images'])} eval_views={len(eval_payload['images'])} \n"
        f"image_batch_size={FLAGS.image_batch_size} \n"
        f"input_gaussians={input_factor_entry['gs_params']['means'].shape[0]} \n"
        f"densified_gaussians={batch_gs[0]['means'].shape[0]} \n"
        f"target_gaussians={target_factor_entry['gs_params']['means'].shape[0]} \n"
        f"input_factor={FLAGS.input_factor} target_factor={FLAGS.target_factor} \n"
        f"alignment={FLAGS.alignment} attribute_init={FLAGS.attribute_init} \n"
        f"means_source={FLAGS.means_source} \n"
        f"means_model_input_features={','.join(means_model.input_features) if means_model is not None else 'none'} \n"
        f"means_model_output_features={','.join(means_model.output_features) if means_model is not None else 'none'} \n"
        f"attribute_model_input_features={','.join(attribute_model.input_features)} \n"
        f"attribute_model_output_features={','.join(attribute_model.output_features)} \n"
    )
    print(training_brief)
    logger.info(training_brief)

    os.makedirs(os.path.join(FLAGS.output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(FLAGS.output_dir, "checkpoints"), exist_ok=True)

    init_batch_images = gpu_utils.move_to_device([train_payload["images"]], device)
    gt_imgs_uint8 = [(img[..., :3] * 255).detach().cpu().numpy().astype(np.uint8) for img in init_batch_images[0]]
    gt_grid = cv2.cvtColor(make_grid(gt_imgs_uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(FLAGS.output_dir, "train", "00000000_gt.png"), gt_grid)

    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(range(resume_from_step, total_steps))
    for step in pbar:
        train_payload = _build_split_payload(
            dataset,
            scene["idx"],
            scene["scene_name"],
            target_factor_entry,
            split="train",
            image_batch_size=FLAGS.image_batch_size,
        )
        batch_cameras = gpu_utils.move_to_device([train_payload["cameras"]], device)
        batch_images = gpu_utils.move_to_device([train_payload["images"]], device)

        with torch.cuda.amp.autocast(enabled=enable_amp):
            out_gs = _forward_two_stage(
                attribute_model=attribute_model,
                input_gs=batch_gs[0],
                scene_idx=batch_scene_idx[0],
                target_gs=target_gs,
                means_source=FLAGS.means_source,
                means_model=means_model,
            )
            pred_imgs, _ = gs_utils.rasterize_gaussians_to_multiimgs(out_gs, batch_cameras[0])

            image_l1 = 0
            lpips_loss = 0
            train_psnr = 0
            num_images = len(pred_imgs)
            for pred_img, gt_img in zip(pred_imgs, batch_images[0]):
                gt_rgb = gt_img[..., :3]
                image_l1 += (pred_img - gt_rgb).abs().mean()
                train_psnr += psnr(pred_img.unsqueeze(0), gt_rgb.unsqueeze(0)).mean()
                if lpips_loss_func is not None:
                    lpips_loss += lpips_loss_func(pred_img.unsqueeze(0), gt_rgb.unsqueeze(0)).mean()

            image_l1 = image_l1 / num_images * image_l1_loss_weight
            train_psnr = train_psnr / num_images
            total_loss = image_l1
            if lpips_loss_func is not None:
                lpips_loss = lpips_loss / num_images * lpips_loss_weight
                total_loss = total_loss + lpips_loss

        if enable_amp:
            scaler.scale(total_loss).backward()
            if grad_clip_norm > 0:
                for optimizer in optimizers:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(_trainable_parameters(active_models), grad_clip_norm)
            for optimizer in optimizers:
                scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(_trainable_parameters(active_models), grad_clip_norm)
            for optimizer in optimizers:
                optimizer.step()

        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        for scheduler in schedulers:
            scheduler.step()

        lpips_value = lpips_loss.item() if lpips_loss_func is not None else 0.0
        pbar.set_postfix(
            {
                "loss": f"{total_loss.item():.4f}",
                "l1": f"{image_l1.item():.4f}",
                "lpips": f"{lpips_value:.4f}",
                "psnr": f"{train_psnr.item():.2f}",
                "attr_lr": f"{attribute_optimizer.param_groups[0]['lr']:.2e}",
            }
        )

        if empty_cache_fre > 0 and (step + 1) % empty_cache_fre == 0:
            torch.cuda.empty_cache()

        if step % log_interval == 0:
            log_msg = (
                f"step={step} total={total_loss.item():.6f} "
                f"l1={image_l1.item():.6f} psnr={train_psnr.item():.4f} "
                f"attr_lr={attribute_optimizer.param_groups[0]['lr']:.8f}"
            )
            if means_model is not None:
                log_msg += f" means_lr={means_optimizer.param_groups[0]['lr']:.8f}"
            if lpips_loss_func is not None:
                log_msg += f" lpips={lpips_loss.item():.6f}"
            logger.info(log_msg)

        if step % log_image_interval == 0:
            pred_imgs_uint8 = [(im * 255).detach().cpu().numpy().astype(np.uint8) for im in pred_imgs]
            pred_grid = cv2.cvtColor(make_grid(pred_imgs_uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(FLAGS.output_dir, "train", f"{step:08d}_pred.png"), pred_grid)

        if step % eval_interval == 0:
            eval_dir = os.path.join(FLAGS.output_dir, "eval", f"{step:08d}")
            metrics, metrics_input = evaluate_single_scene(
                attribute_model=attribute_model,
                input_gs=batch_gs[0],
                gt_gs=target_factor_entry["gs_params"],
                scene_idx=eval_payload["scene_idx"],
                scene_name=eval_payload["scene_name"],
                eval_images=eval_images,
                eval_cameras=eval_cameras,
                image_names=eval_payload["images_name"],
                output_dir=eval_dir,
                eval_chunk_size=eval_chunk_size,
                compare_with_input=FLAGS.compare_with_input,
                save_viewer=FLAGS.save_viewer,
                output_gt=(step == 0),
                target_gs_for_means=target_gs,
                means_source=FLAGS.means_source,
                means_model=means_model,
            )
            metric_str = " ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            logger.info(f"Eval step {step}: {metric_str}")
            if FLAGS.compare_with_input:
                metric_str = " ".join([f"{k}: {v:.4f}" for k, v in metrics_input.items()])
                logger.info(f"Eval input step {step}: {metric_str}")

        if (step + 1) % save_interval == 0:
            torch.save(
                _checkpoint_payload(attribute_model, means_model, FLAGS.means_source),
                os.path.join(FLAGS.output_dir, "checkpoints", f"model_{step:08d}.pth"),
            )

    torch.save(
        _checkpoint_payload(attribute_model, means_model, FLAGS.means_source),
        os.path.join(FLAGS.output_dir, "checkpoints", "model_last.pth"),
    )

    final_eval_dir = os.path.join(FLAGS.output_dir, FLAGS.eval_subdir)
    metrics, metrics_input = evaluate_single_scene(
        attribute_model=attribute_model,
        input_gs=batch_gs[0],
        gt_gs=target_factor_entry["gs_params"],
        scene_idx=eval_payload["scene_idx"],
        scene_name=eval_payload["scene_name"],
        eval_images=eval_images,
        eval_cameras=eval_cameras,
        image_names=eval_payload["images_name"],
        output_dir=final_eval_dir,
        eval_chunk_size=eval_chunk_size,
        compare_with_input=FLAGS.compare_with_input,
        save_viewer=FLAGS.save_viewer,
        output_gt=True,
        target_gs_for_means=target_gs,
        means_source=FLAGS.means_source,
        means_model=means_model,
    )

    metric_str = " ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    logger.info(f"Final eval: {metric_str}")
    if FLAGS.compare_with_input:
        metric_str = " ".join([f"{k}: {v:.4f}" for k, v in metrics_input.items()])
        logger.info(f"Final eval input: {metric_str}")



if __name__ == "__main__":
    app.run(main)
