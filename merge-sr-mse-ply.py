"""Merge final PLY predictions from single-attribute SR-MSE overfit runs."""

import json
import os
import re
from pathlib import Path

import cv2
import gin
import numpy as np
import torch
from absl import app, flags
from tqdm import tqdm

from utils import gpu_utils, gs_utils
from utils.gpu_utils import seed_everything
from utils.gs_utils import make_grid
from utils.log_utils import ProcessSafeLogger
from utils.loss_utils import SUPPORTED_GS_KEYS
from utils.metrics import MetricComputer
from utils.sr_dataset_utils import build_dataset, find_scene_index

SUPPORTED_KEYS = tuple(SUPPORTED_GS_KEYS)

flags.DEFINE_string("runs_root", None, "Folder containing direct single-attribute run folders.")
flags.DEFINE_string("source_eval_subdir", "eval_final", "Evaluation folder containing source PLYs.")
flags.DEFINE_string("output_dir", None, "Merged evaluation output folder.")
flags.DEFINE_string("scene_name", "", "Scene name; empty infers it from source viewer folders.")
flags.DEFINE_integer("input_factor", -1, "Input factor; -1 infers it from source run names.")
flags.DEFINE_integer("target_factor", -1, "Target factor; -1 infers it from source run names.")
flags.DEFINE_integer("eval_chunk_size", 16, "Views rendered per evaluation chunk.")
flags.DEFINE_boolean("compare_with_input", True, "Write GT, input, and merged comparison images.")
flags.DEFINE_boolean("save_viewer", True, "Write viewer PLY files and camera metadata.")
flags.DEFINE_multi_string("gin_file", None, "Gin config files used to load the target scene.")
flags.DEFINE_multi_string("gin_param", "", "Gin parameter overrides used to load the target scene.")

FLAGS = flags.FLAGS


@gin.configurable
def set_seed(seed=42):
    seed_everything(seed)

def _single_loss_feature(run_dir):
    log_path = run_dir / "overfit.log"
    if not log_path.is_file():
        return None
    matches = re.findall(r"^loss_features=([^\s]+)", log_path.read_text(), flags=re.MULTILINE)
    return matches[0] if len(matches) == 1 and matches[0] in SUPPORTED_KEYS else None


def discover_runs(runs_root, source_eval_subdir):
    root = Path(runs_root)
    candidates = {key: [] for key in SUPPORTED_KEYS}
    for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        key = _single_loss_feature(run_dir)
        if key is None:
            continue
        output_plys = list((run_dir / source_eval_subdir / "viewer").glob("*/point_cloud/output.ply"))
        if len(output_plys) != 1:
            raise ValueError(f"Expected one final output PLY for {run_dir}, found {len(output_plys)}")
        input_ply = output_plys[0].with_name("input.ply")
        if not input_ply.is_file():
            raise ValueError(f"Missing source input PLY for {run_dir}: {input_ply}")
        candidates[key].append({"run": run_dir, "output": output_plys[0], "input": input_ply})
    missing = [key for key, values in candidates.items() if not values]
    duplicates = [key for key, values in candidates.items() if len(values) > 1]
    if missing or duplicates:
        raise ValueError(f"Invalid single-attribute runs: missing={missing}, duplicates={duplicates}")
    return {key: candidates[key][0] for key in SUPPORTED_KEYS}


def infer_factors(sources):
    factors = set()
    for source in sources.values():
        match = re.search(r"(?:^|_)if(\d+)_tf(\d+)(?:_|$)", source["run"].name)
        if match is None:
            raise ValueError(f"Cannot infer factors from run name: {source['run'].name}")
        factors.add((int(match.group(1)), int(match.group(2))))
    if len(factors) != 1:
        raise ValueError(f"Source runs disagree on factors: {sorted(factors)}")
    return next(iter(factors))


def infer_scene_name(sources):
    scene_names = {source["output"].parent.parent.name for source in sources.values()}
    if len(scene_names) != 1:
        raise ValueError(f"Source runs disagree on viewer scenes: {sorted(scene_names)}")
    return next(iter(scene_names))


def merge_sources(sources):
    reference_input = gs_utils.load_ply_forviewer(sources["means"]["input"])
    merged = {key: value.clone() for key, value in reference_input.items()}
    for feature, source in sources.items():
        source_input = gs_utils.load_ply_forviewer(source["input"])
        for key in SUPPORTED_KEYS:
            if source_input[key].shape != reference_input[key].shape or not torch.allclose(source_input[key], reference_input[key]):
                raise ValueError(f"Input Gaussian ordering differs for {feature}: {source['input']}")
        predicted = gs_utils.load_ply_forviewer(source["output"])
        if predicted[feature].shape != merged[feature].shape:
            raise ValueError(f"Prediction shape mismatch for {feature}: {source['output']}")
        merged[feature] = predicted[feature]
    return reference_input, merged

def prepare_uint8(rendered, images):
    target = torch.stack(images, dim=0)
    if target.shape[-1] == 4:
        rendered = rendered * target[..., 3].unsqueeze(-1)
        target = target[..., :3]
    return (rendered * 255).to(torch.uint8), (target * 255).to(torch.uint8)

def evaluate_merged(merged, input_gs, target_gs, images, image_names, cameras, scene_idx, scene_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    pred_dir = os.path.join(output_dir, "pred", scene_name)
    os.makedirs(pred_dir, exist_ok=True)
    compare_dir = os.path.join(output_dir, "compare", scene_name)
    if FLAGS.compare_with_input:
        os.makedirs(compare_dir, exist_ok=True)
    device = torch.device("cuda")
    merged = gpu_utils.move_to_device(merged, device)
    input_gs = gpu_utils.move_to_device(input_gs, device)
    target_gs = gpu_utils.move_to_device(target_gs, device)
    metric_computer = MetricComputer()
    pred_preview, gt_preview, compare_preview = [], [], []
    chunk_size = min(FLAGS.eval_chunk_size or len(images), len(images))
    for start in tqdm(range(0, len(images), chunk_size), desc="Evaluating merged PLY"):
        end = min(start + chunk_size, len(images))
        chunk_images = gpu_utils.move_to_device(images[start:end], device)
        chunk_cameras = {key: (value[start:end] if key == "camera_to_worlds" else value) for key, value in cameras.items()}
        chunk_cameras = gpu_utils.move_to_device(chunk_cameras, device)
        with torch.no_grad():
            pred_images, _ = gs_utils.rasterize_gaussians_to_multiimgs(merged, chunk_cameras)
            pred_uint8, gt_uint8 = prepare_uint8(torch.stack(pred_images, dim=0), chunk_images)
            if FLAGS.compare_with_input:
                input_images, _ = gs_utils.rasterize_gaussians_to_multiimgs(input_gs, chunk_cameras)
                input_uint8, _ = prepare_uint8(torch.stack(input_images, dim=0), chunk_images)
        metric_computer.update(pred_uint8, gt_uint8, name=f"{start:06d}")
        for offset, (pred_image, gt_image) in enumerate(zip(pred_uint8, gt_uint8)):
            image_id = start + offset
            filename = image_names[image_id]
            cv2.imwrite(os.path.join(pred_dir, filename), pred_image.cpu().numpy()[..., ::-1])
            if FLAGS.compare_with_input:
                comparison = np.concatenate([gt_image.cpu().numpy(), input_uint8[offset].cpu().numpy(), pred_image.cpu().numpy()], axis=1)
                cv2.imwrite(os.path.join(compare_dir, filename), comparison[..., ::-1])
                if len(compare_preview) < 9:
                    compare_preview.append(comparison)
            if len(pred_preview) < 9:
                pred_preview.append(pred_image.cpu().numpy())
                gt_preview.append(gt_image.cpu().numpy())

    metrics = metric_computer.finalize()
    metric_computer.write_to_file(os.path.join(output_dir, "metrics.json"))
    cv2.imwrite(os.path.join(output_dir, f"scene{scene_idx}_pred.png"), make_grid(pred_preview)[..., ::-1])
    cv2.imwrite(os.path.join(output_dir, f"scene{scene_idx}_gt.png"), make_grid(gt_preview)[..., ::-1])
    if FLAGS.compare_with_input and compare_preview:
        cv2.imwrite(os.path.join(output_dir, f"scene{scene_idx}_compare.png"), make_grid(compare_preview)[..., ::-1])

    if FLAGS.save_viewer:
        viewer_dir = os.path.join(output_dir, "viewer", scene_name)
        os.makedirs(viewer_dir, exist_ok=True)
        sh_degree = int(round((merged["features_rest"].shape[1] + 1) ** 0.5 - 1))
        gs_utils.prepare_viewer(cameras, viewer_dir, sh_degree)
        point_cloud_dir = os.path.join(viewer_dir, "point_cloud")
        gs_utils.export_ply_forviewer(input_gs, os.path.join(point_cloud_dir, "input.ply"))
        gs_utils.export_ply_forviewer(merged, os.path.join(point_cloud_dir, "output.ply"))
        gs_utils.export_ply_forviewer(target_gs, os.path.join(point_cloud_dir, "gt.ply"))
    gs_utils.export_ply_forviewer(merged, os.path.join(output_dir, "merged_output.ply"))
    return metrics

def main(argv):
    del argv
    if FLAGS.runs_root is None:
        raise ValueError("--runs_root is required")
    if not FLAGS.gin_file:
        raise ValueError("At least one --gin_file is required to load the target scene")
    sources = discover_runs(FLAGS.runs_root, FLAGS.source_eval_subdir)
    inferred_input_factor, inferred_target_factor = infer_factors(sources)
    inferred_scene_name = infer_scene_name(sources)
    input_factor = FLAGS.input_factor if FLAGS.input_factor >= 0 else inferred_input_factor
    target_factor = FLAGS.target_factor if FLAGS.target_factor >= 0 else inferred_target_factor
    scene_name = FLAGS.scene_name or inferred_scene_name
    output_dir = FLAGS.output_dir or os.path.join(FLAGS.runs_root, "merged_eval")
    os.makedirs(output_dir, exist_ok=True)
    logger = ProcessSafeLogger(os.path.join(output_dir, "merge.log")).get_logger()

    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param, skip_unknown=True)
    set_seed()
    dataset = build_dataset()
    scene_idx = find_scene_index(dataset, scene_name)
    scene = dataset.load_scene(scene_idx)
    target_entry = scene["factor_data"][target_factor]
    images, image_names, cameras = dataset.load_factor_views(target_entry)
    reference_input, merged = merge_sources(sources)
    if merged["means"].shape[0] != target_entry["gs_params"]["means"].shape[0]:
        raise ValueError("Merged PLY Gaussian count does not match the target scene")
    metrics = evaluate_merged(
        merged, reference_input, target_entry["gs_params"], images, image_names, cameras,
        scene["idx"], scene["scene_name"], output_dir,
    )
    manifest = {
        "scene_name": scene["scene_name"],
        "input_factor": input_factor,
        "target_factor": target_factor,
        "source_eval_subdir": FLAGS.source_eval_subdir,
        "sources": {
            key: {name: str(value) for name, value in source.items()}
            for key, source in sources.items()
        },
        "metrics": metrics,
    }
    with open(os.path.join(output_dir, "merge_manifest.json"), "w") as handle:
        json.dump(manifest, handle, indent=2)
    logger.info("Merged scene=%s metrics=%s", scene["scene_name"], metrics)
    print(f"Merged scene={scene['scene_name']} metrics={metrics}")
if __name__ == "__main__":
    app.run(main)


