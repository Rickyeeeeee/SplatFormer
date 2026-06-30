import csv
import glob
import importlib.util
import json
import os
from pathlib import Path

import cv2
import gin
import numpy as np
import torch
from absl import app, flags

from models.feature_flow_predictor import GSFlowPredictor
from utils import gpu_utils
from utils.log_utils import ProcessSafeLogger
from utils.sr_densify_utils import build_densified_input_gs


def _load_gsfm_lib():
    lib_path = Path(__file__).with_name("overfit-sr-gsfm.py")
    spec = importlib.util.spec_from_file_location("overfit_sr_gsfm_lib", lib_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gsfm = _load_gsfm_lib()

flags.DEFINE_string(
    "checkpoint",
    "",
    "Model checkpoint to evaluate. Defaults to output_dir/checkpoints/model_last.pth, then latest model_*.pth.",
)
flags.DEFINE_string("eval_output_dir", "", "Evaluation output directory. Defaults to output_dir/eval_flow_steps.")
flags.DEFINE_string("eval_flow_steps", "1-10", "Comma-separated flow steps and/or ranges, e.g. '1-10' or '1,2,5,10'.")
flags.DEFINE_string("device", "cuda", "Torch device for evaluation.")
flags.DEFINE_boolean("strict_checkpoint", True, "Use strict model.load_state_dict.")
flags.DEFINE_boolean("save_step_overview", True, "Save one overview image containing each step preview grid.")
flags.DEFINE_boolean(
    "load_train_config",
    True,
    "Load output_dir/config.gin when present so standalone eval uses the training run config.",
)

FLAGS = flags.FLAGS


def _parse_eval_flow_steps(spec):
    steps = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            start = int(left)
            end = int(right)
            stride = 1 if end >= start else -1
            steps.extend(range(start, end + stride, stride))
        else:
            steps.append(int(part))
    if len(steps) == 0:
        raise ValueError("eval_flow_steps did not contain any steps")
    if any(step <= 0 for step in steps):
        raise ValueError(f"All eval flow steps must be positive, got {steps}")
    return steps


def _resolve_checkpoint(output_dir, checkpoint):
    if checkpoint:
        return checkpoint

    ckpt_dir = os.path.join(output_dir, "checkpoints")
    last_ckpt = os.path.join(ckpt_dir, "model_last.pth")
    if os.path.exists(last_ckpt):
        return last_ckpt

    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "model_*.pth")))
    if len(ckpts) > 0:
        return ckpts[-1]

    raise FileNotFoundError(
        f"No checkpoint found. Pass --checkpoint or place model_last.pth/model_*.pth under {ckpt_dir}"
    )


def _state_dict_from_checkpoint(checkpoint_path):
    payload = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(payload, dict):
        for key in ["state_dict", "model", "model_state_dict"]:
            if key in payload and isinstance(payload[key], dict):
                payload = payload[key]
                break
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")

    if any(key.startswith("module.") for key in payload.keys()):
        payload = {key.removeprefix("module."): value for key, value in payload.items()}
    return payload


def _write_summary(output_dir, records):
    json_path = os.path.join(output_dir, "metrics_summary.json")
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2)

    metric_keys = sorted({key for record in records for key in record["metrics"].keys()})
    input_metric_keys = sorted({key for record in records for key in record["metrics_input"].keys()})
    csv_path = os.path.join(output_dir, "metrics_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["flow_steps"]
            + metric_keys
            + [f"input_{key}" for key in input_metric_keys]
            + ["output_dir"]
        )
        for record in records:
            writer.writerow(
                [record["flow_steps"]]
                + [record["metrics"].get(key, "") for key in metric_keys]
                + [record["metrics_input"].get(key, "") for key in input_metric_keys]
                + [record["output_dir"]]
            )
    return json_path, csv_path


def _label_image(image, label):
    top = 36
    out = np.zeros((image.shape[0] + top, image.shape[1], 3), dtype=np.uint8)
    out[top:] = image
    cv2.putText(out, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (235, 235, 235), 2, cv2.LINE_AA)
    return out


def _save_step_overview(output_dir, records, scene_idx):
    rows = []
    for record in records:
        preview_path = os.path.join(record["output_dir"], f"scene{scene_idx}_pred.png")
        if not os.path.exists(preview_path):
            continue
        image = cv2.imread(preview_path, cv2.IMREAD_COLOR)
        if image is None:
            continue
        rows.append(_label_image(image, f"flow_steps={record['flow_steps']}"))

    if len(rows) == 0:
        return None

    width = max(row.shape[1] for row in rows)
    padded_rows = []
    for row in rows:
        if row.shape[1] == width:
            padded_rows.append(row)
            continue
        pad = np.zeros((row.shape[0], width - row.shape[1], 3), dtype=np.uint8)
        padded_rows.append(np.concatenate([row, pad], axis=1))

    overview = np.concatenate(padded_rows, axis=0)
    overview_path = os.path.join(output_dir, "flow_steps_overview.png")
    cv2.imwrite(overview_path, overview)
    return overview_path


def main(argv):
    del argv

    config_files = list(FLAGS.gin_file or [])
    train_config = os.path.join(FLAGS.output_dir, "config.gin")
    if FLAGS.load_train_config and os.path.exists(train_config):
        config_files = [train_config]
    gin.parse_config_files_and_bindings(config_files, FLAGS.gin_param)
    flow_cfg = gsfm._resolve_flow_cfg()
    gsfm.set_seed()

    eval_steps = _parse_eval_flow_steps(FLAGS.eval_flow_steps)
    eval_output_dir = FLAGS.eval_output_dir or os.path.join(FLAGS.output_dir, "eval_flow_steps")
    os.makedirs(eval_output_dir, exist_ok=True)

    logger = ProcessSafeLogger(os.path.join(eval_output_dir, "eval.log")).get_logger()
    checkpoint_path = _resolve_checkpoint(FLAGS.output_dir, FLAGS.checkpoint)

    device = torch.device(FLAGS.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    dataset = gsfm._build_dataset()
    scene_idx = gsfm._find_scene_index(dataset, FLAGS.scene_name)
    scene = dataset.load_scene(scene_idx)
    input_factor_entry = scene["factor_data"][FLAGS.input_factor]
    target_factor_entry = scene["factor_data"][FLAGS.target_factor]

    # Mirror overfit-sr-gsfm.py setup order. The unused train payload matters when
    # the dataset uses random background color because it advances the RNG before
    # constructing the eval payload.
    _train_payload = gsfm._build_split_payload(
        dataset,
        scene["idx"],
        scene["scene_name"],
        target_factor_entry,
        split="train",
    )
    eval_payload = gsfm._build_split_payload(
        dataset,
        scene["idx"],
        scene["scene_name"],
        target_factor_entry,
        split="test",
    )
    eval_images = eval_payload["images"]
    eval_cameras = eval_payload["cameras"]
    eval_chunk_size = dataset.image_per_scene if dataset.image_per_scene is not None else len(eval_images)
    if eval_chunk_size <= 0:
        eval_chunk_size = len(eval_images)

    target_gs_raw = gpu_utils.move_to_device(target_factor_entry["gs_params"], device)

    model = GSFlowPredictor().to(device)
    model.load_state_dict(_state_dict_from_checkpoint(checkpoint_path), strict=FLAGS.strict_checkpoint)
    model.eval()

    loss_features = gsfm._parse_loss_features(FLAGS.loss_features, model, target_gs_raw)
    fixed_attribute_keys = gsfm._fixed_attribute_keys(loss_features, target_gs_raw)

    input_gs_raw = build_densified_input_gs(
        input_factor_entry=input_factor_entry,
        target_factor_entry=target_factor_entry,
        alignment=FLAGS.alignment,
        attribute_init=FLAGS.attribute_init,
        emd_eps=FLAGS.emd_eps,
        emd_iters=FLAGS.emd_iters,
        device=device,
        output_dir=eval_output_dir,
        gt_attribute_keys=fixed_attribute_keys,
    )
    input_gs_raw = gsfm._copy_gt_attributes(input_gs_raw, target_gs_raw, fixed_attribute_keys)
    source_flow_gs = gsfm.raw_to_flow_gs(input_gs_raw, flow_cfg["flow_space"])
    target_flow_gs = gsfm.raw_to_flow_gs(target_gs_raw, flow_cfg["flow_space"])
    source_flow_gs = gsfm._apply_fixed_flow_attributes(source_flow_gs, target_flow_gs, fixed_attribute_keys)

    with open(os.path.join(eval_output_dir, "config.gin"), "w") as f:
        f.writelines(gin.operative_config_str())

    run_config = {
        "checkpoint": checkpoint_path,
        "scene_idx": int(scene["idx"]),
        "scene_name": scene["scene_name"],
        "input_factor": int(FLAGS.input_factor),
        "target_factor": int(FLAGS.target_factor),
        "alignment": FLAGS.alignment,
        "attribute_init": FLAGS.attribute_init,
        "loss_features": loss_features,
        "fixed_attribute_keys": fixed_attribute_keys,
        "flow_space": flow_cfg["flow_space"],
        "eval_flow_steps": eval_steps,
        "eval_views": len(eval_images),
        "loaded_train_config": train_config if FLAGS.load_train_config and os.path.exists(train_config) else None,
        "gin_files": config_files,
    }
    with open(os.path.join(eval_output_dir, "eval_config.json"), "w") as f:
        json.dump(run_config, f, indent=2)

    logger.info(
        "Evaluating checkpoint=%s scene=%s idx=%s flow_space=%s flow_steps=%s loss_features=%s fixed=%s config=%s",
        checkpoint_path,
        scene["scene_name"],
        scene["idx"],
        flow_cfg["flow_space"],
        eval_steps,
        ",".join(loss_features),
        ",".join(fixed_attribute_keys) if fixed_attribute_keys else "none",
        train_config if FLAGS.load_train_config and os.path.exists(train_config) else ",".join(config_files),
    )

    records = []
    for idx, flow_steps in enumerate(eval_steps):
        step_dir = os.path.join(eval_output_dir, f"flow_steps_{flow_steps:02d}")
        metrics, metrics_input = gsfm.evaluate_single_scene(
            model=model,
            input_gs=input_gs_raw,
            source_flow_gs=source_flow_gs,
            gt_gs=target_gs_raw,
            scene_idx=eval_payload["scene_idx"],
            scene_name=eval_payload["scene_name"],
            eval_images=eval_images,
            eval_cameras=eval_cameras,
            image_names=eval_payload["images_name"],
            output_dir=step_dir,
            flow_steps=int(flow_steps),
            flow_space=flow_cfg["flow_space"],
            fixed_raw_gs=target_gs_raw,
            fixed_flow_gs=target_flow_gs,
            fixed_attribute_keys=fixed_attribute_keys,
            eval_chunk_size=eval_chunk_size,
            compare_with_input=FLAGS.compare_with_input,
            save_viewer=FLAGS.save_viewer,
            save_residuals=FLAGS.save_residuals,
            output_gt=(idx == 0),
        )
        record = {
            "flow_steps": int(flow_steps),
            "metrics": metrics,
            "metrics_input": metrics_input,
            "output_dir": step_dir,
        }
        records.append(record)
        metric_str = " ".join([f"{key}: {value:.4f}" for key, value in metrics.items()])
        logger.info("flow_steps=%s %s", flow_steps, metric_str)

    summary_json, summary_csv = _write_summary(eval_output_dir, records)
    overview_path = _save_step_overview(eval_output_dir, records, scene["idx"]) if FLAGS.save_step_overview else None

    print(f"Wrote summary JSON: {summary_json}")
    print(f"Wrote summary CSV: {summary_csv}")
    if overview_path is not None:
        print(f"Wrote overview image: {overview_path}")


if __name__ == "__main__":
    app.run(main)
