import os
import hashlib
import random
from contextlib import contextmanager
from pathlib import Path

import torch.distributed as dist

import torch
import numpy as np
import math
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import torch
import lpips
import json
from utils import gpu_utils, gs_utils

class MetricComputer:
    def __init__(self, forward_bs=8):
        lpips_fn = lpips.LPIPS(net='vgg', verbose=False).to('cuda')
        self.metrics = {
            'psnr': lambda x,y: psnr(x,y).squeeze(), #(N,)
            'ssim': lambda x, y: ssim(x.permute(0,3,1,2),y.permute(0,3,1,2), window_size=11, size_average=False), #(N,)
            'lpips': lambda x,y: lpips_fn(x.permute((0,3,1,2)),y.permute(0,3,1,2),normalize=True).squeeze() #(N,)
        }
        self.results = {metric: [] for metric in self.metrics.keys()}
        self.results_dict = {}
        self.forward_bs = 16

    def update(self, img1s, img2s, name):
        if name not in self.results_dict:
            self.results_dict[name] = {}
        # Metrics always operate on float images in the [0, 1] range.
        img1s = img1s.float()
        img2s = img2s.float()
        if img1s.max() > 1: #255
            img1s = img1s/255.0
        if img2s.max() > 1:
            img2s = img2s/255.0
        for metric, fn in self.metrics.items():
            if metric=='lpips':
                # Concerning OOD issue, we need to split imgs into batches
                total_n = img1s.size(0)
                n_batches = math.ceil(total_n/self.forward_bs)
                for i in range(n_batches): #But later I found that this is not the issue
                    start = i*self.forward_bs
                    end = min((i+1)*self.forward_bs, total_n)
                    res = fn(img1s[start:end], img2s[start:end])
                    if res.dim()==0:
                        res = res.unsqueeze(0)
                    self.results[metric].append(res)
            else:
                res = fn(img1s, img2s)
                if res.dim()==0:
                    res = res.unsqueeze(0)
                self.results[metric].append(res)
            # turn 0-dim to 1-dim
            self.results_dict[name][metric] = [r.item() for r in res] #self.results[metric][-1].mean().item() #?? (per-scene) (per-img?)


    def update_value(self, key, value, name):
        if key in self.results:
            self.results[key].append(value)
        else:
            self.results[key] = [value]
        if name not in self.results_dict:
            self.results_dict[name] = {}
        self.results_dict[name][key] = value.item()

    def sum(self):
        self.summation = {}
        for metric in self.results.keys():
            if max([x.dim() for x in self.results[metric]]) == 0:
                self.summation[metric] = sum(self.results[metric])
            else:
                self.summation[metric] = torch.cat(self.results[metric]).sum()
        return self.summation
    
    def concat(self):
        self.concatenation = {}
        for metric in self.results.keys():
            self.concatenation[metric] = torch.cat(self.results[metric])
        return self.concatenation
    
    def finalize(self):
        #compute the mean of the results
        self.reduced = {}
        for metric in self.results.keys():
            self.reduced[metric] = torch.cat(self.results[metric]).mean().item()
        return self.reduced
    
    def write_to_file(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.results_dict, f, indent=4)
    
def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2, m=None):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True, keep_featuremap=False):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if keep_featuremap:
        return ssim_map.mean(1) # [bs, H, W]
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def render_gs_average_metrics(gs, images, cameras, chunk_size, device):
    metric_computer = MetricComputer()
    num_views = len(images)
    if num_views == 0:
        raise ValueError("Cannot compute render metrics with zero views")
    if chunk_size is None or chunk_size <= 0:
        chunk_size = num_views
    chunk_size = min(chunk_size, num_views)

    with torch.no_grad():
        for start in range(0, num_views, chunk_size):
            end = min(start + chunk_size, num_views)
            chunk_images = gpu_utils.move_to_device(images[start:end], device)
            chunk_cameras = {
                key: (value[start:end] if key == "camera_to_worlds" else value)
                for key, value in cameras.items()
            }
            chunk_cameras = gpu_utils.move_to_device(chunk_cameras, device)

            pred_imgs, _ = gs_utils.rasterize_gaussians_to_multiimgs(gs, chunk_cameras)
            pred_imgs = torch.stack(pred_imgs, dim=0)
            gt_imgs = torch.stack(chunk_images, dim=0)

            if gt_imgs.shape[-1] == 4:
                masks = gt_imgs[..., 3].unsqueeze(-1)
                pred_imgs = (pred_imgs * masks * 255).to(torch.uint8)
                gt_imgs = (gt_imgs[..., :3] * 255).to(torch.uint8)
            else:
                pred_imgs = (pred_imgs * 255).to(torch.uint8)
                gt_imgs = (gt_imgs * 255).to(torch.uint8)

            metric_computer.update(pred_imgs, gt_imgs, name=f"{start:06d}_{end:06d}")

    return metric_computer.finalize()


def write_densify_stage_render_metrics(output_dir, stage_gs, images, cameras, chunk_size, device):
    stage_dir = os.path.join(output_dir, "densify_init")
    os.makedirs(stage_dir, exist_ok=True)
    metrics_by_ply = {}
    for ply_name, gs in stage_gs.items():
        metrics_by_ply[ply_name] = render_gs_average_metrics(gs, images, cameras, chunk_size, device)
    with open(os.path.join(stage_dir, "render_metrics.json"), "w") as f:
        json.dump(metrics_by_ply, f, indent=2)
    return metrics_by_ply


# Test reports use complete accumulated tensors; legacy chunk serialization stays unchanged.
METRIC_NAMES = ("psnr", "ssim", "lpips")


def scene_metric_report(source, scene_idx, scene_name, image_names, computer):
    values = {key: torch.cat([value.reshape(-1) for value in computer.results[key]]).detach().cpu().double().tolist()
              for key in METRIC_NAMES}
    if any(len(values[key]) != len(image_names) for key in METRIC_NAMES):
        raise ValueError(f"Metric/image count mismatch for {scene_name}")
    images = [{"image_id": index, "image_name": str(name), **{key: values[key][index] for key in METRIC_NAMES}}
              for index, name in enumerate(image_names)]
    return {"source": source, "scene_idx": int(scene_idx), "scene_name": scene_name, "num_images": len(images),
            "mean": {key: float(np.mean(values[key])) for key in METRIC_NAMES} if images else None, "images": images}


def write_metric_json(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w") as handle:
        json.dump(report, handle, indent=2)
    temporary.replace(path)


def metric_scene_summary(report, path, root):
    return {**{key: report[key] for key in ("scene_idx", "scene_name", "num_images", "mean")},
            "metrics_file": os.path.relpath(path, root)}


def dataset_metric_report(source, scenes, excluded=()):
    scenes = sorted(scenes, key=lambda item: item["scene_idx"])
    if len({item["scene_idx"] for item in scenes}) != len(scenes) or len({item["scene_name"] for item in scenes}) != len(scenes):
        raise ValueError("Duplicate scenes in distributed evaluation results")
    return {"source": source, "averaging": "mean_of_scene_means", "num_scenes": len(scenes),
            "num_images": sum(item["num_images"] for item in scenes),
            "mean": {key: float(np.mean([item["mean"][key] for item in scenes])) for key in METRIC_NAMES} if scenes else None,
            "scenes": scenes, "num_excluded_scenes": len(excluded), "excluded_scenes": list(excluded)}


def gather_metric_records(records):
    if not dist.is_initialized():
        return records
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, records)
    return [item for shard in gathered for item in shard]


@contextmanager
def preserve_metric_rng(device):
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def render_test_baseline(source, gs, scene, target_data, chunk_size, device):
    with preserve_metric_rng(device), torch.no_grad():
        computer = MetricComputer()
        gs = gpu_utils.move_to_device(gs, device)
        images = target_data["images"]
        if not images:
            raise ValueError(f"Evaluation has zero views: {scene['scene_name']}")
        chunk_size = min(chunk_size, len(images)) if chunk_size is not None and chunk_size > 0 else len(images)
        for start in range(0, len(images), chunk_size):
            end = min(start + chunk_size, len(images))
            cameras = {key: value[start:end] if key == "camera_to_worlds" else value
                       for key, value in target_data["cameras"].items()}
            cameras = gpu_utils.move_to_device(cameras, device)
            predictions, _ = gs_utils.rasterize_gaussians_to_multiimgs(gs, cameras)
            predictions = torch.stack(predictions)
            targets = torch.stack(gpu_utils.move_to_device(images[start:end], device))
            if targets.shape[-1] == 4:
                predictions = predictions * targets[..., 3:4]
            predictions = (predictions * 255).to(torch.uint8)
            targets = (targets[..., :3] * 255).to(torch.uint8)
            computer.update(predictions, targets, name=str(start))
        return scene_metric_report(source, scene["scene_idx"], scene["scene_name"], target_data["images_name"], computer)


def test_baseline_signature(dataset, indices, options, sources):
    attributes = ("dataset_root", "scene_list", "src_resolution", "tgt_resolution", "remove_outlier_ndevs",
                  "max_gs_num", "background_color", "coordinate_frame", "coordinate_frame_version")
    config = {name: getattr(dataset, name, None) for name in attributes}
    config.update(options)
    artifacts = []
    scenes = []
    for index in indices:
        info = dataset.folders[index]
        scenes.append({"scene_idx": index, "scene_name": info["scene_name"]})
        paths = info.get("resolution_paths", {})
        roots = set()
        for resolution_paths in paths.values():
            roots.update(resolution_paths[key] for key in ("image_dir", "sparse_dir") if key in resolution_paths)
            if "gsplat_dir" in resolution_paths:
                roots.add(os.path.join(resolution_paths["gsplat_dir"], "ckpts"))
        if "target_fit_lr_to_hr" in sources:
            roots.add(os.path.join(dataset.pretrained_path("fit_lr_to_hr", info["scene_name"], dataset.src_resolution), "ckpts"))
        if options.get("alignment") == "fit_hr_to_lr":
            roots.add(os.path.join(dataset.pretrained_path("fit_hr_to_lr", info["scene_name"], dataset.tgt_resolution), "ckpts"))
        for root in sorted(roots):
            path = Path(root)
            artifacts.append((str(path), path.exists()))
            if path.exists():
                for entry in sorted(path.rglob("*")):
                    if entry.is_file():
                        stat = entry.stat()
                        artifacts.append((str(entry), stat.st_size, stat.st_mtime_ns))
    payload = {"version": 1, "config": config, "scenes": scenes, "sources": list(sources), "artifacts": artifacts}
    # JSON round-trip gives manifests stable representations for tuple-valued configuration.
    payload = json.loads(json.dumps(payload, default=str))
    payload["fingerprint"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return payload


class TestBaselineReports:
    """Cache fixed test render metrics; only summaries participate in distributed collectives."""

    def __init__(self, root, dataset, indices, options, sources):
        self.root = Path(root)
        self.sources = tuple(sources)
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.records = {source: [] for source in sources}
        decision = [None]
        self.signature = None
        if self.rank == 0:
            try:
                signature = test_baseline_signature(dataset, indices, options, sources)
                manifest_path = self.root / "manifest.json"
                reusable = False
                if manifest_path.exists():
                    try:
                        manifest = json.loads(manifest_path.read_text())
                        reusable = manifest.get("signature") == signature and self.complete(manifest)
                    except (OSError, ValueError, KeyError, TypeError):
                        reusable = False
                self.signature = signature
                decision[0] = {"reuse": reusable}
            except Exception as error:
                decision[0] = {"error": str(error)}
        if dist.is_initialized():
            dist.broadcast_object_list(decision, src=0)
        if "error" in decision[0]:
            raise RuntimeError(f"Cannot prepare test baselines: {decision[0]['error']}")
        self.reuse = decision[0]["reuse"]

    def complete(self, manifest):
        if not manifest.get("complete") or manifest.get("excluded_scenes"):
            return False
        expected = {item["scene_name"] for item in manifest["signature"]["scenes"]}
        for source in self.sources:
            report = json.loads((self.root / f"metrics_{source}.json").read_text())
            if report["source"] != source or {item["scene_name"] for item in report["scenes"]} != expected:
                return False
            if report != dataset_metric_report(source, report["scenes"], report["excluded_scenes"]):
                return False
            for item in report["scenes"]:
                detail = json.loads((self.root / item["metrics_file"]).read_text())
                if detail["source"] != source or len(detail["images"]) != item["num_images"] or detail["mean"] != item["mean"]:
                    return False
                if [image["image_name"] for image in detail["images"]] != manifest["views"][item["scene_name"]]:
                    return False
                if [image["image_id"] for image in detail["images"]] != list(range(item["num_images"])):
                    return False
                if metric_scene_summary(detail, self.root / item["metrics_file"], self.root) != item:
                    return False
                if any(any(key not in image for key in METRIC_NAMES) for image in detail["images"]):
                    return False
        return True

    def scene(self, scene, target_data, gaussian_sets, chunk_size, device):
        means = {}
        summaries = {}
        for source in self.sources:
            path = self.root / "scenes" / scene["scene_name"] / f"metrics_{source}.json"
            if self.reuse:
                report = json.loads(path.read_text())
                if [image["image_name"] for image in report["images"]] != list(target_data["images_name"]):
                    raise ValueError(f"Cached test views changed for {scene['scene_name']}")
            else:
                report = render_test_baseline(source, gaussian_sets[source], scene, target_data, chunk_size, device)
                write_metric_json(path, report)
            means[source] = report["mean"]
            summaries[source] = metric_scene_summary(report, path, self.root)
        # Commit a scene only after every baseline succeeded.
        for source, summary in summaries.items():
            self.records[source].append(summary)
        return means

    def finish(self, excluded=()):
        reports = {}
        for source in self.sources:
            records = gather_metric_records(self.records[source])
            reports[source] = dataset_metric_report(source, records, excluded)
        if self.rank == 0 and (not self.reuse or excluded):
            for source, report in reports.items():
                write_metric_json(self.root / f"metrics_{source}.json", report)
            views = {}
            for item in reports[self.sources[0]]["scenes"]:
                detail = json.loads((self.root / item["metrics_file"]).read_text())
                views[item["scene_name"]] = [image["image_name"] for image in detail["images"]]
            write_metric_json(self.root / "manifest.json", {"signature": self.signature, "complete": True,
                              "views": views, "excluded_scenes": list(excluded)})
        return reports

    def links(self, directory):
        return {source: os.path.relpath(self.root / f"metrics_{source}.json", directory) for source in self.sources}


_LOGGED_TEST_BASELINES = set()


def log_test_baselines(wandb, root, reports):
    if wandb is None or wandb.run is None or (dist.is_initialized() and dist.get_rank() != 0):
        return
    identity = (id(wandb.run), os.path.abspath(root))
    if identity in _LOGGED_TEST_BASELINES:
        return
    values = {f"eval_baselines/{source}/{key}": value for source, report in reports.items()
              for key, value in (report["mean"] or {}).items()}
    wandb.log(values, commit=False)
    _LOGGED_TEST_BASELINES.add(identity)
