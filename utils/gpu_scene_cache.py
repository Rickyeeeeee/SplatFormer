"""Optional compact GPU cache for normalized noise-to-GS training scenes."""

import gc
import math
import time

import torch
import torch.distributed as dist

from dataset.scene_loader import SceneMicrobatchSampler
from utils.gs_normalization import STATISTIC_KEYS
from utils.noise_to_gs_training import prepare_fixed_count_target


class GPUSceneCache:
    def __init__(self, tensors, offsets, sampler, resolution, preprocessed=False):
        self.tensors = tensors
        self.offsets = offsets
        self.sampler = sampler
        self.resolution = resolution
        self.num_workers = 0
        self.preprocessed = preprocessed

    def __iter__(self):
        for indices in self.sampler:
            batch = []
            for index in indices:
                start, end = self.offsets[index:index + 2]
                gaussians = {key: value[start:end] for key, value in self.tensors.items()}
                batch.append({"data": {self.resolution: {"gs_params": gaussians}}})
            yield batch


def build_gpu_scene_cache(dataset, sizes, seed, rank, world_size, sh_degree, device, reserve_gib, logger,
                          standardizer=None, coordinate_mode=None, preprocessed=False):
    """Build a full per-rank cache, or coordinate a fallback on memory pressure."""
    if not math.isfinite(reserve_gib) or reserve_gib < 0:
        raise ValueError("GPU cache headroom must be nonnegative")
    start_time = time.monotonic()
    width = 14 + 3 * ((sh_degree + 1) ** 2 - 1)
    estimate = len(dataset.folders) * dataset.max_gs_num * width * 4
    scratch_estimate = dataset.max_gs_num * (width * 4 * 3 + 128) if preprocessed else 0
    free_bytes, _ = torch.cuda.mem_get_info(device)
    reserve = int(reserve_gib * (1024 ** 3))
    cache = None
    status, reason = "ok", ""
    if free_bytes < estimate + scratch_estimate + reserve:
        status = "memory"
        reason = f"rank {rank}: free={free_bytes} estimated_cache={estimate} scratch={scratch_estimate} reserve={reserve} bytes"
    else:
        try:
            # Preprocessed caches allocate final storage once and prepare one scene at a time.
            parts = {key: [] for key in STATISTIC_KEYS}
            offsets = [0]
            tensors = {}
            gaussians = prepared = value = None
            for index in range(len(dataset.folders)):
                scene = dataset.load_scene(index, sample_views=True)
                gaussians = scene["data"][dataset.tgt_resolution]["gs_params"]
                if sh_degree == 0:
                    gaussians["features_rest"] = gaussians["features_rest"][:, :0, :]
                if preprocessed:
                    gaussians = {key: gaussians[key].to(device) for key in STATISTIC_KEYS}
                    prepared = standardizer.encode(prepare_fixed_count_target(gaussians, dataset.max_gs_num, coordinate_mode))
                    if not tensors:
                        for key, value in prepared.items():
                            tensors[key] = torch.empty((len(dataset.folders) * dataset.max_gs_num,) + value.shape[1:],
                                                       dtype=value.dtype, device=device)
                    start = index * dataset.max_gs_num
                    for key, value in prepared.items():
                        tensors[key][start:start + dataset.max_gs_num].copy_(value)
                    offsets.append(start + dataset.max_gs_num)
                    gaussians = prepared = value = None
                else:
                    for key in parts:
                        parts[key].append(gaussians[key].contiguous())
                    offsets.append(offsets[-1] + len(gaussians["means"]))
            if not preprocessed:
                for key, values in parts.items():
                    tensors[key] = torch.cat(values, dim=0).to(device)
                    parts[key] = []
            sampler = SceneMicrobatchSampler(len(dataset.folders), sizes, rank, world_size,
                                             dataset.split_across_gpus, seed)
            cache = GPUSceneCache(tensors, offsets, sampler, dataset.tgt_resolution, preprocessed=preprocessed)
            actual = sum(value.numel() * value.element_size() for value in tensors.values())
            free_after, _ = torch.cuda.mem_get_info(device)
            if free_after < reserve:
                status = "memory"
                reason = f"rank {rank}: free after cache={free_after} reserve={reserve} bytes"
            if logger is not None:
                logger.info("gpu_scene_cache rank=%d preprocessed=%s estimate_bytes=%d scratch_bytes=%d actual_bytes=%d free_before=%d free_after=%d build_seconds=%.2f",
                            rank, preprocessed, estimate, scratch_estimate, actual, free_bytes, free_after, time.monotonic() - start_time)
        except torch.cuda.OutOfMemoryError:
            status, reason = "memory", f"rank {rank}: CUDA out of memory while building scene cache"
        except Exception as error:
            status, reason = "error", f"rank {rank}: {type(error).__name__}: {error}"
    # Release partial allocations before communicating a failure.
    if status != "ok":
        cache = None
        gaussians = prepared = value = None
        if 'tensors' in locals():
            tensors.clear()
        if 'parts' in locals():
            parts.clear()
        gc.collect()
        torch.cuda.empty_cache()
    outcomes = [(status, reason)]
    if world_size > 1:
        outcomes = [None] * world_size
        dist.all_gather_object(outcomes, (status, reason))
    errors = [message for kind, message in outcomes if kind == "error"]
    if errors:
        raise RuntimeError("GPU scene cache failed: " + "; ".join(errors))
    failures = [message for kind, message in outcomes if kind == "memory"]
    if failures:
        cache = None
        gaussians = prepared = value = None
        if 'tensors' in locals():
            tensors.clear()
        gc.collect()
        torch.cuda.empty_cache()
        if logger is not None:
            logger.warning("GPU scene cache disabled; using normal loader: %s", "; ".join(failures))
    return cache
