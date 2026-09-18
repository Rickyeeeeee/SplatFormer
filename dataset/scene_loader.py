"""CPU scene loading and rank-local microbatch sampling shared by SR trainers."""
import csv
import random
from collections import deque
from contextlib import contextmanager

import gin
import numpy as np
import torch


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

    def __init__(self, scene_count, microbatch_sizes, rank=0, world_size=1, split_across_gpus=True, seed=0,
                 scene_sampling="random", scene_counts=None, big_scene_threshold=25000):
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
        if scene_sampling not in {"random", "big_small", "avoid_big"}:
            raise ValueError(f"Unknown scene_sampling: {scene_sampling}")
        self.scene_sampling = scene_sampling
        self.big_indices = set()
        self.small_count = self.big_count = None
        if scene_sampling != "random":
            if scene_counts is None or len(scene_counts) != scene_count or min(scene_counts) < 1:
                raise ValueError("Size-aware sampling requires a positive Gaussian count for every scene")
            if big_scene_threshold <= 0:
                raise ValueError("big_scene_threshold must be positive")
            self.big_indices = {index for index, count in enumerate(scene_counts) if count > big_scene_threshold}
            self.big_count = len(self.big_indices)
            self.small_count = scene_count - self.big_count
        self.scene_indices = [index for index in range(scene_count)
                              if scene_sampling != "avoid_big" or index not in self.big_indices]
        if not self.scene_indices:
            raise ValueError("avoid_big leaves no eligible training scenes; increase big_scene_threshold")

    def __iter__(self):
        epoch = 0
        remaining, small, big = deque(), deque(), deque()
        rng = random.Random(self.seed)
        while True:
            for size in self.microbatch_sizes:
                batch = []
                has_big = False
                while len(batch) < size:
                    if not remaining and not small and not big:
                        if self.split_across_gpus:
                            order = np.asarray(self.scene_indices)[np.random.RandomState(epoch).permutation(len(self.scene_indices))]
                            per_rank = (len(order) + self.world_size - 1) // self.world_size
                            order = np.resize(order, per_rank * self.world_size)
                            order = order[self.rank * per_rank:(self.rank + 1) * per_rank].tolist()
                        else:
                            order = self.scene_indices.copy()
                            rng.shuffle(order)
                        if self.scene_sampling == "big_small":
                            small.extend(index for index in order if index not in self.big_indices)
                            big.extend(index for index in order if index in self.big_indices)
                        else:
                            remaining.extend(order)
                        epoch += 1
                    if self.scene_sampling != "big_small":
                        scene_idx = remaining.popleft()
                    elif size == 1:
                        scene_idx = big.popleft() if rng.randrange(len(small) + len(big)) < len(big) else small.popleft()
                    elif big and not has_big:
                        scene_idx = big.popleft()
                        has_big = True
                    elif small:
                        scene_idx = small.popleft()
                    else:
                        raise ValueError(f"big_small cannot fill microbatch size {size} on rank {self.rank}: insufficient small scenes; use singleton microbatches (grad_accum_steps=batch_size)")
                    batch.append(scene_idx)
                # Keep the big scene last, including when a microbatch crosses epochs.
                if self.scene_sampling == "big_small":
                    batch.sort(key=lambda index: index in self.big_indices)
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


def build_train_loader(dataset, microbatch_sizes, seed, rank=0, world_size=1, *, count_resolution=None,
                       scene_sampling="random", big_scene_threshold=25000, num_workers=2,
                       prefetch_factor=2, pin_memory=True):
    scene_counts = None
    if scene_sampling != "random":
        # CSV counts are upper estimates after finite/outlier filtering; never load checkpoints here.
        resolution = dataset.src_resolution if count_resolution is None else count_resolution
        column = f"res_{resolution}_num_gs"
        with open(dataset.scene_list, newline="") as source:
            reader = csv.DictReader(source)
            if not reader.fieldnames or not {"scene_id", column}.issubset(reader.fieldnames):
                raise ValueError(f"Size-aware sampling requires scene_id and {column} in {dataset.scene_list}")
            counts_by_name = {row["scene_id"].strip(): row.get(column) for row in reader}
        scene_counts = []
        for scene in dataset.folders:
            name = scene["scene_name"]
            try:
                count = int(counts_by_name[name])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Missing or invalid {column} for scene {name} in {dataset.scene_list}") from error
            if count < 1:
                raise ValueError(f"Invalid {column}={count} for scene {name}; expected a positive count")
            scene_counts.append(min(count, dataset.max_gs_num))
    sampler = SceneMicrobatchSampler(len(dataset.folders), microbatch_sizes, rank, world_size, dataset.split_across_gpus, seed,
                                    scene_sampling, scene_counts, big_scene_threshold)
    worker_options = {}
    if num_workers > 0:
        worker_options = dict(multiprocessing_context="spawn", persistent_workers=True, prefetch_factor=prefetch_factor)
    return torch.utils.data.DataLoader(
        SRSceneDataset(dataset), batch_sampler=sampler, collate_fn=collate_scenes,
        num_workers=num_workers, pin_memory=pin_memory, worker_init_fn=seed_loader_worker,
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
