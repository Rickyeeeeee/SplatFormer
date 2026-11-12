"""Tests for the optional GPU-resident Gaussian scene cache."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from dataset.scene_loader import SceneMicrobatchSampler
from utils.data_augmentation import rotate_gaussians
from utils.gpu_scene_cache import build_gpu_scene_cache
from utils.noise_to_gs_training import prepare_fixed_count_target
from sr.noise_to_gs import construct_path


class FakeDataset:
    max_gs_num = 4
    tgt_resolution = 128
    split_across_gpus = True

    def __init__(self, count=3, sh_degree=1):
        self.folders = list(range(count))
        self.sh_degree = sh_degree
        self.loads = 0

    def load_scene(self, index, sample_views=False):
        self.loads += 1
        count = index % 4 + 1
        gs = {"means": torch.arange(count * 3).reshape(count, 3).float() / 10,
              "scales": torch.zeros(count, 3), "opacities": torch.zeros(count, 1),
              "quats": torch.tensor([[1., 0., 0., 0.]]).repeat(count, 1),
              "features_dc": torch.ones(count, 3) * index,
              "features_rest": torch.ones(count, 3, 3)}
        return {"data": {128: {"gs_params": gs}}}


class GPUSceneCacheTests(unittest.TestCase):
    def build(self, dataset, degree=1, rank=0, world=1):
        with mock.patch("torch.cuda.mem_get_info", return_value=(1000000, 1000000)):
            return build_gpu_scene_cache(dataset, [2, 1], 42 + rank, rank, world, degree, "cpu", 0, None)

    def test_preprocessed_targets_match_training_path(self):
        class Standardizer:
            def encode(self, gs):
                return {key: value * 2 + 1 for key, value in gs.items()}

        standardizer = Standardizer()
        for degree in (0, 1):
            for mode in ("current", "morton_rank"):
                with self.subTest(degree=degree, mode=mode):
                    dataset = FakeDataset(count=4)
                    with mock.patch("torch.cuda.mem_get_info", return_value=(1000000, 1000000)):
                        cache = build_gpu_scene_cache(dataset, [8], 42, 0, 1, degree, "cpu", 0, None,
                                                      standardizer=standardizer, coordinate_mode=mode, preprocessed=True)
                    self.assertTrue(cache.preprocessed)
                    self.assertEqual(cache.offsets, [0, 4, 8, 12, 16])
                    batch = next(iter(cache))
                    sampler = iter(SceneMicrobatchSampler(4, [8], 0, 1, True, 42))
                    for index, scene in zip(next(sampler), batch):
                        actual = scene["data"][128]["gs_params"]
                        expected = dataset.load_scene(index)["data"][128]["gs_params"]
                        if degree == 0:
                            expected["features_rest"] = expected["features_rest"][:, :0]
                        expected = standardizer.encode(prepare_fixed_count_target(expected, 4, mode))
                        for key in expected:
                            torch.testing.assert_close(actual[key], expected[key])
                        before = {key: value.clone() for key, value in actual.items()}
                        first, _ = construct_path(actual, 0.5)
                        second, _ = construct_path(actual, 0.7)
                        self.assertFalse(torch.equal(first["means"], second["means"]))
                        for key in before:
                            torch.testing.assert_close(actual[key], before[key])

    def test_preprocessing_oom_falls_back(self):
        dataset = FakeDataset()
        standardizer = mock.Mock()
        standardizer.encode.side_effect = torch.cuda.OutOfMemoryError("oom")
        with mock.patch("torch.cuda.mem_get_info", return_value=(1000000, 1000000)), \
                mock.patch("torch.cuda.empty_cache"):
            cache = build_gpu_scene_cache(dataset, [1], 0, 0, 1, 0, "cpu", 0, None,
                                          standardizer=standardizer, coordinate_mode="current", preprocessed=True)
        self.assertIsNone(cache)

    def test_preprocessed_cache_falls_back_if_other_rank_runs_out_of_memory(self):
        dataset = FakeDataset()
        standardizer = mock.Mock()
        standardizer.encode.side_effect = lambda gs: gs
        def gather(outcomes, local):
            outcomes[:] = [local, ("memory", "rank 1: out of memory")]
        with mock.patch("torch.cuda.mem_get_info", return_value=(1000000, 1000000)), \
                mock.patch("torch.cuda.empty_cache"), \
                mock.patch("torch.distributed.all_gather_object", side_effect=gather):
            cache = build_gpu_scene_cache(dataset, [1], 0, 0, 2, 0, "cpu", 0, None,
                                          standardizer=standardizer, coordinate_mode="morton_rank", preprocessed=True)
        self.assertIsNone(cache)

    def test_values_order_and_rotation_match_uncached(self):
        for degree in (0, 1):
            dataset = FakeDataset()
            cache = self.build(dataset, degree)
            self.assertEqual(dataset.loads, 3)
            sampler = iter(SceneMicrobatchSampler(3, [2, 1], 0, 1, True, 42))
            batches = iter(cache)
            for _ in range(4):
                indices = next(sampler)
                scenes = next(batches)
                for index, scene in zip(indices, scenes):
                    actual = scene["data"][128]["gs_params"]
                    expected = dataset.load_scene(index)["data"][128]["gs_params"]
                    if degree == 0:
                        expected["features_rest"] = expected["features_rest"][:, :0]
                    for key in expected:
                        torch.testing.assert_close(actual[key], expected[key])
                    rotation = torch.tensor([0.9238795, 0., 0., 0.3826834])
                    rotated = rotate_gaussians(actual, rotation, (0.5, 0.5, 0.5))
                    reference = rotate_gaussians(expected, rotation, (0.5, 0.5, 0.5))
                    actual_target = prepare_fixed_count_target(rotated, 4)
                    expected_target = prepare_fixed_count_target(reference, 4)
                    for key in expected:
                        torch.testing.assert_close(actual_target[key], expected_target[key])
                        torch.testing.assert_close(actual[key], expected[key])

    def test_rank_sampler_and_fresh_noise(self):
        dataset = FakeDataset(count=5)
        def gather(outcomes, local):
            outcomes[:] = [("ok", ""), local]
        with mock.patch("torch.distributed.all_gather_object", side_effect=gather):
            cache = self.build(dataset, degree=0, rank=1, world=2)
        sampler = iter(SceneMicrobatchSampler(5, [2, 1], 1, 2, True, 43))
        iterator = iter(cache)
        for _ in range(4):
            indices = next(sampler)
            batch = next(iterator)
            self.assertEqual([int(item["data"][128]["gs_params"]["features_dc"][0, 0]) for item in batch], indices)
        target = prepare_fixed_count_target(next(iterator)[0]["data"][128]["gs_params"], 4)
        first, _ = construct_path(target, 0.5)
        second, _ = construct_path(target, 0.5)
        self.assertFalse(torch.equal(first["means"], second["means"]))

    def test_memory_preflight_and_oom_fall_back(self):
        dataset = FakeDataset()
        with mock.patch("torch.cuda.mem_get_info", return_value=(1, 1)), mock.patch("torch.cuda.empty_cache"):
            self.assertIsNone(build_gpu_scene_cache(dataset, [1], 0, 0, 1, 0, "cpu", 0, None))
        self.assertEqual(dataset.loads, 0)
        with mock.patch("torch.cuda.mem_get_info", return_value=(1000000, 1000000)), \
                mock.patch("torch.cuda.empty_cache"), \
                mock.patch.object(dataset, "load_scene", side_effect=torch.cuda.OutOfMemoryError("oom")):
            self.assertIsNone(build_gpu_scene_cache(dataset, [1], 0, 0, 1, 0, "cpu", 0, None))

    def test_other_errors_propagate_and_remote_failure_causes_fallback(self):
        dataset = FakeDataset()
        with mock.patch("torch.cuda.mem_get_info", return_value=(1000000, 1000000)), \
                mock.patch("torch.cuda.empty_cache"), \
                mock.patch.object(dataset, "load_scene", side_effect=ValueError("bad data")):
            with self.assertRaisesRegex(RuntimeError, "bad data"):
                build_gpu_scene_cache(dataset, [1], 0, 0, 1, 0, "cpu", 0, None)
        def gather(outcomes, local):
            outcomes[:] = [local, ("memory", "rank 1: insufficient memory")]
        with mock.patch("torch.cuda.mem_get_info", return_value=(1000000, 1000000)), \
                mock.patch("torch.cuda.empty_cache"), mock.patch("torch.distributed.all_gather_object", side_effect=gather):
            self.assertIsNone(build_gpu_scene_cache(dataset, [1], 0, 0, 2, 0, "cpu", 0, None))


if __name__ == "__main__":
    unittest.main()
