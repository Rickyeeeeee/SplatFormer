"""Coverage for retained SR scene loading and coordinate normalization."""

import unittest
from unittest import mock
import tempfile
from pathlib import Path

import torch

from dataset.GS_SR import SplatFactoSRDataset
from dataset.scene_loader import SRSceneDataset


class SRSceneLoadingTests(unittest.TestCase):
    def setUp(self):
        self.source = {"means": torch.tensor([[0., 0., 0.], [2., 2., 2.]]),
                       "scales": torch.zeros(2, 3)}
        self.target = {"means": torch.tensor([[1., 1., 1.], [3., 3., 3.]]),
                       "scales": torch.zeros(2, 3)}
        self.camera = torch.eye(4).unsqueeze(0)

    def make_dataset(self, **overrides):
        settings = dict(train_or_test="test", dataset_root="/tmp/fixture", train_scene_list="train.csv",
                        test_scene_list="test.csv", image_per_scene=1, remove_outlier_ndevs=-1,
                        max_gs_num=100, split_across_gpus=False, src_resolution=32, tgt_resolution=128,
                        load_src_images=False, load_tgt_images=False)
        settings.update(overrides)
        with mock.patch("dataset.GS_SR.gs_io.read_scene_names", return_value=["fixture"]):
            return SplatFactoSRDataset(**settings)

    def load_gaussians(self, path):
        return (self.source if "/32/" in path else self.target), "ckpt"

    def test_paired_gaussians_share_source_coordinate_frame(self):
        dataset = self.make_dataset()
        with mock.patch("dataset.GS_SR.gs_io.load_gsplat", side_effect=self.load_gaussians):
            scene = dataset.load_scene(0)
        torch.testing.assert_close(scene["data"][32]["gs_params"]["means"], self.source["means"] / 2)
        torch.testing.assert_close(scene["data"][128]["gs_params"]["means"], self.target["means"] / 2)
        self.assertEqual(scene["coordinate_resolution"], 32)

    def test_source_gaussians_and_target_images_through_both_loaders(self):
        dataset = self.make_dataset(load_tgt_gs=False, load_tgt_images=True)
        meta = {"meta": {"camera_to_worlds": self.camera}}
        views = ([torch.zeros(2, 2, 3)], ["view.png"], {"camera_to_worlds": self.camera})
        with mock.patch("dataset.GS_SR.gs_io.load_gsplat", side_effect=self.load_gaussians) as load, \
             mock.patch.object(dataset, "load_camera_meta", return_value=meta), \
             mock.patch.object(dataset, "load_resolution_views", return_value=views):
            scenes = (next(iter(dataset)), SRSceneDataset(dataset)[0])
        for scene in scenes:
            self.assertIn("gs_params", scene["data"][32])
            self.assertNotIn("gs_params", scene["data"][128])
            self.assertEqual(scene["data"][128]["images_name"], ["view.png"])
        self.assertEqual(load.call_count, 2)

    def test_shapenet_paths_and_checkpoint_override(self):
        dataset = self.make_dataset(dataset_layout="shapenet", image_dataset_root="/images",
                                    gs_dataset_root="/splats")
        scene = "03001627/model"
        paths = dataset.resolution_paths(scene, 128)
        self.assertEqual(paths["colmap_dir"], "/images/shapenet/128/colmap/03001627/model")
        self.assertEqual(paths["gsplat_dir"], "/splats/shapenet/128/gsplat/03001627/model")
        self.assertEqual(dataset.resolution_paths(scene, 128, gs_root="/override")["gsplat_dir"],
                         "/override/shapenet/128/gsplat/03001627/model")
        split = self.make_dataset()
        self.assertEqual(split.resolution_paths(scene, 128, gs_root="/override")["gsplat_dir"],
                         "/override/test-set/objaverse/128/gsplat/03001627/model")

    def test_shapenet_manifests_and_equal_resolution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.csv"
            val = root / "val.csv"
            train.write_text("scene_id\n03001627/train\n")
            val.write_text("scene_id\n03001627/val\n")
            settings = dict(dataset_root=directory, train_scene_list=str(train), test_scene_list=str(val),
                            image_per_scene=1, remove_outlier_ndevs=-1, max_gs_num=100,
                            split_across_gpus=False, dataset_layout="shapenet", src_resolution=128,
                            tgt_resolution=128, load_src_gs=False, load_src_images=False, load_tgt_images=False)
            train_dataset = SplatFactoSRDataset(train_or_test="train", **settings)
            val_dataset = SplatFactoSRDataset(train_or_test="test", **settings)
            self.assertEqual(train_dataset.folders[0]["scene_name"], "03001627/train")
            self.assertEqual(val_dataset.folders[0]["scene_name"], "03001627/val")
            paths = val_dataset.folders[0]["resolution_paths"][128]
            self.assertEqual(paths["gsplat_dir"], str(root / "shapenet/128/gsplat/03001627/val"))
            with mock.patch("dataset.GS_SR.gs_io.load_gsplat", return_value=(self.target, "ckpt")) as load:
                scene = val_dataset.load_scene(0)
            load.assert_called_once_with(paths["gsplat_dir"])
            self.assertEqual(list(scene["data"]), [128])
            self.assertIn("gs_params", scene["data"][128])
            torch.testing.assert_close(scene["data"][128]["gs_params"]["means"], (self.target["means"] - 1) / 2)

    def test_invalid_dataset_layout(self):
        with self.assertRaisesRegex(ValueError, "dataset_layout"):
            self.make_dataset(dataset_layout="unknown")


if __name__ == "__main__":
    unittest.main()
