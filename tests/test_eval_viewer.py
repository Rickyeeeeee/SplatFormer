import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from eval_viewer.index import EvaluationIndex, metric_gain


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def metrics(psnr: float, ssim: float, lpips: float):
    return {"psnr": psnr, "ssim": ssim, "lpips": lpips}


class EvaluationIndexTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.scene = "scene-a"

        reference_scene = {
            "scene_idx": 7,
            "scene_name": self.scene,
            "output_gs": metrics(20.0, 0.7, 0.3),
            "input_gs": metrics(20.0, 0.7, 0.3),
        }
        write_json(self.root / "00000000" / "scene_average_metrics.json", [reference_scene])
        write_json(
            self.root / "00000000" / self.scene / "metrics.json",
            [{"image_id": 0, "image_name": "000.png", **metrics(20.0, 0.7, 0.3)}],
        )
        write_json(
            self.root / "00000000" / self.scene / "metrics_input.json",
            [{"image_id": 0, "image_name": "000.png", **metrics(18.0, 0.6, 0.4)}],
        )

        comparison = np.zeros((2, 6, 3), dtype=np.uint8)
        comparison[:, 0:2] = (0, 0, 255)  # GT is red in RGB.
        comparison[:, 2:4] = (0, 255, 0)  # Input is green.
        comparison[:, 4:6] = (255, 0, 0)
        compare_path = self.root / "00000000" / self.scene / "compare" / "000.png"
        compare_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(compare_path), comparison)

        current_scene = {
            "scene_idx": 7,
            "scene_name": self.scene,
            "output_gs": metrics(24.0, 0.8, 0.2),
        }
        write_json(self.root / "00000010" / "scene_average_metrics.json", [current_scene])
        write_json(
            self.root / "00000010" / self.scene / "metrics.json",
            [{"image_id": 0, "image_name": "000.png", **metrics(23.0, 0.75, 0.25)}],
        )
        pred_path = self.root / "00000010" / self.scene / "pred" / "000.png"
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(pred_path), np.full((2, 2, 3), 127, dtype=np.uint8))

        (self.root / "00000020").mkdir()
        self.index = EvaluationIndex(self.root)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_discovers_only_complete_iterations_and_defaults_to_latest(self) -> None:
        payload = self.index.iterations()
        self.assertEqual(payload["default_iteration"], "00000010")
        self.assertEqual([item["name"] for item in payload["items"]], ["00000010", "00000000"])
        self.assertTrue(any("00000020" in warning for warning in payload["warnings"]))

    def test_scene_and_view_gains_use_positive_is_better_lpips(self) -> None:
        scene = self.index.scenes("00000010")["items"][0]
        self.assertAlmostEqual(scene["gain"]["psnr"], 4.0)
        self.assertAlmostEqual(scene["gain"]["ssim"], 0.1)
        self.assertAlmostEqual(scene["gain"]["lpips"], 0.1)
        view = self.index.views("00000010", self.scene)["items"][0]
        self.assertAlmostEqual(view["gain"]["psnr"], 5.0)
        self.assertAlmostEqual(view["gain"]["ssim"], 0.15)
        self.assertAlmostEqual(view["gain"]["lpips"], 0.15)
        self.assertEqual(view["images"], {"prediction": True, "input": True, "gt": True})

    def test_reference_panels_are_cropped_from_first_two_thirds(self) -> None:
        gt = cv2.imdecode(np.frombuffer(self.index.reference_image(self.scene, "000.png", "gt"), np.uint8), cv2.IMREAD_COLOR)
        input_image = cv2.imdecode(
            np.frombuffer(self.index.reference_image(self.scene, "000.png", "input"), np.uint8), cv2.IMREAD_COLOR
        )
        self.assertEqual(gt.shape[:2], (2, 2))
        self.assertTrue(np.all(gt == (0, 0, 255)))
        self.assertTrue(np.all(input_image == (0, 255, 0)))

    def test_prediction_path_rejects_unknown_and_traversal_views(self) -> None:
        self.assertTrue(self.index.prediction_path("00000010", self.scene, "000.png").is_file())
        with self.assertRaises(KeyError):
            self.index.prediction_path("00000010", self.scene, "../000.png")

    def test_refresh_promotes_newly_completed_iteration(self) -> None:
        write_json(self.root / "00000020" / "scene_average_metrics.json", [])
        self.index.refresh()
        self.assertEqual(self.index.default_iteration, "00000020")

    def test_metric_gain_contract(self) -> None:
        gain = metric_gain(metrics(22.0, 0.8, 0.2), metrics(20.0, 0.7, 0.3))
        self.assertAlmostEqual(gain["psnr"], 2.0)
        self.assertAlmostEqual(gain["ssim"], 0.1)
        self.assertAlmostEqual(gain["lpips"], 0.1)


if __name__ == "__main__":
    unittest.main()
