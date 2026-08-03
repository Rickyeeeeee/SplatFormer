import json
import tempfile
import unittest
from pathlib import Path

from eval_viewer.index import EvaluationIndex


def write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def values(psnr: float, ssim: float, lpips: float):
    return {"psnr": psnr, "ssim": ssim, "lpips": lpips}


class EvaluationMetricSetsTest(unittest.TestCase):
    def test_exposes_current_prediction_and_reference_baseline_sets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene = "scene-a"
            reference = root / "00000000"
            current = root / "00000010"
            input_metrics = values(20.0, 0.70, 0.30)
            low_metrics = values(21.0, 0.72, 0.28)
            high_metrics = values(32.0, 0.96, 0.08)
            prediction_metrics = values(25.0, 0.85, 0.18)

            write(reference / "scene_average_metrics.json", [{
                "scene_idx": 0,
                "scene_name": scene,
                "output_gs": input_metrics,
                "input_gs": input_metrics,
                "gt_low_res_gs": low_metrics,
                "gt_high_res_gs": high_metrics,
            }])
            write(reference / "metrics_input.json", input_metrics)
            write(reference / "metrics_gt_low_res.json", low_metrics)
            write(reference / "metrics_gt_high_res.json", high_metrics)
            write(current / "scene_average_metrics.json", [{
                "scene_idx": 0,
                "scene_name": scene,
                "output_gs": prediction_metrics,
            }])
            write(current / "metrics.json", prediction_metrics)

            view_identity = {"image_id": 0, "image_name": "000.png"}
            write(current / scene / "metrics.json", [{**view_identity, **prediction_metrics}])
            write(reference / scene / "metrics_input.json", [{**view_identity, **input_metrics}])
            write(reference / scene / "metrics_gt_low_res.json", [{**view_identity, **low_metrics}])
            write(reference / scene / "metrics_gt_high_res.json", [{**view_identity, **high_metrics}])

            index = EvaluationIndex(root)
            iteration = next(item for item in index.iterations()["items"] if item["name"] == "00000010")
            self.assertEqual(iteration["metric_sets"], {
                "prediction": prediction_metrics,
                "input": input_metrics,
                "gt_low_res": low_metrics,
                "gt_high_res": high_metrics,
            })

            scene_row = index.scenes("00000010")["items"][0]
            self.assertEqual(scene_row["gt_low_res_metrics"], low_metrics)
            self.assertEqual(scene_row["gt_high_res_metrics"], high_metrics)
            view_row = index.views("00000010", scene)["items"][0]
            self.assertEqual(view_row["gt_low_res_metrics"], low_metrics)
            self.assertEqual(view_row["gt_high_res_metrics"], high_metrics)


if __name__ == "__main__":
    unittest.main()
