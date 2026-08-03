import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from eval_viewer.index import EvaluationDataError, EvaluationIndex


class EvaluationIndexErrorTest(unittest.TestCase):
    def test_rejects_malformed_reference_strip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene = "scene-a"
            iteration = root / "00000000"
            iteration.mkdir()
            (iteration / "scene_average_metrics.json").write_text(
                json.dumps([{"scene_idx": 0, "scene_name": scene, "output_gs": {"psnr": 1, "ssim": 1, "lpips": 1}}]),
                encoding="utf-8",
            )
            compare = iteration / scene / "compare" / "000.png"
            compare.parent.mkdir(parents=True)
            cv2.imwrite(str(compare), np.zeros((3, 5, 3), dtype=np.uint8))

            index = EvaluationIndex(root)
            with self.assertRaises(EvaluationDataError):
                index.reference_image(scene, "000.png", "gt")

    def test_rejects_unknown_sort_field(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            iteration = root / "00000000"
            iteration.mkdir()
            (iteration / "scene_average_metrics.json").write_text("[]", encoding="utf-8")
            index = EvaluationIndex(root)
            with self.assertRaises(ValueError):
                index.scenes("00000000", sort="not-a-metric")


if __name__ == "__main__":
    unittest.main()
