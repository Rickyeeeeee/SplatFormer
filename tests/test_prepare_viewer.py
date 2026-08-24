import json
import tempfile
import unittest
from pathlib import Path

import torch

from utils.gs_utils import prepare_viewer


def _cameras(camera_to_worlds):
    return {
        "camera_to_worlds": camera_to_worlds,
        "width": torch.tensor(512.0),
        "height": torch.tensor(512.0),
        "fx": torch.tensor(400.0),
        "fy": torch.tensor(400.0),
    }


class PrepareViewerTests(unittest.TestCase):
    def test_accepts_equivalent_three_by_four_and_four_by_four_cameras(self):
        camera_4x4 = torch.eye(4).unsqueeze(0)
        camera_4x4[0, :3, 3] = torch.tensor([1.0, 2.0, 3.0])
        camera_3x4 = camera_4x4[:, :3, :4]

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            three_by_four_dir = root / "three_by_four"
            four_by_four_dir = root / "four_by_four"
            three_by_four_dir.mkdir()
            four_by_four_dir.mkdir()
            prepare_viewer(_cameras(camera_3x4), str(three_by_four_dir), 3)
            prepare_viewer(_cameras(camera_4x4), str(four_by_four_dir), 3)

            three_by_four = json.loads(
                (three_by_four_dir / "cameras.json").read_text()
            )
            four_by_four = json.loads(
                (four_by_four_dir / "cameras.json").read_text()
            )

        self.assertEqual(three_by_four, four_by_four)

    def test_rejects_invalid_camera_shape(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "shape"):
                prepare_viewer(
                    _cameras(torch.zeros((1, 3, 3))), temp_dir, sh_degree=3
                )


if __name__ == "__main__":
    unittest.main()
