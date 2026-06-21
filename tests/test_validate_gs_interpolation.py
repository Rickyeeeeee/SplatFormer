import importlib.util
import os
from unittest import mock

import numpy as np
import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODULE_PATH = os.path.join(ROOT, "validate-gs-interpolation.py")
SPEC = importlib.util.spec_from_file_location("validate_gs_interpolation", MODULE_PATH)
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


def test_interpolation_formula_all_keys_and_endpoint():
    x0 = {
        "means": torch.tensor([[0.0, 1.0, 2.0]]),
        "scales": torch.tensor([[1.0, 2.0, 3.0]]),
        "quats": torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
    }
    x1 = {
        "means": torch.tensor([[2.0, 3.0, 4.0]]),
        "scales": torch.tensor([[3.0, 4.0, 5.0]]),
        "quats": torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
    }
    keys = list(x0)

    halfway = VALIDATOR.interpolation_state(x0, x1, 0.5, keys)
    endpoint = VALIDATOR.interpolation_state(x0, x1, 1.0, keys)

    for key in keys:
        torch.testing.assert_close(halfway[key], (x0[key] + x1[key]) * 0.5)
        torch.testing.assert_close(endpoint[key], x1[key])
    assert VALIDATOR.t_label(0.1) == "0.1"
    assert VALIDATOR.t_label(1.0) == "1.0"


def test_two_step_optimization_writes_initial_and_post_update_frames():
    class FakeRecorder:
        instances = []

        def __init__(self, path, fps, first_frame):
            self.path = path
            self.fps = fps
            self.frames = []
            self.closed = False
            self.instances.append(self)

        def write(self, frame):
            self.frames.append(frame.copy())

        def close(self):
            self.closed = True

    class FakeLogger:
        def info(self, message):
            del message

    def fake_optimizer(trainable):
        return torch.optim.SGD(list(trainable.values()), lr=0.1)

    def fake_loss(gs, images, cameras):
        del images, cameras
        return ((gs["means"] - 1.0) ** 2).mean()

    def fake_frame(gs, gt_image, camera, iteration):
        del gs, gt_image, camera
        return np.full((4, 5, 3), iteration, dtype=np.uint8), float(iteration)

    x0 = {"means": torch.zeros((1, 3))}
    with mock.patch.object(VALIDATOR, "build_3DGSoptimizer", side_effect=fake_optimizer), mock.patch.object(
        VALIDATOR.GSPath,
        "render_l1_loss",
        side_effect=fake_loss,
    ), mock.patch.object(VALIDATOR, "render_video_frame", side_effect=fake_frame):
        endpoint, history = VALIDATOR.optimize_endpoint_with_video(
            x0=x0,
            images=[torch.zeros((1, 1, 3))],
            cameras={"camera_to_worlds": torch.eye(4).unsqueeze(0)},
            float_keys=["means"],
            optimization_steps=2,
            video_view_index=0,
            video_fps=30.0,
            video_path="unused.mp4",
            logger=FakeLogger(),
            recorder_factory=FakeRecorder,
        )

    recorder = FakeRecorder.instances[-1]
    assert len(history) == 3
    assert [int(frame[0, 0, 0]) for frame in recorder.frames] == [0, 1, 2]
    assert recorder.closed
    assert endpoint["means"].mean().item() > x0["means"].mean().item()


def test_video_recorder_rejects_size_changes_and_releases_writer():
    class FakeWriter:
        def __init__(self):
            self.frames = []
            self.released = False

        def isOpened(self):
            return True

        def write(self, frame):
            self.frames.append(frame.copy())

        def release(self):
            self.released = True

    writer = FakeWriter()

    def writer_factory(path, fourcc, fps, size):
        del path, fourcc, fps
        assert size == (5, 4)
        return writer

    recorder = VALIDATOR.TrainingVideoRecorder(
        "unused.mp4",
        30.0,
        np.zeros((4, 5, 3), dtype=np.uint8),
        writer_factory=writer_factory,
    )
    recorder.write(np.zeros((4, 5, 3), dtype=np.uint8))
    try:
        recorder.write(np.zeros((5, 5, 3), dtype=np.uint8))
        raise AssertionError("Expected a frame-size validation error")
    except ValueError:
        pass
    finally:
        recorder.close()

    assert len(writer.frames) == 1
    assert writer.released


if __name__ == "__main__":
    test_interpolation_formula_all_keys_and_endpoint()
    test_two_step_optimization_writes_initial_and_post_update_frames()
    test_video_recorder_rejects_size_changes_and_releases_writer()
    print("GS interpolation validator tests passed")
