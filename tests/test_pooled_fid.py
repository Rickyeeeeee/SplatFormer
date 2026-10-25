"""Pooled FID checks without downloading pretrained Inception weights."""

import os
import sys
from types import ModuleType
import unittest
from unittest import mock

import torch

from utils.metrics import PooledFID


class FakeFID:
    def __init__(self, feature, reset_real_features):
        self.feature = feature
        self.reset_real_features = reset_real_features
        self.real_batches = []
        self.generated_batches = []
        self.dtype = None
        self.device = None

    def set_dtype(self, dtype):
        self.dtype = dtype

    def to(self, device):
        self.device = device
        return self

    def update(self, images, real):
        (self.real_batches if real else self.generated_batches).append(images.clone())

    def compute(self):
        return torch.tensor(float(len(self.real_batches) + len(self.generated_batches)))

    def reset(self):
        if self.reset_real_features:
            self.real_batches.clear()
        self.generated_batches.clear()


class PooledFIDTests(unittest.TestCase):
    def test_real_statistics_persist_and_fake_statistics_reset(self):
        root = ModuleType("torchmetrics")
        root.__path__ = []
        image = ModuleType("torchmetrics.image")
        image.__path__ = []
        fid_module = ModuleType("torchmetrics.image.fid")
        fid_module.FrechetInceptionDistance = FakeFID
        modules = {"torchmetrics": root, "torchmetrics.image": image, "torchmetrics.image.fid": fid_module}
        with mock.patch.dict(sys.modules, modules):
            fid = PooledFID(torch.device("cpu"))

        self.assertEqual(fid.metric.feature, 2048)
        self.assertFalse(fid.metric.reset_real_features)
        self.assertEqual(fid.metric.dtype, torch.float64)
        target = torch.arange(3 * 3 * 4 * 3, dtype=torch.uint8).reshape(3, 3, 4, 3)
        fid.update(target[:2], real=True)
        fid.update(target[2:], real=True)
        first = torch.randint(0, 256, (3, 3, 4, 3), dtype=torch.uint8,
                              generator=torch.Generator().manual_seed(3))
        fid.update(first[:1], real=False)
        fid.update(first[1:], real=False)
        self.assertEqual(fid.report(), {"fid": 4.0, "fid_counts": {"real": 3, "generated": 3}})
        self.assertEqual(len(fid.metric.real_batches), 2)
        self.assertEqual(len(fid.metric.generated_batches), 0)
        torch.testing.assert_close(fid.metric.real_batches[0], target[:2].permute(0, 3, 1, 2))
        self.assertTrue(fid.metric.real_batches[0].is_contiguous())

        second = torch.randint(0, 256, (3, 3, 4, 3), dtype=torch.uint8,
                               generator=torch.Generator().manual_seed(4))
        fid.update(second, real=False)
        self.assertFalse(torch.equal(first, second))
        self.assertEqual(fid.report(), {"fid": 3.0, "fid_counts": {"real": 3, "generated": 3}})
        self.assertEqual(len(fid.metric.real_batches), 2)

    @unittest.skipUnless(os.environ.get("RUN_FID_INTEGRATION") == "1", "Pretrained FID integration is opt-in")
    def test_pretrained_inception_produces_finite_fid(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        fid = PooledFID(device)
        real = torch.randint(0, 256, (16, 32, 32, 3), dtype=torch.uint8, device=device)
        generated = torch.randint(0, 256, (16, 32, 32, 3), dtype=torch.uint8, device=device)
        fid.update(real, real=True)
        fid.update(generated, real=False)
        self.assertTrue(torch.isfinite(torch.tensor(fid.report()["fid"])))


if __name__ == "__main__":
    unittest.main()
