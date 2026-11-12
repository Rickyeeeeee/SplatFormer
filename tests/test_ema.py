"""Checks for EMA initialization, updates, and checkpoint compatibility."""

import io
import unittest

import torch
from torch import nn

from utils.ema import create_ema_model, update_ema_model


class EMATests(unittest.TestCase):
    def test_initialization_update_and_checkpoint(self):
        model = nn.BatchNorm1d(2)
        ema = create_ema_model(model, True, 0.9999)
        self.assertFalse(ema.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in ema.parameters()))
        with torch.no_grad():
            model.weight.fill_(3)
            model.running_mean.fill_(4)
            model.num_batches_tracked.fill_(5)
        update_ema_model(ema, model, 0.9999)
        torch.testing.assert_close(ema.weight, torch.full_like(ema.weight, 1.0002))
        torch.testing.assert_close(ema.running_mean, model.running_mean)
        torch.testing.assert_close(ema.num_batches_tracked, model.num_batches_tracked)
        stream = io.BytesIO()
        torch.save(ema.state_dict(), stream)
        stream.seek(0)
        restored = nn.BatchNorm1d(2)
        restored.load_state_dict(torch.load(stream, weights_only=True), strict=True)
        torch.testing.assert_close(restored.weight, ema.weight)

    def test_disabled_and_invalid_decay(self):
        model = nn.Linear(1, 1)
        self.assertIsNone(create_ema_model(model, False, 0.9999))
        for decay in (-0.1, 1.0, float("nan"), float("inf")):
            with self.subTest(decay=decay), self.assertRaises(ValueError):
                create_ema_model(model, False, decay)
