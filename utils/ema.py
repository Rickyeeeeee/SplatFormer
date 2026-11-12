"""Exponential moving average weights for training and evaluation."""

import copy
import math

import torch


def create_ema_model(model, enabled, decay):
    if not math.isfinite(decay) or not 0 <= decay < 1:
        raise ValueError("ema_decay must be finite and in [0, 1)")
    if not enabled:
        return None
    ema_model = copy.deepcopy(model).eval()
    ema_model.requires_grad_(False)
    return ema_model


@torch.no_grad()
def update_ema_model(ema_model, model, decay):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.lerp_(param, 1.0 - decay)
    for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
        ema_buffer.copy_(buffer)
