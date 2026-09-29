"""Differentiable scalar-first quaternion operations for rotation flows.

Principal logarithms choose the shortest rotation. At exactly pi the largest
vector component is made positive; this unavoidable branch is discontinuous.
"""
import math

import torch
from torch.nn import functional as F

from utils.data_augmentation import quaternion_inverse, quaternion_multiply


def rotation_exp(vector):
    """Map rotation vectors (radians) to unit wxyz quaternions."""
    angle = vector.norm(dim=-1, keepdim=True)
    return torch.cat([torch.cos(angle / 2), .5 * torch.sinc(angle / (2 * math.pi)) * vector], dim=-1)


def rotation_log(quaternion):
    """Principal rotation vector, invariant to quaternion sign."""
    q = F.normalize(quaternion, dim=-1)
    largest = q[..., 1:].gather(-1, q[..., 1:].abs().argmax(dim=-1, keepdim=True))
    flip = (q[..., :1] < 0) | ((q[..., :1] == 0) & (largest < 0))
    q = torch.where(flip, -q, q)
    norm = q[..., 1:].norm(dim=-1, keepdim=True)
    angle = 2 * torch.atan2(norm, q[..., :1])
    # The inactive branch is finite too, including at the identity.
    ratio = torch.where(norm < 1e-6, 2 + norm.square() / 3, angle / norm.clamp_min(1e-8))
    return ratio * q[..., 1:]


def relative_rotation(source, target):
    return rotation_log(quaternion_multiply(quaternion_inverse(source), target))


def quaternion_derivative_to_body(q, derivative):
    return 2 * quaternion_multiply(quaternion_inverse(q), derivative)[..., 1:]


def body_to_quaternion_derivative(q, angular):
    return .5 * quaternion_multiply(q, torch.cat([torch.zeros_like(angular[..., :1]), angular], dim=-1))


def integrate_rotation(q, angular, step):
    return F.normalize(quaternion_multiply(q, rotation_exp(step * angular)), dim=-1)
