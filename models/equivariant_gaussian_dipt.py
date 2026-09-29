"""Time-conditioned Gaussian attention with Pointcept serialized neighborhoods.

Geometry is SE(3)-invariant for fixed neighborhoods and unchanged scalar features.
World-axis serialization, rotating SH features, and equivalent local-axis choices
are intentionally not made invariant by this initial model.
"""
import math

import gin
import torch
from torch import nn
from torch.nn import functional as F

from pointcept.models.utils.structure import Point


class TimestepEmbedding(nn.Module):
    def __init__(self, channels, frequency_channels=256):
        super().__init__()
        self.frequency_channels = frequency_channels
        self.mlp = nn.Sequential(nn.Linear(frequency_channels, channels), nn.GELU(), nn.Linear(channels, channels))

    def forward(self, time):
        half = self.frequency_channels // 2
        frequencies = torch.exp(-math.log(10000) * torch.arange(half, device=time.device).float() / half)
        angles = time.float()[:, None] * frequencies
        encoded = torch.cat([angles.cos(), angles.sin()], dim=-1)
        encoded = F.pad(encoded, (0, self.frequency_channels % 2))
        return self.mlp(encoded)


class GeometricAttention(nn.Module):
    """Explicit patch attention; positions and rotations stay differentiable."""

    def __init__(self, channels, num_heads, geometry_channels=16, geometry_length_scale=1.0,
                 attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        if channels % num_heads or geometry_length_scale <= 0 or not math.isfinite(geometry_length_scale):
            raise ValueError("Invalid attention dimensions or geometry length scale")
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.length_scale = geometry_length_scale
        self.qkv = nn.Linear(channels, 3 * channels)
        self.geometry_encoder = nn.Sequential(nn.Linear(12, 32), nn.GELU(), nn.Linear(32, geometry_channels))
        self.geometry_bias = nn.Linear(geometry_channels, num_heads, bias=False)
        self.raw_range = nn.Parameter(torch.full((num_heads,), math.log(math.expm1(1.0 - 1e-6))))
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(channels + num_heads * geometry_channels, channels)
        self.proj_drop = nn.Dropout(proj_drop)

    def attend(self, features, positions, rotations, valid):
        batch, size, channels = features.shape
        q, k, v = self.qkv(features).reshape(batch, size, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        # Disable autocast for physical geometry, edge encoding, and softmax.
        with torch.autocast(device_type=features.device.type, enabled=False):
            positions, rotations = positions.float(), rotations.float()
            displacement = positions[:, None, :, :] - positions[:, :, None, :]
            local = torch.einsum('biwa,bijw->bija', rotations, displacement) / self.length_scale
            relative = torch.einsum('biwa,bjwc->bijac', rotations, rotations)
            edges = self.geometry_encoder(torch.cat([local, relative.flatten(-2)], dim=-1))
            radius = F.softplus(self.raw_range.float()) + 1e-6
            scores = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(self.head_dim)
            scores = scores + self.geometry_bias(edges).permute(0, 3, 1, 2)
            scores = scores - local.square().sum(-1)[:, None] / (2 * radius[None, :, None, None].square())
            scores = scores.masked_fill(~valid[:, None, None, :], float('-inf'))
            weights = self.attn_drop(scores.softmax(-1))
            content = weights @ v.float()
            geometry = torch.einsum('bhij,bije->bhie', weights, edges)
            summary = torch.cat([content, geometry], dim=-1).transpose(1, 2).reshape(batch, size, -1)
        return self.proj_drop(self.proj(summary.to(features.dtype)))

    def forward(self, features, positions, rotations, order, inverse, counts, patch_size):
        # Each patch has at least one real point; scenes never share a patch.
        index_patches, masks = [], []
        left = 0
        for count in counts:
            for start in range(left, left + count, patch_size):
                indices = order[start:min(start + patch_size, left + count)]
                index_patches.append(F.pad(indices, (0, patch_size - len(indices))))
                masks.append(torch.arange(patch_size, device=features.device) < len(indices))
            left += count
        indices, valid = torch.stack(index_patches), torch.stack(masks)
        # Submit all independent patches together without activation recomputation.
        result = self.attend(features[indices], positions[indices], rotations[indices], valid)
        return result[valid][inverse]


class EquivariantDiPTBlock(nn.Module):
    def __init__(self, channels, num_heads, mlp_ratio, drop_path, **attention_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.modulation = nn.Sequential(nn.GELU(), nn.Linear(channels, 6 * channels))
        self.attention = GeometricAttention(channels, num_heads, **attention_kwargs)
        hidden = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(channels, hidden), nn.GELU(), nn.Dropout(attention_kwargs['proj_drop']),
                                 nn.Linear(hidden, channels), nn.Dropout(attention_kwargs['proj_drop']))
        self.drop_path = drop_path

    def residual(self, branch):
        if self.training and self.drop_path:
            keep = 1 - self.drop_path
            branch = branch * branch.new_empty((len(branch), 1)).bernoulli_(keep) / keep
        return branch

    def forward(self, features, time, batch, positions, rotations, order, inverse, counts, patch_size):
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.modulation(time)[batch].chunk(6, dim=-1)
        query = self.norm1(features) * (1 + scale_a) + shift_a
        features = features + gate_a * self.residual(self.attention(query, positions, rotations, order, inverse, counts, patch_size))
        query = self.norm2(features) * (1 + scale_m) + shift_m
        return features + gate_m * self.residual(self.mlp(query))


@gin.configurable
class EquivariantGaussianDiPT(nn.Module):
    """Packed tensor backbone; optional serialized_order fixes neighborhoods.

    Input fields: feat, coord (current), rotations, grid_coord (reference), offset,
    timesteps, and optional serialized_order [orders, N]. Returns a Point with
    updated feat and reusable serialized_order/serialized_inverse metadata.
    """

    def __init__(self, in_channels, depth=12, channels=384, num_head=6, patch_size=256,
                 order=('z', 'z-trans'), mlp_ratio=4, frequency_embedding_size=256,
                 geometry_channels=16, geometry_length_scale=1.0, attn_drop=0.0,
                 proj_drop=0.0, drop_path=0.0, shuffle_orders=True):
        super().__init__()
        self.patch_sizes = (patch_size,) * depth if isinstance(patch_size, int) else tuple(patch_size)
        self.order = (order,) if isinstance(order, str) else tuple(order)
        if depth < 1 or len(self.patch_sizes) != depth or any(size <= 0 for size in self.patch_sizes):
            raise ValueError("Expected one positive patch size per block")
        if channels <= 0 or num_head <= 0 or channels % num_head or not 0 <= drop_path < 1:
            raise ValueError("Invalid channel/head dimensions or drop_path")
        if frequency_embedding_size < 2 or not self.order or set(self.order) - {'z', 'z-trans'}:
            raise ValueError("Expected frequency width >= 2 and z/z-trans orders")
        self.output_dim = channels
        self.shuffle_orders = shuffle_orders
        self.embedding = nn.Sequential(nn.Linear(in_channels, channels), nn.LayerNorm(channels), nn.GELU())
        self.time_embedding = TimestepEmbedding(channels, frequency_embedding_size)
        self.blocks = nn.ModuleList([
            EquivariantDiPTBlock(channels, num_head, mlp_ratio, rate,
                                 geometry_channels=geometry_channels, geometry_length_scale=geometry_length_scale,
                                 attn_drop=attn_drop, proj_drop=proj_drop)
            for rate in torch.linspace(0, drop_path, depth).tolist()
        ])

    def forward(self, data_dict):
        point = Point(data_dict)
        # Pointcept constructs batch indices in inference mode; autograd indexes need a normal tensor.
        point.batch = point.batch.clone()
        counts = torch.diff(F.pad(point.offset, (1, 0))).tolist()
        if not counts or min(counts) <= 0 or sum(counts) != len(point.feat):
            raise ValueError("Offsets must describe nonempty scenes covering all points")
        if 'serialized_order' not in point:
            point.serialization(order=self.order, shuffle_orders=self.training and self.shuffle_orders)
        else:
            order = point.serialized_order
            expected = torch.arange(len(point.feat), device=point.feat.device)
            if order.ndim != 2 or order.shape[0] == 0 or order.shape[1] != len(expected):
                raise ValueError("serialized_order must have shape [orders, N]")
            if not torch.equal(order.sort(dim=1).values, expected.expand_as(order)):
                raise ValueError("Each serialized order must be a permutation")
            if not torch.equal(point.batch[order], point.batch.expand_as(order)):
                raise ValueError("Serialized orders must preserve scene segments")
            point.serialized_inverse = order.argsort(dim=1)
        features = self.embedding(point.feat)
        time = self.time_embedding(point.timesteps)
        for index, block in enumerate(self.blocks):
            row = index % len(point.serialized_order)
            features = block(features, time, point.batch, point.coord, point.rotations,
                             point.serialized_order[row], point.serialized_inverse[row], counts, self.patch_sizes[index])
        point.feat = features
        return point
