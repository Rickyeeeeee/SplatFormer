"""Time-conditioned global attention over packed Gaussian scenes."""

import math
from types import SimpleNamespace

import gin
import torch
from torch import nn
from torch.nn import functional as F

try:
    import flash_attn
except ImportError:
    flash_attn = None


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
        return self.mlp(F.pad(encoded, (0, self.frequency_channels % 2)))


class GlobalAttention(nn.Module):
    def __init__(self, channels, num_heads, attention_backend, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.attention_backend = attention_backend
        self.attn_drop = attn_drop
        self.qkv = nn.Linear(channels, 3 * channels)
        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, features, offset):
        qkv = self.qkv(features).reshape(-1, 3, self.num_heads, self.head_dim)
        dropout_p = self.attn_drop if self.training else 0.0
        if self.attention_backend == "flash":
            if flash_attn is None or not features.is_cuda:
                raise RuntimeError("FlashAttention backend requires flash_attn and CUDA")
            # FlashAttention accepts packed scenes without spatial sorting or padding.
            cu_seqlens = F.pad(offset.to(torch.int32), (1, 0)).contiguous()
            packed = qkv if qkv.dtype in (torch.float16, torch.bfloat16) else qkv.half()
            attended = flash_attn.flash_attn_varlen_qkvpacked_func(
                packed, cu_seqlens, max_seqlen=int(torch.diff(cu_seqlens).max().item()),
                dropout_p=dropout_p,
            ).to(qkv.dtype)
        else:
            # Each SDPA call covers one scene, so no attention mask or padding is needed.
            attended_scenes = []
            left = 0
            for right in offset.tolist():
                q, k, v = qkv[left:right].unbind(dim=1)
                attended_scenes.append(F.scaled_dot_product_attention(
                    q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None],
                    dropout_p=dropout_p,
                )[0].transpose(0, 1))
                left = right
            attended = torch.cat(attended_scenes, dim=0)
        return self.proj_drop(self.proj(attended.reshape_as(features)))


class GlobalDiffusionBlock(nn.Module):
    def __init__(self, channels, num_heads, mlp_ratio, attention_backend, attn_drop, proj_drop):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.modulation = nn.Sequential(nn.GELU(), nn.Linear(channels, 6 * channels))
        self.attention = GlobalAttention(channels, num_heads, attention_backend, attn_drop, proj_drop)
        hidden = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(channels, hidden), nn.GELU(), nn.Dropout(proj_drop),
                                 nn.Linear(hidden, channels), nn.Dropout(proj_drop))

    def forward(self, features, time, batch, offset):
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.modulation(time)[batch].chunk(6, dim=-1)
        query = self.norm1(features) * (1 + scale_a) + shift_a
        features = features + gate_a * self.attention(query, offset)
        query = self.norm2(features) * (1 + scale_m) + shift_m
        return features + gate_m * self.mlp(query)


@gin.configurable
class GlobalDiffusionPointTransformer(nn.Module):
    """Return per-point features after global attention within each scene."""

    def __init__(self, in_channels, depth=12, channels=384, num_head=6, mlp_ratio=4,
                 frequency_embedding_size=256, attention_backend="flash", attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        if depth < 1 or channels <= 0 or num_head <= 0 or channels % num_head or mlp_ratio <= 0:
            raise ValueError("Invalid global transformer dimensions")
        if frequency_embedding_size < 2 or attention_backend not in ("flash", "sdpa"):
            raise ValueError("Expected frequency_embedding_size >= 2 and attention_backend flash or sdpa")
        self.output_dim = channels
        self.embedding = nn.Sequential(nn.Linear(in_channels + 3, channels), nn.LayerNorm(channels), nn.GELU())
        self.timestep_embedding = TimestepEmbedding(channels, frequency_embedding_size)
        self.blocks = nn.ModuleList([
            GlobalDiffusionBlock(channels, num_head, mlp_ratio, attention_backend, attn_drop, proj_drop)
            for _ in range(depth)
        ])

    def forward(self, data_dict):
        features = data_dict["feat"]
        coord = data_dict["coord"]
        offset = data_dict["offset"]
        time = data_dict["timesteps"]
        if offset.ndim != 1 or len(offset) != len(time) or not len(offset) or offset[-1] != len(features):
            raise ValueError("Scene offsets and timesteps must cover all points")
        counts = torch.diff(F.pad(offset, (1, 0)))
        if (counts <= 0).any() or coord.shape != (len(features), 3):
            raise ValueError("Each scene needs points and one reference coordinate per point")
        batch = torch.repeat_interleave(torch.arange(len(offset), device=features.device), counts)
        # Raw reference coordinates supply spatial context without sparse convolution.
        features = self.embedding(torch.cat([features, coord], dim=-1))
        time = self.timestep_embedding(time)
        for block in self.blocks:
            features = block(features, time, batch, offset)
        return SimpleNamespace(feat=features)
