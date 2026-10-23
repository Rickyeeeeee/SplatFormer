"""Classic DiT blocks with global FlashAttention over Gaussian scenes."""

import math

import gin
import torch
from torch import nn

try:
    import flash_attn
except ImportError:
    flash_attn = None


class TimestepEmbedding(nn.Module):
    def __init__(self, channels, frequency_embedding_size=256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(nn.Linear(frequency_embedding_size, channels), nn.SiLU(), nn.Linear(channels, channels))

    def forward(self, time):
        half = self.frequency_embedding_size // 2
        frequencies = torch.exp(-math.log(10000) * torch.arange(half, device=time.device).float() / half)
        angles = time.float()[:, None] * frequencies
        embedding = torch.cat([angles.cos(), angles.sin()], dim=-1)
        if self.frequency_embedding_size % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return self.mlp(embedding)


class GaussianDiTBlock(nn.Module):
    def __init__(self, channels, num_heads, mlp_ratio):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.norm1 = nn.LayerNorm(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(channels, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(channels, 3 * channels)
        self.proj = nn.Linear(channels, channels)
        hidden = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(channels, hidden), nn.GELU(), nn.Linear(hidden, channels))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(channels, 6 * channels))

    def forward(self, features, time, batch, cu_seqlens, max_seqlen):
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.adaLN_modulation(time)[batch].chunk(6, dim=-1)
        query = self.norm1(features) * (1 + scale_a) + shift_a
        qkv = self.qkv(query).reshape(-1, 3, self.num_heads, self.head_dim)
        packed = qkv if qkv.dtype in (torch.float16, torch.bfloat16) else qkv.half()
        # Cumulative scene lengths prevent attention from crossing scene boundaries.
        attended = flash_attn.flash_attn_varlen_qkvpacked_func(
            packed.contiguous(), cu_seqlens, max_seqlen=max_seqlen, dropout_p=0.0
        ).reshape_as(features).to(qkv.dtype)
        features = features + gate_a * self.proj(attended)
        query = self.norm2(features) * (1 + scale_m) + shift_m
        return features + gate_m * self.mlp(query)


class GaussianDiTFinalLayer(nn.Module):
    def __init__(self, channels, out_channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(channels, 2 * channels))
        self.linear = nn.Linear(channels, out_channels)

    def forward(self, features, time, batch):
        shift, scale = self.adaLN_modulation(time)[batch].chunk(2, dim=-1)
        return self.linear(self.norm(features) * (1 + scale) + shift)


@gin.configurable
class GaussianDiT(nn.Module):
    """Predict packed Gaussian velocities with scene-wide DiT attention."""

    def __init__(self, in_channels, out_channels, max_gaussians=1000, depth=12, channels=384,
                 num_heads=6, mlp_ratio=4, frequency_embedding_size=256):
        super().__init__()
        if max_gaussians < 1 or depth < 1 or channels < 1 or num_heads < 1 or channels % num_heads:
            raise ValueError("Invalid Gaussian DiT dimensions or maximum Gaussian count")
        if mlp_ratio <= 0 or frequency_embedding_size < 2:
            raise ValueError("mlp_ratio must be positive and frequency_embedding_size must be at least 2")
        self.max_gaussians = max_gaussians
        self.output_dim = out_channels
        self.feature_embedding = nn.Sequential(nn.Linear(in_channels, channels), nn.GELU(), nn.Linear(channels, channels))
        self.coordinate_embedding = nn.Sequential(nn.Linear(3, channels), nn.GELU(), nn.Linear(channels, channels))
        self.timestep_embedding = TimestepEmbedding(channels, frequency_embedding_size)
        self.blocks = nn.ModuleList([GaussianDiTBlock(channels, num_heads, mlp_ratio) for _ in range(depth)])
        self.final_layer = GaussianDiTFinalLayer(channels, out_channels)

        # AdaLN-Zero starts every transformer block as an identity map.
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    def forward(self, data_dict):
        if flash_attn is None or not data_dict["feat"].is_cuda:
            raise RuntimeError("GaussianDiT requires flash_attn and CUDA")
        feat, coord = data_dict["feat"], data_dict["coord"]
        offset, timesteps = data_dict["offset"], data_dict["timesteps"]
        if offset.ndim != 1 or len(offset) != len(timesteps) or not len(offset) or offset[-1] != len(feat):
            raise ValueError("Scene offsets and timesteps must cover all Gaussians")
        counts = torch.diff(torch.cat([offset.new_zeros(1), offset]))
        if (counts < 1).any() or (counts > self.max_gaussians).any():
            raise ValueError(f"Each scene must contain 1 to {self.max_gaussians} Gaussians")
        if coord.shape != (len(feat), 3):
            raise ValueError("Expected one current coordinate per Gaussian")

        batch = torch.repeat_interleave(torch.arange(len(offset), device=feat.device), counts)
        cu_seqlens = torch.cat([offset.new_zeros(1), offset]).to(torch.int32).contiguous()
        features = self.feature_embedding(feat) + self.coordinate_embedding(coord)
        time = self.timestep_embedding(timesteps)
        max_seqlen = int(counts.max().item())
        for block in self.blocks:
            features = block(features, time, batch, cu_seqlens, max_seqlen)
        return self.final_layer(features, time, batch)
