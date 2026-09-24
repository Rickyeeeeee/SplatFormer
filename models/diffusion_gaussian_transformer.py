"""DiPT point backbone conditioned on continuous Gaussian interpolant time.

Adapted from Pointcept's DiPT v1m1 implementation by Matteo Bastico.
"""

import math
from functools import partial

import gin
import torch
import torch.nn as nn

from pointcept.models.modules import PointSequential
from pointcept.models.point_prompt_training import PDNorm
from pointcept.models.point_transformer_v3 import Block, Embedding
from pointcept.models.utils.misc import offset2bincount
from pointcept.models.utils.structure import Point


class TimestepEmbedding(nn.Module):
    """Embed continuous interpolant time with DiPT's sinusoidal MLP."""

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, t):
        half = self.frequency_embedding_size // 2
        frequencies = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
        arguments = t[:, None].float() * frequencies[None]
        embedding = torch.cat([torch.cos(arguments), torch.sin(arguments)], dim=-1)
        if self.frequency_embedding_size % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return self.mlp(embedding)


class GaussianDiffusionBlock(Block):
    """Apply per-scene time modulation to a Pointcept transformer block."""

    def __init__(self, channels, time_channels, **kwargs):
        super().__init__(channels=channels, **kwargs)
        self.time_projection = nn.Identity() if time_channels == channels else nn.Linear(time_channels, channels)
        self.adaLN_modulation = nn.Sequential(nn.GELU(), nn.Linear(channels, 6 * channels))

    def forward(self, point: Point):
        modulation = self.adaLN_modulation(self.time_projection(point.time_condition))
        modulation = torch.repeat_interleave(modulation, offset2bincount(point.offset), dim=0)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(6, dim=1)

        shortcut = point.feat
        point = self.cpe(point)
        point.feat = shortcut + point.feat
        shortcut = point.feat
        if self.pre_norm:
            point = self.norm1(point)
        point.feat = point.feat * (1 + scale_msa) + shift_msa
        # Pointcept caches patch padding on Point; invalidate it when patch size changes.
        patch_size = self.attn.patch_size if self.attn.enable_flash else min(offset2bincount(point.offset).min().item(), self.attn.patch_size_max)
        if point.get("attention_patch_size") != patch_size:
            for key in ("pad", "unpad", "cu_seqlens_key"):
                point.pop(key, None)
            for key in list(point.keys()):
                if key.startswith("rel_pos_"):
                    point.pop(key)
            point.attention_patch_size = patch_size
        point = self.drop_path(self.attn(point))
        point.feat = shortcut + gate_msa * point.feat
        if not self.pre_norm:
            point = self.norm1(point)

        shortcut = point.feat
        if self.pre_norm:
            point = self.norm2(point)
        point.feat = point.feat * (1 + scale_mlp) + shift_mlp
        point = self.drop_path(self.mlp(point))
        point.feat = shortcut + gate_mlp * point.feat
        if not self.pre_norm:
            point = self.norm2(point)
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point


@gin.configurable
class DiffusionGaussianTransformer(nn.Module):
    """Return per-point features from DiPT blocks using only time conditioning."""

    def __init__(
        self,
        in_channels,
        order=("z", "z-trans"),
        depth=12,
        channels=768,
        num_head=12,
        patch_size=48,
        mlp_ratio=4,
        frequency_embedding_size=256,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=True,
        shuffle_orders_eval=None,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("Gaussian",),
        pdnorm_condition="Gaussian",
    ):
        super().__init__()
        self.patch_size = (patch_size,) * depth if isinstance(patch_size, int) else tuple(patch_size)
        self.num_head = (num_head,) * depth if isinstance(num_head, int) else tuple(num_head)
        self.channels = (channels,) * depth if isinstance(channels, int) else tuple(channels)
        if not (len(self.patch_size) == len(self.num_head) == len(self.channels) == depth):
            raise ValueError("patch_size, num_head, and channels must each have depth entries")
        if any(p <= 0 for p in self.patch_size) or any(h <= 0 or c <= 0 or c % h for c, h in zip(self.channels, self.num_head)):
            raise ValueError("patch sizes must be positive and channels must be divisible by positive head counts")
        self.order = (order,) if isinstance(order, str) else tuple(order)
        self.shuffle_orders = shuffle_orders
        self.shuffle_orders_eval = shuffle_orders if shuffle_orders_eval is None else shuffle_orders_eval
        self.output_dim = self.channels[-1]
        self.pdnorm_condition = pdnorm_condition
        self.pdnorm_adaptive = pdnorm_adaptive and (pdnorm_bn or pdnorm_ln)
        if (pdnorm_bn or pdnorm_ln) and pdnorm_condition not in pdnorm_conditions:
            raise ValueError("pdnorm_condition must appear in pdnorm_conditions")

        # Keep normalization labels separate from the numerical time condition.
        bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        ln_layer = nn.LayerNorm
        pdnorm_args = dict(
            context_channels=self.channels[0], conditions=tuple(pdnorm_conditions),
            decouple=pdnorm_decouple, adaptive=pdnorm_adaptive,
        )
        if pdnorm_bn:
            bn_layer = partial(PDNorm, norm_layer=partial(bn_layer, affine=pdnorm_affine), **pdnorm_args)
        if pdnorm_ln:
            ln_layer = partial(PDNorm, norm_layer=partial(nn.LayerNorm, elementwise_affine=pdnorm_affine), **pdnorm_args)
        self.timestep_embedding = TimestepEmbedding(self.channels[0], frequency_embedding_size)
        self.embedding = Embedding(
            in_channels=in_channels,
            embed_channels=self.channels[0],
            norm_layer=bn_layer,
            act_layer=nn.GELU,
        )

        # DiPT keeps the point count fixed through a flat stack of conditioned blocks.
        self.enc = PointSequential()
        for index, block_drop_path in enumerate(torch.linspace(0, drop_path, depth).tolist()):
            if index and self.channels[index] != self.channels[index - 1]:
                self.enc.add(nn.Linear(self.channels[index - 1], self.channels[index]), name=f"project{index + 1}")
            self.enc.add(
                GaussianDiffusionBlock(
                    channels=self.channels[index],
                    time_channels=self.channels[0],
                    num_heads=self.num_head[index],
                    patch_size=self.patch_size[index],
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    drop_path=block_drop_path,
                    norm_layer=ln_layer,
                    act_layer=nn.GELU,
                    pre_norm=pre_norm,
                    order_index=index % len(self.order),
                    cpe_indice_key=f"block{index + 1}",
                    enable_rpe=enable_rpe,
                    enable_flash=enable_flash,
                    upcast_attention=upcast_attention,
                    upcast_softmax=upcast_softmax,
                ),
                name=f"block{index + 1}",
            )

    def forward(self, data_dict):
        point = Point(data_dict)
        shuffle_orders = self.shuffle_orders if self.training else self.shuffle_orders_eval
        point.serialization(order=self.order, shuffle_orders=shuffle_orders)
        point.sparsify()
        point.time_condition = self.timestep_embedding(point.timesteps)
        point.condition = self.pdnorm_condition
        if self.pdnorm_adaptive:
            point.context = torch.repeat_interleave(point.time_condition, offset2bincount(point.offset), dim=0)
        point = self.embedding(point)
        return self.enc(point)
