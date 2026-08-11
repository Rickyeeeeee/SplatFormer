import importlib
import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn
import gin


def _load_local_time_point_transformer():
    from pointcept.models.builder import MODELS

    registered = MODELS.get("PT-v3m1-Time")
    if registered is not None:
        return registered

    module_name = "pointcept.models.point_transformer_v3.point_transformer_v3m1_time"
    try:
        module = importlib.import_module(module_name)
        return module.PointTransformerV3Time
    except (ImportError, AttributeError):
        pass

    module_path = (
        Path(__file__).resolve().parents[1]
        / "Pointcept"
        / "pointcept"
        / "models"
        / "point_transformer_v3"
        / "point_transformer_v3m1_time.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except KeyError:
        registered = MODELS.get("PT-v3m1-Time")
        if registered is not None:
            return registered
        raise
    return module.PointTransformerV3Time


PointTransformerV3 = _load_local_time_point_transformer()


@gin.configurable
class PointTransformerV3FlowModel(nn.Module):
    def __init__(
        self,
        in_channels,
        enable_flash,
        enc_dim,
        output_dim,
        turn_off_bn,
        stride,
        embedding_type="MLP",
        enc_depths=(2, 2, 2, 6, 2),
        enc_num_head=(2, 4, 8, 16, 32),
        dec_depths=(2, 2, 2, 2),
        dec_num_head=(4, 4, 8, 16),
        dec_channels=None,
        enc_channels=None,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pretrained_ckpt=None,
        T_dim=-1,
        drop_path=0.3,
        shuffle_orders=True,
        shuffle_orders_eval=None,
    ):
        super(PointTransformerV3FlowModel, self).__init__()
        self.T_dim = T_dim
        if dec_channels is None:
            if output_dim == 64:
                self.dec_channels = (64, 64, 128, 256)
            elif output_dim == 128:
                self.dec_channels = (128, 128, 256, 256)
            elif output_dim == 96:
                self.dec_channels = (96, 96, 128, 256)
            else:
                raise ValueError("Unsupported output_dim")
        else:
            self.dec_channels = dec_channels

        if enc_channels is None:
            if enc_dim == 32:
                enc_channels = (32, 64, 128, 256, 512)
            elif enc_dim == 64:
                enc_channels = (64, 96, 128, 256, 512)
            else:
                raise ValueError("Unsupported enc_dim")

        if enable_flash:
            enc_patch_size = (1024, 1024, 1024, 1024, 1024)[:len(enc_channels)]
            dec_patch_size = (1024, 1024, 1024, 1024)[:len(self.dec_channels)]
        else:
            enc_patch_size = (128, 128, 128, 128, 128)[:len(enc_channels)]
            dec_patch_size = (128, 128, 128, 128)[:len(self.dec_channels)]

        self.backbone = PointTransformerV3(
            in_channels=in_channels,
            embedding_type=embedding_type,
            order=("z", "z-trans", "hilbert", "hilbert-trans"),
            stride=stride,
            enc_depths=enc_depths,
            enc_channels=enc_channels,
            enc_num_head=enc_num_head,
            enc_patch_size=enc_patch_size,
            dec_depths=dec_depths,
            dec_channels=self.dec_channels,
            dec_num_head=dec_num_head,
            dec_patch_size=dec_patch_size,
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            drop_path=drop_path,
            shuffle_orders=shuffle_orders,
            shuffle_orders_eval=shuffle_orders_eval,
            pre_norm=True,
            enable_rpe=False,
            enable_flash=enable_flash,
            upcast_attention=False,
            upcast_softmax=False,
            cls_mode=False,
            pdnorm_bn=pdnorm_bn,
            turn_off_bn=turn_off_bn,
            pdnorm_ln=pdnorm_ln,
            pdnorm_decouple=True,
            pdnorm_adaptive=False,
            pdnorm_affine=True,
            pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
            T_dim=T_dim,
        )
        self.output_dim = self.dec_channels[0]

        if pretrained_ckpt is not None:
            sd = torch.load(pretrained_ckpt, map_location="cpu")["state_dict"]
            sd = {k.replace("module.backbone.", ""): v for k, v in sd.items() if "backbone." in k}
            load_sd = {}
            for k, v in self.backbone.state_dict().items():
                if k not in sd:
                    print(f"Key {k} not found in pretrained model")
                    continue
                if v.shape != sd[k].shape:
                    print(f"Shape mismatch for {k}, {v.shape} != {sd[k].shape}, train the weight from scratch")
                    continue
                load_sd[k] = sd[k]
            msg = self.backbone.load_state_dict(load_sd, strict=False)
            print(f"Loaded pretrained model from {pretrained_ckpt}", msg)

    def forward(self, x):
        return self.backbone(x)
