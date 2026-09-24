import math

import gin
import torch
@gin.configurable
def build_3DGSoptimizer(gs_params, lr_dict, optimizer_type, optimizer_params):
    params_lr = []
    for param in gs_params:
        lr = lr_dict.get(param, lr_dict['base'])
        params_lr.append({'params': gs_params[param], 'lr': lr})
    if optimizer_type.lower() == 'adam':
        optimizer = torch.optim.Adam(params_lr, 
                                     lr = lr_dict['base'],
                                     **optimizer_params)
    elif optimizer_type.lower() == 'sgd':
        optimizer = torch.optim.SGD(params_lr, 
                                    lr = lr_dict['base'])
    else:
        raise NotImplementedError
    return optimizer

@gin.configurable
def build_optimizer(model, 
                    lr_dict: gin.REQUIRED, 
                    optimizer_type: gin.REQUIRED,
                    optimizer_params, use_zero=False):
    # ZeRO inspects groups before optimizer initialization, so parameters must be reusable lists.
    params_lr = []
    if hasattr(model, 'backbone') and hasattr(model, 'features_outputhead'):
        if getattr(model, 'backbone_type', None) != 'empty':
            params_lr.append({'params': list(model.backbone.parameters()), 'lr': lr_dict['backbone']})
        for feature in model.features_outputhead.keys():
            lr = lr_dict.get(feature, lr_dict['base'])
            params_lr.append({'params': list(model.features_outputhead[feature].parameters()), 'lr': lr})
    else:
        for param in model.parameters():
            lr = lr_dict.get(param, lr_dict['base'])
            params_lr.append({'params': [param], 'lr': lr})

    if optimizer_type.lower() == 'adam':
        optimizer_class = torch.optim.Adam
        defaults = dict(lr=lr_dict['base'], **optimizer_params)
    elif optimizer_type.lower() == 'sgd':
        optimizer_class = torch.optim.SGD
        defaults = dict(lr=lr_dict['base'])
    else:
        raise NotImplementedError

    # Shard optimizer state only for explicitly enabled multi-rank runs.
    if use_zero and torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
        from torch.distributed.optim import ZeroRedundancyOptimizer

        optimizer = ZeroRedundancyOptimizer(params_lr, optimizer_class=optimizer_class,
                                            overlap_with_ddp=False, parameters_as_bucket_view=False, **defaults)
    else:
        optimizer = optimizer_class(params_lr, **defaults)
    return optimizer

@gin.configurable
def build_scheduler(optimizer, schedule, total_step, warmup_step=0, warmup_start_factor=1.0 / 30.0):
    if total_step <= 0:
        raise ValueError("total_step must be positive")
    if warmup_step < 0 or warmup_step >= total_step:
        raise ValueError("warmup_step must be in [0, total_step)")
    if not 0 < warmup_start_factor <= 1:
        raise ValueError("warmup_start_factor must be in (0, 1]")
    if warmup_step > 0:
        decay_steps = total_step - warmup_step

        def lr_multiplier(step):
            if step < warmup_step:
                return warmup_start_factor + (1.0 - warmup_start_factor) * step / warmup_step
            progress = min((step - warmup_step) / decay_steps, 1.0)
            if schedule == 'constant':
                return 1.0
            if schedule == 'linear':
                return 1.0 - progress
            if schedule == 'cosine':
                return 0.5 * (1.0 + math.cos(math.pi * progress))
            raise NotImplementedError(f"Unsupported schedule {schedule!r}")

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_multiplier)
    if schedule == 'constant':
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1)
    elif schedule == 'linear':
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1-step/total_step)
    elif schedule == 'cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_step)
    elif schedule == 'exponential':
        raise ValueError
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=exponential_gamma)
    else:
        raise NotImplementedError
    return lr_scheduler
