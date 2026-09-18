import random

import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_cpu(data):
    if torch.is_tensor(data):
        return data.detach().cpu()
    if isinstance(data, dict):
        return {key: to_cpu(value) for key, value in data.items()}
    if isinstance(data, list):
        return [to_cpu(value) for value in data]
    if isinstance(data, tuple):
        return tuple(to_cpu(value) for value in data)
    return data


def move_to_device(data, device):
    if data is None:
        return None
    elif isinstance(data, (list, tuple)):
        return [move_to_device(d, device) for d in data]
    elif isinstance(data, dict):
        return {key: move_to_device(value, device) for key, value in data.items()}
    else:
        return data.to(device)

def move_training_data(data, device, non_blocking=False):
    if torch.is_tensor(data):
        return data.to(device, non_blocking=non_blocking)
    if isinstance(data, dict):
        return {key: move_training_data(value, device, non_blocking) for key, value in data.items()}
    if isinstance(data, (list, tuple)):
        return type(data)(move_training_data(value, device, non_blocking) for value in data)
    return data
