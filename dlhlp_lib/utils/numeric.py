import numpy as np
import torch


def numpy_exist_nan(x: np.array):
    return np.any(np.isnan(x))


def torch_exist_nan(x: torch.Tensor):
    return (x != x).any()
