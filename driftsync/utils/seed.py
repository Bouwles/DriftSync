"""
Reproducibility Utilities
==========================
Seed all random number generators for reproducible experiments.
"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set seeds for Python, NumPy, and PyTorch (CPU + CUDA).

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device_str: str = "auto") -> torch.device:
    """
    Resolve device string to a torch.device.

    Args:
        device_str: "auto", "cpu", "cuda", or "mps".

    Returns:
        torch.device instance.
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)
