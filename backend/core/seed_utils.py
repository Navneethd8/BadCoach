"""
Reproducibility: set all RNG seeds so training runs are deterministic.
Call at the start of training (before creating model/dataloaders).
"""
import random
import os

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set Python, NumPy, and PyTorch seeds for reproducible training.
    Also enables deterministic cuDNN (may slightly reduce speed on GPU).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # MPS (Apple Silicon) doesn't have a seed API; CPU/CUDA are the main ones.
    os.environ["PYTHONHASHSEED"] = str(seed)
