"""
Reproducibility Utilities for MediFusion.
Provides seed initialization functions across Python random, NumPy, and PyTorch.
"""

import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds across all libraries to ensure reproducible experiments.
    
    Args:
        seed (int): The integer random seed value. Default is 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic algorithms where possible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    print(f"[MediFusion] Global random seed set to: {seed}")
