"""
Loss Function Module for MediFusion.

Provides loss function implementations and factory methods for 14-class multi-label
Chest X-Ray pathology classification using PyTorch BCEWithLogitsLoss.
"""

from typing import Optional
import torch
import torch.nn as nn


def get_loss_function(
    weighted: bool = False,
    pos_weight: Optional[torch.Tensor] = None,
    device: Optional[str] = None
) -> nn.Module:
    """
    Factory function to construct multi-label loss functions.

    Args:
        weighted (bool): Whether to apply positive class weights to handle imbalance.
        pos_weight (Optional[torch.Tensor]): Tensor of shape (14,) containing positive class weights.
        device (Optional[str]): Device to place pos_weight tensor on ('cuda' or 'cpu').

    Returns:
        nn.Module: Instantiated PyTorch loss function.
    """
    if weighted:
        if pos_weight is None:
            raise ValueError("Positive class weight tensor 'pos_weight' must be provided when weighted=True!")
        if device is not None:
            pos_weight = pos_weight.to(device)
        print(f"[MediFusion Loss] Using Weighted BCEWithLogitsLoss with pos_weight shape {pos_weight.shape}")
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    print("[MediFusion Loss] Using Standard Unweighted BCEWithLogitsLoss")
    return nn.BCEWithLogitsLoss()
