"""
Loss Function Module for MediFusion.

Provides loss function implementations and factory methods for 14-class multi-label
Chest X-Ray pathology classification using PyTorch BCEWithLogitsLoss.
"""

from typing import Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.data.label_encoder import LabelEncoder


def compute_class_pos_weights(
    df_train: pd.DataFrame,
    label_encoder: LabelEncoder,
    label_column: str = "Finding Labels"
) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, Dict[str, int]]]:
    """
    Calculate per-class positive weights exclusively from the training split DataFrame.

    Calculates:
        pos_weight[c] = negative_count[c] / positive_count[c]

    Args:
        df_train (pd.DataFrame): Training split DataFrame containing ground truth labels.
        label_encoder (LabelEncoder): Initialized LabelEncoder instance.
        label_column (str): Column name containing string finding labels.

    Returns:
        Tuple[torch.Tensor, Dict[str, float], Dict[str, Dict[str, int]]]:
            - pos_weight_tensor: PyTorch float32 tensor of shape (14,).
            - pos_weight_dict: Dictionary mapping class name -> pos_weight value.
            - class_counts_dict: Dictionary mapping class name -> {'positive': count, 'negative': count}.
    """
    if label_column not in df_train.columns:
        raise KeyError(f"Column '{label_column}' not found in training DataFrame!")

    num_samples = len(df_train)
    num_classes = label_encoder.num_classes

    # Multi-hot encode all rows in training DataFrame
    target_matrix = np.zeros((num_samples, num_classes), dtype=np.float32)
    for i, label_str in enumerate(df_train[label_column]):
        target_matrix[i] = label_encoder.encode(str(label_str))

    pos_counts = target_matrix.sum(axis=0)
    neg_counts = num_samples - pos_counts

    # Handle edge cases (zero positive samples) safely to avoid zero division
    pos_weights = np.zeros(num_classes, dtype=np.float32)
    for c in range(num_classes):
        if pos_counts[c] > 0:
            pos_weights[c] = float(neg_counts[c] / pos_counts[c])
        else:
            pos_weights[c] = 1.0

    pos_weight_tensor = torch.tensor(pos_weights, dtype=torch.float32)

    pos_weight_dict = {}
    class_counts_dict = {}
    for c, class_name in enumerate(label_encoder.classes):
        pos_weight_dict[class_name] = float(pos_weights[c])
        class_counts_dict[class_name] = {
            "positive": int(pos_counts[c]),
            "negative": int(neg_counts[c])
        }

    return pos_weight_tensor, pos_weight_dict, class_counts_dict


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

