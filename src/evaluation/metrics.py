"""
Multi-Label Evaluation Metrics Module for MediFusion.

Computes comprehensive multi-label evaluation metrics:
    - Macro & Per-Class ROC-AUC
    - Macro & Per-Class Precision, Recall (Sensitivity), Specificity, F1-Score
    - Macro & Per-Class PR-AUC (Average Precision)

Gracefully handles edge cases where classes have zero positive or zero negative samples,
returning NaN for undefined metrics without crashing.
"""

from typing import Dict, List, Union, Tuple, Any
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, average_precision_score

from configs.config import EXPECTED_PATHOLOGIES, DEFAULT_THRESHOLD


def compute_sigmoid(logits: np.ndarray) -> np.ndarray:
    """
    Apply Sigmoid activation function to raw unnormalized logits.

    Args:
        logits (np.ndarray): Unnormalized model output array.

    Returns:
        np.ndarray: Probabilities in range [0.0, 1.0].
    """
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -88.0, 88.0)))


def calculate_multilabel_metrics(
    y_pred: Union[torch.Tensor, np.ndarray],
    y_true: Union[torch.Tensor, np.ndarray],
    class_names: List[str] = EXPECTED_PATHOLOGIES,
    threshold: float = DEFAULT_THRESHOLD,
    is_logits: bool = True
) -> Dict[str, Any]:
    """
    Calculate comprehensive multi-label metrics across 14 pathology classes.

    Args:
        y_pred (Union[torch.Tensor, np.ndarray]): Model outputs of shape (N, 14) (logits or probabilities).
        y_true (Union[torch.Tensor, np.ndarray]): Ground truth multi-hot binary targets of shape (N, 14).
        class_names (List[str]): List of 14 pathology class names.
        threshold (float): Decision threshold for converting probabilities to binary predictions (default 0.5).
        is_logits (bool): Whether y_pred contains raw unnormalized logits.

    Returns:
        Dict[str, Any]: Dictionary containing macro metrics and per-class detail dictionaries.
    """
    # 1. Convert torch Tensors to numpy arrays
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    # 2. Compute probabilities
    if is_logits:
        probs = compute_sigmoid(y_pred)
    else:
        probs = np.clip(y_pred, 0.0, 1.0)

    # 3. Compute binary predictions
    binary_preds = (probs >= threshold).astype(np.float32)

    num_classes = len(class_names)
    per_class_auc: Dict[str, float] = {}
    per_class_pr_auc: Dict[str, float] = {}
    per_class_precision: Dict[str, float] = {}
    per_class_recall: Dict[str, float] = {}
    per_class_specificity: Dict[str, float] = {}
    per_class_f1: Dict[str, float] = {}

    auc_list = []
    pr_auc_list = []
    precision_list = []
    recall_list = []
    specificity_list = []
    f1_list = []

    # 4. Iterate per pathology class
    for i, class_name in enumerate(class_names):
        y_t = y_true[:, i]
        p_score = probs[:, i]
        y_p = binary_preds[:, i]

        # Check if class has both positive and negative samples
        unique_targets = np.unique(y_t)

        # A. ROC-AUC Calculation
        if len(unique_targets) == 2:
            try:
                auc_val = float(roc_auc_score(y_t, p_score))
            except Exception:
                auc_val = np.nan
        else:
            # Undefined if only 1 class present in ground truth (e.g. all 0s or all 1s)
            auc_val = np.nan

        per_class_auc[class_name] = auc_val
        if not np.isnan(auc_val):
            auc_list.append(auc_val)

        # B. PR-AUC Calculation (Average Precision)
        if len(unique_targets) == 2 and np.sum(y_t) > 0:
            try:
                pr_auc_val = float(average_precision_score(y_t, p_score))
            except Exception:
                pr_auc_val = np.nan
        else:
            pr_auc_val = np.nan

        per_class_pr_auc[class_name] = pr_auc_val
        if not np.isnan(pr_auc_val):
            pr_auc_list.append(pr_auc_val)

        # C. Confusion Matrix Elements (TP, FP, TN, FN)
        tp = np.sum((y_t == 1) & (y_p == 1))
        fp = np.sum((y_t == 0) & (y_p == 1))
        tn = np.sum((y_t == 0) & (y_p == 0))
        fn = np.sum((y_t == 1) & (y_p == 0))

        # Precision
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        # Recall / Sensitivity
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        # Specificity
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        # F1 Score
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        per_class_precision[class_name] = prec
        per_class_recall[class_name] = rec
        per_class_specificity[class_name] = spec
        per_class_f1[class_name] = f1

        precision_list.append(prec)
        recall_list.append(rec)
        specificity_list.append(spec)
        f1_list.append(f1)

    # 5. Compute Macro Metrics (using np.nanmean to ignore NaN classes)
    macro_auc = float(np.nanmean(auc_list)) if len(auc_list) > 0 else np.nan
    macro_pr_auc = float(np.nanmean(pr_auc_list)) if len(pr_auc_list) > 0 else np.nan
    macro_precision = float(np.mean(precision_list))
    macro_recall = float(np.mean(recall_list))
    macro_specificity = float(np.mean(specificity_list))
    macro_f1 = float(np.mean(f1_list))

    return {
        "macro_roc_auc": macro_auc,
        "macro_pr_auc": macro_pr_auc,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_specificity": macro_specificity,
        "macro_f1": macro_f1,
        "threshold": threshold,
        "valid_auc_classes_count": len(auc_list),
        "per_class_roc_auc": per_class_auc,
        "per_class_pr_auc": per_class_pr_auc,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "per_class_specificity": per_class_specificity,
        "per_class_f1": per_class_f1,
    }
