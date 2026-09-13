"""
Unit Test Suite for Model & Training Pipeline in MediFusion.

Tests:
    A. Model input shape [2, 1, 224, 224] -> output shape [2, 14].
    B. Model produces finite logits (no NaN or Inf).
    C. Parameter count is in expected approximate range (~2.1M parameters).
    D. BCEWithLogitsLoss accepts logits [batch, 14] and targets [batch, 14].
    E. One forward + backward pass works cleanly.
    F. CPU execution works even when CUDA is unavailable.
    G. Metric function works on a synthetic multi-label example.
    H. Metric function does not crash when one class has no positive samples (returns NaN gracefully).
    I. Dataset -> DataLoader -> Model integration works for a small sample count.
"""

import pytest
import numpy as np
import pandas as pd
import torch
from PIL import Image

from configs.config import EXPECTED_PATHOLOGIES, IMAGE_SIZE
from src.models.medcxrnet import MedCXRNet, count_parameters
from src.training.loss import get_loss_function, compute_class_pos_weights
from src.evaluation.metrics import calculate_multilabel_metrics, compute_sigmoid
from src.data.label_encoder import LabelEncoder
from src.data.transforms import get_baseline_transforms
from src.data.dataset import NIHChestXRayDataset
from torch.utils.data import DataLoader


def test_model_input_output_shape():
    """Test A: Input [2, 1, 224, 224] produces output logits of shape [2, 14]."""
    model = MedCXRNet()
    x = torch.randn(2, 1, 224, 224)
    out = model(x)
    assert out.shape == (2, 14)


def test_model_finite_logits():
    """Test B: Model produces finite logits without NaN or Inf."""
    model = MedCXRNet()
    x = torch.randn(2, 1, 224, 224)
    out = model(x)
    assert torch.isfinite(out).all()


def test_model_parameter_count():
    """Test C: Parameter count is in expected approximate range (~392K parameters with GAP)."""
    model = MedCXRNet()
    counts = count_parameters(model)
    total_params = counts["total_parameters"]
    # Verify parameter count is between 300K and 500K (391,918 params)
    assert 300_000 <= total_params <= 500_000, f"Unexpected parameter count: {total_params}"


def test_loss_function_shape_compatibility():
    """Test D: BCEWithLogitsLoss accepts logits [2, 14] and targets [2, 14]."""
    criterion = get_loss_function(weighted=False)
    logits = torch.randn(2, 14)
    targets = torch.randint(0, 2, (2, 14)).float()
    loss = criterion(logits, targets)
    assert loss.dim() == 0  # Scalar loss tensor
    assert torch.isfinite(loss)


def test_forward_backward_pass():
    """Test E: One forward + backward pass executes without error."""
    model = MedCXRNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = get_loss_function(weighted=False)

    x = torch.randn(2, 1, 224, 224)
    targets = torch.randint(0, 2, (2, 14)).float()

    optimizer.zero_grad()
    logits = model(x)
    loss = criterion(logits, targets)
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)


def test_cpu_execution_fallback():
    """Test F: Model runs on CPU explicitly."""
    model = MedCXRNet().to("cpu")
    x = torch.randn(1, 1, 224, 224, device="cpu")
    logits = model(x)
    assert logits.device.type == "cpu"
    assert logits.shape == (1, 14)


def test_metrics_calculation_synthetic():
    """Test G: Metric function works on a synthetic multi-label example."""
    np.random.seed(42)
    y_true = np.random.randint(0, 2, size=(20, 14)).astype(np.float32)
    y_pred = np.random.randn(20, 14).astype(np.float32)  # Logits

    metrics = calculate_multilabel_metrics(
        y_pred=y_pred,
        y_true=y_true,
        class_names=EXPECTED_PATHOLOGIES,
        threshold=0.5,
        is_logits=True
    )

    assert "macro_roc_auc" in metrics
    assert "per_class_roc_auc" in metrics
    assert len(metrics["per_class_roc_auc"]) == 14
    assert 0.0 <= metrics["macro_precision"] <= 1.0


def test_metrics_calculation_missing_positive_class():
    """Test H: Metric function does not crash when one class has 0 positive targets."""
    y_true = np.zeros((10, 14), dtype=np.float32)
    # Give class 0 positive targets, leave class 1 with all zeros
    y_true[:5, 0] = 1.0
    y_pred = np.random.randn(10, 14).astype(np.float32)

    metrics = calculate_multilabel_metrics(
        y_pred=y_pred,
        y_true=y_true,
        class_names=EXPECTED_PATHOLOGIES,
        threshold=0.5,
        is_logits=True
    )

    # Class 1 should return NaN for ROC-AUC
    assert np.isnan(metrics["per_class_roc_auc"]["Cardiomegaly"])
    # Macro ROC-AUC should still evaluate using valid class 0
    assert not np.isnan(metrics["macro_roc_auc"])


def test_dataloader_model_integration(tmp_path):
    """Test I: Dataset -> DataLoader -> Model integration for 4 small synthetic samples."""
    label_encoder = LabelEncoder(EXPECTED_PATHOLOGIES)
    img_map = {}
    records = []

    for i in range(4):
        img_name = f"0000000{i}_000.png"
        img_path = str(tmp_path / img_name)
        Image.fromarray(np.uint8(np.random.randint(0, 256, (224, 224)))).save(img_path)
        img_map[img_name] = img_path
        records.append({
            "Image Index": img_name,
            "Patient ID": i + 1,
            "Finding Labels": "Atelectasis|Effusion" if i % 2 == 0 else "No Finding"
        })

    df_dummy = pd.DataFrame(records)
    dataset = NIHChestXRayDataset(
        df=df_dummy,
        image_dir_map=img_map,
        label_encoder=label_encoder,
        transform=get_baseline_transforms(IMAGE_SIZE)
    )

    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    model = MedCXRNet()

    for batch in loader:
        images = batch["image"]
        targets = batch["target"]
        logits = model(images)

        assert images.shape == (2, 1, 224, 224)
        assert targets.shape == (2, 14)
        assert logits.shape == (2, 14)


def test_compute_class_pos_weights():
    """Test J: Verify calculation of pos_weight = negative_count / positive_count from DataFrame."""
    label_encoder = LabelEncoder(EXPECTED_PATHOLOGIES)
    records = []
    # Create 10 dummy samples: 2 positive for Atelectasis (8 negative), 1 positive for Effusion (9 negative)
    for i in range(10):
        if i < 2:
            lbl = "Atelectasis"
        elif i == 2:
            lbl = "Effusion"
        else:
            lbl = "No Finding"
        records.append({"Finding Labels": lbl})

    df_dummy = pd.DataFrame(records)
    pos_weight_tensor, pos_weight_dict, class_counts = compute_class_pos_weights(df_dummy, label_encoder)

    assert pos_weight_tensor.shape == (14,)
    assert class_counts["Atelectasis"]["positive"] == 2
    assert class_counts["Atelectasis"]["negative"] == 8
    assert np.isclose(pos_weight_dict["Atelectasis"], 8.0 / 2.0)

    assert class_counts["Effusion"]["positive"] == 1
    assert class_counts["Effusion"]["negative"] == 9
    assert np.isclose(pos_weight_dict["Effusion"], 9.0 / 1.0)


def test_weighted_bce_loss_forward_backward():
    """Test K: Verify Weighted BCEWithLogitsLoss forward and backward pass."""
    pos_weight = torch.tensor([5.0] * 14, dtype=torch.float32)
    criterion = get_loss_function(weighted=True, pos_weight=pos_weight)

    model = MedCXRNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    x = torch.randn(2, 1, 224, 224)
    targets = torch.randint(0, 2, (2, 14)).float()

    optimizer.zero_grad()
    logits = model(x)
    loss = criterion(logits, targets)
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)
    assert loss.dim() == 0


def test_compute_sigmoid_numerical_stability():
    """Test L: Verify compute_sigmoid handles extreme logits without raising RuntimeWarning."""
    extreme_logits = np.array([-1000.0, -100.0, 0.0, 100.0, 1000.0], dtype=np.float32)
    probs = compute_sigmoid(extreme_logits)

    assert probs.shape == (5,)
    assert probs[0] == 0.0
    assert probs[2] == 0.5
    assert probs[4] == 1.0

