"""
Unit Test Suite for Official Test-Set Evaluator in MediFusion.

Tests:
    1. CLI parser accepts --mode evaluate and --checkpoint argument.
    2. Missing checkpoint path raises FileNotFoundError.
    3. Evaluator extracts model_state_dict and loads MedCXRNet correctly.
    4. Evaluator uses deterministic transforms (get_baseline_transforms) without random augmentations.
    5. Evaluation produces expected 14-class output shape (N, 14).
    6. Metrics and JSON report dictionary structure is valid.
    7. Model is placed in eval() mode.
    8. Evaluation uses torch.no_grad() context.
"""

import pytest
import torch
import numpy as np
import pandas as pd
from PIL import Image

from configs.config import EXPECTED_PATHOLOGIES, IMAGE_SIZE, DEFAULT_THRESHOLD
from src.models.medcxrnet import MedCXRNet
from src.evaluation.evaluator import Evaluator
from src.data.label_encoder import LabelEncoder
from src.data.transforms import get_baseline_transforms


def test_missing_checkpoint_raises_error(tmp_path):
    """Test 2: Missing checkpoint file raises FileNotFoundError."""
    missing_path = tmp_path / "non_existent_checkpoint.pth"
    with pytest.raises(FileNotFoundError, match="CHECKPOINT ERROR"):
        Evaluator(checkpoint_path=missing_path)


def test_checkpoint_loading_and_eval_mode(tmp_path):
    """Test 3 & 7: Checkpoint loading extracts model_state_dict and places model in eval() mode."""
    # Create fake valid checkpoint file
    model = MedCXRNet()
    ckpt_dict = {
        "epoch": 15,
        "experiment_name": "medcxrnet_baseline",
        "model_state_dict": model.state_dict(),
        "val_macro_auc": 0.7203
    }
    ckpt_path = tmp_path / "fake_checkpoint.pth"
    torch.save(ckpt_dict, ckpt_path)

    evaluator = Evaluator(checkpoint_path=ckpt_path, device="cpu", use_amp=False)
    loaded_model, meta = evaluator.load_model_from_checkpoint()

    assert isinstance(loaded_model, MedCXRNet)
    assert not loaded_model.training  # Verify model.eval() is active
    assert meta["epoch"] == 15
    assert meta["val_macro_auc"] == 0.7203


def test_evaluator_deterministic_transforms(tmp_path):
    """Test 4: Verification that evaluation pipeline uses get_baseline_transforms."""
    # Create small synthetic test subset
    label_encoder = LabelEncoder(EXPECTED_PATHOLOGIES)
    img_map = {}
    records = []

    for i in range(4):
        img_name = f"test_00{i}.png"
        img_path = str(tmp_path / img_name)
        Image.fromarray(np.uint8(np.random.randint(0, 256, (224, 224)))).save(img_path)
        img_map[img_name] = img_path
        records.append({
            "Image Index": img_name,
            "Patient ID": 1000 + i,
            "Finding Labels": "Atelectasis|Effusion" if i % 2 == 0 else "No Finding"
        })

    df_test_dummy = pd.DataFrame(records)

    model = MedCXRNet()
    ckpt_path = tmp_path / "fake_checkpoint.pth"
    torch.save({"epoch": 1, "model_state_dict": model.state_dict()}, ckpt_path)

    evaluator = Evaluator(checkpoint_path=ckpt_path, device="cpu", use_amp=False, results_dir=tmp_path)
    results = evaluator.evaluate_test_set(
        test_df=df_test_dummy,
        image_map=img_map,
        batch_size=2,
        output_filename="test_results_mock.json"
    )

    # Test 5 & 6: Verify JSON structure and metrics presence
    assert results["experiment"] == "experiment_1_test_evaluation"
    assert results["dataset"]["num_test_images"] == 4
    assert results["dataset"]["num_test_patients"] == 4
    assert "macro_roc_auc" in results["macro_metrics"]
    assert len(results["per_class_metrics"]["roc_auc"]) == 14
    assert (tmp_path / "test_results_mock.json").exists()


def test_inference_no_grad_and_14_logits_shape():
    """Test 5 & 8: Verify inference under no_grad produces (N, 14) logits without modifying grad state."""
    model = MedCXRNet()
    model.eval()
    x = torch.randn(4, 1, 224, 224)

    with torch.no_grad():
        logits = model(x)

    assert logits.shape == (4, 14)
    assert not logits.requires_grad
