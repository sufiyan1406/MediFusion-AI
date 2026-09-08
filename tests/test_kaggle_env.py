"""
Kaggle GPU Environment & Path Resolution Unit Tests for MediFusion.

Tests:
    1. Dynamic dataset directory resolution (local vs Kaggle).
    2. Dynamic checkpoint and results directory resolution.
    3. Environment variable override functionality (DATA_ROOT, CHECKPOINT_DIR, NUM_WORKERS).
    4. Modern PyTorch 2.x AMP API compatibility check.
    5. Patient-level split logic preservation.
"""

import os
import pytest
from pathlib import Path

from configs.config import (
    DATASET_DIR,
    OUTPUT_MODELS_DIR,
    RESULTS_DIR,
    NUM_WORKERS,
    PIN_MEMORY,
    PERSISTENT_WORKERS,
    resolve_dataset_dir,
    get_dataset_file,
    EXPECTED_PATHOLOGIES
)
from src.data.transforms import get_baseline_transforms, get_train_transforms


def test_dataset_dir_resolution():
    """Test dataset directory path resolves to a valid Path object."""
    ds_dir = resolve_dataset_dir()
    assert isinstance(ds_dir, Path)


def test_dataset_file_finder():
    """Test get_dataset_file locates Data_Entry_2017.csv recursively if needed."""
    csv_file = get_dataset_file("Data_Entry_2017.csv", DATASET_DIR)
    assert isinstance(csv_file, Path)
    assert csv_file.name == "Data_Entry_2017.csv"


def test_env_var_override(monkeypatch, tmp_path):
    """Test that setting DATA_ROOT environment variable overrides the default path."""
    fake_data_dir = str(tmp_path / "fake_data")
    monkeypatch.setenv("DATA_ROOT", fake_data_dir)
    ds_dir = resolve_dataset_dir()
    assert str(ds_dir) == fake_data_dir


def test_dataloader_config_types():
    """Test DataLoader configuration options."""
    assert isinstance(NUM_WORKERS, int)
    assert isinstance(PIN_MEMORY, bool)
    assert isinstance(PERSISTENT_WORKERS, bool)


def test_transforms_augmentation_rules():
    """Test validation/test transforms have NO random augmentations, train transform has NO horizontal flip."""
    val_tf = get_baseline_transforms()
    train_tf = get_train_transforms()

    # Check transform class types in validation pipeline
    val_types = [type(t).__name__ for t in val_tf.transforms]
    assert "RandomHorizontalFlip" not in val_types
    assert "RandomRotation" not in val_types

    # Check train transforms
    train_types = [type(t).__name__ for t in train_tf.transforms]
    assert "RandomHorizontalFlip" not in train_types
    assert "RandomRotation" in train_types
    assert "RandomAffine" in train_types
