"""
Foundation Unit Test Suite for MediFusion ML Foundation.

Tests:
    1. LabelEncoder encoding, multi-hot parsing, and invalid label error raising.
    2. Baseline image preprocessing transforms.
    3. PyTorch Dataset lazy sample retrieval and tensor shapes.
    4. Patient-level grouping and zero patient leakage across Train/Val/Test splits.
"""

import os
import pytest
import numpy as np
import pandas as pd
import torch

from configs.config import EXPECTED_PATHOLOGIES, IMAGE_SIZE
from src.data.label_encoder import LabelEncoder
from src.data.transforms import get_baseline_transforms
from src.data.dataset import NIHChestXRayDataset
from src.data.split import create_patient_level_splits


@pytest.fixture
def label_encoder():
    return LabelEncoder(EXPECTED_PATHOLOGIES)


def test_label_encoder_single_disease(label_encoder):
    """Test encoding a single pathology string."""
    vec = label_encoder.encode("Cardiomegaly")
    assert vec.shape == (14,)
    assert vec[label_encoder.class_to_idx["Cardiomegaly"]] == 1.0
    assert np.sum(vec) == 1.0


def test_label_encoder_multiple_diseases(label_encoder):
    """Test encoding a pipe-separated string with multiple pathologies."""
    vec = label_encoder.encode("Cardiomegaly|Effusion")
    assert vec.shape == (14,)
    assert vec[label_encoder.class_to_idx["Cardiomegaly"]] == 1.0
    assert vec[label_encoder.class_to_idx["Effusion"]] == 1.0
    assert np.sum(vec) == 2.0


def test_label_encoder_no_finding(label_encoder):
    """Test encoding 'No Finding' returns an all-zero target vector."""
    vec = label_encoder.encode("No Finding")
    assert vec.shape == (14,)
    assert np.sum(vec) == 0.0


def test_label_encoder_unknown_label_fails(label_encoder):
    """Test that an unknown label raises a ValueError."""
    with pytest.raises(ValueError, match="UNRECOGNIZED LABEL DETECTED"):
        label_encoder.encode("Cardiomegaly|UnknownDisease")


def test_baseline_transforms():
    """Test image preprocessing transform pipeline shapes."""
    from PIL import Image
    # Create a dummy 500x500 grayscale image
    img = Image.fromarray(np.uint8(np.random.randint(0, 256, (500, 500))))
    transform = get_baseline_transforms(IMAGE_SIZE)
    tensor = transform(img)
    
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 224, 224)
    # Range check for normalized values (mean 0.5, std 0.5 maps [0, 1] to [-1, 1])
    assert tensor.min() >= -1.0
    assert tensor.max() <= 1.0


def test_dataset_sample_retrieval(label_encoder, tmp_path):
    """Test lazy loading and sample dictionary structure of NIHChestXRayDataset."""
    from PIL import Image
    # Create temporary fake image file
    fake_img_name = "00000001_000.png"
    fake_img_path = str(tmp_path / fake_img_name)
    Image.fromarray(np.uint8(np.random.randint(0, 256, (1024, 1024)))).save(fake_img_path)

    df_dummy = pd.DataFrame([{
        "Image Index": fake_img_name,
        "Patient ID": 1,
        "Finding Labels": "Cardiomegaly|Effusion"
    }])

    image_dir_map = {fake_img_name: fake_img_path}
    transform = get_baseline_transforms(IMAGE_SIZE)

    dataset = NIHChestXRayDataset(
        df=df_dummy,
        image_dir_map=image_dir_map,
        label_encoder=label_encoder,
        transform=transform
    )

    assert len(dataset) == 1
    sample = dataset[0]

    assert sample["image"].shape == (1, 224, 224)
    assert sample["target"].shape == (14,)
    assert sample["image_filename"] == fake_img_name
    assert sample["patient_id"] == 1
    assert sample["target"][label_encoder.class_to_idx["Cardiomegaly"]] == 1.0
    assert sample["target"][label_encoder.class_to_idx["Effusion"]] == 1.0


def test_patient_level_splitting_no_leakage(tmp_path):
    """Test that create_patient_level_splits creates 0 patient overlap across splits."""
    # Create dummy split files
    train_val_file = tmp_path / "train_val_list.txt"
    test_file = tmp_path / "test_list.txt"

    train_val_file.write_text("img1.png\nimg2.png\nimg3.png\nimg4.png\n")
    test_file.write_text("img5.png\nimg6.png\n")

    df_dummy = pd.DataFrame([
        {"Image Index": "img1.png", "Patient ID": 101, "Finding Labels": "Effusion"},
        {"Image Index": "img2.png", "Patient ID": 101, "Finding Labels": "Atelectasis"},
        {"Image Index": "img3.png", "Patient ID": 102, "Finding Labels": "No Finding"},
        {"Image Index": "img4.png", "Patient ID": 103, "Finding Labels": "Pneumonia"},
        {"Image Index": "img5.png", "Patient ID": 104, "Finding Labels": "Edema"},
        {"Image Index": "img6.png", "Patient ID": 105, "Finding Labels": "Hernia"},
    ])

    df_train, df_val, df_test = create_patient_level_splits(
        df=df_dummy,
        train_val_list_path=str(train_val_file),
        test_list_path=str(test_file),
        val_patient_ratio=0.33,
        random_seed=42
    )

    train_patients = set(df_train["Patient ID"])
    val_patients = set(df_val["Patient ID"])
    test_patients = set(df_test["Patient ID"])

    assert len(train_patients.intersection(val_patients)) == 0
    assert len(train_patients.intersection(test_patients)) == 0
    assert len(val_patients.intersection(test_patients)) == 0
