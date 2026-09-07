"""
Preprocessing and Data Augmentation Transformation Pipelines for MediFusion.

Standardizes Chest X-Ray images into 1-channel 224x224 grayscale tensors.
Provides conservative training data augmentation and deterministic validation/test pipelines.
"""

from typing import Tuple
from torchvision import transforms
from configs.config import IMAGE_SIZE, AUG_ROTATION_DEGREES, AUG_TRANSLATION, AUG_COLOR_JITTER


def get_baseline_transforms(image_size: Tuple[int, int] = IMAGE_SIZE) -> transforms.Compose:
    """
    Construct the deterministic transformation pipeline for Validation and Testing.

    Steps:
        1. Resize image to 224x224 pixels.
        2. Convert PIL Image to PyTorch FloatTensor (scales values to [0.0, 1.0]).
        3. Normalize 1-channel grayscale values: (input - 0.5) / 0.5 -> range [-1.0, 1.0].

    Args:
        image_size (Tuple[int, int]): Height and Width dimensions for resizing.

    Returns:
        transforms.Compose: PyTorch torchvision transform pipeline.
    """
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])


def get_train_transforms(image_size: Tuple[int, int] = IMAGE_SIZE) -> transforms.Compose:
    """
    Construct the conservative training data augmentation pipeline.

    Augmentations:
        - Resize to 224x224
        - Random Rotation (+/- 7 degrees)
        - Random Affine Translation (+/- 4%)
        - ColorJitter Brightness & Contrast (+/- 10%)
        - ToTensor
        - Normalize (mean=[0.5], std=[0.5])

    Note: Horizontal flipping is STRICTLY OMITTED to preserve thoracic anatomical positioning.

    Args:
        image_size (Tuple[int, int]): Height and Width dimensions for resizing.

    Returns:
        transforms.Compose: PyTorch torchvision training transform pipeline.
    """
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomRotation(degrees=AUG_ROTATION_DEGREES),
        transforms.RandomAffine(degrees=0, translate=AUG_TRANSLATION),
        transforms.ColorJitter(brightness=AUG_COLOR_JITTER[0], contrast=AUG_COLOR_JITTER[1]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
