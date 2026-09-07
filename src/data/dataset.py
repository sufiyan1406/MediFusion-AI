"""
PyTorch Dataset Implementation for MediFusion.

Loads Chest X-Ray images lazily on-demand from disk, converts them to single-channel grayscale,
applies baseline image transformations, and returns the image tensor along with 14-class multi-hot target vectors.
"""

import os
from typing import Callable, Dict, Optional, Tuple, Any
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

from src.data.label_encoder import LabelEncoder


class NIHChestXRayDataset(Dataset):
    """
    PyTorch Dataset for NIH ChestX-ray14 single-channel multi-label pathology classification.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        image_dir_map: Dict[str, str],
        label_encoder: LabelEncoder,
        transform: Optional[Callable] = None,
    ):
        """
        Initialize the Dataset.

        Args:
            df (pd.DataFrame): DataFrame containing metadata ('Image Index', 'Patient ID', 'Finding Labels').
            image_dir_map (Dict[str, str]): Dictionary mapping image filenames to absolute file paths on disk.
            label_encoder (LabelEncoder): Instance of LabelEncoder to encode string labels into multi-hot targets.
            transform (Optional[Callable]): Optional torchvision transform pipeline to apply to images.
        """
        self.df = df.reset_index(drop=True)
        self.image_dir_map = image_dir_map
        self.label_encoder = label_encoder
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retrieve a single sample lazily on-the-fly.

        Args:
            idx (int): Sample index.

        Returns:
            Dict containing:
                - 'image': PyTorch Tensor of shape (1, 224, 224)
                - 'target': PyTorch Tensor of shape (14,)
                - 'image_filename': str
                - 'patient_id': int
        """
        row = self.df.iloc[idx]
        img_filename = row["Image Index"]
        patient_id = row["Patient ID"]
        finding_label_str = row["Finding Labels"]

        # Look up exact file path on disk
        if img_filename not in self.image_dir_map:
            raise FileNotFoundError(
                f"MISSING FILE ERROR: Image '{img_filename}' listed in metadata was not found on disk!"
            )

        img_path = self.image_dir_map[img_filename]

        # 1. Open image lazily and convert consistently to single-channel grayscale ('L')
        try:
            with Image.open(img_path) as img:
                img_grayscale = img.convert("L")
        except Exception as e:
            raise IOError(f"CORRUPT IMAGE ERROR: Failed to open image at '{img_path}'. Error: {e}")

        # 2. Apply torchvision baseline transform pipeline
        if self.transform is not None:
            image_tensor = self.transform(img_grayscale)
        else:
            # Fallback basic tensor conversion if no transform provided
            arr = np.array(img_grayscale, dtype=np.float32) / 255.0
            image_tensor = torch.from_numpy(arr).unsqueeze(0)  # (1, H, W)

        # 3. Encode label string into 14-element multi-hot target vector
        multi_hot_target = self.label_encoder.encode(finding_label_str)
        target_tensor = torch.from_numpy(multi_hot_target)  # (14,) float32

        return {
            "image": image_tensor,
            "target": target_tensor,
            "image_filename": img_filename,
            "patient_id": int(patient_id),
        }
