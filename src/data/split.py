"""
Patient-Level Data Splitting and Integrity Validation Module for MediFusion.

Enforces strict patient-level separation across Train, Validation, and Test sets
to prevent data leakage caused by multiple scans per patient.
Uses the official NIH split lists and performs patient-grouped splitting for Train/Val.
"""

import os
from typing import Dict, Set, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def build_image_disk_mapping(dataset_dir: str) -> Dict[str, str]:
    """
    Recursively index all PNG images under the dataset directory to map filename -> absolute path.

    Args:
        dataset_dir (str): Path to root raw dataset folder.

    Returns:
        Dict[str, str]: Dictionary mapping image filename (e.g. '00000001_000.png') to full disk path.
    """
    image_map = {}
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.lower().endswith(".png"):
                if file in image_map:
                    raise ValueError(f"DUPLICATE FILENAME DETECTED ON DISK: '{file}' found in multiple subfolders!")
                image_map[file] = os.path.join(root, file)
    return image_map


def validate_dataset_integrity(
    df: pd.DataFrame,
    image_map: Dict[str, str],
    train_val_filenames: Set[str],
    test_filenames: Set[str]
) -> None:
    """
    Run strict validation checks on metadata, disk files, and official split lists.
    Fails loudly if any integrity issue is found.
    """
    print("[MediFusion] Running Dataset Integrity & Validation Checks...")

    # Check 1: Duplicate filenames in CSV
    if df["Image Index"].duplicated().any():
        dups = df[df["Image Index"].duplicated()]["Image Index"].tolist()
        raise ValueError(f"DATA INTEGRITY FAILURE: Found duplicate image filenames in metadata CSV: {dups[:5]}")

    # Check 2: Missing image files on disk
    all_csv_images = set(df["Image Index"])
    missing_on_disk = all_csv_images - set(image_map.keys())
    if missing_on_disk:
        raise FileNotFoundError(
            f"DATA INTEGRITY FAILURE: {len(missing_on_disk)} images listed in Data_Entry_2017.csv are missing on disk!"
        )

    # Check 3: Check official split lists match CSV images
    split_images = train_val_filenames.union(test_filenames)
    missing_in_splits = all_csv_images - split_images
    if missing_in_splits:
        raise ValueError(
            f"DATA INTEGRITY FAILURE: {len(missing_in_splits)} images in CSV are not accounted for in train_val_list or test_list!"
        )

    # Check 4: Check patient overlap between official train_val and official test lists
    df_train_val = df[df["Image Index"].isin(train_val_filenames)]
    df_test = df[df["Image Index"].isin(test_filenames)]

    train_val_patients = set(df_train_val["Patient ID"])
    test_patients = set(df_test["Patient ID"])

    overlap = train_val_patients.intersection(test_patients)
    if overlap:
        raise ValueError(
            f"DATA LEAKAGE DETECTED! Found {len(overlap)} overlapping patients between official train_val and test lists!"
        )

    print("[MediFusion] Dataset Integrity Checks PASSED! Zero missing files, zero duplicate filenames, zero patient overlap.")


def create_patient_level_splits(
    df: pd.DataFrame,
    train_val_list_path: str,
    test_list_path: str,
    val_patient_ratio: float = 0.15,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create reproducible Train, Validation, and Test DataFrames with strict patient-level separation.

    Steps:
        1. Read official `train_val_list.txt` and `test_list.txt`.
        2. Assign official `test_list.txt` records to Test DataFrame.
        3. Partition official `train_val` records into Train and Validation using GroupShuffleSplit on 'Patient ID'.
        4. Verify zero patient overlap across Train, Validation, and Test sets.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (df_train, df_val, df_test)
    """
    # Read official split lists
    with open(train_val_list_path, "r") as f:
        train_val_filenames = set(line.strip() for line in f if line.strip())

    with open(test_list_path, "r") as f:
        test_filenames = set(line.strip() for line in f if line.strip())

    # Create official Test subset
    df_test = df[df["Image Index"].isin(test_filenames)].copy().reset_index(drop=True)

    # Create official Train/Val subset
    df_train_val = df[df["Image Index"].isin(train_val_filenames)].copy().reset_index(drop=True)

    # GroupShuffleSplit based on 'Patient ID' to ensure no patient overlap between Train and Val
    gss = GroupShuffleSplit(n_splits=1, test_size=val_patient_ratio, random_state=random_seed)
    groups = df_train_val["Patient ID"]

    train_idx, val_idx = next(gss.split(df_train_val, groups=groups))

    df_train = df_train_val.iloc[train_idx].copy().reset_index(drop=True)
    df_val = df_train_val.iloc[val_idx].copy().reset_index(drop=True)

    # Final Patient Leakage Verification
    train_patients = set(df_train["Patient ID"])
    val_patients = set(df_val["Patient ID"])
    test_patients = set(df_test["Patient ID"])

    tv_overlap = train_patients.intersection(val_patients)
    tt_overlap = train_patients.intersection(test_patients)
    vt_overlap = val_patients.intersection(test_patients)

    if tv_overlap or tt_overlap or vt_overlap:
        raise ValueError(
            f"CRITICAL PATIENT LEAKAGE ERROR:\n"
            f"  Train-Val Overlap: {len(tv_overlap)} patients\n"
            f"  Train-Test Overlap: {len(tt_overlap)} patients\n"
            f"  Val-Test Overlap: {len(vt_overlap)} patients"
        )

    print(
        f"[MediFusion] Patient-Level Splits Created Successfully:\n"
        f"  • Train Set: {len(df_train)} images across {len(train_patients)} unique patients\n"
        f"  • Val Set:   {len(df_val)} images across {len(val_patients)} unique patients\n"
        f"  • Test Set:  {len(df_test)} images across {len(test_patients)} unique patients"
    )

    return df_train, df_val, df_test
