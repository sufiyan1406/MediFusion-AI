"""
Dataset Validation and Statistics Generator Script for MediFusion.

Performs strict integrity checks on metadata and dataset disk images, creates reproducible
patient-level Train/Val/Test splits, calculates pathology distributions, and exports a
machine-readable summary to reports/results/dataset_split_statistics.json.
"""

import sys
import json
from pathlib import Path
import pandas as pd

# Add project root directory to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import (
    DATASET_DIR,
    METADATA_CSV_PATH,
    TRAIN_VAL_LIST_PATH,
    TEST_LIST_PATH,
    EXPECTED_PATHOLOGIES,
    VAL_PATIENT_RATIO,
    RANDOM_SEED,
    RESULTS_DIR,
)
from src.utils.seed import set_seed
from src.data.label_encoder import LabelEncoder
from src.data.split import build_image_disk_mapping, validate_dataset_integrity, create_patient_level_splits


def compute_split_statistics(df_split: pd.DataFrame, split_name: str, label_encoder: LabelEncoder) -> dict:
    """
    Calculate detailed statistics for a dataset split.

    Args:
        df_split (pd.DataFrame): DataFrame for the specific split.
        split_name (str): Name of the split ('train', 'validation', or 'test').
        label_encoder (LabelEncoder): LabelEncoder instance.

    Returns:
        dict: Summary statistics dictionary.
    """
    total_images = len(df_split)
    unique_patients = df_split["Patient ID"].nunique()

    # Split finding labels
    label_lists = df_split["Finding Labels"].apply(lambda x: [t.strip() for t in str(x).split("|") if t.strip()])
    label_counts_per_image = label_lists.apply(len)

    # Co-occurring multi-finding count (images with >= 2 pathology findings)
    # Note: 'No Finding' counts as 1 label in string, but 0 pathology findings
    multi_finding_images = label_lists.apply(
        lambda tags: len([t for t in tags if t != "No Finding"]) >= 2
    ).sum()

    no_finding_count = label_lists.apply(lambda tags: "No Finding" in tags).sum()

    # Pathology class frequencies
    pathology_stats = {}
    for pathology in EXPECTED_PATHOLOGIES:
        count = label_lists.apply(lambda tags: pathology in tags).sum()
        pct = (count / total_images) * 100.0 if total_images > 0 else 0.0
        pathology_stats[pathology] = {
            "count": int(count),
            "percentage": round(pct, 4)
        }

    return {
        "split_name": split_name,
        "total_images": int(total_images),
        "unique_patients": int(unique_patients),
        "no_finding_images": int(no_finding_count),
        "multi_finding_images": int(multi_finding_images),
        "pathology_counts": pathology_stats
    }


def main():
    print("=" * 70)
    print(" MediFusion Foundation — Data Validation & Split Generator")
    print("=" * 70)

    # 1. Set global random seed
    set_seed(RANDOM_SEED)

    # 2. Initialize label encoder and validate expected pathology set
    label_encoder = LabelEncoder(EXPECTED_PATHOLOGIES)

    # 3. Read metadata CSV
    if not METADATA_CSV_PATH.exists():
        raise FileNotFoundError(f"Metadata file not found at '{METADATA_CSV_PATH}'!")

    print(f"\n[1/5] Loading metadata CSV from '{METADATA_CSV_PATH}'...")
    df_raw = pd.read_csv(METADATA_CSV_PATH)
    print(f"      Loaded {len(df_raw)} records across {df_raw['Patient ID'].nunique()} unique patients.")

    # 4. Validate metadata labels
    print("\n[2/5] Validating finding labels against expected pathology schema...")
    label_encoder.validate_dataframe(df_raw)

    # 5. Build image disk mapping and run integrity checks
    print(f"\n[3/5] Indexing raw images in '{DATASET_DIR}'...")
    image_map = build_image_disk_mapping(str(DATASET_DIR))
    print(f"      Found {len(image_map)} PNG files on disk.")

    with open(TRAIN_VAL_LIST_PATH, "r") as f:
        train_val_filenames = set(line.strip() for line in f if line.strip())

    with open(TEST_LIST_PATH, "r") as f:
        test_filenames = set(line.strip() for line in f if line.strip())

    validate_dataset_integrity(df_raw, image_map, train_val_filenames, test_filenames)

    # 6. Create patient-level splits
    print("\n[4/5] Creating patient-grouped Train, Validation, and Test splits...")
    df_train, df_val, df_test = create_patient_level_splits(
        df=df_raw,
        train_val_list_path=str(TRAIN_VAL_LIST_PATH),
        test_list_path=str(TEST_LIST_PATH),
        val_patient_ratio=VAL_PATIENT_RATIO,
        random_seed=RANDOM_SEED,
    )

    # 7. Compute statistics for all splits
    print("\n[5/5] Computing multi-label class distributions across splits...")
    train_stats = compute_split_statistics(df_train, "train", label_encoder)
    val_stats = compute_split_statistics(df_val, "validation", label_encoder)
    test_stats = compute_split_statistics(df_test, "test", label_encoder)

    overall_stats = {
        "dataset_name": "NIH ChestX-ray14",
        "random_seed": RANDOM_SEED,
        "val_patient_ratio": VAL_PATIENT_RATIO,
        "pathologies": EXPECTED_PATHOLOGIES,
        "train": train_stats,
        "validation": val_stats,
        "test": test_stats
    }

    # Export machine-readable JSON results file
    output_json_path = RESULTS_DIR / "dataset_split_statistics.json"
    with open(output_json_path, "w") as f:
        json.dump(overall_stats, f, indent=4)

    print(f"\n[MediFusion SUCCESS] Split statistics exported to '{output_json_path}'.")
    print("=" * 70)


if __name__ == "__main__":
    main()
