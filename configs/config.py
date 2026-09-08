"""
MediFusion Configuration File
Centralized ML parameters and paths for Chest X-Ray Multi-Label Classification.

Fully cross-platform: dynamically resolves dataset, model checkpoint, and results paths
for both local Windows environments and Kaggle Linux GPU environments.
"""

import os
import torch
from pathlib import Path

# Project Root Directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_dataset_dir() -> Path:
    """
    Dynamically resolve the root dataset directory.
    Checks environment variables first, then Kaggle mount paths, then local fallback.
    """
    env_dir = os.environ.get("DATA_ROOT") or os.environ.get("DATASET_DIR")
    if env_dir:
        return Path(env_dir)

    # Kaggle environment check
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        # Search recursively for Data_Entry_2017.csv inside /kaggle/input
        for csv_path in kaggle_input.rglob("Data_Entry_2017.csv"):
            return csv_path.parent

    # Local fallback
    return PROJECT_ROOT / "DataSet"


def get_dataset_file(filename: str, dataset_dir: Path) -> Path:
    """
    Locate a dataset metadata file recursively if not directly present at root.
    """
    direct_path = dataset_dir / filename
    if direct_path.exists():
        return direct_path

    # Search recursively inside dataset_dir
    found_files = list(dataset_dir.rglob(filename))
    if found_files:
        return found_files[0]

    return direct_path


# -----------------------------------------------------------------------------
# Data & Storage Paths (Cross-Platform / Kaggle-Aware)
# -----------------------------------------------------------------------------
DATASET_DIR = resolve_dataset_dir()
METADATA_CSV_PATH = get_dataset_file("Data_Entry_2017.csv", DATASET_DIR)
BBOX_CSV_PATH = get_dataset_file("BBox_List_2017.csv", DATASET_DIR)
TRAIN_VAL_LIST_PATH = get_dataset_file("train_val_list.txt", DATASET_DIR)
TEST_LIST_PATH = get_dataset_file("test_list.txt", DATASET_DIR)

# Output & Artifact Directories
def resolve_output_dir(env_name: str, local_default: Path, kaggle_default: Path) -> Path:
    env_val = os.environ.get(env_name)
    if env_val:
        return Path(env_val)
    if Path("/kaggle/working").exists():
        return kaggle_default
    return local_default


OUTPUT_MODELS_DIR = resolve_output_dir(
    "CHECKPOINT_DIR",
    PROJECT_ROOT / "models",
    Path("/kaggle/working/models")
)

REPORTS_DIR = resolve_output_dir(
    "REPORTS_DIR",
    PROJECT_ROOT / "reports",
    Path("/kaggle/working/reports")
)

RESULTS_DIR = resolve_output_dir(
    "RESULTS_DIR",
    REPORTS_DIR / "results",
    Path("/kaggle/working/reports/results")
)

FIGURES_DIR = REPORTS_DIR / "figures"

# Ensure output directories exist safely
os.makedirs(OUTPUT_MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Target Labels Specification (Exact 14 Pathology Order)
# -----------------------------------------------------------------------------
EXPECTED_PATHOLOGIES = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]

NUM_CLASSES = len(EXPECTED_PATHOLOGIES)  # Exactly 14 classes

# -----------------------------------------------------------------------------
# Preprocessing & Image Hyperparameters
# -----------------------------------------------------------------------------
IMAGE_SIZE = (224, 224)  # Height x Width downsampling for local & Kaggle GPU efficiency
NUM_CHANNELS = 1         # Single-channel grayscale
IMAGE_MEAN = [0.5]
IMAGE_STD = [0.5]

# Data Augmentation Parameters (Training Only)
AUG_ROTATION_DEGREES = 7
AUG_TRANSLATION = (0.04, 0.04)
AUG_COLOR_JITTER = (0.10, 0.10)

# -----------------------------------------------------------------------------
# Model & Training Hyperparameters (BASELINE - MedCXRNet Scratch Training)
# -----------------------------------------------------------------------------
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 16))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", 1e-4))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", 1e-4))
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", 15))
VAL_PATIENT_RATIO = 0.15

# LR Scheduler & Early Stopping
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 2
EARLY_STOPPING_PATIENCE = 4

# Classification Threshold
DEFAULT_THRESHOLD = 0.5

# Reproducibility
RANDOM_SEED = 42

# -----------------------------------------------------------------------------
# System Hardware, DataLoader & Mixed Precision Configuration
# -----------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = torch.cuda.is_available()

# Configurable DataLoader settings for Kaggle GPU throughput
DEFAULT_NUM_WORKERS = 4 if Path("/kaggle/input").exists() else 2
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", DEFAULT_NUM_WORKERS))
PIN_MEMORY = os.environ.get("PIN_MEMORY", "true").lower() == "true" if torch.cuda.is_available() else False
PERSISTENT_WORKERS = os.environ.get("PERSISTENT_WORKERS", "true").lower() == "true" if NUM_WORKERS > 0 else False
