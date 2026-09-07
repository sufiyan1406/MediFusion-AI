"""
MediFusion Configuration File
Centralized ML parameters and paths for the Chest X-Ray Multi-Label Baseline.
All values are baseline settings and clearly documented for beginner friendliness.
"""

import os
import torch
from pathlib import Path

# Project Root Directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# -----------------------------------------------------------------------------
# Data Paths (Raw Dataset is treated as strictly READ-ONLY)
# -----------------------------------------------------------------------------
DATASET_DIR = PROJECT_ROOT / "DataSet"
METADATA_CSV_PATH = DATASET_DIR / "Data_Entry_2017.csv"
BBOX_CSV_PATH = DATASET_DIR / "BBox_List_2017.csv"
TRAIN_VAL_LIST_PATH = DATASET_DIR / "train_val_list.txt"
TEST_LIST_PATH = DATASET_DIR / "test_list.txt"

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
IMAGE_SIZE = (224, 224)  # Height x Width downsampling for local GPU efficiency
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
BATCH_SIZE = 16           # Batch size suitable for 4GB VRAM
LEARNING_RATE = 1e-4      # Initial learning rate for AdamW optimizer
WEIGHT_DECAY = 1e-4       # L2 Weight Decay
NUM_EPOCHS = 15           # Standard initial training epochs
VAL_PATIENT_RATIO = 0.15  # 15% of patients from official train_val list reserved for validation

# LR Scheduler & Early Stopping
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 2
EARLY_STOPPING_PATIENCE = 4

# Classification Threshold
DEFAULT_THRESHOLD = 0.5

# Reproducibility
RANDOM_SEED = 42

# System Hardware & Mixed Precision
NUM_WORKERS = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = torch.cuda.is_available()

# -----------------------------------------------------------------------------
# Output & Artifact Directories
# -----------------------------------------------------------------------------
OUTPUT_MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
RESULTS_DIR = REPORTS_DIR / "results"
FIGURES_DIR = REPORTS_DIR / "figures"

os.makedirs(OUTPUT_MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
