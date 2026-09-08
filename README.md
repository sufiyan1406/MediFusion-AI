# MediFusion — Deep Learning-Based Chest X-Ray Pathology Classification

MediFusion is an academic deep learning project focused on multi-label disease detection and abnormality classification from chest radiographs using the NIH ChestX-ray14 dataset.

---

## 📁 Project Architecture & Structure

```
MediFusion/
├── DataSet/                      # Existing raw NIH dataset (READ-ONLY)
│   ├── Data_Entry_2017.csv
│   ├── BBox_List_2017.csv
│   ├── train_val_list.txt
│   ├── test_list.txt
│   └── images_001 ... images_012
├── configs/
│   └── config.py                 # Centralized baseline hyperparameters & paths
├── docs/
│   ├── dataset.md                # Empirical dataset inspection report (Step 1)
│   └── methodology.md            # Multi-label math, MedCXRNet & evaluation workflow
├── src/
│   ├── data/
│   │   ├── dataset.py            # Lazy-loading PyTorch NIHChestXRayDataset
│   │   ├── label_encoder.py      # Multi-hot 14-class pathology parser
│   │   ├── split.py              # Patient-level split & leakage validator
│   │   └── transforms.py         # 1-channel 224x224 grayscale & augmentation
│   ├── models/
│   │   └── medcxrnet.py          # Custom 4-stage MedCXRNet CNN (~392K params)
│   ├── training/
│   │   ├── loss.py               # BCEWithLogitsLoss factory
│   │   └── trainer.py            # PyTorch Trainer with AMP & checkpointing
│   ├── evaluation/
│   │   ├── evaluator.py          # Official test-set evaluator engine
│   │   └── metrics.py            # Multi-label Macro & Per-Class ROC-AUC engine
│   └── utils/
│       └── seed.py               # Reproducibility random seed setup
├── scripts/
│   ├── cuda_diagnostic.py        # Hardware, CUDA & path resolution diagnostic
│   ├── validate_and_split_data.py # Data validation & split generator
│   └── train.py                  # CLI entry point (--mode exp0 / train / evaluate)
├── tests/
│   ├── test_foundation.py        # 7 foundation unit tests (Passed)
│   ├── test_kaggle_env.py        # 5 Kaggle environment tests (Passed)
│   ├── test_model_pipeline.py    # 9 model & pipeline unit tests (Passed)
│   └── test_evaluation.py        # 5 evaluator unit tests (Passed)
├── models/
│   └── medcxrnet_baseline_best.pth # Best trained checkpoint (Val Macro AUC: 0.7203)
├── reports/
│   └── results/
│       ├── dataset_split_statistics.json
│       ├── experiment_1_baseline_results.json
│       └── experiment_1_test_results.json (to be generated upon evaluation)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚡ Running Tests & Experiments

### 1. Run Unit Test Suite
Run the complete PyTest test suite (26 unit tests):
```bash
python -m pytest tests/ -v
```

### 2. Environment & CUDA Hardware Diagnostic
Run hardware, path resolution, and DataLoader diagnostic:
```bash
python scripts/train.py --mode env-check
```

### 3. Dry-Run Fast Pipeline Check
Run 2-step fast GPU/CPU execution verification on real dataset samples:
```bash
python scripts/train.py --mode dry-run
```

### 4. Experiment 0 (Sanity Check / Overfit 100 Samples)
Run 5-epoch sanity check on 100 training samples:
```bash
python scripts/train.py --mode exp0 --epochs 5 --num-samples 100
```

### 5. Experiment 1 (Full Baseline Scratch Training)
Run full baseline training across the NIH ChestX-ray14 dataset:
```bash
python scripts/train.py --mode train --epochs 15 --batch-size 16 --lr 1e-4
```

### 6. Official Test-Set Evaluation
Evaluate a trained checkpoint against the official NIH ChestX-ray14 test set (25,596 images / 2,797 patients):
```bash
python scripts/train.py --mode evaluate --checkpoint models/medcxrnet_baseline_best.pth
```
Output results will be exported to `reports/results/experiment_1_test_results.json`.

---

## 🔬 Workflow & Test Set Isolation Rules

```
TRAIN (73,916 images) -> VAL (12,608 images) -> LOCK CHECKPOINT -> OFFICIAL TEST (25,596 images)
```

- **Validation Set:** Used exclusively for hyperparameter tuning and model selection (`best_val_macro_auc = 0.7203`).
- **Official Test Set:** Reserved strictly for final evaluation of locked checkpoints. Test metrics are **NEVER** used to tune future experiments.
