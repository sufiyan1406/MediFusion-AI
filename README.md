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
│   └── methodology.md            # Multi-label math & MedCXRNet methodology
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py            # Lazy-loading PyTorch NIHChestXRayDataset class
│   │   ├── label_encoder.py      # Multi-hot 14-class pathology parser & validator
│   │   ├── split.py              # Patient-level split & leakage validator
│   │   └── transforms.py         # 1-channel 224x224 grayscale & augmentation pipeline
│   ├── models/
│   │   ├── __init__.py
│   │   └── medcxrnet.py          # Custom 4-stage MedCXRNet CNN (~2.1M params)
│   ├── training/
│   │   ├── __init__.py
│   │   ├── loss.py               # BCEWithLogitsLoss factory
│   │   └── trainer.py            # PyTorch Trainer with AMP, AUC tracking, checkpointing
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py            # Multi-label Macro & Per-Class ROC-AUC calculator
│   └── utils/
│       ├── __init__.py
│       └── seed.py               # Reproducibility random seed initialization
├── scripts/
│   ├── validate_and_split_data.py # Validation & split statistics generator
│   └── train.py                  # CLI entry point for Experiment 0 & Baseline training
├── tests/
│   ├── test_foundation.py        # PyTest suite for ML foundation (Step 2)
│   └── test_model_pipeline.py    # PyTest suite for MedCXRNet, Loss, Trainer & Metrics (Step 4A)
├── reports/
│   ├── figures/
│   └── results/
│       └── dataset_split_statistics.json
├── requirements.txt              # Minimal core ML dependencies
├── .gitignore
└── README.md
```

---

## ⚡ Running Tests & Experiments

### 1. Run Unit Tests
Run the complete PyTest test suite (foundation + model & pipeline tests):
```bash
python -m pytest tests/ -v
```

### 2. Run Experiment 0 (Sanity Check / Overfit 100 Samples)
Run a small 5-epoch sanity check on 100 training samples to verify the end-to-end training pipeline:
```bash
python scripts/train.py --mode exp0 --epochs 5 --num-samples 100
```
Results and model checkpoints will be exported to `reports/results/experiment_0_sanity_check/results.json` and `models/exp0/`.

### 3. Run Experiment 1 (Full Baseline Scratch Training)
Run full baseline training across the NIH ChestX-ray14 dataset:
```bash
python scripts/train.py --mode train --epochs 15 --batch-size 16 --lr 1e-4
```
Checkpoints will be saved to `models/medcxrnet_baseline_best.pth` and `models/medcxrnet_baseline_latest.pth`.

---

## 🔬 Core Architecture Highlights

1. **Academic Scratch Training:** `MedCXRNet` (~2.1M parameters) trained strictly from scratch with Kaiming (He) Normal weight initialization. No ImageNet pre-trained weights.
2. **14 Multi-Label Output Logits:** Sigmoid activation applied independently to each output. Normal scans (`No Finding`) map to all-zero target vectors `[0, 0, ..., 0]`.
3. **Zero Patient Leakage:** Patient-level grouping guarantees 0 patient overlap across Train, Validation, and Test sets.
4. **VRAM Optimized:** $224 \times 224$ single-channel grayscale input + Automatic Mixed Precision (`torch.cuda.amp`) fits comfortably within RTX 2050 4GB VRAM (<1.2 GB required).
