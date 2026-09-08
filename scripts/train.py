"""
CLI Training Entry Point for MediFusion.

Supports:
    - Mode 'env-check': Hardware, CUDA, and path resolution diagnostic
    - Mode 'dry-run': Fast 2-iteration pipeline validation on real dataset samples
    - Mode 'exp0': Pipeline sanity check / overfit on small subset (100 images for 5 epochs)
    - Mode 'train': Full baseline scratch training on NIH ChestX-ray14 dataset
"""

import sys
import json
import argparse
from pathlib import Path
import pandas as pd

import torch
from torch.utils.data import DataLoader

# Add project root directory to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import (
    DATASET_DIR,
    METADATA_CSV_PATH,
    TRAIN_VAL_LIST_PATH,
    TEST_LIST_PATH,
    EXPECTED_PATHOLOGIES,
    IMAGE_SIZE,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_EPOCHS,
    VAL_PATIENT_RATIO,
    SCHEDULER_FACTOR,
    SCHEDULER_PATIENCE,
    EARLY_STOPPING_PATIENCE,
    RANDOM_SEED,
    NUM_WORKERS,
    PIN_MEMORY,
    PERSISTENT_WORKERS,
    DEVICE,
    USE_AMP,
    OUTPUT_MODELS_DIR,
    RESULTS_DIR
)
from src.utils.seed import set_seed
from src.data.label_encoder import LabelEncoder
from src.data.transforms import get_baseline_transforms, get_train_transforms
from src.data.dataset import NIHChestXRayDataset
from src.data.split import build_image_disk_mapping, create_patient_level_splits
from src.models.medcxrnet import MedCXRNet, count_parameters
from src.training.loss import get_loss_function
from src.training.trainer import Trainer
from scripts.cuda_diagnostic import run_cuda_diagnostic


def parse_args():
    parser = argparse.ArgumentParser(description="MediFusion Training Script")
    parser.add_argument(
        "--mode",
        type=str,
        default="exp0",
        choices=["env-check", "dry-run", "exp0", "train"],
        help="Mode: 'env-check' diagnostic, 'dry-run' fast 2-step check, 'exp0' sanity check, 'train' full training."
    )
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size.")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate.")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of samples for Experiment 0 or Dry Run.")
    parser.add_argument("--disable-early-stopping", action="store_true", help="Disable early stopping.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed.")
    return parser.parse_args()


def run_dry_run(args):
    """
    Run Dry-Run Mode: Fast 2-iteration forward/backward execution on real dataset samples.
    """
    print("=" * 70)
    print(" MediFusion — Dry-Run Validation (Fast Kaggle Pipeline Verification)")
    print("=" * 70)

    set_seed(args.seed)
    label_encoder = LabelEncoder(EXPECTED_PATHOLOGIES)

    # 1. Verify Dataset Paths
    if not METADATA_CSV_PATH.exists():
        raise FileNotFoundError(f"Metadata file not found at '{METADATA_CSV_PATH}'!")

    df_raw = pd.read_csv(METADATA_CSV_PATH)
    image_map = build_image_disk_mapping(str(DATASET_DIR))

    df_train, df_val, _ = create_patient_level_splits(
        df=df_raw,
        train_val_list_path=str(TRAIN_VAL_LIST_PATH),
        test_list_path=str(TEST_LIST_PATH),
        val_patient_ratio=VAL_PATIENT_RATIO,
        random_seed=args.seed
    )

    # Take 16 real samples for 1 batch iteration
    df_dry = df_train.iloc[:16].reset_index(drop=True)
    dataset = NIHChestXRayDataset(
        df=df_dry,
        image_dir_map=image_map,
        label_encoder=label_encoder,
        transform=get_train_transforms(IMAGE_SIZE)
    )

    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    # 2. Instantiate Model, Loss, Optimizer
    model = MedCXRNet().to(DEVICE)
    criterion = get_loss_function(weighted=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    device_type = "cuda" if "cuda" in str(DEVICE) else "cpu"
    use_amp = USE_AMP if (torch.cuda.is_available() and device_type == "cuda") else False

    try:
        scaler = torch.amp.GradScaler(device_type, enabled=use_amp)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # 3. Perform 2 Forward + Backward Iterations
    print(f"[Dry-Run] Executing 2 forward/backward iterations on device '{DEVICE}' (AMP={use_amp})...")
    model.train()

    for i in range(2):
        for batch in loader:
            images = batch["image"].to(DEVICE, non_blocking=PIN_MEMORY)
            targets = batch["target"].to(DEVICE, non_blocking=PIN_MEMORY)

            optimizer.zero_grad()

            try:
                ctx = torch.amp.autocast(device_type, enabled=use_amp)
            except Exception:
                ctx = torch.cuda.amp.autocast(enabled=use_amp)

            with ctx:
                logits = model(images)
                loss = criterion(logits, targets)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            print(f"  Iteration {i + 1}/2 | Loss: {loss.item():.4f} | Logits Shape: {list(logits.shape)}")

    print(f"\n[MediFusion Dry-Run SUCCESS] Pipeline validated! Device: {DEVICE}, Dataset Path: '{DATASET_DIR}'.")
    print("=" * 70)


def run_experiment_0(args):
    """
    Run Experiment 0: Pipeline Sanity Check on ~100 images for 5 epochs.
    """
    print("=" * 70)
    print(" MediFusion — Experiment 0: Pipeline Sanity Check")
    print("=" * 70)

    set_seed(args.seed)
    epochs = args.epochs if args.epochs is not None else 5

    exp0_dir = RESULTS_DIR / "experiment_0_sanity_check"
    exp0_models_dir = OUTPUT_MODELS_DIR / "exp0"
    exp0_dir.mkdir(parents=True, exist_ok=True)
    exp0_models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    label_encoder = LabelEncoder(EXPECTED_PATHOLOGIES)
    df_raw = pd.read_csv(METADATA_CSV_PATH)
    image_map = build_image_disk_mapping(str(DATASET_DIR))

    df_train, df_val, _ = create_patient_level_splits(
        df=df_raw,
        train_val_list_path=str(TRAIN_VAL_LIST_PATH),
        test_list_path=str(TEST_LIST_PATH),
        val_patient_ratio=VAL_PATIENT_RATIO,
        random_seed=args.seed
    )

    df_train_sub = df_train.iloc[:args.num_samples].reset_index(drop=True)
    df_val_sub = df_val.iloc[:min(30, len(df_val))].reset_index(drop=True)

    train_dataset = NIHChestXRayDataset(
        df=df_train_sub,
        image_dir_map=image_map,
        label_encoder=label_encoder,
        transform=get_train_transforms(IMAGE_SIZE)
    )

    val_dataset = NIHChestXRayDataset(
        df=df_val_sub,
        image_dir_map=image_map,
        label_encoder=label_encoder,
        transform=get_baseline_transforms(IMAGE_SIZE)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    # 2. Model & Loss & Optimizer
    model = MedCXRNet()
    param_counts = count_parameters(model)
    print(f"[Model] MedCXRNet initialized with {param_counts['trainable_parameters']:,} trainable parameters.")

    criterion = get_loss_function(weighted=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)

    # 3. Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=None,
        device=DEVICE,
        use_amp=USE_AMP,
        checkpoint_dir=exp0_models_dir,
        experiment_name="exp0"
    )

    history = trainer.fit(
        epochs=epochs,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        disable_early_stopping=True
    )

    summary_path = exp0_dir / "results.json"
    with open(summary_path, "w") as f:
        json.dump({
            "experiment": "Experiment 0: Sanity Check",
            "num_train_samples": len(df_train_sub),
            "epochs": epochs,
            "history": history
        }, f, indent=4)

    print(f"\n[Experiment 0 SUCCESS] Results exported to '{summary_path}'.")
    print("=" * 70)


def run_full_training(args):
    """
    Run Experiment 1: Baseline MedCXRNet Training on Full Dataset.
    """
    print("=" * 70)
    print(" MediFusion — Experiment 1: Full Baseline Training")
    print("=" * 70)

    set_seed(args.seed)
    epochs = args.epochs if args.epochs is not None else NUM_EPOCHS

    # 1. Load Data
    label_encoder = LabelEncoder(EXPECTED_PATHOLOGIES)
    df_raw = pd.read_csv(METADATA_CSV_PATH)
    image_map = build_image_disk_mapping(str(DATASET_DIR))

    df_train, df_val, _ = create_patient_level_splits(
        df=df_raw,
        train_val_list_path=str(TRAIN_VAL_LIST_PATH),
        test_list_path=str(TEST_LIST_PATH),
        val_patient_ratio=VAL_PATIENT_RATIO,
        random_seed=args.seed
    )

    train_dataset = NIHChestXRayDataset(
        df=df_train,
        image_dir_map=image_map,
        label_encoder=label_encoder,
        transform=get_train_transforms(IMAGE_SIZE)
    )

    val_dataset = NIHChestXRayDataset(
        df=df_val,
        image_dir_map=image_map,
        label_encoder=label_encoder,
        transform=get_baseline_transforms(IMAGE_SIZE)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS
    )

    # 2. Model & Loss & Optimizer
    model = MedCXRNet()
    param_counts = count_parameters(model)
    print(f"[Model] MedCXRNet initialized with {param_counts['trainable_parameters']:,} trainable parameters.")

    criterion = get_loss_function(weighted=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE
    )

    # 3. Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=DEVICE,
        use_amp=USE_AMP,
        checkpoint_dir=OUTPUT_MODELS_DIR,
        experiment_name="medcxrnet_baseline"
    )

    history = trainer.fit(
        epochs=epochs,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        disable_early_stopping=args.disable_early_stopping
    )

    summary_path = RESULTS_DIR / "experiment_1_baseline_results.json"
    with open(summary_path, "w") as f:
        json.dump({
            "experiment": "Experiment 1: Baseline MedCXRNet Scratch Training",
            "epochs": epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "param_counts": param_counts,
            "history": history
        }, f, indent=4)

    print(f"\n[Experiment 1 SUCCESS] Full Baseline results exported to '{summary_path}'.")
    print("=" * 70)


def main():
    args = parse_args()
    if args.mode == "env-check":
        run_cuda_diagnostic()
    elif args.mode == "dry-run":
        run_dry_run(args)
    elif args.mode == "exp0":
        run_experiment_0(args)
    elif args.mode == "train":
        run_full_training(args)


if __name__ == "__main__":
    main()
