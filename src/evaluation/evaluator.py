"""
Official Test-Set Evaluator Module for MediFusion.

Loads trained model checkpoints (e.g. models/medcxrnet_baseline_best.pth), runs deterministic inference
on the official NIH ChestX-ray14 test split (25,596 images / 2,797 patients), computes macro and per-class
multi-label metrics, and exports reports/results/experiment_1_test_results.json.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from configs.config import (
    DATASET_DIR,
    METADATA_CSV_PATH,
    TEST_LIST_PATH,
    EXPECTED_PATHOLOGIES,
    IMAGE_SIZE,
    BATCH_SIZE,
    DEFAULT_THRESHOLD,
    NUM_WORKERS,
    PIN_MEMORY,
    DEVICE,
    USE_AMP,
    RESULTS_DIR
)
from src.utils.seed import set_seed
from src.data.label_encoder import LabelEncoder
from src.data.transforms import get_baseline_transforms
from src.data.dataset import NIHChestXRayDataset
from src.data.split import build_image_disk_mapping
from src.models.medcxrnet import MedCXRNet, count_parameters
from src.evaluation.metrics import calculate_multilabel_metrics


class Evaluator:
    """
    Evaluator engine for running model checkpoints against the official test set.
    """

    def __init__(
        self,
        checkpoint_path: Path,
        device: str = DEVICE,
        use_amp: bool = USE_AMP,
        threshold: float = DEFAULT_THRESHOLD,
        results_dir: Path = RESULTS_DIR
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = str(device)
        self.device_type = "cuda" if "cuda" in self.device else "cpu"
        self.use_amp = use_amp if (torch.cuda.is_available() and self.device_type == "cuda") else False
        self.threshold = threshold
        self.results_dir = Path(results_dir)

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"CHECKPOINT ERROR: Checkpoint file not found at '{self.checkpoint_path}'!")

        os.makedirs(self.results_dir, exist_ok=True)

    def _get_autocast_context(self):
        """Helper to get PyTorch 2.x autocast context."""
        try:
            return torch.amp.autocast(self.device_type, enabled=self.use_amp)
        except Exception:
            return torch.cuda.amp.autocast(enabled=self.use_amp)

    def load_model_from_checkpoint(self) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Load weights from checkpoint into MedCXRNet architecture.
        Checkpoint is treated as strictly READ-ONLY.

        Returns:
            Tuple[nn.Module, Dict[str, Any]]: (Loaded MedCXRNet model in eval mode, Checkpoint metadata).
        """
        print(f"[Evaluator] Loading checkpoint from '{self.checkpoint_path}'...")
        checkpoint_dict = torch.load(self.checkpoint_path, map_location=self.device)

        if "model_state_dict" not in checkpoint_dict:
            raise KeyError(
                f"CHECKPOINT COMPATIBILITY ERROR: Missing 'model_state_dict' in checkpoint '{self.checkpoint_path}'!"
            )

        model = MedCXRNet()
        # Strict state_dict loading
        model.load_state_dict(checkpoint_dict["model_state_dict"], strict=True)
        model = model.to(self.device)
        model.eval()

        metadata = {
            "epoch": checkpoint_dict.get("epoch", "unknown"),
            "val_macro_auc": checkpoint_dict.get("val_macro_auc", checkpoint_dict.get("best_val_macro_auc", float("nan"))),
            "experiment_name": checkpoint_dict.get("experiment_name", "unknown")
        }

        print(f"  • Successfully loaded MedCXRNet from Checkpoint (Epoch: {metadata['epoch']}, Val Macro AUC: {metadata['val_macro_auc']})")
        return model, metadata

    def evaluate_test_set(
        self,
        test_df: Optional[pd.DataFrame] = None,
        image_map: Optional[Dict[str, str]] = None,
        batch_size: int = BATCH_SIZE,
        output_filename: str = "experiment_1_test_results.json"
    ) -> Dict[str, Any]:
        """
        Execute evaluation against the official test set.

        Args:
            test_df (Optional[pd.DataFrame]): Optional DataFrame override for unit testing.
            image_map (Optional[Dict[str, str]]): Optional image disk path mapping for unit testing.
            batch_size (int): Batch size for inference.
            output_filename (str): JSON output filename.

        Returns:
            Dict[str, Any]: Detailed evaluation metrics dictionary.
        """
        label_encoder = LabelEncoder(EXPECTED_PATHOLOGIES)

        # 1. Load test dataset if not provided
        if test_df is None or image_map is None:
            if not METADATA_CSV_PATH.exists():
                raise FileNotFoundError(f"Metadata CSV not found at '{METADATA_CSV_PATH}'!")
            if not TEST_LIST_PATH.exists():
                raise FileNotFoundError(f"Test list file not found at '{TEST_LIST_PATH}'!")

            df_raw = pd.read_csv(METADATA_CSV_PATH)
            image_map = build_image_disk_mapping(str(DATASET_DIR))

            with open(TEST_LIST_PATH, "r") as f:
                test_filenames = set(line.strip() for line in f if line.strip())

            test_df = df_raw[df_raw["Image Index"].isin(test_filenames)].copy().reset_index(drop=True)

            num_images = len(test_df)
            num_patients = test_df["Patient ID"].nunique()

            print(f"[Evaluator] Loaded Official NIH Test Split: {num_images} images across {num_patients} unique patients.")
            
            # Verify official counts (25,596 images / 2,797 patients)
            if num_images != 25596 or num_patients != 2797:
                print(f"  [Notice] Test count notice: Evaluated {num_images} images ({num_patients} patients). Official NIH baseline is 25,596 images / 2,797 patients.")

        num_images = len(test_df)
        num_patients = test_df["Patient ID"].nunique()

        # 2. Construct Deterministic Test Dataset & DataLoader
        test_dataset = NIHChestXRayDataset(
            df=test_df,
            image_dir_map=image_map,
            label_encoder=label_encoder,
            transform=get_baseline_transforms(IMAGE_SIZE)  # Deterministic transforms
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY
        )

        # 3. Load Model from Checkpoint
        model, ckpt_meta = self.load_model_from_checkpoint()
        param_counts = count_parameters(model)

        # 4. Perform Inference under torch.no_grad()
        print(f"[Evaluator] Running deterministic test inference on device '{self.device}' (AMP={self.use_amp})...")
        all_logits = []
        all_targets = []

        with torch.no_grad():
            for batch in test_loader:
                images = batch["image"].to(self.device, non_blocking=PIN_MEMORY)
                targets = batch["target"].to(self.device, non_blocking=PIN_MEMORY)

                with self._get_autocast_context():
                    logits = model(images)

                all_logits.append(logits.cpu())
                all_targets.append(targets.cpu())

        concat_logits = torch.cat(all_logits, dim=0)
        concat_targets = torch.cat(all_targets, dim=0)

        # 5. Compute Metrics
        metrics = calculate_multilabel_metrics(
            y_pred=concat_logits,
            y_true=concat_targets,
            class_names=EXPECTED_PATHOLOGIES,
            threshold=self.threshold,
            is_logits=True
        )

        # 6. Format Structured Results JSON
        results = {
            "experiment": "experiment_1_test_evaluation",
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_epoch": ckpt_meta["epoch"],
            "checkpoint_val_macro_auc": ckpt_meta["val_macro_auc"],
            "dataset": {
                "split": "official_test",
                "num_test_images": int(num_images),
                "num_test_patients": int(num_patients)
            },
            "model": {
                "name": "MedCXRNet",
                "parameters": param_counts["trainable_parameters"]
            },
            "evaluation": {
                "device": self.device,
                "amp": self.use_amp,
                "threshold": self.threshold
            },
            "macro_metrics": {
                "macro_roc_auc": metrics["macro_roc_auc"],
                "macro_pr_auc": metrics["macro_pr_auc"],
                "macro_precision": metrics["macro_precision"],
                "macro_recall_sensitivity": metrics["macro_recall"],
                "macro_specificity": metrics["macro_specificity"],
                "macro_f1": metrics["macro_f1"],
                "valid_auc_classes_count": metrics["valid_auc_classes_count"]
            },
            "per_class_metrics": {
                "roc_auc": metrics["per_class_roc_auc"],
                "pr_auc": metrics["per_class_pr_auc"],
                "precision": metrics["per_class_precision"],
                "recall_sensitivity": metrics["per_class_recall"],
                "specificity": metrics["per_class_specificity"],
                "f1": metrics["per_class_f1"]
            }
        }

        # 7. Export JSON Results
        output_json_path = self.results_dir / output_filename
        with open(output_json_path, "w") as f:
            json.dump(results, f, indent=4)

        print(f"\n[MediFusion Evaluator SUCCESS] Test evaluation results saved to '{output_json_path}'.")
        print(f"  • Official Test Macro ROC-AUC: {metrics['macro_roc_auc']:.4f}")

        return results
