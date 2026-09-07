"""
PyTorch Training Engine for MediFusion.

Implements clean, modular training and validation loops with support for:
    - PyTorch Automatic Mixed Precision (AMP) via torch.cuda.amp.autocast and GradScaler
    - Validation Macro ROC-AUC tracking
    - Learning rate scheduling via ReduceLROnPlateau
    - Checkpoint saving (latest_checkpoint.pth & best_checkpoint.pth)
    - Early stopping
"""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from configs.config import OUTPUT_MODELS_DIR, DEVICE, USE_AMP, DEFAULT_THRESHOLD
from src.evaluation.metrics import calculate_multilabel_metrics


class Trainer:
    """
    Trainer engine managing training, validation, AMP scaling, metrics calculation, and checkpointing.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        device: str = DEVICE,
        use_amp: bool = USE_AMP,
        checkpoint_dir: Path = OUTPUT_MODELS_DIR,
        experiment_name: str = "baseline"
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        
        # Disable AMP if CUDA is unavailable
        self.use_amp = use_amp if (torch.cuda.is_available() and device != "cpu") else False
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.checkpoint_dir = Path(checkpoint_dir)
        self.experiment_name = experiment_name
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.best_val_macro_auc = -1.0
        self.best_epoch = 0

    def train_epoch(self) -> float:
        """
        Execute one training epoch.

        Returns:
            float: Average training loss across batches.
        """
        self.model.train()
        running_loss = 0.0
        num_batches = len(self.train_loader)

        for batch in self.train_loader:
            images = batch["image"].to(self.device)
            targets = batch["target"].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass with AMP autocast
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, targets)

            # Scaled backward pass and optimizer step
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / num_batches if num_batches > 0 else 0.0
        return avg_train_loss

    def validate_epoch(self, threshold: float = DEFAULT_THRESHOLD) -> Tuple[float, Dict[str, Any]]:
        """
        Execute one validation epoch and calculate multi-label metrics.

        Args:
            threshold (float): Decision threshold for binary metrics.

        Returns:
            Tuple[float, Dict[str, Any]]: (Average validation loss, Multi-label metrics dictionary).
        """
        if self.val_loader is None:
            return 0.0, {"macro_roc_auc": 0.0}

        self.model.eval()
        running_loss = 0.0
        num_batches = len(self.val_loader)

        all_logits = []
        all_targets = []

        with torch.no_grad():
            for batch in self.val_loader:
                images = batch["image"].to(self.device)
                targets = batch["target"].to(self.device)

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    logits = self.model(images)
                    loss = self.criterion(logits, targets)

                running_loss += loss.item()
                all_logits.append(logits.cpu())
                all_targets.append(targets.cpu())

        avg_val_loss = running_loss / num_batches if num_batches > 0 else 0.0

        if all_logits:
            concat_logits = torch.cat(all_logits, dim=0)
            concat_targets = torch.cat(all_targets, dim=0)
            val_metrics = calculate_multilabel_metrics(
                y_pred=concat_logits,
                y_true=concat_targets,
                threshold=threshold,
                is_logits=True
            )
        else:
            val_metrics = {"macro_roc_auc": 0.0}

        return avg_val_loss, val_metrics

    def save_checkpoint(
        self,
        epoch: int,
        val_macro_auc: float,
        is_best: bool = False,
        filename_prefix: str = "checkpoint"
    ) -> Path:
        """
        Save model, optimizer, scheduler, and scaler checkpoint.

        Args:
            epoch (int): Current epoch number.
            val_macro_auc (float): Validation Macro ROC-AUC achieved.
            is_best (bool): Whether this is the best checkpoint so far.
            filename_prefix (str): Prefix string for checkpoint file.

        Returns:
            Path: Path to saved checkpoint file.
        """
        checkpoint_dict = {
            "epoch": epoch,
            "experiment_name": self.experiment_name,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
            "scaler_state_dict": self.scaler.state_dict() if self.use_amp else None,
            "best_val_macro_auc": self.best_val_macro_auc,
            "val_macro_auc": val_macro_auc,
        }

        # Save latest checkpoint
        latest_path = self.checkpoint_dir / f"{self.experiment_name}_latest.pth"
        torch.save(checkpoint_dict, latest_path)

        if is_best:
            best_path = self.checkpoint_dir / f"{self.experiment_name}_best.pth"
            torch.save(checkpoint_dict, best_path)
            print(f"  [Checkpoint] Saved NEW BEST model checkpoint to '{best_path}' (Val Macro AUC: {val_macro_auc:.4f})")

        return latest_path

    def fit(
        self,
        epochs: int,
        early_stopping_patience: int = 4,
        disable_early_stopping: bool = False
    ) -> Dict[str, Any]:
        """
        Main training loop over multiple epochs.

        Args:
            epochs (int): Number of training epochs.
            early_stopping_patience (int): Number of epochs to wait for improvement before early stopping.
            disable_early_stopping (bool): If True, disables early stopping (e.g. for Experiment 0).

        Returns:
            Dict[str, Any]: Training history dictionary.
        """
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_macro_auc": [],
            "val_metrics": []
        }

        patience_counter = 0

        print(f"\n[MediFusion Trainer] Starting training for {epochs} epochs on device '{self.device}' (AMP={self.use_amp})...")

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch()
            val_loss, val_metrics = self.validate_epoch()
            val_macro_auc = val_metrics.get("macro_roc_auc", 0.0)

            # Handle NaN Macro AUC safely
            auc_display = val_macro_auc if not np.isnan(val_macro_auc) else 0.0

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_macro_auc"].append(auc_display)
            history["val_metrics"].append(val_metrics)

            print(
                f"Epoch {epoch:02d}/{epochs:02d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Macro AUC: {auc_display:.4f}"
            )

            # Scheduler step monitoring Val Macro AUC if ReduceLROnPlateau
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(auc_display)
                else:
                    self.scheduler.step()

            # Checkpoint tracking
            is_best = auc_display > self.best_val_macro_auc
            if is_best:
                self.best_val_macro_auc = auc_display
                self.best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1

            self.save_checkpoint(epoch=epoch, val_macro_auc=auc_display, is_best=is_best)

            # Early Stopping Check
            if not disable_early_stopping and patience_counter >= early_stopping_patience:
                print(
                    f"\n[Early Stopping Triggered] Stopping early at epoch {epoch}. "
                    f"Best Val Macro AUC was {self.best_val_macro_auc:.4f} at epoch {self.best_epoch}."
                )
                break

        print(f"[MediFusion Trainer] Training Complete. Best Val Macro AUC: {self.best_val_macro_auc:.4f} at epoch {self.best_epoch}.")
        return history
