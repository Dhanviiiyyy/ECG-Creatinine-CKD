"""
training/trainer.py
====================
Training loop for the ECG-to-serum-creatinine regression task.

Design
------
* AdamW optimiser with weight decay (Loshchilov & Hutter, ICLR 2019).
* CosineAnnealingLR scheduler (Loshchilov & Hutter, ICLR 2017).
* Early stopping on val_mae with configurable patience (default 15 epochs).
* Gradient clipping max_norm=1.0 to prevent exploding gradients.
* Mixed-precision training (torch.amp) when CUDA is available.
* Best checkpoint saved on lowest val_mae.
* All metrics logged per epoch to a CSV for post-hoc analysis.
* Reproducible: seeds set globally before training.

References
----------
Optimiser  : Loshchilov, I. and Hutter, F. (2019). Decoupled Weight Decay
             Regularisation. ICLR 2019.
Scheduler  : Loshchilov, I. and Hutter, F. (2017). SGDR: Stochastic Gradient
             Descent with Warm Restarts. ICLR 2017.
Grad clip  : Pascanu, R. et al. (2013). On the difficulty of training RNNs.
             ICML 2013.
"""

from __future__ import annotations

import csv
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from evaluation.metrics import evaluate_all

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training configuration (dataclass)
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """
    All training hyperparameters in one place for reproducibility.
    Values documented with literature justification.
    """
    # Optimisation
    lr:               float = 3e-4   # AdamW default (Loshchilov & Hutter 2019)
    weight_decay:     float = 1e-4
    max_epochs:       int   = 100
    grad_clip:        float = 1.0    # max gradient norm (Pascanu et al., 2013)

    # Scheduler (CosineAnnealingLR, T_max = max_epochs)
    eta_min:          float = 1e-6   # minimum LR at end of cosine cycle

    # Early stopping
    patience:         int   = 15     # epochs without val_mae improvement
    min_delta:        float = 1e-4   # minimum improvement to reset patience

    # Misc
    seed:             int   = 42
    use_amp:          bool  = True   # mixed precision (CUDA only)

    # Notebook & runtime convenience fields
    checkpoint_dir:   Optional[Path | str] = None
    device:           Optional[str | torch.device] = None
    epochs:           Optional[int] = None
    learning_rate:    Optional[float] = None

    def __init__(
        self,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        max_epochs: int = 100,
        grad_clip: float = 1.0,
        eta_min: float = 1e-6,
        patience: int = 15,
        min_delta: float = 1e-4,
        seed: int = 42,
        use_amp: bool = True,
        checkpoint_dir: Optional[Path | str] = None,
        device: Optional[str | torch.device] = None,
        epochs: Optional[int] = None,
        learning_rate: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        effective_epochs = epochs if epochs is not None else max_epochs
        effective_lr = learning_rate if learning_rate is not None else lr

        self.lr = float(effective_lr)
        self.learning_rate = self.lr
        self.weight_decay = float(weight_decay)
        self.max_epochs = int(effective_epochs)
        self.epochs = self.max_epochs
        self.grad_clip = float(grad_clip)
        self.eta_min = float(eta_min)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.seed = int(seed)
        self.use_amp = bool(use_amp)
        self.checkpoint_dir = checkpoint_dir
        self.device = device

        for k, v in kwargs.items():
            setattr(self, k, v)


# Alias for backward and notebook compatibility
TrainerConfig = TrainConfig


# ---------------------------------------------------------------------------
# Training state (mutable)
# ---------------------------------------------------------------------------

@dataclass
class _State:
    best_val_mae:   float = float("inf")
    best_epoch:     int   = 0
    patience_left:  int   = 0
    epoch_logs:     list  = field(default_factory=list)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Manages the full training loop for any ECG-creatinine regression model.

    Parameters
    ----------
    model          : nn.Module  (CNNBaseline1D, ResNet1D34, or ECGFMLinearProbe/LoRA)
    loss_fn        : nn.Module, optional (from training/losses.py)
    train_loader   : DataLoader
    val_loader     : DataLoader
    cfg            : TrainConfig, optional
    checkpoint_dir : Path | str, optional  Directory to save best model checkpoint.
    criterion      : nn.Module, optional (alias for loss_fn)
    optimizer      : torch.optim.Optimizer, optional
    scheduler      : Any, optional
    device         : torch.device | str, optional
    """

    def __init__(
        self,
        model:          nn.Module,
        loss_fn:        Optional[nn.Module] = None,
        train_loader:   Optional[DataLoader] = None,
        val_loader:     Optional[DataLoader] = None,
        cfg:            Optional[TrainConfig] = None,
        checkpoint_dir: Optional[Path | str] = None,
        *,
        criterion:      Optional[nn.Module] = None,
        optimizer:      Optional[torch.optim.Optimizer] = None,
        scheduler:      Optional[Any] = None,
        device:         Optional[torch.device | str] = None,
        **kwargs:       Any,
    ) -> None:
        self.model = model

        # Support both loss_fn and criterion
        resolved_loss = loss_fn if loss_fn is not None else criterion
        if resolved_loss is None:
            raise ValueError("Must provide either 'loss_fn' or 'criterion' to Trainer.")
        self.loss_fn   = resolved_loss
        self.criterion = resolved_loss

        if train_loader is None or val_loader is None:
            raise ValueError("Both 'train_loader' and 'val_loader' must be provided to Trainer.")
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.cfg          = cfg if cfg is not None else TrainConfig()

        # Checkpoint directory: prioritize explicit arg, then cfg, then default
        resolved_ckpt = checkpoint_dir if checkpoint_dir is not None else self.cfg.checkpoint_dir
        self.checkpoint_dir = Path(resolved_ckpt) if resolved_ckpt is not None else Path("checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Device
        resolved_device = device if device is not None else self.cfg.device
        if resolved_device is not None:
            self.device = torch.device(resolved_device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)
        self.loss_fn.to(self.device)
        logger.info("Training on device: %s", self.device)

        # Seed
        self._set_seed(self.cfg.seed)

        # Optimiser + scheduler (accept custom if provided)
        if optimizer is not None:
            self.optimiser = optimizer
        else:
            self.optimiser = AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.cfg.lr,
                weight_decay=self.cfg.weight_decay,
            )
        self.optimizer = self.optimiser

        if scheduler is not None:
            self.scheduler = scheduler
        else:
            self.scheduler = CosineAnnealingLR(
                self.optimiser,
                T_max=self.cfg.max_epochs,
                eta_min=self.cfg.eta_min,
            )

        # Mixed precision (CUDA only)
        self._use_amp = self.cfg.use_amp and (self.device.type == "cuda")
        if hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=self._use_amp)
        else:
            self.scaler = GradScaler(enabled=self._use_amp)

        # Logging CSV
        self._log_path = self.checkpoint_dir / "training_log.csv"
        self._state    = _State(patience_left=self.cfg.patience)

    @property
    def best_val_loss(self) -> float:
        """Best validation loss recorded during training."""
        val_losses = [float(row["val_loss"]) for row in self._state.epoch_logs if "val_loss" in row]
        if val_losses:
            return float(min(val_losses))
        return float(self._state.best_val_mae)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def fit(self) -> dict:
        """
        Run the full training loop.

        Returns
        -------
        dict with keys 'best_val_mae', 'best_val_loss', 'best_epoch',
        'epoch_logs', 'train_loss', 'val_loss'.
        """
        logger.info("Starting training: max_epochs=%d, patience=%d",
                    self.cfg.max_epochs, self.cfg.patience)

        with open(self._log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "train_loss", "val_loss",
                "val_mae", "val_rmse", "val_r2", "val_pearson_r", "lr",
            ])

        for epoch in range(1, self.cfg.max_epochs + 1):
            t0 = time.time()

            train_loss             = self._train_epoch()
            val_loss, val_metrics  = self._eval_epoch()
            val_mae                = val_metrics["mae"]

            if hasattr(self.scheduler, "get_last_lr"):
                lr_now = float(self.scheduler.get_last_lr()[0])
            else:
                lr_now = float(self.optimiser.param_groups[0]["lr"])
            self.scheduler.step()

            elapsed = time.time() - t0
            logger.info(
                "Epoch %3d/%d | loss %.4f | val_loss %.4f | "
                "MAE %.4f | RMSE %.4f | R2 %.4f | LR %.2e | %.1fs",
                epoch, self.cfg.max_epochs,
                train_loss, val_loss,
                val_mae, val_metrics["rmse"], val_metrics["r2"],
                lr_now, elapsed,
            )

            row = {
                "epoch": epoch, "train_loss": train_loss,
                "val_loss": val_loss, "lr": lr_now, **val_metrics,
            }
            self._state.epoch_logs.append(row)
            self._append_csv_row(row)

            # Checkpoint + early stopping
            improved = val_mae < self._state.best_val_mae - self.cfg.min_delta
            if improved:
                self._state.best_val_mae  = val_mae
                self._state.best_epoch    = epoch
                self._state.patience_left = self.cfg.patience
                self._save_checkpoint(epoch, val_mae)
            else:
                self._state.patience_left -= 1
                if self._state.patience_left <= 0:
                    logger.info(
                        "Early stopping at epoch %d (best val_mae=%.4f @ epoch %d).",
                        epoch, self._state.best_val_mae, self._state.best_epoch,
                    )
                    break

        train_losses = [float(row["train_loss"]) for row in self._state.epoch_logs if "train_loss" in row]
        val_losses   = [float(row["val_loss"]) for row in self._state.epoch_logs if "val_loss" in row]
        best_val_loss = float(min(val_losses)) if val_losses else float(self._state.best_val_mae)

        return {
            "best_val_mae":  self._state.best_val_mae,
            "best_val_loss": best_val_loss,
            "best_epoch":    self._state.best_epoch,
            "epoch_logs":    self._state.epoch_logs,
            "train_loss":    train_losses,
            "val_loss":      val_losses,
        }

    def _autocast_context(self):
        if hasattr(torch.amp, "autocast"):
            return torch.amp.autocast(device_type=self.device.type, enabled=self._use_amp)
        return autocast(enabled=self._use_amp)

    # ------------------------------------------------------------------
    # Train one epoch
    # ------------------------------------------------------------------

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches  = 0

        for batch in self.train_loader:
            waveform = batch["waveform"].to(self.device)   # [B, 12, 5000]
            label    = batch["label"].to(self.device)      # [B]

            self.optimiser.zero_grad(set_to_none=True)

            with self._autocast_context():
                pred = self.model(waveform)                # [B, 1]
                loss = self.loss_fn(pred, label)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimiser)
            nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.cfg.grad_clip
            )
            self.scaler.step(self.optimiser)
            self.scaler.update()

            total_loss += loss.item()
            n_batches  += 1

        return total_loss / max(n_batches, 1)

    # ------------------------------------------------------------------
    # Evaluate one epoch
    # ------------------------------------------------------------------

    def _eval_epoch(self) -> tuple[float, dict]:
        self.model.eval()
        total_loss  = 0.0
        all_preds:  list[np.ndarray] = []
        all_labels: list[np.ndarray] = []
        all_ages:   list[np.ndarray] = []
        all_sexes:  list[np.ndarray] = []
        n_batches   = 0

        with torch.no_grad():
            for batch in self.val_loader:
                waveform = batch["waveform"].to(self.device)
                label    = batch["label"].to(self.device)

                with self._autocast_context():
                    pred = self.model(waveform)
                    loss = self.loss_fn(pred, label)

                total_loss  += loss.item()
                n_batches   += 1

                all_preds.append(pred.squeeze(-1).cpu().numpy())
                all_labels.append(label.cpu().numpy())
                all_ages.append(batch["age_years"].numpy())
                all_sexes.append(batch["sex"].numpy())

        preds  = np.concatenate(all_preds)
        labels = np.concatenate(all_labels)
        ages   = np.concatenate(all_ages)
        sexes  = np.concatenate(all_sexes)

        # Age/sex available if all recorded (not -1 placeholder)
        has_demo = bool((ages >= 0).all() and (sexes >= 0).all())

        metrics = evaluate_all(
            labels, preds,
            age=ages  if has_demo else None,
            sex=sexes if has_demo else None,
        )

        return total_loss / max(n_batches, 1), metrics

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _save_checkpoint(self, epoch: int, val_mae: float) -> None:
        path = self.checkpoint_dir / "best_checkpoint.pth"
        torch.save({
            "epoch":       epoch,
            "val_mae":     val_mae,
            "model_state": self.model.state_dict(),
            "optim_state": self.optimiser.state_dict(),
            "cfg":         self.cfg,
        }, path)
        logger.info("Checkpoint saved -> %s  (val_mae=%.4f)", path, val_mae)

    def _append_csv_row(self, row: dict) -> None:
        with open(self._log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writerow(row)

    @staticmethod
    def _set_seed(seed: int) -> None:
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark     = False
