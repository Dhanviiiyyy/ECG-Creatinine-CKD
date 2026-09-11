"""
run_training.py
================
CLI entry point for training ECG-to-serum-creatinine models.

Usage
-----
# Train the 1D-CNN baseline (default)
python run_training.py --model cnn

# Train 1D ResNet-34
python run_training.py --model resnet34

# Train ECG-FM linear probe (requires backbone weights)
python run_training.py --model ecgfm_probe

# Train ECG-FM LoRA adapter (requires backbone weights)
python run_training.py --model ecgfm_lora

# Custom config
python run_training.py --model resnet34 --lr 1e-4 --epochs 50 --batch-size 16

Colab usage
-----------
    !python run_training.py --model cnn --hdf5 data/processed/ecg_creatinine.h5
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sized

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def build_model(name: str):
    """Instantiate the requested model architecture."""
    if name == "cnn":
        from models.cnn_baseline import CNNBaseline1D
        model = CNNBaseline1D()
        logger.info("Model: CNNBaseline1D  params=%d", model.count_parameters())
        return model

    if name == "resnet34":
        from models.resnet1d import ResNet1D34
        model = ResNet1D34()
        logger.info("Model: ResNet1D34  params=%d", model.count_parameters())
        return model

    if name == "ecgfm_probe":
        from models.ecg_fm_probe import ECGFMLinearProbe, MockECGFMBackbone
        backbone = MockECGFMBackbone()
        logger.warning(
            "Using MockECGFMBackbone. Replace with load_stmem_backbone() "
            "when real weights are available."
        )
        model = ECGFMLinearProbe(backbone)
        logger.info("Model: ECGFMLinearProbe  trainable_params=%d",
                    model.count_parameters())
        return model

    if name == "ecgfm_lora":
        from models.ecg_fm_probe import ECGFMLoRA, MockECGFMBackbone
        backbone = MockECGFMBackbone()
        logger.warning(
            "Using MockECGFMBackbone. Replace with load_stmem_backbone() "
            "when real weights are available."
        )
        model = ECGFMLoRA(backbone, lora_rank=8, lora_alpha=16.0)
        logger.info("Model: ECGFMLoRA  trainable_params=%d",
                    model.count_parameters())
        return model

    raise ValueError(
        f"Unknown model '{name}'. Choose: cnn, resnet34, ecgfm_probe, ecgfm_lora."
    )


def main(args: argparse.Namespace) -> None:
    from config.paths import Paths, load_config
    from datasets.ecg_dataset import build_dataloaders
    from training.losses import build_loss
    from training.trainer import Trainer, TrainConfig

    # -- Config -----------------------------------------------------------
    cfg_yaml = load_config(args.config)
    paths    = Paths(cfg_yaml)

    hdf5_path = Path(args.hdf5) if args.hdf5 else paths.hdf5_file
    if not hdf5_path.exists():
        logger.error(
            "HDF5 not found: %s\nRun 'python run_pipeline.py' first.", hdf5_path
        )
        sys.exit(1)

    ckpt_dir = Path(args.checkpoint_dir) / args.model
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # -- Data -------------------------------------------------------------
    logger.info("Loading dataset from: %s", hdf5_path)
    loaders = build_dataloaders(
        hdf5_path=hdf5_path,
        batch_size=args.batch_size,
        num_workers=args.workers,
        preload=args.preload,
        seed=int(cfg_yaml["reproducibility"]["global_seed"]),
    )
    train_ds = loaders["train"].dataset
    val_ds   = loaders["val"].dataset
    test_ds  = loaders["test"].dataset
    logger.info("Dataset splits: train=%d  val=%d  test=%d",
                len(train_ds) if isinstance(train_ds, Sized) else 0,
                len(val_ds)   if isinstance(val_ds, Sized) else 0,
                len(test_ds)  if isinstance(test_ds, Sized) else 0)

    # -- Model + loss -----------------------------------------------------
    model   = build_model(args.model)
    loss_fn = build_loss(args.loss, delta=args.huber_delta)

    train_cfg = TrainConfig(
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.epochs,
        patience=args.patience,
        seed=int(cfg_yaml["reproducibility"]["global_seed"]),
        use_amp=not args.no_amp,
    )

    # -- Train ------------------------------------------------------------
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        cfg=train_cfg,
        checkpoint_dir=ckpt_dir,
    )
    results = trainer.fit()

    logger.info("Training complete.")
    logger.info("  Best val MAE : %.4f mg/dL", results["best_val_mae"])
    logger.info("  Best epoch   : %d",          results["best_epoch"])
    logger.info("  Checkpoint   : %s",           ckpt_dir / "best_checkpoint.pth")
    logger.info("  Training log : %s",           ckpt_dir / "training_log.csv")


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ECG-creatinine model.")
    p.add_argument("--model",     default="cnn",
                   choices=["cnn","resnet34","ecgfm_probe","ecgfm_lora"])
    p.add_argument("--hdf5",      default=None,
                   help="Path to ecg_creatinine.h5 (overrides config).")
    p.add_argument("--config",    default=None,
                   help="Path to pipeline_config.yaml.")
    p.add_argument("--checkpoint-dir", default="checkpoints",
                   help="Directory for model checkpoints.")
    p.add_argument("--epochs",    type=int,   default=100)
    p.add_argument("--lr",        type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size",   type=int,   default=32)
    p.add_argument("--patience",     type=int,   default=15)
    p.add_argument("--loss",         default="huber",
                   choices=["huber","mse","mae"])
    p.add_argument("--huber-delta",  type=float, default=1.0)
    p.add_argument("--workers",      type=int,   default=0)
    p.add_argument("--preload",      action="store_true")
    p.add_argument("--no-amp",       action="store_true",
                   help="Disable mixed-precision training.")
    return p.parse_args()


if __name__ == "__main__":
    main(_parse())
