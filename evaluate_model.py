"""
evaluate_model.py
=================
Comprehensive evaluation CLI for ECG-to-Creatinine models.

Directly implements the comparative evaluation standards from:
- Holmstrom et al. (2023) Communications Medicine (eGFR < 60 CKD screening)
- Tsai et al. (2025) BMC Medical Informatics (Cross-metric evaluation)
- Lopez Alcaraz & Strodthoff (2024/2025) (Demographic baseline comparison)

Usage:
------
python evaluate_model.py --model cnn --checkpoint checkpoints/cnn/best_checkpoint.pth
python evaluate_model.py --model resnet34 --checkpoint checkpoints/resnet34/best_checkpoint.pth
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config.paths import Paths, load_config
from datasets.ecg_dataset import ECGCreatinineDataset
from evaluation.baseline_demographics import evaluate_demographic_baseline
from evaluation.metrics import evaluate_all, ckd_screening_metrics
from run_training import build_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def evaluate_checkpoint(
    model_name: str,
    checkpoint_path: Path,
    hdf5_path: Path,
    device: torch.device,
) -> dict:
    # 1. Build and load model
    model = build_model(model_name)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state", ckpt.get("model_state_dict", ckpt))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 2. Build test loader
    test_ds = ECGCreatinineDataset(hdf5_path, split="test", preload=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    preds_list, labels_list, ages_list, sexes_list = [], [], [], []

    with torch.no_grad():
        for batch in test_loader:
            x = batch["waveform"].to(device)
            out = model(x)
            preds_list.extend(out.squeeze(-1).cpu().numpy())
            labels_list.extend(batch["label"].numpy())
            ages_list.extend(batch["age_years"].numpy())
            sexes_list.extend(batch["sex"].numpy())

    y_pred = np.array(preds_list, dtype=np.float64)
    y_true = np.array(labels_list, dtype=np.float64)
    ages   = np.array(ages_list, dtype=np.float64)
    sexes  = np.array(sexes_list, dtype=np.int32)

    # 3. Compute all metrics
    metrics = evaluate_all(y_true, y_pred, age=ages, sex=sexes)
    screening = ckd_screening_metrics(y_true, y_pred, age=ages, sex=sexes, threshold=60.0)
    metrics["screening"] = screening
    return metrics


def main(args: argparse.Namespace) -> None:
    cfg_yaml = load_config(args.config)
    paths    = Paths(cfg_yaml)
    hdf5_path = Path(args.hdf5) if args.hdf5 else paths.hdf5_file
    ckpt_path = Path(args.checkpoint)

    if not ckpt_path.exists():
        logger.error("Checkpoint not found: %s", ckpt_path)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    logger.info("Evaluating model: %s on device: %s", args.model, device)

    # 1. Evaluate DL Model
    dl_metrics = evaluate_checkpoint(args.model, ckpt_path, hdf5_path, device)

    # 2. Evaluate Demographic Baseline (Age + Sex)
    demo_results = evaluate_demographic_baseline(hdf5_path)
    demo_metrics = demo_results["demographic_baseline"]

    # 3. Formatted Publication Summary Table
    print("\n" + "=" * 76)
    print("      PUBLICATION BENCHMARK: 12-LEAD ECG vs. DEMOGRAPHIC BASELINE")
    print("=" * 76)
    print(f"{'Metric':<32} | {'Demographics [Age,Sex]':<20} | {f'ECG Model ({args.model})':<18}")
    print("-" * 76)
    print(f"{'Creatinine MAE (mg/dL)':<32} | {demo_metrics['mae']:<20.4f} | {dl_metrics['mae']:<18.4f}")
    print(f"{'Creatinine RMSE (mg/dL)':<32} | {demo_metrics['rmse']:<20.4f} | {dl_metrics['rmse']:<18.4f}")
    print(f"{'Creatinine R^2':<32} | {demo_metrics['r2']:<20.4f} | {dl_metrics['r2']:<18.4f}")
    print(f"{'Creatinine Pearson r':<32} | {demo_metrics['pearson_r']:<20.4f} | {dl_metrics['pearson_r']:<18.4f}")
    print("-" * 76)
    print(f"{'eGFR MAE (mL/min/1.73m^2)':<32} | {demo_metrics.get('egfr_mae', float('nan')):<20.4f} | {dl_metrics.get('egfr_mae', float('nan')):<18.4f}")
    print(f"{'CKD 6-Stage Accuracy':<32} | {demo_metrics.get('ckd_stage_acc', float('nan')):<20.4f} | {dl_metrics.get('ckd_stage_acc', float('nan')):<18.4f}")
    print("-" * 76)
    print(f"{'CKD Screening (eGFR < 60) AUROC':<32} | {demo_metrics['screening']['auroc']:<20.4f} | {dl_metrics['screening']['auroc']:<18.4f}")
    print(f"{'CKD Screening Sensitivity':<32} | {demo_metrics['screening']['sensitivity']:<20.4f} | {dl_metrics['screening']['sensitivity']:<18.4f}")
    print(f"{'CKD Screening Specificity':<32} | {demo_metrics['screening']['specificity']:<20.4f} | {dl_metrics['screening']['specificity']:<18.4f}")
    print(f"{'CKD Screening F1-Score':<32} | {demo_metrics['screening']['f1']:<20.4f} | {dl_metrics['screening']['f1']:<18.4f}")
    print("=" * 76)

    # Save to JSON
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "model": args.model,
            "checkpoint": str(ckpt_path),
            "dl_metrics": dl_metrics,
            "demographic_baseline": demo_metrics,
        }
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Evaluation results saved to: %s", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ECG-to-creatinine model vs demographic baseline.")
    parser.add_argument("--model", required=True, choices=["cnn", "resnet34", "ecgfm_probe", "ecgfm_lora"])
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint.pth")
    parser.add_argument("--hdf5", default=None, help="Path to ecg_creatinine.h5")
    parser.add_argument("--config", default=None, help="Path to pipeline_config.yaml")
    parser.add_argument("--output", default="reports/evaluation_results.json", help="Output JSON path")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    main(parser.parse_args())
