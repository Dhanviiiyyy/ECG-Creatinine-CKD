"""
run_pipeline.py
================
Master orchestrator for the ECG-Creatinine data pipeline (Step 1).

Runs all pipeline stages in sequence:
  Stage 0: Generate synthetic dataset (if mode == synthetic)
  Stage 3: Match ECG records to creatinine labs (temporal matching + splits)
  Stage 4: Preprocess signals (resample, bandpass, notch, QC, z-score)
  Stage 5: Build HDF5 dataset
  Stage 6: Validate dataset (QC report)

Usage (local)
-------------
    python run_pipeline.py
    python run_pipeline.py --config config/pipeline_config.yaml --skip-validate

Usage (Colab)
-------------
    !python run_pipeline.py
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.paths import Paths, load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _banner(stage: str, title: str) -> None:
    width = 70
    logger.info("=" * width)
    logger.info("  %s  |  %s", stage, title)
    logger.info("=" * width)


def _load_script(script_path: Path):
    """Dynamically load a pipeline script as a module."""
    spec = importlib.util.spec_from_file_location(script_path.stem, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for {script_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(config_path: str | None = None, skip_validate: bool = False) -> None:
    cfg = load_config(config_path)
    paths = Paths(cfg)
    paths.make_all_dirs()
    mode = cfg["dataset"]["mode"]

    total_start = time.time()
    pipeline_dir = _PROJECT_ROOT / "pipeline"

    # -------------------------------------------------------------------------
    # Stage 0: Synthetic data generation
    # -------------------------------------------------------------------------
    if mode == "synthetic":
        _banner("STAGE 0", "Generating Synthetic Dataset")
        from data.synthetic.generate_synthetic_dataset import generate_synthetic_dataset
        generate_synthetic_dataset(
            n_records=int(cfg["dataset"]["n_records"]),
            n_subjects=int(cfg["dataset"]["pilot_subject_pool"]),
            seed=int(cfg["reproducibility"]["global_seed"]),
            fs=float(cfg["signal"]["target_fs_hz"]),
            n_samples=int(cfg["signal"]["n_samples"]),
            delta_t_hours=float(cfg["temporal_matching"]["delta_t_hours"]),
            output_dir=paths.synthetic_dir,
        )
    else:
        logger.info("Mode='mimic': skipping synthetic generation.")

    # -------------------------------------------------------------------------
    # Stage 3: Temporal matching + split assignment
    # -------------------------------------------------------------------------
    _banner("STAGE 3", "Temporal Matching & Patient-Level Split Assignment")
    stage3 = _load_script(pipeline_dir / "03_match_ecg_creatinine.py")
    stage3.run(cfg, paths)

    # -------------------------------------------------------------------------
    # Stage 4: Signal preprocessing
    # -------------------------------------------------------------------------
    _banner("STAGE 4", "Signal Preprocessing (Resample + Bandpass + Notch + QC + Z-score)")
    stage4 = _load_script(pipeline_dir / "04_preprocess_signals.py")
    stage4.run(cfg, paths)

    # -------------------------------------------------------------------------
    # Stage 5: Build HDF5
    # -------------------------------------------------------------------------
    _banner("STAGE 5", "Building HDF5 Dataset")
    stage5 = _load_script(pipeline_dir / "05_build_hdf5.py")
    stage5.run(cfg, paths)

    # -------------------------------------------------------------------------
    # Stage 6: Validation & QC report
    # -------------------------------------------------------------------------
    if not skip_validate:
        _banner("STAGE 6", "Dataset Validation & QC Report")
        stage6 = _load_script(pipeline_dir / "06_validate_dataset.py")
        stage6.run(cfg, paths)
        logger.info("QC report: %s", paths.qc_report)

    elapsed = time.time() - total_start
    logger.info("")
    logger.info("Pipeline complete in %.1f s.", elapsed)
    logger.info("HDF5 dataset: %s", paths.hdf5_file)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full ECG-Creatinine data pipeline.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--skip-validate", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(config_path=args.config, skip_validate=args.skip_validate)
