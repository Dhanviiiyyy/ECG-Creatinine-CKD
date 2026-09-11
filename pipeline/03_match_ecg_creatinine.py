"""
pipeline/03_match_ecg_creatinine.py
=====================================
Step 3 of the ECG-Creatinine prediction pipeline.

Loads the ECG manifest and lab creatinine table, performs temporal matching
within the configured delta_t window, assigns patient-level train/val/test
splits, verifies there is no subject-level data leakage, and saves the
cohort manifest to Parquet.

Usage
-----
    python pipeline/03_match_ecg_creatinine.py [--config path/to/config.yaml]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.paths import Paths, load_config
from ecg_creatinine.temporal_matching import (
    assign_patient_split,
    match_ecg_to_creatinine,
    verify_no_subject_overlap,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def run(cfg: dict, paths: Paths) -> pd.DataFrame:
    """Execute temporal matching and split assignment.

    Parameters
    ----------
    cfg:
        Parsed pipeline configuration.
    paths:
        Resolved project paths.

    Returns
    -------
    pd.DataFrame
        Matched and split-assigned cohort.
    """
    mode = cfg["dataset"]["mode"]
    tm_cfg = cfg["temporal_matching"]
    split_cfg = cfg["splits"]
    seed = cfg["reproducibility"]["global_seed"]

    # -- Load manifests --
    if mode == "synthetic":
        ecg_manifest_path = paths.synthetic_ecg_manifest
        lab_path = paths.synthetic_lab_manifest
    else:
        raise NotImplementedError("MIMIC mode not yet implemented. Set mode='synthetic'.")

    logger.info("Loading ECG manifest from: %s", ecg_manifest_path)
    ecg_df = pd.read_parquet(ecg_manifest_path)
    # Rename to expected columns if needed
    if "ecg_time" not in ecg_df.columns and "ecg_timestamp" in ecg_df.columns:
        ecg_df = ecg_df.rename(columns={"ecg_timestamp": "ecg_time"})

    logger.info("Loading lab creatinine from: %s", lab_path)
    lab_df = pd.read_parquet(lab_path)

    logger.info(
        "Input: %d ECG records, %d lab draws.",
        len(ecg_df), len(lab_df),
    )

    # -- Temporal matching --
    matched = match_ecg_to_creatinine(
        ecg_manifest=ecg_df,
        lab_df=lab_df,
        delta_t_hours=float(tm_cfg["delta_t_hours"]),
        strategy=str(tm_cfg["strategy"]),
    )

    if matched.empty:
        logger.error("No matched pairs found. Check delta_t and input data.")
        sys.exit(1)

    # -- Patient-level split assignment --
    matched = assign_patient_split(
        matched_df=matched,
        train_frac=float(split_cfg["train"]),
        val_frac=float(split_cfg["val"]),
        test_frac=float(split_cfg["test"]),
        seed=seed,
        stratify_on=str(split_cfg["stratify_on"]),
    )

    # -- Leakage verification --
    verify_no_subject_overlap(matched)

    # -- Save --
    paths.data_interim.mkdir(parents=True, exist_ok=True)
    out_path = paths.cohort_matched
    matched.to_parquet(out_path, index=False)
    logger.info(
        "Saved matched cohort (%d records) -> %s",
        len(matched), out_path,
    )

    # Summary
    for split_name in ["train", "val", "test"]:
        sub = matched.loc[matched["split"] == split_name]
        assert isinstance(sub, pd.DataFrame)
        cr_vals = sub["creatinine_mgdl"].to_numpy()
        subj_vals = sub["subject_id"].to_numpy()
        logger.info(
            "  %s: %d records, %d subjects, creatinine %.2f +/- %.2f mg/dL",
            split_name, len(sub), int(len(np.unique(subj_vals))),
            float(np.mean(cr_vals)) if len(cr_vals) > 0 else 0.0,
            float(np.std(cr_vals)) if len(cr_vals) > 0 else 0.0,
        )

    return matched


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Match ECG records to creatinine labs.")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to pipeline_config.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = load_config(args.config)
    paths = Paths(cfg)
    run(cfg, paths)
