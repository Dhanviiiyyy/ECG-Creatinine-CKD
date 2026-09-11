"""
pipeline/04_preprocess_signals.py
==================================
Step 4 of the ECG-Creatinine prediction pipeline.

Loads the matched cohort, reads raw ECG waveforms (from .npy for synthetic
mode, or WFDB for MIMIC mode), applies the full signal preprocessing pipeline
(resample -> bandpass -> notch -> QC -> z-score), and saves processed
waveforms back to .npy files for use by the HDF5 builder.

The processing results (paths, QC status) are appended to the cohort parquet
as new columns and re-saved.

Usage
-----
    python pipeline/04_preprocess_signals.py [--config path/to/config.yaml]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.paths import Paths, load_config
from ecg_creatinine.signal_processing import preprocess_ecg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

_PROCESSED_WAVEFORM_DIR_NAME = "preprocessed"


def _load_raw_waveform(study_id: str, paths: Paths, mode: str) -> tuple[np.ndarray, float]:
    """Load raw ECG waveform and return (signal [12, N], orig_fs).

    Parameters
    ----------
    study_id:
        Study identifier string.
    paths:
        Resolved project paths.
    mode:
        'synthetic' | 'mimic'.

    Returns
    -------
    signal: np.ndarray [12, N] float32 in mV
    orig_fs: float  original sampling frequency
    """
    if mode == "synthetic":
        npy_path = paths.synthetic_waveforms_dir / f"{study_id}.npy"
        signal = np.load(npy_path).astype(np.float32)
        orig_fs = 500.0  # synthetic data generated at 500 Hz
        return signal, orig_fs
    else:
        raise NotImplementedError("MIMIC WFDB loading not yet implemented.")


def run(cfg: dict, paths: Paths) -> None:
    """Apply preprocessing to all records in the matched cohort.

    Parameters
    ----------
    cfg:
        Parsed pipeline configuration.
    paths:
        Resolved project paths.
    """
    mode = cfg["dataset"]["mode"]
    sig_cfg = cfg["signal"]

    # Load cohort
    cohort = pd.read_parquet(paths.cohort_matched)
    logger.info("Loaded cohort: %d records.", len(cohort))

    # Output directory for processed waveforms
    proc_dir = paths.data_processed / _PROCESSED_WAVEFORM_DIR_NAME
    proc_dir.mkdir(parents=True, exist_ok=True)

    qc_results = []
    processed_paths = []

    n_rejected = 0
    for _, row in tqdm(cohort.iterrows(), total=len(cohort), desc="Preprocessing ECGs"):
        study_id = str(row["study_id"])
        try:
            raw_signal, orig_fs = _load_raw_waveform(study_id, paths, mode)
        except FileNotFoundError:
            logger.warning("Waveform not found for study_id=%s, skipping.", study_id)
            qc_results.append(False)
            processed_paths.append(None)
            n_rejected += 1
            continue

        result = preprocess_ecg(raw_signal, orig_fs, sig_cfg)

        # Record QC result
        qc_pass = result["qc"]["pass"]
        qc_results.append(qc_pass)

        if not qc_pass:
            n_rejected += 1
            processed_paths.append(None)
            failed_leads = np.where(~result["qc"]["lead_mask"])[0].tolist()
            logger.debug(
                "QC FAIL study_id=%s: leads %s out of amplitude range.",
                study_id, failed_leads,
            )
            continue

        # Save processed waveforms
        out_norm = proc_dir / f"{study_id}_norm.npy"
        out_raw  = proc_dir / f"{study_id}_raw.npy"
        np.save(out_norm, result["waveform"])
        np.save(out_raw,  result["waveform_raw"])
        processed_paths.append(str(out_norm))

    cohort["qc_pass"] = qc_results
    cohort["processed_path"] = processed_paths

    # Filter to QC-passing records
    n_before = len(cohort)
    cohort = cohort[cohort["qc_pass"]].reset_index(drop=True)
    logger.info(
        "QC filtering: %d -> %d records (rejected %d / %.1f%%).",
        n_before, len(cohort), n_rejected,
        100.0 * n_rejected / max(n_before, 1),
    )

    # Overwrite cohort with QC info
    cohort.to_parquet(paths.cohort_matched, index=False)
    logger.info("Updated cohort saved -> %s", paths.cohort_matched)

    # Split summary
    for split_name in ["train", "val", "test"]:
        sub = cohort[cohort["split"] == split_name]
        logger.info("  %s: %d QC-passing records.", split_name, len(sub))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess ECG waveforms.")
    parser.add_argument("--config", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = load_config(args.config)
    paths = Paths(cfg)
    run(cfg, paths)
