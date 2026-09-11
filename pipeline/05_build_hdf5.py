"""
pipeline/05_build_hdf5.py
===========================
Step 5 of the ECG-Creatinine prediction pipeline.

Reads processed waveforms and the matched cohort manifest, writes the final
HDF5 dataset with the schema defined in ecg_creatinine/hdf5_utils.py, and
embeds full pipeline metadata for reproducibility.

Usage
-----
    python pipeline/05_build_hdf5.py [--config path/to/config.yaml]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.paths import Paths, load_config
from ecg_creatinine.hdf5_utils import HDF5Writer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _get_git_hash() -> str:
    """Return the current git commit hash, or 'unknown' if not in a repo."""
    try:
        import git
        repo = git.Repo(search_parent_directories=True)
        return repo.head.commit.hexsha[:12]
    except Exception:
        return "unknown"


def _compute_cohort_stats(cohort: pd.DataFrame) -> dict:
    """Compute per-split creatinine statistics for metadata embedding."""
    stats = {}
    for split_name in ["train", "val", "test"]:
        split_mask = cohort["split"] == split_name
        sub = cohort.loc[split_mask, "creatinine_mgdl"].to_numpy()
        subj_col = cohort.loc[split_mask, "subject_id"]
        stats[split_name] = {
            "n_records": int(len(sub)),
            "n_subjects": int(subj_col.nunique()),
            "mean": float(np.mean(sub)) if len(sub) > 0 else 0.0,
            "std": float(np.std(sub)) if len(sub) > 0 else 0.0,
            "min": float(np.min(sub)) if len(sub) > 0 else 0.0,
            "q25": float(np.percentile(sub, 25)) if len(sub) > 0 else 0.0,
            "median": float(np.median(sub)) if len(sub) > 0 else 0.0,
            "q75": float(np.percentile(sub, 75)) if len(sub) > 0 else 0.0,
            "max": float(np.max(sub)) if len(sub) > 0 else 0.0,
        }
    return stats


def run(cfg: dict, paths: Paths) -> None:
    """Build the HDF5 dataset from the preprocessed cohort.

    Parameters
    ----------
    cfg:
        Parsed pipeline configuration.
    paths:
        Resolved project paths.
    """
    hdf5_cfg = cfg["hdf5"]
    sig_cfg = cfg["signal"]
    seed = cfg["reproducibility"]["global_seed"]
    lead_names = sig_cfg["lead_names"]
    git_hash = _get_git_hash()

    # Load cohort (QC-filtered, with preprocessed paths)
    cohort = pd.read_parquet(paths.cohort_matched)
    assert isinstance(cohort, pd.DataFrame)
    # Keep only QC-passing records with valid processed paths
    cohort = cohort[cohort["qc_pass"] & cohort["processed_path"].notna()].reset_index(drop=True)
    assert isinstance(cohort, pd.DataFrame)
    logger.info("Building HDF5 from %d records.", len(cohort))

    # Prepare config YAML string for embedding
    config_yaml_str = yaml.dump(cfg, default_flow_style=False)

    cohort_stats = _compute_cohort_stats(cohort)

    paths.data_processed.mkdir(parents=True, exist_ok=True)

    with HDF5Writer(
        paths.hdf5_file,
        cfg=cfg,
        compression=str(hdf5_cfg["compression"]),
        compression_opts=int(hdf5_cfg["compression_opts"]),
    ) as writer:

        for _, row in tqdm(cohort.iterrows(), total=len(cohort), desc="Writing HDF5"):
            study_id = str(row["study_id"])
            processed_path = Path(str(row["processed_path"]))
            raw_path = Path(str(row["processed_path"]).replace("_norm.npy", "_raw.npy"))

            # Load preprocessed waveforms
            waveform     = np.load(processed_path).astype(np.float32)  # z-scored
            waveform_raw = np.load(raw_path).astype(np.float32)       # filtered, not z-scored

            meta = {
                "ecg_time":        str(row.get("ecg_time", "")),
                "lab_time":        str(row.get("lab_time", "")),
                "delta_t_hours":   float(row.get("delta_t_hours", 0.0)),
                "lead_names":      lead_names,
                "fs":              float(sig_cfg["target_fs_hz"]),
                "pipeline_version":"0.1.0",
                "seed":            seed,
                "git_hash":        git_hash,
                # Demographics for eGFR (Inker et al., 2021 CKD-EPI)
                "age_years":       int(row.get("age_years", -1)),
                "sex":             int(row.get("sex", -1)),  # 0=female, 1=male
            }

            writer.write_record(
                split=str(row["split"]),
                study_id=study_id,
                subject_id=int(row["subject_id"]),
                waveform=waveform,
                waveform_raw=waveform_raw,
                label=float(row["creatinine_mgdl"]),
                meta=meta,
            )

        writer.write_metadata(cohort_stats, str(config_yaml_str))

    # File size report
    hdf5_size_mb = paths.hdf5_file.stat().st_size / 1024**2
    logger.info(
        "HDF5 dataset written: %s  (%.1f MB)",
        paths.hdf5_file, hdf5_size_mb,
    )
    for split_name, stats in cohort_stats.items():
        logger.info(
            "  %s: %d records, creatinine median=%.2f [%.2f-%.2f] mg/dL",
            split_name, stats["n_records"],
            stats["median"], stats["q25"], stats["q75"],
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build HDF5 dataset.")
    parser.add_argument("--config", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = load_config(args.config)
    paths = Paths(cfg)
    run(cfg, paths)
