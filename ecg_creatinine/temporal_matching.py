"""
ecg_creatinine/temporal_matching.py
=====================================
Patient-aware temporal matching of ECG records to laboratory creatinine values.

Design goals (publication rigour)
----------------------------------
* The train/val/test split is assigned at the subject_id level BEFORE any
  waveform is loaded, preventing any form of data leakage.
* Only the temporally nearest creatinine draw within the configured delta-t window
  is retained per ECG study.
* The actual delta-t (hours) for each pair is stored in the output manifest so
  reviewers can audit the matching quality.
* Preserves demographic attributes (age_years, sex) and study metadata throughout
  the cohort matching and splitting pipeline.
"""

from __future__ import annotations

import datetime
import logging
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core matching
# ---------------------------------------------------------------------------

def match_ecg_to_creatinine(
    ecg_manifest: pd.DataFrame,
    lab_df: pd.DataFrame,
    delta_t_hours: float = 6.0,
    strategy: str = "nearest",
) -> pd.DataFrame:
    """Match each ECG record to the nearest creatinine draw within delta-t.

    Complexity: O(N log N) via pd.merge_asof.
    Replaces the prior O(N·M) iterrows implementation.

    Parameters
    ----------
    ecg_manifest:
        DataFrame with columns: [study_id, subject_id, ecg_time].
        May optionally include demographic columns: [age_years, sex].
        ecg_time must be a timezone-naive datetime64.
    lab_df:
        DataFrame with columns: [subject_id, charttime, creatinine_mgdl].
        charttime must be timezone-naive datetime64.
    delta_t_hours:
        Maximum |ECG_time - lab_time| in hours to consider a valid pair.
    strategy:
        'nearest'      -- closest lab draw (in time) within the window.
        'first_before' -- most recent lab draw strictly before ECG time.
        'first_after'  -- earliest lab draw strictly after ECG time.

    Returns
    -------
    pd.DataFrame
        Matched cohort with columns:
        [study_id, subject_id, ecg_time, lab_time, creatinine_mgdl,
         delta_t_hours, ... (any extra columns preserved from ecg_manifest)]
        'split' column is NOT set here -- call assign_patient_split() next.
    """
    direction: Literal["backward", "forward", "nearest"]
    if strategy == "nearest":
        direction = "nearest"
    elif strategy == "first_before":
        direction = "backward"
    elif strategy == "first_after":
        direction = "forward"
    else:
        raise ValueError(
            f"Unknown matching strategy: '{strategy}'. "
            f"Choose from ['nearest', 'first_before', 'first_after']."
        )

    ecg_manifest = ecg_manifest.copy()
    lab_df = lab_df.copy()

    # Ensure datetime types and timezone-naive
    ecg_manifest["ecg_time"] = pd.to_datetime(ecg_manifest["ecg_time"])
    lab_df["charttime"]      = pd.to_datetime(lab_df["charttime"])

    if hasattr(ecg_manifest["ecg_time"].dt, "tz") and ecg_manifest["ecg_time"].dt.tz is not None:
        ecg_manifest["ecg_time"] = ecg_manifest["ecg_time"].dt.tz_localize(None)
    if hasattr(lab_df["charttime"].dt, "tz") and lab_df["charttime"].dt.tz is not None:
        lab_df["charttime"] = lab_df["charttime"].dt.tz_localize(None)

    tolerance = datetime.timedelta(hours=float(delta_t_hours))
    
    # Make subject_id the same integer type in both DataFrames
    ecg_manifest["subject_id"] = ecg_manifest["subject_id"].astype("int64")
    lab_df["subject_id"] = lab_df["subject_id"].astype("int64")

    # Identify metadata columns to preserve
    extra_cols = [
        c for c in ecg_manifest.columns
        if c not in ("study_id", "subject_id", "ecg_time")
    ]

    # merge_asof requires both DataFrames sorted by the temporal merge key
    ecg_sorted = ecg_manifest.sort_values("ecg_time").reset_index(drop=True)
    lab_sorted = lab_df.sort_values("charttime").reset_index(drop=True)

    merged = pd.merge_asof(
        ecg_sorted,
        lab_sorted[["subject_id", "charttime", "creatinine_mgdl"]],
        by="subject_id",
        left_on="ecg_time",
        right_on="charttime",
        direction=direction,
        tolerance=tolerance,
    )

    # Drop rows with no match (charttime is NaT after merge_asof)
    merged = merged.dropna(subset=["charttime"]).reset_index(drop=True)

    base_cols = [
        "study_id", "subject_id", "ecg_time", "lab_time",
        "creatinine_mgdl", "delta_t_hours",
    ]

    if merged.empty:
        logger.warning("No ECG-creatinine pairs found within delta_t=%.1f h.", delta_t_hours)
        all_cols = base_cols + [c for c in extra_cols if c not in base_cols]
        return pd.DataFrame(columns=all_cols)

    # Compute signed delta_t (hours): positive = lab after ECG
    merged["delta_t_hours"] = (
        (merged["charttime"] - merged["ecg_time"]).dt.total_seconds() / 3600.0
    )

    merged = merged.rename(columns={"charttime": "lab_time"})
    target_cols = [c for c in base_cols if c in merged.columns] + [
        c for c in extra_cols if c in merged.columns and c not in base_cols
    ]
    matched = merged[target_cols].copy()
    assert isinstance(matched, pd.DataFrame)

    logger.info(
        "Matched %d ECG-creatinine pairs from %d ECG records "
        "(delta_t <= %.1f h, strategy=%s).",
        len(matched), len(ecg_manifest), delta_t_hours, strategy,
    )
    return matched


# ---------------------------------------------------------------------------
# Patient-level split assignment
# ---------------------------------------------------------------------------

def assign_patient_split(
    matched_df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
    stratify_on: str = "creatinine_quartile",
) -> pd.DataFrame:
    """Assign train/val/test split at the patient (subject_id) level.

    Critically, assignment is on SUBJECTS, not individual ECG records.
    A subject's *all* records go to the same split, preventing leakage.

    Parameters
    ----------
    matched_df:
        Output of match_ecg_to_creatinine().
    train_frac, val_frac, test_frac:
        Fractional sizes.  Must sum to 1.0.
    seed:
        Random seed for reproducibility.
    stratify_on:
        'creatinine_quartile' -- stratify subjects by their median creatinine
        quartile so each split has similar CKD severity distribution.

    Returns
    -------
    pd.DataFrame
        matched_df with added 'split' column: 'train' | 'val' | 'test'.
    """
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
        raise ValueError("Split fractions must sum to 1.0.")

    rng = np.random.default_rng(seed)
    df = matched_df.copy()

    # Aggregate one row per subject: median creatinine as stratifier
    subject_stats = (
        df.groupby("subject_id")["creatinine_mgdl"]
        .median()
        .reset_index()
        .rename(columns={"creatinine_mgdl": "median_creatinine"})
    )

    # Robust quartile stratification handling duplicate edges or small counts
    try:
        subject_stats["creatinine_quartile"] = pd.qcut(
            subject_stats["median_creatinine"], q=4, labels=False, duplicates="drop"
        )
    except (ValueError, IndexError):
        subject_stats["creatinine_quartile"] = 0

    unique_strata = sorted(subject_stats["creatinine_quartile"].dropna().unique())

    # Shuffle subjects within each quartile stratum, then split
    subject_splits: dict[int, str] = {}
    for q in unique_strata:
        stratum = subject_stats.loc[subject_stats["creatinine_quartile"] == q, "subject_id"].to_numpy().copy()
        rng.shuffle(stratum)
        n = len(stratum)
        n_train = max(1, round(n * train_frac))
        n_val   = max(1, round(n * val_frac))
        for sid in stratum[:n_train]:
            subject_splits[int(sid)] = "train"
        for sid in stratum[n_train : n_train + n_val]:
            subject_splits[int(sid)] = "val"
        for sid in stratum[n_train + n_val :]:
            subject_splits[int(sid)] = "test"

    df["split"] = df["subject_id"].map(lambda x: subject_splits.get(int(x), "train"))

    # Log split statistics
    for split_name in ["train", "val", "test"]:
        subset = df.loc[df["split"] == split_name]
        assert isinstance(subset, pd.DataFrame)
        subj_col = subset["subject_id"].to_numpy()
        cr_col = subset["creatinine_mgdl"].to_numpy()
        n_subjects = int(len(np.unique(subj_col)))
        logger.info(
            "Split '%s': %d records from %d unique subjects "
            "| creatinine %.2f +/- %.2f mg/dL",
            split_name,
            len(subset),
            n_subjects,
            float(np.mean(cr_col)) if len(cr_col) > 0 else 0.0,
            float(np.std(cr_col)) if len(cr_col) > 1 else 0.0,
        )

    return df


# ---------------------------------------------------------------------------
# Leakage verification
# ---------------------------------------------------------------------------

def verify_no_subject_overlap(matched_df: pd.DataFrame) -> bool:
    """Assert that no subject_id appears in more than one split.

    Parameters
    ----------
    matched_df:
        DataFrame with 'subject_id' and 'split' columns.

    Returns
    -------
    bool
        True if clean (no overlap), raises AssertionError otherwise.
    """
    pivot = matched_df.groupby("subject_id")["split"].nunique()
    assert isinstance(pivot, pd.Series)
    leakers = pivot[pivot > 1]
    assert isinstance(leakers, pd.Series)
    if not leakers.empty:
        raise AssertionError(
            f"DATA LEAKAGE DETECTED: {len(leakers)} subject(s) appear in "
            f"multiple splits: {leakers.index.tolist()}"
        )
    logger.info("Leakage check PASSED: no subject appears in multiple splits.")
    return True


__all__ = [
    "match_ecg_to_creatinine",
    "assign_patient_split",
    "verify_no_subject_overlap",
]
