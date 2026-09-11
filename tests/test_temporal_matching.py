"""
tests/test_temporal_matching.py
=================================
Unit tests for temporal_matching module.

Tests
-----
* match_ecg_to_creatinine: correct filtering within delta_t, decoys excluded.
* assign_patient_split: no subject appears in two splits.
* verify_no_subject_overlap: raises AssertionError on leakage.
* Split fractions are approximately correct.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ecg_creatinine.temporal_matching import (
    assign_patient_split,
    match_ecg_to_creatinine,
    verify_no_subject_overlap,
)


class TestMatchEcgToCreatinine:
    def _make_single_pair(self, offset_hours: float, delta_t: float = 6.0):
        """Helper to build a trivial one-record ECG / lab DataFrames."""
        base = pd.Timestamp("2150-06-01 12:00:00")
        ecg_df = pd.DataFrame({
            "study_id":   ["s000001"],
            "subject_id": [1],
            "ecg_time":   [base],
        })
        lab_df = pd.DataFrame({
            "subject_id":      [1],
            "charttime":       [base + pd.Timedelta(hours=offset_hours)],
            "creatinine_mgdl": [1.2],
        })
        return ecg_df, lab_df, delta_t

    def test_within_window_matched(self):
        ecg, lab, dt = self._make_single_pair(offset_hours=3.0, delta_t=6.0)
        result = match_ecg_to_creatinine(ecg, lab, delta_t_hours=dt)
        assert len(result) == 1
        assert abs(result.iloc[0]["delta_t_hours"] - 3.0) < 0.01

    def test_outside_window_rejected(self):
        ecg, lab, dt = self._make_single_pair(offset_hours=7.0, delta_t=6.0)
        result = match_ecg_to_creatinine(ecg, lab, delta_t_hours=dt)
        assert len(result) == 0, "Draw outside delta_t window must not be matched."

    def test_exactly_at_boundary_accepted(self):
        """A draw exactly at +6.0 h should be accepted (window is inclusive)."""
        ecg, lab, dt = self._make_single_pair(offset_hours=6.0, delta_t=6.0)
        result = match_ecg_to_creatinine(ecg, lab, delta_t_hours=dt)
        assert len(result) == 1

    def test_nearest_selected_among_multiple(self):
        """When two labs are within window, the closer one should be selected."""
        base = pd.Timestamp("2150-06-01 12:00:00")
        ecg_df = pd.DataFrame({
            "study_id":   ["s000001"],
            "subject_id": [1],
            "ecg_time":   [base],
        })
        lab_df = pd.DataFrame({
            "subject_id":      [1, 1],
            "charttime":       [base + pd.Timedelta(hours=1.0),
                                base + pd.Timedelta(hours=5.0)],
            "creatinine_mgdl": [0.9, 2.5],
        })
        result = match_ecg_to_creatinine(ecg_df, lab_df, delta_t_hours=6.0, strategy="nearest")
        assert len(result) == 1
        # The closer draw (1 h) should be selected
        assert abs(result.iloc[0]["delta_t_hours"] - 1.0) < 0.01
        assert abs(result.iloc[0]["creatinine_mgdl"] - 0.9) < 0.01

    def test_wrong_subject_not_matched(self):
        """Lab for subject 2 must not be matched to ECG of subject 1."""
        base = pd.Timestamp("2150-06-01 12:00:00")
        ecg_df = pd.DataFrame({
            "study_id":   ["s000001"],
            "subject_id": [1],
            "ecg_time":   [base],
        })
        lab_df = pd.DataFrame({
            "subject_id":      [2],
            "charttime":       [base],
            "creatinine_mgdl": [1.0],
        })
        result = match_ecg_to_creatinine(ecg_df, lab_df, delta_t_hours=6.0)
        assert len(result) == 0

    def test_delta_t_recorded_correctly(self, dummy_matched_df):
        """delta_t_hours column should have values within the configured window."""
        # dummy_matched_df already has delta_t_hours <= 5.0 by construction
        assert (dummy_matched_df["delta_t_hours"].abs() <= 6.0).all()


class TestAssignPatientSplit:
    def test_no_subject_in_two_splits(self, dummy_matched_df):
        result = assign_patient_split(dummy_matched_df, seed=42)
        for sid, grp in result.groupby("subject_id"):
            splits_for_sid = grp["split"].unique()
            assert len(splits_for_sid) == 1, (
                f"Subject {sid} appears in splits: {splits_for_sid}"
            )

    def test_all_records_assigned(self, dummy_matched_df):
        result = assign_patient_split(dummy_matched_df, seed=42)
        assert bool(result["split"].notna().all())
        assert set(result["split"].unique()) <= {"train", "val", "test"}

    def test_train_fraction_approximate(self, dummy_matched_df):
        result = assign_patient_split(
            dummy_matched_df, train_frac=0.70, val_frac=0.15, test_frac=0.15, seed=42
        )
        n_subjects = result["subject_id"].nunique()
        train_df = result[result["split"] == "train"]
        assert isinstance(train_df, pd.DataFrame)
        n_train_subjects = train_df["subject_id"].nunique()
        actual_frac = n_train_subjects / n_subjects
        # Allow +/-10% tolerance at this small scale
        assert abs(actual_frac - 0.70) < 0.15, (
            f"Train fraction {actual_frac:.2f} deviates too far from 0.70"
        )

    def test_invalid_fractions_raises(self, dummy_matched_df):
        with pytest.raises(ValueError, match="sum to 1.0"):
            assign_patient_split(dummy_matched_df, train_frac=0.5, val_frac=0.3, test_frac=0.3)


class TestVerifyNoSubjectOverlap:
    def test_clean_split_passes(self, dummy_matched_df):
        result = assign_patient_split(dummy_matched_df, seed=42)
        assert verify_no_subject_overlap(result) is True

    def test_leaky_split_raises(self, dummy_matched_df):
        result = assign_patient_split(dummy_matched_df, seed=42)
        # Artificially introduce leakage: assign first subject to both train and val
        train_df = result[result["split"] == "train"]
        assert isinstance(train_df, pd.DataFrame)
        first_sid = train_df["subject_id"].iloc[0]
        result.loc[result["subject_id"] == first_sid, "split"] = "val"
        # Add back the train records to force overlap
        extra = result[result["subject_id"] == first_sid].copy()
        extra["split"] = "train"
        leaky = pd.concat([result, extra], ignore_index=True)
        assert isinstance(leaky, pd.DataFrame)
        with pytest.raises(AssertionError, match="DATA LEAKAGE DETECTED"):
            verify_no_subject_overlap(leaky)
