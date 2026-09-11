"""
tests/conftest.py
==================
Shared pytest fixtures for the ECG-Creatinine pipeline tests.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sys
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.paths import load_config


@pytest.fixture(scope="session")
def cfg():
    """Load the pipeline configuration once per test session."""
    return load_config()


@pytest.fixture
def rng():
    """Seeded NumPy Generator for reproducible test data."""
    return np.random.default_rng(42)


@pytest.fixture
def dummy_ecg_12lead(rng):
    """Return a synthetic 12-lead ECG array [12, 5000] with realistic structure."""
    n_leads, n_samples = 12, 5000
    fs = 500.0
    t = np.linspace(0, 10, n_samples)
    signal = np.zeros((n_leads, n_samples), dtype=np.float32)
    for i in range(n_leads):
        # Simple sum of sinusoids as stand-in for ECG
        signal[i] = (
            0.5 * np.sin(2 * np.pi * 1.2 * t)    # ~heart rate
            + 0.1 * np.sin(2 * np.pi * 5.0 * t)
            + 0.02 * rng.standard_normal(n_samples)
        ).astype(np.float32)
    return signal


@pytest.fixture
def dummy_ecg_with_noise(dummy_ecg_12lead, rng):
    """ECG with 60 Hz powerline noise and baseline wander for filter testing."""
    t = np.linspace(0, 10, 5000)
    signal = dummy_ecg_12lead.copy()
    for i in range(12):
        signal[i] += (0.2 * np.sin(2 * np.pi * 60.0 * t)).astype(np.float32)  # powerline
        signal[i] += (0.1 * np.sin(2 * np.pi * 0.2 * t)).astype(np.float32)   # baseline wander
    return signal


@pytest.fixture
def dummy_matched_df():
    """Minimal matched cohort DataFrame for split/matching tests."""
    n = 100
    rng = np.random.default_rng(0)
    subject_ids = np.repeat(np.arange(1, 51), 2)  # 50 subjects x 2 ECGs
    np.random.shuffle(subject_ids)
    creatinine = np.clip(rng.lognormal(mean=0.0, sigma=0.4, size=n), 0.1, 10.0)
    base_time = pd.Timestamp("2150-01-01")
    ecg_times = [base_time + pd.Timedelta(hours=float(i * 24)) for i in range(n)]
    lab_times = [t + pd.Timedelta(hours=float(rng.uniform(-5, 5))) for t in ecg_times]
    return pd.DataFrame({
        "study_id":         [f"s{i:06d}" for i in range(n)],
        "subject_id":       subject_ids,
        "ecg_time":         ecg_times,
        "lab_time":         lab_times,
        "creatinine_mgdl":  creatinine.astype(np.float32),
        "delta_t_hours":    rng.uniform(-5, 5, n).astype(np.float32),
    })


@pytest.fixture
def tmp_hdf5(tmp_path, dummy_matched_df):
    """Write a minimal HDF5 file to a temporary directory for reader tests."""
    import h5py
    hdf5_path = tmp_path / "test.h5"
    rng = np.random.default_rng(0)

    from ecg_creatinine.temporal_matching import assign_patient_split
    df = assign_patient_split(dummy_matched_df, seed=42)

    with h5py.File(hdf5_path, "w") as fh:
        fh.attrs["schema_version"] = "1.0.0"
        import json
        for split in ["train", "val", "test"]:
            sub = df[df["split"] == split]
            grp = fh.require_group(split)
            for _, row in sub.iterrows():
                sid = str(row["study_id"])
                rec = grp.require_group(sid)
                wf = rng.standard_normal((12, 5000)).astype(np.float32)
                rec.create_dataset("waveform", data=wf)
                rec.create_dataset("waveform_raw", data=wf)
                rec.create_dataset("label", data=np.float32(float(str(row["creatinine_mgdl"]))))
                rec.attrs["subject_id"]    = int(str(row["subject_id"]))
                rec.attrs["study_id"]      = sid
                rec.attrs["ecg_time"]      = ""
                rec.attrs["lab_time"]      = ""
                rec.attrs["delta_t_hours"] = float(str(row["delta_t_hours"]))
                rec.attrs["lead_names"]    = json.dumps(
                    ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
                )
                rec.attrs["fs"]            = 500.0
        meta = fh.require_group("metadata")
        meta.create_dataset("cohort_stats", data="{}")
        meta.create_dataset("config_snapshot", data="")

    return hdf5_path
