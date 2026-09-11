# ECG-Based Serum Creatinine Prediction — Data Pipeline (Step 1)

**IIT Hyderabad Semester Project**
Supervised by **Prof. Amit Acharyya** & **Dr. Pabitra Das**

---

## Overview

This repository implements Step 1 of a peer-reviewed biomedical AI pipeline:
predicting continuous serum creatinine from raw 12-lead ECG waveforms
(`[12 x 5000]`) using MIMIC-IV-ECG matched with labevents (itemid 50912).

Downstream models (Steps 2-4):
- From-scratch 1D-CNN baseline
- 1D ResNet-34
- ECG-FM / ST-MEM Foundation Model (frozen probe + LoRA adapter)

---

## Project Structure

```
.
|-- config/
|   |-- pipeline_config.yaml      # All reproducibility parameters (SINGLE SOURCE OF TRUTH)
|   |-- paths.py                  # Centralised I/O path resolution
|
|-- data/
|   |-- synthetic/                # Synthetic demo dataset (pre-PhysioNet)
|   |   |-- generate_synthetic_dataset.py
|   |   |-- ecg_manifest.parquet  (generated)
|   |   |-- lab_creatinine.parquet(generated)
|   |   |-- waveforms/            (generated .npy files)
|   |-- raw/                      (MIMIC-IV WFDB files, when available)
|   |-- interim/                  (cohort_matched.parquet)
|   |-- processed/                (preprocessed waveforms + HDF5)
|
|-- ecg_creatinine/               # Core library (clean-room implementation)
|   |-- signal_processing.py      # Butterworth bandpass, notch, QC, z-score
|   |-- temporal_matching.py      # Patient-level temporal matching + split
|   |-- hdf5_utils.py             # HDF5 schema reader/writer
|
|-- pipeline/
|   |-- 03_match_ecg_creatinine.py
|   |-- 04_preprocess_signals.py
|   |-- 05_build_hdf5.py
|   |-- 06_validate_dataset.py
|
|-- tests/
|   |-- conftest.py               # Shared fixtures
|   |-- test_signal_filter.py     # Filter frequency response + pipeline
|   |-- test_temporal_matching.py # Leakage + matching correctness
|   |-- test_hdf5_schema.py       # Schema integrity
|
|-- reports/                      # QC HTML report (generated)
|-- run_pipeline.py               # Single-entry master runner
|-- requirements.txt
```

---

## Quick Start (Synthetic Mode)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline (synthetic data, 5000 records)
python run_pipeline.py

# 3. View QC report
open reports/dataset_qc_report.html   # macOS / Colab browser
```

---

## Signal Processing

| Step | Method | Parameters |
|------|--------|-----------|
| Resample | `scipy.signal.resample_poly` (GCD-reduced rational ratio) | 500 Hz target |
| Bandpass | 4th-order zero-phase Butterworth (`sosfiltfilt`) | 0.5 - 40 Hz |
| Notch | Zero-phase IIR notch (`iirnotch` -> `zpk2sos` -> `sosfiltfilt`) | 60 Hz, Q=30 |
| QC | Peak-to-peak amplitude check per lead | 0.05 - 20.0 mV |
| Normalise | Per-lead z-score (mean=0, std=1) | stored alongside raw |

---

## HDF5 Schema (v1.0.0)

```
/train/<study_id>/
    waveform        float32 [12, 5000]   z-scored, filtered
    waveform_raw    float32 [12, 5000]   filtered, not z-scored (mV)
    label           float32 scalar       serum creatinine (mg/dL)
    attrs: subject_id, ecg_time, lab_time, delta_t_hours,
           lead_names, fs=500, pipeline_version, git_hash, seed
/val/  (same)
/test/ (same)
/metadata/
    cohort_stats    JSON (per-split N, creatinine mean/std/quartiles)
    config_snapshot YAML (full pipeline_config.yaml)
```

---

## Key Methodological Safeguards

- **Patient-level splitting**: `subject_id` assignment happens at Stage 3,
  BEFORE any waveform is loaded, preventing data leakage.
- **Leakage assertion**: `verify_no_subject_overlap()` will raise
  `AssertionError` if any subject appears in multiple splits.
- **Temporal window**: +/-6 h (strict physiological fidelity).
- **Reproducibility**: every output embeds seed, pipeline version, git hash,
  and a full YAML config snapshot.

---

## Running Tests

```bash
pytest tests/ -v --tb=short
pytest tests/ -v --tb=short --cov=ecg_creatinine --cov-report=term-missing
```

---

## Switching to Real MIMIC-IV Data

1. Obtain PhysioNet credentialed access to `mimic-iv-ecg/1.0`
2. Set `ECG_PROJECT_ROOT` environment variable
3. In `config/pipeline_config.yaml`, change `dataset.mode: "mimic"`
4. Run `python run_pipeline.py`

---

## Citation / Attribution

All code written from first principles. No code copied from external repositories.
Filter design follows standard digital signal processing theory (Proakis & Manolakis, 2006).
Synthetic creatinine distribution parameterised from NHANES 2017-2018 and
Coresh et al. (2007) JAMA CKD prevalence estimates.

---

## Academic Integrity Statement

This codebase was written clean-room for academic publication purposes.
All external references are cited in module docstrings. No raw code chunks
were copied from GitHub or other repositories without explicit license compliance.
