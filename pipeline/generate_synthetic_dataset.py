"""
data/synthetic/generate_synthetic_dataset.py
=============================================
Synthetic 12-lead ECG dataset generator for pipeline development.

This module generates physiologically plausible synthetic ECG waveforms paired
with synthetic serum creatinine labels.  It is intended for use ONLY during
the development phase (before PhysioNet MIMIC-IV-ECG access is obtained).

ECG Generation Model
---------------------
Each beat is modelled as a sum of five Gaussian pulses, representing:
  P wave, Q deflection, R peak, S deflection, T wave.
This is the standard parameterisation for synthetic ECG generation, originally
described in:
  McSharry, P.E. et al. (2003). A dynamical model for generating synthetic
  electrocardiogram signals. IEEE Trans Biomed Eng, 50(3), 289-294.
Note: We do NOT use the ODE model from that paper; we use only the
Gaussian-pulse parametrisation idea (well-known in the literature) to produce
plausible waveform shapes from scratch.

CKD-ECG Relationship (physiological basis)
-------------------------------------------
In patients with chronic kidney disease (CKD) and elevated creatinine:
  - Hyperkalaemia causes: peaked T waves, widened QRS, flattened P waves
    (Ref: Mount, D.B. (2014). Fluid and Electrolyte Disturbances.
     Harrison Principles of Internal Medicine.)
  - QTc prolongation from electrolyte disturbances
These correlations are encoded statistically in the synthetic generator to give
the model a physiologically grounded signal to learn from.

USAGE
-----
    python data/synthetic/generate_synthetic_dataset.py --n_records 5000 --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# -- Project root resolution ------------------------------------------------
_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.paths import Paths, load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Physiological constants
# =============================================================================

# Standard 12-lead order
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

# Gaussian pulse parameters per lead.
# Each row: [P_amp, Q_amp, R_amp, S_amp, T_amp] in normalised units.
# Values approximate typical ECG morphology for each lead.
# Negative amplitude = downward deflection.
# Source: textbook ECG morphology (Goldberger, Clinical Electrocardiography, 2017)
_LEAD_AMPLITUDES = np.array([
    # P      Q       R       S       T      lead
    [ 0.10, -0.03,  0.80,  -0.15,  0.40],  # I
    [ 0.15, -0.05,  1.20,  -0.20,  0.50],  # II
    [ 0.05, -0.02,  0.40,  -0.05,  0.30],  # III
    [-0.15,  0.05, -1.00,   0.20, -0.40],  # aVR
    [ 0.08, -0.02,  0.40,  -0.08,  0.20],  # aVL
    [ 0.10, -0.04,  0.80,  -0.13,  0.40],  # aVF
    [-0.05, -0.10,  0.25,  -0.80, -0.10],  # V1 (rS pattern)
    [ 0.05, -0.08,  0.50,  -0.60,  0.30],  # V2
    [ 0.10, -0.05,  0.90,  -0.35,  0.45],  # V3
    [ 0.12, -0.04,  1.20,  -0.20,  0.55],  # V4
    [ 0.10, -0.03,  1.50,  -0.10,  0.50],  # V5
    [ 0.08, -0.02,  1.00,  -0.05,  0.35],  # V6
], dtype=np.float32)  # shape (12, 5)

# Gaussian widths (in seconds) for [P, Q, R, S, T]
_WAVE_WIDTHS_S = np.array([0.080, 0.030, 0.020, 0.030, 0.160], dtype=np.float32)

# Timing offsets from R-peak centre (in seconds) for [P, Q, R, S, T]
_WAVE_OFFSETS_S = np.array([-0.160, -0.045, 0.000, 0.045, 0.180], dtype=np.float32)

# Leads where T-wave becomes peaked in hyperkalemia (CKD effect)
_HYPERKALEMIA_T_LEADS = [6, 7, 8, 9]  # V1-V4 (0-indexed)


# =============================================================================
# Beat generator
# =============================================================================

def _gaussian(t: np.ndarray, centre: float, width: float, amplitude: float) -> np.ndarray:
    """Return a Gaussian pulse evaluated at times t.

    f(t) = amplitude * exp(-0.5 * ((t - centre) / width)^2)
    """
    return amplitude * np.exp(-0.5 * ((t - centre) / (width + 1e-9)) ** 2)


def generate_single_beat(
    fs: float,
    rr_interval_s: float,
    lead_amplitudes: np.ndarray,
    creatinine: float,
    rng: np.random.Generator,
    qrs_width_factor: float = 1.0,
    noise_std: float = 0.02,
) -> np.ndarray:
    """Generate one 12-lead ECG beat as a sum of Gaussian pulses.

    Parameters
    ----------
    fs:
        Sampling frequency (Hz).
    rr_interval_s:
        RR interval (beat duration) in seconds.
    lead_amplitudes:
        Array [12, 5] of per-lead Gaussian amplitudes.
    creatinine:
        Serum creatinine (mg/dL) -- modulates T-wave amplitude and QRS width.
    rng:
        NumPy Generator for reproducibility.
    qrs_width_factor:
        Scale factor for QRS width (>1 = wider QRS, used for CKD simulation).
    noise_std:
        Standard deviation of Gaussian noise added to beat (mV).

    Returns
    -------
    np.ndarray
        Shape (12, n_samples_per_beat) float32.
    """
    n_samples = int(rr_interval_s * fs)
    t = np.linspace(0, rr_interval_s, n_samples, endpoint=False, dtype=np.float32)
    r_centre = rr_interval_s / 2.0  # R peak at midpoint of RR interval

    # CKD modulation factors
    # Creatinine > 1.5 mg/dL: mild hyperkalemia correlation
    # Creatinine > 3.0 mg/dL: moderate hyperkalemia
    ckd_factor = np.clip((creatinine - 0.8) / 3.0, 0.0, 1.0)  # 0 (normal) to 1 (severe)

    # Adjust wave offsets relative to R peak centre
    offsets = _WAVE_OFFSETS_S + r_centre  # absolute time positions
    widths = _WAVE_WIDTHS_S.copy()
    widths[1:4] *= qrs_width_factor  # Q, R, S widen with CKD

    beat = np.zeros((12, n_samples), dtype=np.float32)
    for lead_idx in range(12):
        amps = lead_amplitudes[lead_idx].copy()

        # CKD: peaked T waves in precordial leads V1-V4
        if lead_idx in _HYPERKALEMIA_T_LEADS:
            amps[4] *= (1.0 + 0.6 * ckd_factor)  # T-wave up to 60% taller

        # CKD: P-wave flattening (generalised across all leads)
        amps[0] *= (1.0 - 0.4 * ckd_factor)

        for wave_idx in range(5):
            beat[lead_idx] += _gaussian(
                t,
                centre=offsets[wave_idx],
                width=widths[wave_idx],
                amplitude=float(amps[wave_idx]),
            )

        # Add Gaussian noise
        beat[lead_idx] += rng.normal(0.0, noise_std, n_samples).astype(np.float32)

    return beat


def generate_ecg_waveform(
    n_samples: int,
    fs: float,
    heart_rate_bpm: float,
    creatinine: float,
    rng: np.random.Generator,
    hr_variability: float = 0.05,
) -> np.ndarray:
    """Generate a full 12-lead ECG waveform by concatenating beats.

    Parameters
    ----------
    n_samples:
        Total number of samples per lead (e.g. 5000 for 10 s at 500 Hz).
    fs:
        Sampling frequency (Hz).
    heart_rate_bpm:
        Mean heart rate (beats per minute).
    creatinine:
        Serum creatinine (mg/dL) for CKD modulation.
    rng:
        NumPy Generator.
    hr_variability:
        Fractional RR interval variability (simulates HRV).

    Returns
    -------
    np.ndarray
        Shape (12, n_samples) float32 ECG waveform.
    """
    # Add small per-patient amplitude perturbation
    amp_scale = rng.uniform(0.7, 1.3)
    lead_amps = _LEAD_AMPLITUDES * amp_scale

    # Add baseline wander (low-frequency sinusoid, <0.5 Hz)
    duration_s = n_samples / fs
    t_full = np.linspace(0, duration_s, n_samples, endpoint=False, dtype=np.float32)
    wander_freq = rng.uniform(0.1, 0.45)  # Hz -- below bandpass cutoff (will be removed by filter)
    wander_amp  = rng.uniform(0.0, 0.15)  # mV
    baseline_wander = (wander_amp * np.sin(2 * np.pi * wander_freq * t_full)).astype(np.float32)

    # Add simulated 60 Hz powerline noise (will be removed by notch filter)
    pl_amp = rng.uniform(0.0, 0.05)
    powerline = (pl_amp * np.sin(2 * np.pi * 60.0 * t_full)).astype(np.float32)

    # QRS width factor (elevated creatinine -> wider QRS)
    qrs_width_factor = 1.0 + 0.15 * np.clip((creatinine - 1.5) / 4.0, 0.0, 1.0)

    # Mean RR interval with beat-to-beat HRV
    rr_mean_s = 60.0 / heart_rate_bpm

    signal = np.zeros((12, n_samples), dtype=np.float32)
    sample_pos = 0

    while sample_pos < n_samples:
        # Beat-to-beat RR variability
        rr_s = rr_mean_s * rng.uniform(1 - hr_variability, 1 + hr_variability)
        beat = generate_single_beat(
            fs=fs,
            rr_interval_s=rr_s,
            lead_amplitudes=lead_amps,
            creatinine=creatinine,
            rng=rng,
            qrs_width_factor=float(qrs_width_factor),
            noise_std=0.02,
        )
        n_beat = beat.shape[1]
        end_pos = min(sample_pos + n_beat, n_samples)
        n_copy  = end_pos - sample_pos
        signal[:, sample_pos:end_pos] = beat[:, :n_copy]
        sample_pos = end_pos

    # Add baseline wander and powerline noise to all leads
    for lead_idx in range(12):
        signal[lead_idx] += baseline_wander + powerline

    return signal


# =============================================================================
# Creatinine distribution
# =============================================================================

def sample_creatinine_values(
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample realistic serum creatinine values from a mixture distribution.

    Mixture components (approximate population proportions):
      - 65% healthy: log-normal centred at 0.9 mg/dL, sigma=0.2
      - 25% mild CKD: log-normal centred at 1.6 mg/dL, sigma=0.3
      - 10% severe CKD: log-normal centred at 4.0 mg/dL, sigma=0.5

    Distribution parameters are estimated from:
      NHANES (2017-2018) serum creatinine reference ranges and CKD prevalence.
      Coresh, J. et al. (2007). Prevalence of Chronic Kidney Disease in the
      United States. JAMA, 298(17), 2038-2047.

    Parameters
    ----------
    n: Number of values to sample.
    rng: NumPy Generator.

    Returns
    -------
    np.ndarray shape (n,) float32, values in mg/dL.
    """
    probs = [0.65, 0.25, 0.10]
    params = [
        (np.log(0.90), 0.20),  # healthy: mu, sigma in log-space
        (np.log(1.60), 0.30),  # mild CKD
        (np.log(4.00), 0.50),  # severe CKD
    ]
    component = rng.choice(len(probs), size=n, p=probs)
    values = np.zeros(n, dtype=np.float32)
    for idx, (mu, sigma) in enumerate(params):
        mask = component == idx
        if mask.any():
            values[mask] = np.exp(rng.normal(mu, sigma, mask.sum())).astype(np.float32)

    # Clip to physiologic range
    values = np.clip(values, 0.1, 20.0)
    return values


# =============================================================================
# Timestamp generation
# =============================================================================

def generate_timestamps(
    n_ecg: int,
    n_subjects: int,
    delta_t_hours_max: float,
    rng: np.random.Generator,
    base_date: str = "2150-01-01",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic ECG and lab timestamps respecting the +/-dt constraint.

    Parameters
    ----------
    n_ecg:
        Number of ECG records.
    n_subjects:
        Number of unique subjects in the cohort.
    delta_t_hours_max:
        Maximum |ECG_time - lab_time| in hours.
    rng:
        NumPy Generator.
    base_date:
        Synthetic start date (future date avoids confusion with real MIMIC data).

    Returns
    -------
    ecg_df:
        DataFrame [study_id, subject_id, ecg_time, heart_rate_bpm].
    lab_df:
        DataFrame [subject_id, charttime, creatinine_mgdl].
    """
    base = pd.Timestamp(base_date)
    subject_ids = np.arange(1, n_subjects + 1)

    # Assign subjects to ECG records (some subjects have multiple ECGs)
    assigned_subjects = rng.choice(subject_ids, size=n_ecg, replace=True)

    # ECG times: spread over ~2 years
    ecg_offsets_h = rng.uniform(0, 24 * 365 * 2, size=n_ecg)
    ecg_times = [base + pd.Timedelta(hours=float(h)) for h in ecg_offsets_h]

    # Heart rates: 50-110 bpm (slightly elevated in CKD due to anemia)
    heart_rates = rng.uniform(50, 110, size=n_ecg).astype(np.float32)

    # Patient demographics: age 18-90, sex 50/50 (per-subject, stable)
    subject_ages = {int(sid): int(rng.integers(18, 91))  for sid in subject_ids}
    subject_sexes= {int(sid): int(rng.integers(0, 2))    for sid in subject_ids}

    ecg_df = pd.DataFrame({
        "study_id":       [f"s{i:08d}" for i in range(1, n_ecg + 1)],
        "subject_id":     assigned_subjects,
        "ecg_time":       ecg_times,
        "heart_rate_bpm": heart_rates,
        "age_years":      [subject_ages[int(s)]  for s in assigned_subjects],
        "sex":            [subject_sexes[int(s)] for s in assigned_subjects],
    })


    # Lab records: for each ECG, create a matching lab draw within +/-delta_t
    # Plus some decoy lab draws outside the window (to test matching robustness)
    lab_records = []
    creatinine_per_subject: dict[int, float] = {}

    # Assign a stable creatinine level per subject (with small temporal variation)
    base_creatinine = sample_creatinine_values(n_subjects, rng)
    for sid, cr in zip(subject_ids, base_creatinine):
        creatinine_per_subject[int(sid)] = float(cr)

    for _, row in ecg_df.iterrows():
        sid = int(str(row["subject_id"]))
        t_ecg = row["ecg_time"]
        base_cr = creatinine_per_subject[sid]

        # Matching lab draw: within +/-delta_t_hours_max
        offset_h = rng.uniform(-delta_t_hours_max * 0.9, delta_t_hours_max * 0.9)
        cr_noise = rng.normal(0, 0.05 * base_cr)  # small temporal variation
        lab_records.append({
            "subject_id":       sid,
            "charttime":        t_ecg + pd.Timedelta(hours=float(offset_h)),
            "creatinine_mgdl":  np.clip(base_cr + cr_noise, 0.1, 20.0),
        })

        # Decoy draws outside the window (negative controls for matching test)
        n_decoys = int(rng.integers(0, 4))
        for _ in range(n_decoys):
            decoy_offset = rng.choice([-1, 1]) * rng.uniform(
                delta_t_hours_max * 1.2, delta_t_hours_max * 5
            )
            lab_records.append({
                "subject_id":       sid,
                "charttime":        t_ecg + pd.Timedelta(hours=float(decoy_offset)),
                "creatinine_mgdl":  float(np.clip(base_cr + rng.normal(0, 0.1), 0.1, 20.0)),
            })

    lab_df = pd.DataFrame(lab_records)
    return ecg_df, lab_df


# =============================================================================
# Main generator
# =============================================================================

def generate_synthetic_dataset(
    n_records: int = 5000,
    n_subjects: int = 3000,
    seed: int = 42,
    fs: float = 500.0,
    n_samples: int = 5000,
    delta_t_hours: float = 6.0,
    output_dir: Path | None = None,
) -> None:
    """Generate and save the full synthetic ECG-creatinine dataset.

    Outputs
    -------
    <output_dir>/ecg_manifest.parquet
        One row per ECG record with metadata.
    <output_dir>/lab_creatinine.parquet
        Synthetic lab draws (creatinine) including decoys.
    <output_dir>/waveforms/<study_id>.npy
        Raw (pre-filter) ECG waveform [12, 5000] float32 per record.
    """
    rng = np.random.default_rng(seed)

    if output_dir is None:
        cfg = load_config()
        p = Paths(cfg)
        output_dir = p.synthetic_dir
    output_dir = Path(output_dir)
    waveform_dir = output_dir / "waveforms"
    waveform_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating synthetic dataset: n_records=%d, n_subjects=%d, seed=%d",
                n_records, n_subjects, seed)

    # -- 1. Generate timestamps and lab records ------------------------------
    ecg_df, lab_df = generate_timestamps(
        n_ecg=n_records,
        n_subjects=n_subjects,
        delta_t_hours_max=delta_t_hours,
        rng=rng,
    )

    # -- 2. Generate waveforms -----------------------------------------------
    logger.info("Generating ECG waveforms (this may take a few minutes)...")

    # Build subject -> creatinine lookup from lab_df (use median per subject)
    subject_cr = (
        lab_df.groupby("subject_id")["creatinine_mgdl"]
        .median()
        .to_dict()
    )

    for idx, (_, row) in enumerate(ecg_df.iterrows()):
        study_id = str(row["study_id"])
        sid = int(str(row["subject_id"]))
        cr = float(subject_cr.get(sid, 1.0))
        hr = float(str(row["heart_rate_bpm"]))

        waveform = generate_ecg_waveform(
            n_samples=n_samples,
            fs=fs,
            heart_rate_bpm=hr,
            creatinine=cr,
            rng=rng,
        )

        out_path = waveform_dir / f"{study_id}.npy"
        np.save(out_path, waveform)

        if (idx + 1) % 500 == 0:
            logger.info("  Generated %d / %d waveforms.", idx + 1, n_records)

    # -- 3. Save manifests ---------------------------------------------------
    ecg_df.to_parquet(output_dir / "ecg_manifest.parquet", index=False)
    lab_df.to_parquet(output_dir / "lab_creatinine.parquet", index=False)

    # Save run metadata
    meta = {
        "generator":     "generate_synthetic_dataset.py",
        "n_records":     n_records,
        "n_subjects":    n_subjects,
        "seed":          seed,
        "fs_hz":         fs,
        "n_samples":     n_samples,
        "delta_t_hours": delta_t_hours,
        "schema":        "v1.0.0",
    }
    with open(output_dir / "synthetic_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info(
        "Synthetic dataset complete.\n"
        "  ECG manifest  : %s\n"
        "  Lab manifest  : %s\n"
        "  Waveforms     : %s/\n"
        "  N records     : %d",
        output_dir / "ecg_manifest.parquet",
        output_dir / "lab_creatinine.parquet",
        waveform_dir,
        n_records,
    )


# =============================================================================
# CLI entrypoint
# =============================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic ECG-creatinine dataset for pipeline development."
    )
    parser.add_argument("--n_records",  type=int, default=5000)
    parser.add_argument("--n_subjects", type=int, default=3000)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory (default: from config)")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    generate_synthetic_dataset(
        n_records=args.n_records,
        n_subjects=args.n_subjects,
        seed=args.seed,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )
