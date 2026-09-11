"""
ecg_creatinine/signal_processing.py
=====================================
From-scratch ECG signal processing utilities.

All filter designs are derived using scipy.signal primitives only.
No third-party ECG library is used, ensuring clean-room implementation
suitable for peer-reviewed publication.

Filter design methodology
--------------------------
Bandpass (0.5-40 Hz):
    4th-order Butterworth IIR, implemented in second-order sections (SOS)
    for numerical stability. Applied zero-phase via sosfiltfilt (forward +
    backward pass cancels phase distortion). Design follows standard digital
    filter design as described in Proakis & Manolakis (2006).

Powerline notch (60 Hz):
    IIR notch filter designed via scipy.signal.iirnotch, converted to SOS
    and applied zero-phase. Quality factor Q=30 yields 2 Hz bandwidth.

Resampling:
    scipy.signal.resample_poly with GCD-reduced rational ratio to avoid
    aliasing and preserve waveform morphology.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.signal import (
    butter,
    iirnotch,
    resample_poly,
    sosfiltfilt,
    zpk2sos,
)
from math import gcd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

def resample_signal(
    signal: np.ndarray,
    orig_fs: float,
    target_fs: float,
) -> np.ndarray:
    """Resample a multi-lead ECG signal to target_fs using polyphase filtering.

    Parameters
    ----------
    signal:
        Input array of shape (n_leads, n_samples) or (n_samples,).
    orig_fs:
        Original sampling frequency in Hz.
    target_fs:
        Target sampling frequency in Hz.

    Returns
    -------
    np.ndarray
        Resampled signal with shape (n_leads, n_new_samples) or (n_new_samples,).
    """
    if orig_fs == target_fs:
        return signal.copy()

    up = int(target_fs)
    down = int(orig_fs)
    common = gcd(up, down)
    up //= common
    down //= common

    if signal.ndim == 1:
        return resample_poly(signal, up, down).astype(np.float32)
    # Multi-lead: apply per-lead
    resampled = np.stack(
        [resample_poly(signal[i], up, down) for i in range(signal.shape[0])],
        axis=0,
    )
    return resampled.astype(np.float32)


# ---------------------------------------------------------------------------
# Bandpass filter
# ---------------------------------------------------------------------------

def design_bandpass_sos(
    low_hz: float,
    high_hz: float,
    fs: float,
    order: int = 4,
) -> np.ndarray:
    """Design a zero-phase Butterworth bandpass filter in SOS form.

    Parameters
    ----------
    low_hz:
        Lower -3 dB cutoff frequency (Hz).
    high_hz:
        Upper -3 dB cutoff frequency (Hz).
    fs:
        Sampling frequency (Hz).
    order:
        Filter order (applied as order//2 in each direction for bandpass).

    Returns
    -------
    np.ndarray
        Second-order sections array, shape (n_sections, 6).
    """
    nyq = fs / 2.0
    if low_hz <= 0 or high_hz >= nyq:
        raise ValueError(
            f"Cutoff frequencies must satisfy 0 < low_hz={low_hz} < "
            f"high_hz={high_hz} < Nyquist={nyq:.1f} Hz."
        )
    sos = butter(
        order,
        [low_hz / nyq, high_hz / nyq],
        btype="bandpass",
        output="sos",
    )
    return np.asarray(sos, dtype=np.float64)


def apply_bandpass(
    signal: np.ndarray,
    low_hz: float = 0.5,
    high_hz: float = 40.0,
    fs: float = 500.0,
    order: int = 4,
) -> np.ndarray:
    """Apply zero-phase bandpass filter to a (multi-lead) ECG signal.

    Parameters
    ----------
    signal:
        Array of shape (n_leads, n_samples) or (n_samples,).
    low_hz, high_hz:
        Passband cutoff frequencies (Hz).
    fs:
        Sampling frequency (Hz).
    order:
        Butterworth filter order.

    Returns
    -------
    np.ndarray
        Filtered signal, same shape as input, float32.
    """
    sos = design_bandpass_sos(low_hz, high_hz, fs, order)
    if signal.ndim == 1:
        return sosfiltfilt(sos, signal).astype(np.float32)
    return np.stack(
        [sosfiltfilt(sos, signal[i]) for i in range(signal.shape[0])],
        axis=0,
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# Powerline notch filter
# ---------------------------------------------------------------------------

def design_notch_sos(
    notch_hz: float,
    quality_factor: float,
    fs: float,
) -> np.ndarray:
    """Design a zero-phase IIR notch filter in SOS form.

    Parameters
    ----------
    notch_hz:
        Notch centre frequency (Hz).  Typically 60 Hz (US) or 50 Hz (EU).
    quality_factor:
        Q = f0 / bandwidth.  Q=30 -> 2 Hz bandwidth at 60 Hz.
    fs:
        Sampling frequency (Hz).

    Returns
    -------
    np.ndarray
        SOS array, shape (1, 6).
    """
    w0 = notch_hz / (fs / 2.0)
    b, a = iirnotch(w0, quality_factor)
    # Convert ba -> zpk -> sos for numerical stability
    from scipy.signal import tf2zpk
    z, p, k = tf2zpk(b, a)
    sos = zpk2sos(z, p, k)
    return np.asarray(sos, dtype=np.float64)


def apply_notch(
    signal: np.ndarray,
    notch_hz: float = 60.0,
    quality_factor: float = 30.0,
    fs: float = 500.0,
) -> np.ndarray:
    """Apply zero-phase notch filter to a (multi-lead) ECG signal.

    Parameters
    ----------
    signal:
        Array of shape (n_leads, n_samples) or (n_samples,).
    notch_hz:
        Centre frequency to attenuate (Hz).
    quality_factor:
        Q factor (Q=30 gives ~2 Hz bandwidth).
    fs:
        Sampling frequency (Hz).

    Returns
    -------
    np.ndarray
        Filtered signal, same shape as input, float32.
    """
    sos = design_notch_sos(notch_hz, quality_factor, fs)
    if signal.ndim == 1:
        return sosfiltfilt(sos, signal).astype(np.float32)
    return np.stack(
        [sosfiltfilt(sos, signal[i]) for i in range(signal.shape[0])],
        axis=0,
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# Amplitude QC
# ---------------------------------------------------------------------------

def check_lead_quality(
    signal: np.ndarray,
    min_pp_mv: float = 0.05,
    max_pp_mv: float = 20.0,
) -> dict[str, Any]:
    """Check per-lead amplitude quality.

    Parameters
    ----------
    signal:
        Array of shape (n_leads, n_samples) in mV.
    min_pp_mv, max_pp_mv:
        Peak-to-peak amplitude thresholds.

    Returns
    -------
    dict with keys:
        'pass'      : bool -- True if all leads pass
        'lead_mask' : bool array (n_leads,) -- True per passing lead
        'pp_mv'     : float array (n_leads,) -- peak-to-peak per lead
    """
    pp = signal.max(axis=1) - signal.min(axis=1)  # (n_leads,)
    lead_mask = (pp >= min_pp_mv) & (pp <= max_pp_mv)
    return {
        "pass": bool(lead_mask.all()),
        "lead_mask": lead_mask,
        "pp_mv": pp,
    }


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def z_score_normalize(
    signal: np.ndarray,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-lead z-score normalisation.

    Parameters
    ----------
    signal:
        Array of shape (n_leads, n_samples).
    eps:
        Small constant to avoid division by zero for flat leads.

    Returns
    -------
    normalised : np.ndarray  -- z-scored signal, float32
    means      : np.ndarray  -- per-lead mean (n_leads,)
    stds       : np.ndarray  -- per-lead std  (n_leads,)
    """
    means = signal.mean(axis=1, keepdims=True)  # (n_leads, 1)
    stds = signal.std(axis=1, keepdims=True).clip(min=eps)
    normalised = ((signal - means) / stds).astype(np.float32)
    return normalised, means.squeeze(), stds.squeeze()


# ---------------------------------------------------------------------------
# Full preprocessing pipeline
# ---------------------------------------------------------------------------

def preprocess_ecg(
    raw_signal: np.ndarray,
    orig_fs: float,
    cfg_signal: dict[str, Any],
) -> dict[str, Any]:
    """Apply the full preprocessing pipeline to one 12-lead ECG record.

    Steps (in order):
      1. Resample to target_fs
      2. Bandpass filter  (0.5-40 Hz)
      3. Notch filter     (60 Hz)
      4. Amplitude QC
      5. Z-score normalise per lead

    Parameters
    ----------
    raw_signal:
        Raw ECG array, shape (n_leads, n_raw_samples), in mV.
    orig_fs:
        Original sampling frequency (Hz).
    cfg_signal:
        Signal section of pipeline_config.yaml.

    Returns
    -------
    dict with keys:
        'waveform'      : np.ndarray [12, 5000] float32 -- normalised
        'waveform_raw'  : np.ndarray [12, 5000] float32 -- filtered, not normalised
        'qc'            : dict -- amplitude QC results
        'means'         : np.ndarray [12] -- per-lead mean (before normalisation)
        'stds'          : np.ndarray [12] -- per-lead std
        'fs'            : float -- target sampling frequency
    """
    target_fs = float(cfg_signal["target_fs_hz"])
    bp = cfg_signal["bandpass"]
    notch = cfg_signal["notch"]
    qc_cfg = cfg_signal["amplitude_qc"]

    # 1. Resample
    signal = resample_signal(raw_signal, orig_fs, target_fs)

    # 2. Bandpass
    signal = apply_bandpass(
        signal,
        low_hz=float(bp["low_hz"]),
        high_hz=float(bp["high_hz"]),
        fs=target_fs,
        order=int(bp["order"]),
    )

    # 3. Notch
    signal = apply_notch(
        signal,
        notch_hz=float(notch["freq_hz"]),
        quality_factor=float(notch["quality_factor"]),
        fs=target_fs,
    )

    # 4. QC
    qc = check_lead_quality(
        signal,
        min_pp_mv=float(qc_cfg["min_peak_to_peak_mv"]),
        max_pp_mv=float(qc_cfg["max_peak_to_peak_mv"]),
    )

    # 5. Normalise
    normalised, means, stds = z_score_normalize(signal)

    # Truncate / pad to exact n_samples
    n_target = int(cfg_signal["n_samples"])
    def _fit(arr: np.ndarray) -> np.ndarray:
        if arr.shape[1] >= n_target:
            return arr[:, :n_target]
        pad = n_target - arr.shape[1]
        return np.pad(arr, ((0, 0), (0, pad)), mode="constant")

    return {
        "waveform":     _fit(normalised),
        "waveform_raw": _fit(signal),
        "qc":           qc,
        "means":        means,
        "stds":         stds,
        "fs":           target_fs,
    }
