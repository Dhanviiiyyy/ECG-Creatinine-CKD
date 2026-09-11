"""
tests/test_signal_filter.py
=============================
Unit tests for signal_processing module.

Tests
-----
* Bandpass filter: verify passband gain ~0 dB, stopband attenuation >= 20 dB.
* Notch filter: verify 60 Hz attenuation >= 20 dB.
* Resample: verify output length and that no NaN introduced.
* Z-score: verify per-lead mean ~0, std ~1.
* Full preprocess_ecg: verify output shape [12, 5000], finite values.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import freqz

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ecg_creatinine.signal_processing import (
    apply_bandpass,
    apply_notch,
    design_bandpass_sos,
    design_notch_sos,
    preprocess_ecg,
    resample_signal,
    z_score_normalize,
)


FS = 500.0
N = 5000


class TestBandpassFilter:
    def test_passband_gain(self):
        """Mid-band gain (e.g. at 10 Hz) should be close to 0 dB (gain ~1)."""
        sos = design_bandpass_sos(0.5, 40.0, FS, order=4)
        # Evaluate frequency response
        from scipy.signal import sosfreqz
        w, h = sosfreqz(sos, worN=4096, fs=FS)
        w_arr, h_arr = np.asarray(w), np.asarray(h)
        # Find gain at 10 Hz
        idx_10 = int(np.argmin(np.abs(w_arr - 10.0)))
        gain_db = float(20 * np.log10(np.abs(h_arr[idx_10]) + 1e-12))
        assert gain_db > -3.0, f"Passband gain at 10 Hz too low: {gain_db:.1f} dB"

    def test_stopband_below_lowcut(self):
        """Signal at 0.1 Hz (below 0.5 Hz cutoff) must be attenuated >= 20 dB."""
        from scipy.signal import sosfreqz
        sos = design_bandpass_sos(0.5, 40.0, FS, order=4)
        w, h = sosfreqz(sos, worN=8192, fs=FS)
        w_arr, h_arr = np.asarray(w), np.asarray(h)
        idx = int(np.argmin(np.abs(w_arr - 0.1)))
        gain_db = float(20 * np.log10(np.abs(h_arr[idx]) + 1e-12))
        assert gain_db < -20.0, f"Stopband gain at 0.1 Hz too high: {gain_db:.1f} dB"

    def test_stopband_above_highcut(self):
        """Signal at 80 Hz (above 40 Hz cutoff) must be attenuated >= 20 dB."""
        from scipy.signal import sosfreqz
        sos = design_bandpass_sos(0.5, 40.0, FS, order=4)
        w, h = sosfreqz(sos, worN=8192, fs=FS)
        w_arr, h_arr = np.asarray(w), np.asarray(h)
        idx = int(np.argmin(np.abs(w_arr - 80.0)))
        gain_db = float(20 * np.log10(np.abs(h_arr[idx]) + 1e-12))
        assert gain_db < -20.0, f"Stopband gain at 80 Hz too high: {gain_db:.1f} dB"

    def test_output_shape_single_lead(self):
        sig = np.random.default_rng(0).standard_normal(N).astype(np.float32)
        out = apply_bandpass(sig, fs=FS)
        assert out.shape == (N,)

    def test_output_shape_multi_lead(self, dummy_ecg_12lead):
        out = apply_bandpass(dummy_ecg_12lead, fs=FS)
        assert out.shape == (12, N)

    def test_output_dtype(self, dummy_ecg_12lead):
        out = apply_bandpass(dummy_ecg_12lead, fs=FS)
        assert out.dtype == np.float32

    def test_no_nan_in_output(self, dummy_ecg_with_noise):
        out = apply_bandpass(dummy_ecg_with_noise, fs=FS)
        assert np.isfinite(out).all()

    def test_in_band_signal_passes_through(self):
        """Signal at 10 Hz (well within 0.5-40 Hz passband) should NOT be removed."""
        t = np.linspace(0, 10, N)
        sig_10hz = (0.5 * np.sin(2 * np.pi * 10.0 * t)).astype(np.float32)
        out = apply_bandpass(sig_10hz[np.newaxis], fs=FS)
        # Energy at 10 Hz should be largely preserved
        assert out.std() > 0.1, "Bandpass incorrectly removed in-band 10 Hz signal."


class TestNotchFilter:
    def test_notch_attenuation_at_60hz(self):
        """60 Hz component must be attenuated >= 20 dB by the notch filter."""
        from scipy.signal import sosfreqz
        sos = design_notch_sos(60.0, quality_factor=30.0, fs=FS)
        w, h = sosfreqz(sos, worN=8192, fs=FS)
        w_arr, h_arr = np.asarray(w), np.asarray(h)
        idx = int(np.argmin(np.abs(w_arr - 60.0)))
        gain_db = float(20 * np.log10(np.abs(h_arr[idx]) + 1e-12))
        assert gain_db < -20.0, f"Notch gain at 60 Hz: {gain_db:.1f} dB (should be < -20 dB)"

    def test_notch_passband_preserved(self):
        """Signal at 10 Hz should NOT be significantly attenuated by notch."""
        from scipy.signal import sosfreqz
        sos = design_notch_sos(60.0, quality_factor=30.0, fs=FS)
        w, h = sosfreqz(sos, worN=8192, fs=FS)
        w_arr, h_arr = np.asarray(w), np.asarray(h)
        idx = int(np.argmin(np.abs(w_arr - 10.0)))
        gain_db = float(20 * np.log10(np.abs(h_arr[idx]) + 1e-12))
        assert gain_db > -3.0, f"Notch incorrectly attenuates 10 Hz: {gain_db:.1f} dB"

    def test_notch_removes_powerline_from_signal(self, dummy_ecg_with_noise):
        """After notch, 60 Hz energy should be significantly reduced."""
        from scipy.signal import welch
        sig_1lead = dummy_ecg_with_noise[0]  # 1D
        filtered = apply_notch(sig_1lead[np.newaxis], fs=FS)[0]
        f_orig, psd_orig = welch(sig_1lead, fs=FS, nperseg=512)
        f_filt, psd_filt = welch(filtered,  fs=FS, nperseg=512)
        idx_60 = np.argmin(np.abs(f_orig - 60.0))
        ratio = psd_filt[idx_60] / (psd_orig[idx_60] + 1e-12)
        assert ratio < 0.1, f"Notch filter did not reduce 60 Hz sufficiently (ratio={ratio:.3f})"


class TestResample:
    def test_resample_1000hz_to_500hz(self):
        sig = np.random.default_rng(1).standard_normal((12, 10000)).astype(np.float32)
        out = resample_signal(sig, orig_fs=1000.0, target_fs=500.0)
        assert out.shape == (12, 5000)

    def test_resample_250hz_to_500hz(self):
        sig = np.random.default_rng(1).standard_normal((12, 2500)).astype(np.float32)
        out = resample_signal(sig, orig_fs=250.0, target_fs=500.0)
        assert out.shape == (12, 5000)

    def test_resample_same_fs(self):
        sig = np.random.default_rng(1).standard_normal((12, 5000)).astype(np.float32)
        out = resample_signal(sig, orig_fs=500.0, target_fs=500.0)
        np.testing.assert_array_equal(out, sig)

    def test_resample_no_nan(self):
        sig = np.random.default_rng(2).standard_normal((12, 10000)).astype(np.float32)
        out = resample_signal(sig, orig_fs=1000.0, target_fs=500.0)
        assert np.isfinite(out).all()


class TestZScore:
    def test_mean_near_zero(self, dummy_ecg_12lead):
        norm, means, stds = z_score_normalize(dummy_ecg_12lead)
        assert np.allclose(norm.mean(axis=1), 0.0, atol=1e-5)

    def test_std_near_one(self, dummy_ecg_12lead):
        norm, means, stds = z_score_normalize(dummy_ecg_12lead)
        assert np.allclose(norm.std(axis=1), 1.0, atol=1e-4)

    def test_means_stds_returned_correctly(self, dummy_ecg_12lead):
        norm, means, stds = z_score_normalize(dummy_ecg_12lead)
        assert means.shape == (12,)
        assert stds.shape == (12,)


class TestFullPreprocessPipeline:
    def test_output_shape(self, dummy_ecg_12lead, cfg):
        result = preprocess_ecg(dummy_ecg_12lead, orig_fs=500.0, cfg_signal=cfg["signal"])
        assert result["waveform"].shape == (12, 5000)
        assert result["waveform_raw"].shape == (12, 5000)

    def test_output_dtype(self, dummy_ecg_12lead, cfg):
        result = preprocess_ecg(dummy_ecg_12lead, orig_fs=500.0, cfg_signal=cfg["signal"])
        assert result["waveform"].dtype == np.float32

    def test_no_nan(self, dummy_ecg_12lead, cfg):
        result = preprocess_ecg(dummy_ecg_12lead, orig_fs=500.0, cfg_signal=cfg["signal"])
        assert np.isfinite(result["waveform"]).all()
        assert np.isfinite(result["waveform_raw"]).all()

    def test_qc_dict_present(self, dummy_ecg_12lead, cfg):
        result = preprocess_ecg(dummy_ecg_12lead, orig_fs=500.0, cfg_signal=cfg["signal"])
        assert "qc" in result
        assert "pass" in result["qc"]
        assert "pp_mv" in result["qc"]
