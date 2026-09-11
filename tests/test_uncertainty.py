"""
tests/test_uncertainty.py
==========================
Unit tests for evaluation/uncertainty.py (MC-Dropout uncertainty).

Reference: Gal & Ghahramani (2016), ICML 2016.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.uncertainty import (
    calibration_plot_data,
    count_dropout_layers,
    coverage_at_confidence,
    mc_dropout_predict,
)
from models.cnn_baseline import CNNBaseline1D
from models.resnet1d import ResNet1D34


BATCH  = 4
DEVICE = "cpu"


def _ecg_batch(batch: int = BATCH) -> torch.Tensor:
    return torch.randn(batch, 12, 5000)


# ---------------------------------------------------------------------------
# Test 1: Output shapes
# ---------------------------------------------------------------------------

class TestOutputShapes:
    def test_shapes_cnn(self):
        model = CNNBaseline1D()
        out   = mc_dropout_predict(model, _ecg_batch(), n_samples=50, device=DEVICE)

        assert out["mean"].shape    == (BATCH,),        f"mean shape wrong: {out['mean'].shape}"
        assert out["std"].shape     == (BATCH,),        f"std shape wrong: {out['std'].shape}"
        assert out["p10"].shape     == (BATCH,),        f"p10 shape wrong: {out['p10'].shape}"
        assert out["p90"].shape     == (BATCH,),        f"p90 shape wrong: {out['p90'].shape}"
        assert out["samples"].shape == (50, BATCH),     f"samples shape wrong: {out['samples'].shape}"


# ---------------------------------------------------------------------------
# Test 2: std > 0 for model with Dropout (CNNBaseline1D has Dropout p=0.5)
# ---------------------------------------------------------------------------

class TestStdPositive:
    def test_std_positive_with_dropout(self):
        model = CNNBaseline1D()   # has Dropout p=0.5 in FC layers
        out   = mc_dropout_predict(model, _ecg_batch(), n_samples=50, device=DEVICE)
        assert out["std"].mean() > 1e-6, (
            f"Expected positive std for Dropout model, got {out['std'].mean():.2e}"
        )

    def test_p10_lt_p90(self):
        model = CNNBaseline1D()
        out   = mc_dropout_predict(model, _ecg_batch(batch=8), n_samples=30)
        assert (out["p10"] <= out["p90"]).all(), "p10 must be <= p90"


# ---------------------------------------------------------------------------
# Test 3: Approximate coverage check
# ---------------------------------------------------------------------------

class TestCoverageAtConfidence:
    def test_coverage_at_1sigma_approximately_correct(self):
        """
        Generate synthetic y_true = mean + z * std * noise (noise ~ N(0,1)).
        For z=1.0, coverage at z=1.0 confidence should be between 0.55 and 0.85
        (the exact 68.3% Gaussian interval, ±slack for finite samples).
        """
        rng       = np.random.default_rng(42)
        N         = 500
        z_noise   = 1.0
        mean_pred = rng.uniform(0.5, 3.0, N)
        std_pred  = rng.uniform(0.1, 0.5, N)
        y_true    = mean_pred + z_noise * std_pred * rng.standard_normal(N)

        cov = coverage_at_confidence(y_true, mean_pred, std_pred, z=z_noise)
        assert 0.55 <= cov <= 0.85, (
            f"Coverage at z=1.0 should be ~68%, got {cov:.3f}"
        )

    def test_high_z_high_coverage(self):
        """At z=3.0 (>99% CI), coverage should be high."""
        rng       = np.random.default_rng(0)
        N         = 200
        mean_pred = rng.uniform(0.5, 3.0, N)
        std_pred  = rng.uniform(0.1, 0.5, N)
        y_true    = mean_pred + std_pred * rng.standard_normal(N)
        cov       = coverage_at_confidence(y_true, mean_pred, std_pred, z=3.0)
        assert cov >= 0.90, f"Expected >= 0.90 coverage at z=3, got {cov:.3f}"


# ---------------------------------------------------------------------------
# Test 4: count_dropout_layers
# ---------------------------------------------------------------------------

class TestCountDropoutLayers:
    def test_cnn_has_one_dropout(self):
        """CNNBaseline1D has one Dropout(0.5) in the FC head."""
        n = count_dropout_layers(CNNBaseline1D())
        assert n == 1, f"Expected 1 Dropout in CNNBaseline1D, got {n}"

    def test_resnet_has_two_dropouts(self):
        """ResNet1D34 has two Dropout layers in the regression head."""
        n = count_dropout_layers(ResNet1D34())
        assert n == 2, f"Expected 2 Dropouts in ResNet1D34, got {n}"

    def test_no_dropout_model(self):
        """A model with no Dropout should return 0."""
        import torch.nn as nn
        tiny = nn.Sequential(nn.Linear(10, 1))
        assert count_dropout_layers(tiny) == 0

    def test_zero_p_dropout_not_counted(self):
        """Dropout(p=0.0) should NOT be counted (not stochastic)."""
        import torch.nn as nn
        model = nn.Sequential(nn.Dropout(p=0.0), nn.Linear(10, 1))
        assert count_dropout_layers(model) == 0


# ---------------------------------------------------------------------------
# Test 5: calibration_plot_data
# ---------------------------------------------------------------------------

class TestCalibrationPlotData:
    def test_output_keys(self):
        rng     = np.random.default_rng(7)
        y_true  = rng.uniform(0.5, 5.0, 50)
        samples = rng.uniform(0.5, 5.0, (30, 50))
        result  = calibration_plot_data(y_true, samples)
        assert "confidence_levels"   in result
        assert "empirical_coverages" in result

    def test_lengths_match(self):
        rng     = np.random.default_rng(7)
        y_true  = rng.uniform(0.5, 5.0, 50)
        samples = rng.uniform(0.5, 5.0, (30, 50))
        result  = calibration_plot_data(y_true, samples)
        assert len(result["confidence_levels"]) == len(result["empirical_coverages"])

    def test_coverages_in_0_1(self):
        rng     = np.random.default_rng(99)
        y_true  = rng.normal(1.0, 0.5, 100)
        samples = rng.normal(1.0, 0.5, (50, 100))
        result  = calibration_plot_data(y_true, samples)
        for c in result["empirical_coverages"]:
            assert 0.0 <= c <= 1.0, f"Coverage {c} out of [0,1]"

    def test_mc_dropout_model_run(self):
        """Full end-to-end: CNNBaseline1D -> mc_dropout_predict -> calibration_plot_data."""
        model   = CNNBaseline1D().eval()
        # run in train mode for stochasticity
        out     = mc_dropout_predict(model, _ecg_batch(), n_samples=20, device=DEVICE)
        y_fake  = out["mean"] + np.random.default_rng(0).normal(0, 0.3, BATCH)
        result  = calibration_plot_data(y_fake, out["samples"])
        assert len(result["confidence_levels"]) == 6

    def test_n_samples_too_small_raises(self):
        model = CNNBaseline1D()
        with pytest.raises(ValueError, match="n_samples"):
            mc_dropout_predict(model, _ecg_batch(), n_samples=1)
