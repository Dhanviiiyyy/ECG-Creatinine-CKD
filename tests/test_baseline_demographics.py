"""
tests/test_baseline_demographics.py
=====================================
Unit tests for evaluation/baseline_demographics.py.
Tests cover all three baselines: MeanBaseline, DemographicBaseline,
and DemographicBaselineExtended.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.baseline_demographics import (
    DemographicBaseline,
    DemographicBaselineExtended,
    MeanBaseline,
)


# ===========================================================================
# Original tests (preserved unchanged)
# ===========================================================================

def test_demographic_baseline_fit_predict():
    age = np.array([20, 40, 60, 80], dtype=np.float64)
    sex = np.array([0,  1,  0,  1], dtype=np.int32)
    # y = 0.5 + 0.01 * age + 0.2 * sex
    y   = 0.5 + 0.01 * age + 0.2 * sex

    model = DemographicBaseline()
    model.fit(age, sex, y)

    preds = model.predict(age, sex)
    assert np.allclose(preds, y, atol=1e-3)


def test_demographic_baseline_not_fitted():
    model = DemographicBaseline()
    with pytest.raises(ValueError, match="not fitted"):
        model.predict(np.array([50]), np.array([1]))


# ===========================================================================
# FIX 6: New tests for MeanBaseline and DemographicBaselineExtended
# ===========================================================================

def test_mean_baseline_constant_prediction():
    """MeanBaseline must return the training mean for every test sample."""
    train_cr = np.array([0.8, 1.0, 1.2, 1.5, 2.0], dtype=np.float64)
    expected_mean = float(np.mean(train_cr))

    model = MeanBaseline()
    model.fit(train_cr)

    preds = model.predict(n=10)
    assert preds.shape == (10,), "predict must return array of length n"
    assert (preds == expected_mean).all(), (
        f"All predictions must equal training mean {expected_mean:.4f}"
    )


def test_mean_baseline_not_fitted_raises():
    model = MeanBaseline()
    with pytest.raises(ValueError, match="not fitted"):
        model.predict(5)


def test_extended_baseline_weight_count():
    """DemographicBaselineExtended must have exactly 5 weights."""
    age = np.linspace(20, 80, 50)
    sex = np.random.RandomState(0).randint(0, 2, 50).astype(float)
    y   = 0.5 + 0.01 * age + 0.2 * sex + 0.0001 * age ** 2

    model = DemographicBaselineExtended()
    model.fit(age, sex, y)

    assert model.weights is not None
    assert len(model.weights) == 5, (
        f"Extended baseline must have 5 weights, got {len(model.weights)}"
    )


def test_extended_baseline_better_than_linear_on_quadratic():
    """Extended baseline should fit quadratic age relationship better than linear."""
    rng = np.random.default_rng(42)
    age = rng.uniform(20, 80, 200)
    sex = rng.integers(0, 2, 200).astype(float)
    # True relationship has a clear age^2 term
    y   = 0.3 + 0.005 * age + 0.0002 * age**2 + 0.1 * sex + rng.normal(0, 0.05, 200)

    lin_bl = DemographicBaseline().fit(age, sex, y)
    ext_bl = DemographicBaselineExtended().fit(age, sex, y)

    # On the same training data, extended should have lower or equal residuals
    err_lin = np.mean((lin_bl.predict(age, sex) - y) ** 2)
    err_ext = np.mean((ext_bl.predict(age, sex) - y) ** 2)
    assert err_ext <= err_lin + 1e-6, (
        f"Extended baseline (MSE={err_ext:.6f}) should be <= linear (MSE={err_lin:.6f})"
    )


def test_extended_baseline_not_fitted_raises():
    model = DemographicBaselineExtended()
    with pytest.raises(ValueError, match="not fitted"):
        model.predict(np.array([50.0]), np.array([1.0]))
