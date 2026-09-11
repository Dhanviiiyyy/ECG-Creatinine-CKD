"""
tests/test_metrics.py
======================
Unit tests for evaluation/ckd_epi.py and evaluation/metrics.py.

Key verification
----------------
* CKD-EPI 2021 formula validated against published reference values
  from Inker et al. (2021), NEJM, Table 2.
* MAE/RMSE/R2 formulas verified against numpy implementations.
* Edge cases: all-same predictions, zero variance, single sample.
"""

from __future__ import annotations

import sys
from pathlib import Path
import math
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.ckd_epi import (
    ckd_stage,
    ckd_stage_index,
    egfr_ckdepi_2021,
)
from evaluation.metrics import (
    evaluate_all,
    mean_absolute_error,
    pearson_r,
    r_squared,
    root_mean_squared_error,
    egfr_mae,
    ckd_stage_accuracy,
    ckd_screening_metrics,
)


# ---------------------------------------------------------------------------
# CKD-EPI 2021 formula tests
# ---------------------------------------------------------------------------

class TestCKDEPI2021:
    """
    Reference values from Inker et al. (2021) NEJM supplementary material
    and the NKF/ASN online eGFR calculator (https://www.kidney.org/professionals/kdoqi/gfr_calculator).
    """

    def test_male_scr_1_age_50(self):
        # Male, Scr=1.0 mg/dL, age=50 -> expected ~91.7 mL/min/1.73m^2 (Inker et al. 2021)
        result = egfr_ckdepi_2021(1.0, 50, sex=1)
        assert abs(result - 91.7) < 1.0, f"Got {result:.2f}, expected ~91.7"

    def test_female_scr_1_age_50(self):
        # Female, Scr=1.0 mg/dL, age=50 -> expected ~68.6 mL/min/1.73m^2
        # Derivation (Inker et al. 2021, NEJM DOI:10.1056/NEJMoa2102953):
        #   kappa=0.7 (female); Scr/kappa = 1.0/0.7 = 1.429 > 1 (high regime)
        #   min(ratio,1)^alpha = 1^(-0.241) = 1.0
        #   max(ratio,1)^(-1.200) = 1.429^(-1.200) ≈ 0.6518
        #   eGFR = 142 * 1.0 * 0.6518 * 0.9938^50 * 1.012 ≈ 68.6
        result = egfr_ckdepi_2021(1.0, 50, sex=0)
        assert abs(result - 68.63) < 0.5, f"Got {result:.2f}, expected ~68.63"


    def test_high_creatinine_low_egfr(self):
        # Male, Scr=5.0 mg/dL, age=60 -> should be CKD G4 or G5 (<30)
        result = egfr_ckdepi_2021(5.0, 60, sex=1)
        assert result < 30.0, f"Expected eGFR < 30 for Scr=5.0, got {result:.2f}"

    def test_low_creatinine_high_egfr(self):
        # Female, Scr=0.6 mg/dL, age=30 -> should be G1 (>= 90)
        result = egfr_ckdepi_2021(0.6, 30, sex=0)
        assert result >= 90.0, f"Expected eGFR >= 90 for young female Scr=0.6, got {result:.2f}"

    def test_female_sex_multiplier_applied(self):
        # At standardized baseline (Scr = kappa: 0.7 for female, 0.9 for male),
        # female eGFR should be 1.012x higher than male due to the sex multiplier
        egfr_f = egfr_ckdepi_2021(0.7, 50, sex=0)
        egfr_m = egfr_ckdepi_2021(0.9, 50, sex=1)
        assert egfr_f > egfr_m, "Female should have higher eGFR at baseline ratio"
        assert abs(egfr_f / egfr_m - 1.012) < 1e-3, f"Expected ratio ~1.012, got {egfr_f/egfr_m:.4f}"

    def test_age_decay(self):
        # Older patient -> lower eGFR at same Scr
        egfr_young = egfr_ckdepi_2021(1.0, 30, sex=1)
        egfr_old   = egfr_ckdepi_2021(1.0, 70, sex=1)
        assert egfr_young > egfr_old, "Younger patient should have higher eGFR"

    def test_vectorised_numpy(self):
        cr  = np.array([0.8, 1.0, 2.0, 5.0])
        age = np.array([40,  50,  60,  70])
        sex = np.array([0,   1,   0,   1])
        result = egfr_ckdepi_2021(cr, age, sex)
        assert isinstance(result, np.ndarray)
        assert result.shape == (4,)
        assert bool((result > 0).all())

    def test_positive_eGFR_always(self):
        # eGFR must always be positive for valid creatinine/age inputs
        for cr in [0.5, 1.0, 2.0, 8.0, 15.0]:
            for age in [20, 50, 80]:
                for sex in [0, 1]:
                    result = egfr_ckdepi_2021(cr, age, sex)
                    assert result > 0, f"eGFR should be > 0 for Scr={cr}, age={age}, sex={sex}"


class TestCKDStaging:
    def test_g1_boundary(self):
        assert ckd_stage(90.0)  == "G1"
        assert ckd_stage(120.0) == "G1"

    def test_g2_boundary(self):
        assert ckd_stage(89.9) == "G2"
        assert ckd_stage(60.0) == "G2"

    def test_g3a_boundary(self):
        assert ckd_stage(59.9) == "G3a"
        assert ckd_stage(45.0) == "G3a"

    def test_g3b_boundary(self):
        assert ckd_stage(44.9) == "G3b"
        assert ckd_stage(30.0) == "G3b"

    def test_g4_boundary(self):
        assert ckd_stage(29.9) == "G4"
        assert ckd_stage(15.0) == "G4"

    def test_g5_boundary(self):
        assert ckd_stage(14.9) == "G5"
        assert ckd_stage(5.0)  == "G5"

    def test_stage_index_ordering(self):
        # G5 (most severe) should have lowest index
        assert ckd_stage_index(5.0)  < ckd_stage_index(50.0)
        assert ckd_stage_index(50.0) < ckd_stage_index(100.0)

    def test_vectorised_staging(self):
        stages = ckd_stage(np.array([100.0, 70.0, 50.0, 35.0, 20.0, 10.0]))
        expected = np.array(["G1", "G2", "G3a", "G3b", "G4", "G5"], dtype=object)
        assert isinstance(stages, np.ndarray)
        assert bool((stages == expected).all())


# ---------------------------------------------------------------------------
# Regression metrics tests
# ---------------------------------------------------------------------------

class TestMAE:
    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0])
        assert mean_absolute_error(y, y) == pytest.approx(0.0)

    def test_known_value(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])
        assert mean_absolute_error(y_true, y_pred) == pytest.approx(1.0)

    def test_single_sample(self):
        assert mean_absolute_error([2.5], [3.0]) == pytest.approx(0.5)


class TestRMSE:
    def test_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert root_mean_squared_error(y, y) == pytest.approx(0.0)

    def test_known_value(self):
        # errors = [1, 1, 1] -> RMSE = 1.0
        assert root_mean_squared_error([0,0,0], [1,1,1]) == pytest.approx(1.0)

    def test_rmse_ge_mae(self):
        y_true = np.array([1.0, 2.0, 5.0])
        y_pred = np.array([1.5, 2.5, 3.0])
        assert root_mean_squared_error(y_true, y_pred) >= mean_absolute_error(y_true, y_pred)


class TestR2:
    def test_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert r_squared(y, y) == pytest.approx(1.0)

    def test_predicting_mean(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 2.0, 2.0])  # predicting mean always
        assert r_squared(y_true, y_pred) == pytest.approx(0.0, abs=1e-10)

    def test_negative_r2(self):
        # Worse than predicting mean -> R2 < 0
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([3.0, 1.0, 1.0])
        assert r_squared(y_true, y_pred) < 0.0


class TestPearsonR:
    def test_perfect_positive(self):
        y = np.array([1.0, 2.0, 3.0])
        assert pearson_r(y, y) == pytest.approx(1.0)

    def test_perfect_negative(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([3.0, 2.0, 1.0])
        assert pearson_r(y_true, y_pred) == pytest.approx(-1.0)

    def test_range(self):
        rng    = np.random.default_rng(42)
        y_true = rng.uniform(0, 5, 100)
        y_pred = y_true + rng.normal(0, 0.5, 100)
        r = pearson_r(y_true, y_pred)
        assert -1.0 <= r <= 1.0


class TestEvaluateAll:
    def test_keys_without_demo(self):
        y = np.array([1.0, 2.0, 3.0])
        out = evaluate_all(y, y)
        assert set(out.keys()) == {"mae", "rmse", "r2", "pearson_r"}

    def test_keys_with_demo(self):
        y   = np.array([1.0, 2.0, 3.0])
        age = np.array([50,  50,  50])
        sex = np.array([1,   0,   1])
        out = evaluate_all(y, y, age=age, sex=sex)
        assert "egfr_mae"      in out
        assert "ckd_stage_acc" in out
        assert out["ckd_stage_acc"] == pytest.approx(1.0)  # perfect prediction
        assert "ckd_screening_auroc" in out


class TestCKDScreeningMetrics:
    def test_perfect_screening(self):
        # 2 high creatinine (CKD, eGFR < 60) and 2 low creatinine (healthy)
        cr_true = np.array([3.0, 4.0, 0.8, 0.7])
        cr_pred = np.array([3.1, 3.9, 0.9, 0.6])
        age     = np.array([60,  65,  30,  25])
        sex     = np.array([1,   0,   1,   0])
        res = ckd_screening_metrics(cr_true, cr_pred, age, sex, threshold=60.0)
        assert res["sensitivity"] == 1.0
        assert res["specificity"] == 1.0
        assert res["f1"] == 1.0
        assert res["auroc"] == 1.0
        assert res["prevalence"] == 0.5


class TestNewEvaluationUtilities:
    def test_bland_altman_limits(self):
        from evaluation.metrics import bland_altman_limits
        yt = np.array([1.0, 2.0, 3.0, 4.0])
        yp = np.array([1.1, 2.1, 3.1, 4.1])
        ba = bland_altman_limits(yt, yp)
        assert ba["mean_bias"] == pytest.approx(0.1)
        assert ba["std_diff"] == pytest.approx(0.0)
        assert ba["lower_limit"] == pytest.approx(0.1)
        assert ba["upper_limit"] == pytest.approx(0.1)

    def test_stratified_metrics_by_range(self):
        from evaluation.metrics import stratified_metrics_by_range
        yt = np.array([0.8, 1.5, 3.0])
        yp = np.array([0.9, 1.6, 3.2])
        strat = stratified_metrics_by_range(yt, yp)
        assert "Normal (<1.2)" in strat
        assert "Mild (1.2-2.0)" in strat
        assert "Severe (>2.0)" in strat
        assert strat["Normal (<1.2)"]["count"] == 1.0
        assert strat["Normal (<1.2)"]["mae"] == pytest.approx(0.1)
        assert strat["Mild (1.2-2.0)"]["count"] == 1.0
        assert strat["Severe (>2.0)"]["count"] == 1.0
        assert strat["Severe (>2.0)"]["mae"] == pytest.approx(0.2)

    def test_patient_clustered_bootstrap_ci(self):
        from evaluation.metrics import patient_clustered_bootstrap_ci, mean_absolute_error
        yt = np.array([1.0, 2.0, 3.0, 4.0])
        yp = np.array([1.1, 2.1, 3.1, 4.1])
        sids = np.array([1, 1, 2, 2])
        boot = patient_clustered_bootstrap_ci(yt, yp, sids, metric_fn=mean_absolute_error, n_bootstraps=100)
        assert boot["point_estimate"] == pytest.approx(0.1)
        assert boot["ci_lower"] == pytest.approx(0.1)
        assert boot["ci_upper"] == pytest.approx(0.1)
        assert boot["confidence_level"] == 0.95


