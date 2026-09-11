"""
evaluation/baseline_demographics.py
=====================================
Tabular demographic baselines for ECG-creatinine regression.

Purpose
-------
Mandatory peer-review control benchmarks. These baselines demonstrate whether
12-lead ECG waveforms provide statistically significant incremental prognostic
value over demographic priors alone (addressing the reviewer challenge:
"Is the ECG model just learning sex from waveforms?").

Three baselines (FIX 6 — all fitted on train, evaluated on test)
-----------------------------------------------------------------
1. MeanBaseline            -- predicts training-set mean for every sample.
                             Sets the absolute performance floor.
2. DemographicBaseline     -- closed-form Ridge: [1, age, sex] -> creatinine.
3. DemographicBaselineExtended -- Ridge with [1, age, sex, age^2, age*sex]
                                  tests whether nonlinear demographic
                                  correlations alone explain model performance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np

from evaluation.metrics import ckd_screening_metrics, evaluate_all


# ===========================================================================
# Baseline 1: Population mean
# ===========================================================================

class MeanBaseline:
    """
    Population-mean baseline: predicts the training-set mean creatinine for
    every test sample. Establishes the absolute performance floor.
    Any useful model must outperform this.

    fit(creatinine)  -- stores self.mean_ = float(np.mean(creatinine)).
    predict(n)       -- returns np.full(n, self.mean_, dtype=np.float64).
    """

    def __init__(self) -> None:
        self.mean_: float | None = None

    def fit(self, creatinine: np.ndarray) -> "MeanBaseline":
        """Compute and store the training-set mean creatinine."""
        self.mean_ = float(np.mean(np.asarray(creatinine, dtype=np.float64)))
        return self

    def predict(self, n: int) -> np.ndarray:
        """Return array of length n all equal to self.mean_."""
        if self.mean_ is None:
            raise ValueError("MeanBaseline is not fitted yet. Call fit() first.")
        return np.full(n, self.mean_, dtype=np.float64)


# ===========================================================================
# Baseline 2: Linear demographic baseline  [1, age, sex]
# ===========================================================================

class DemographicBaseline:
    """
    Closed-form Ridge/OLS regression mapping [1, Age, Sex] -> Creatinine.

    Design matrix X: [N, 3]  (intercept, age, sex)
    Closed-form: weights = (X^T X + alpha * I)^{-1} X^T y
    Intercept is not regularised (standard convention).
    """

    def __init__(self, alpha: float = 1e-4) -> None:
        self.alpha = alpha
        self.weights: np.ndarray | None = None

    def fit(
        self,
        age: np.ndarray,
        sex: np.ndarray,
        creatinine: np.ndarray,
    ) -> "DemographicBaseline":
        """Fit linear weights: y = beta_0 + beta_1 * age + beta_2 * sex."""
        age = np.asarray(age, dtype=np.float64).ravel()
        sex = np.asarray(sex, dtype=np.float64).ravel()
        y   = np.asarray(creatinine, dtype=np.float64).ravel()

        X = np.column_stack([np.ones_like(age), age, sex])

        reg_eye       = self.alpha * np.eye(X.shape[1])
        reg_eye[0, 0] = 0.0  # do not regularize intercept
        self.weights  = np.linalg.solve(X.T @ X + reg_eye, X.T @ y)
        return self

    def predict(self, age: np.ndarray, sex: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise ValueError("Model is not fitted yet. Call fit() first.")
        age = np.asarray(age, dtype=np.float64).ravel()
        sex = np.asarray(sex, dtype=np.float64).ravel()
        X   = np.column_stack([np.ones_like(age), age, sex])
        return np.clip(X @ self.weights, 0.1, 25.0)


# ===========================================================================
# Baseline 3: Extended demographic baseline  [1, age, sex, age^2, age*sex]
# ===========================================================================

class DemographicBaselineExtended:
    """
    Extended demographic baseline with quadratic age and age-sex interaction.

    Design matrix X: [N, 5]
        columns: [1, age, sex, age^2, age*sex]

    Tests whether nonlinear demographic correlations alone explain model
    performance, closing the standard reviewer challenge about confounding.

    Closed-form Ridge: weights = (X^T X + alpha * I)^{-1} X^T y
    Intercept is not regularised.
    """

    def __init__(self, alpha: float = 1e-4) -> None:
        self.alpha = alpha
        self.weights: np.ndarray | None = None  # length 5

    def _design_matrix(
        self, age: np.ndarray, sex: np.ndarray
    ) -> np.ndarray:
        """Build the 5-column design matrix."""
        age = np.asarray(age, dtype=np.float64).ravel()
        sex = np.asarray(sex, dtype=np.float64).ravel()
        return np.column_stack([
            np.ones_like(age),
            age,
            sex,
            age ** 2,
            age * sex,
        ])

    def fit(
        self,
        age: np.ndarray,
        sex: np.ndarray,
        creatinine: np.ndarray,
    ) -> "DemographicBaselineExtended":
        """Fit 5-parameter model including age^2 and age*sex interaction."""
        y         = np.asarray(creatinine, dtype=np.float64).ravel()
        X         = self._design_matrix(age, sex)
        reg_eye   = self.alpha * np.eye(X.shape[1])
        reg_eye[0, 0] = 0.0
        self.weights = np.linalg.solve(X.T @ X + reg_eye, X.T @ y)
        return self

    def predict(self, age: np.ndarray, sex: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise ValueError("Model is not fitted yet. Call fit() first.")
        X = self._design_matrix(age, sex)
        return np.clip(X @ self.weights, 0.1, 25.0)


# ===========================================================================
# HDF5 utilities
# ===========================================================================

def extract_split_demographics(
    hdf5_path: str | Path, split: str
) -> dict[str, np.ndarray]:
    """Extract age, sex, and creatinine labels from a HDF5 split group."""
    hdf5_path = Path(hdf5_path)
    ages, sexes, labels = [], [], []

    with h5py.File(hdf5_path, "r") as fh:
        grp = fh[split]
        if isinstance(grp, h5py.Group):
            for sid in grp.keys():
                item = grp[sid]
                if isinstance(item, h5py.Group):
                    label_ds = item["label"]
                    if isinstance(label_ds, h5py.Dataset):
                        labels.append(float(label_ds[()]))
                    attrs = item.attrs
                    ages.append(float(attrs.get("age_years", 50.0)))
                    sexes.append(int(attrs.get("sex", 1)))

    return {
        "age":        np.array(ages,   dtype=np.float64),
        "sex":        np.array(sexes,  dtype=np.int32),
        "creatinine": np.array(labels, dtype=np.float64),
    }


# ===========================================================================
# Master evaluation function (FIX 6: all three baselines)
# ===========================================================================

def evaluate_demographic_baseline(hdf5_path: str | Path) -> dict[str, Any]:
    """
    Fit all three baselines on the train split and evaluate on the test split.

    Returns
    -------
    dict with keys:
        'mean_baseline'                -- MeanBaseline metrics
        'demographic_baseline'         -- DemographicBaseline metrics
        'extended_demographic_baseline'-- DemographicBaselineExtended metrics
    Each value is a dict of MAE, RMSE, R2, Pearson-r, eGFR-MAE, CKD-stage-acc.
    """
    train = extract_split_demographics(hdf5_path, "train")
    test  = extract_split_demographics(hdf5_path, "test")

    n_test = len(test["creatinine"])

    # -- Baseline 1: Mean --------------------------------------------------
    mean_bl = MeanBaseline().fit(train["creatinine"])
    m1 = evaluate_all(
        test["creatinine"], mean_bl.predict(n_test),
        age=test["age"], sex=test["sex"],
    )

    # -- Baseline 2: Linear demographics -----------------------------------
    demo_bl = DemographicBaseline().fit(
        train["age"], train["sex"], train["creatinine"]
    )
    preds2 = demo_bl.predict(test["age"], test["sex"])
    m2 = evaluate_all(
        test["creatinine"], preds2,
        age=test["age"], sex=test["sex"],
    )
    m2["screening"] = ckd_screening_metrics(
        test["creatinine"], preds2, age=test["age"], sex=test["sex"], threshold=60.0
    )
    if demo_bl.weights is not None:
        m2["weights"] = {
            "intercept": float(demo_bl.weights[0]),
            "beta_age":  float(demo_bl.weights[1]),
            "beta_sex":  float(demo_bl.weights[2]),
        }

    # -- Baseline 3: Extended demographics ---------------------------------
    ext_bl = DemographicBaselineExtended().fit(
        train["age"], train["sex"], train["creatinine"]
    )
    preds3 = ext_bl.predict(test["age"], test["sex"])
    m3 = evaluate_all(
        test["creatinine"], preds3,
        age=test["age"], sex=test["sex"],
    )

    return {
        "mean_baseline":                 m1,
        "demographic_baseline":          m2,
        "extended_demographic_baseline": m3,
    }
