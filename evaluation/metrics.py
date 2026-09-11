"""
evaluation/metrics.py
======================
Evaluation metrics for the ECG-to-serum-creatinine regression task.

All formulas are standard regression metrics computed from first principles
using NumPy. PyTorch tensors are accepted and silently converted.

Primary metric   : MAE on serum creatinine (mg/dL)
Secondary        : RMSE, R-squared, Pearson r
Clinical metrics : eGFR-MAE, CKD stage classification accuracy
                   (both derived via egfr_ckdepi_2021 from evaluation/ckd_epi.py)

Clinical grounding
------------------
A creatinine MAE of ~0.2 mg/dL corresponds to a clinically meaningful difference
at the G2/G3a boundary (eGFR ~60 mL/min/1.73m^2, creatinine ~1.0-1.2 mg/dL).
The eGFR-MAE and CKD stage accuracy metrics translate model performance
directly into the language used by nephrologists and primary care physicians.
"""

from __future__ import annotations
from typing import Any

import numpy as np

try:
    import torch as _torch
    _TORCH_AVAILABLE = True
except ImportError:
    _torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

from evaluation.ckd_epi import ckd_stage, egfr_ckdepi_2021


def _to_np(x: Any) -> np.ndarray:
    if _TORCH_AVAILABLE and _torch is not None and isinstance(x, _torch.Tensor):
        return x.detach().cpu().numpy().ravel().astype(np.float64)
    return np.asarray(x, dtype=np.float64).ravel()


# ---------------------------------------------------------------------------
# Core regression metrics
# ---------------------------------------------------------------------------

def mean_absolute_error(y_true: Any, y_pred: Any) -> float:
    """
    Mean Absolute Error.
        MAE = (1/N) * sum |y_true_i - y_pred_i|
    Units: mg/dL (creatinine) or mL/min/1.73m^2 (eGFR).
    """
    return float(np.mean(np.abs(_to_np(y_true) - _to_np(y_pred))))


def root_mean_squared_error(y_true: Any, y_pred: Any) -> float:
    """
    Root Mean Squared Error.
        RMSE = sqrt((1/N) * sum (y_true_i - y_pred_i)^2)
    """
    return float(np.sqrt(np.mean((_to_np(y_true) - _to_np(y_pred)) ** 2)))


def r_squared(y_true: Any, y_pred: Any) -> float:
    """
    Coefficient of determination R^2.
        R^2 = 1 - SS_res / SS_tot
    R^2=1: perfect. R^2=0: predicting the mean. R^2<0: worse than mean.
    """
    yt, yp = _to_np(y_true), _to_np(y_pred)
    ss_tot = np.sum((yt - yt.mean()) ** 2)
    if ss_tot < 1e-12:
        return float("nan")
    return float(1.0 - np.sum((yt - yp) ** 2) / ss_tot)


def pearson_r(y_true: Any, y_pred: Any) -> float:
    """
    Pearson correlation coefficient.
        r = cov(y_true, y_pred) / (std(y_true) * std(y_pred))
    Range: [-1, 1].
    """
    yt, yp = _to_np(y_true), _to_np(y_pred)
    if len(yt) < 2:
        return float("nan")
    return float(np.corrcoef(yt, yp)[0, 1])


# ---------------------------------------------------------------------------
# Clinical metrics
# ---------------------------------------------------------------------------

def egfr_mae(
    creatinine_true: Any,
    creatinine_pred: Any,
    age: Any,
    sex: Any,
) -> float:
    """
    MAE on eGFR derived from predicted creatinine via 2021 CKD-EPI.
    (Inker et al., 2021, NEJM, DOI: 10.1056/NEJMoa2102953)

    Parameters
    ----------
    creatinine_true, creatinine_pred : array-like (mg/dL)
    age : array-like (years)
    sex : array-like (0=female, 1=male)

    Returns
    -------
    float  MAE on eGFR in mL/min/1.73 m^2
    """
    ct = _to_np(creatinine_true)
    cp = _to_np(creatinine_pred)
    a  = _to_np(age).astype(np.float64)
    s  = _to_np(sex).astype(np.int32)
    return mean_absolute_error(egfr_ckdepi_2021(ct, a, s),
                               egfr_ckdepi_2021(cp, a, s))


def ckd_stage_accuracy(
    creatinine_true: Any,
    creatinine_pred: Any,
    age: Any,
    sex: Any,
) -> float:
    """
    Fraction of samples where predicted creatinine maps to the correct
    KDIGO 2012 CKD stage (G1-G5) via the 2021 CKD-EPI eGFR.

    Parameters
    ----------
    creatinine_true, creatinine_pred : array-like (mg/dL)
    age : array-like (years)
    sex : array-like (0=female, 1=male)

    Returns
    -------
    float  Accuracy in [0, 1]
    """
    ct = _to_np(creatinine_true)
    cp = _to_np(creatinine_pred)
    a  = _to_np(age).astype(np.float64)
    s  = _to_np(sex).astype(np.int32)
    stage_t = ckd_stage(egfr_ckdepi_2021(ct, a, s))
    stage_p = ckd_stage(egfr_ckdepi_2021(cp, a, s))
    return float(np.mean(stage_t == stage_p))


def ckd_screening_metrics(
    creatinine_true: Any,
    creatinine_pred: Any,
    age: Any,
    sex: Any,
    threshold: float = 60.0,
) -> dict[str, float]:
    """
    Clinical CKD screening metrics (KDIGO Stage G3+ detection: eGFR < 60 mL/min/1.73m^2).
    Directly implements the clinical evaluation protocol from:
    - Holmstrom et al. (2023) Communications Medicine (DOI: 10.1038/s43856-023-00278-w)
    - Tsai et al. (2025) BMC Medical Informatics (DOI: 10.1186/s12911-025-03278-z)

    Parameters
    ----------
    creatinine_true, creatinine_pred : array-like (mg/dL)
    age : array-like (years)
    sex : array-like (0=female, 1=male)
    threshold : float (default: 60.0 mL/min/1.73m^2)

    Returns
    -------
    dict with keys:
        auroc, sensitivity, specificity, ppv, npv, f1, prevalence
    """
    ct = _to_np(creatinine_true)
    cp = _to_np(creatinine_pred)
    a  = _to_np(age).astype(np.float64)
    s  = _to_np(sex).astype(np.int32)

    egfr_t = egfr_ckdepi_2021(ct, a, s)
    egfr_p = egfr_ckdepi_2021(cp, a, s)

    y_true = np.asarray(egfr_t < threshold, dtype=np.int32)
    y_pred = np.asarray(egfr_p < threshold, dtype=np.int32)

    # Risk score: lower eGFR / higher creatinine corresponds to higher CKD probability
    risk_score = -egfr_p

    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    prevalence = float(np.mean(y_true))

    # AUROC via Mann-Whitney U / rank calculation
    if n_pos > 0 and n_neg > 0:
        try:
            from scipy.stats import rankdata
            ranks = rankdata(risk_score)
            u_stat = np.sum(ranks[y_true == 1]) - n_pos * (n_pos + 1) / 2.0
            auroc = float(u_stat / (n_pos * n_neg))
        except Exception:
            auroc = float("nan")
    else:
        auroc = float("nan")

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))

    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
    ppv  = float(tp / (tp + fp)) if (tp + fp) > 0 else float("nan")
    npv  = float(tn / (tn + fn)) if (tn + fn) > 0 else float("nan")
    f1   = float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else float("nan")

    return {
        "auroc": auroc,
        "sensitivity": sens,
        "specificity": spec,
        "ppv": ppv,
        "npv": npv,
        "f1": f1,
        "prevalence": prevalence,
    }


# ---------------------------------------------------------------------------
# Comprehensive evaluation (single call for all metrics)
# ---------------------------------------------------------------------------

def evaluate_all(
    creatinine_true: Any,
    creatinine_pred: Any,
    age: Any | None = None,
    sex: Any | None = None,
) -> dict[str, Any]:
    """
    Compute all regression and clinical metrics in one call.

    Parameters
    ----------
    creatinine_true, creatinine_pred : array-like (mg/dL)
    age, sex : array-like or None.
        If None, eGFR-based metrics are skipped.

    Returns
    -------
    dict with keys:
        mae, rmse, r2, pearson_r
        egfr_mae, ckd_stage_acc, ckd_screening_auroc, ckd_screening_sens, ...
        (only when age and sex provided)
    """
    out: dict[str, float] = {
        "mae":       mean_absolute_error(creatinine_true, creatinine_pred),
        "rmse":      root_mean_squared_error(creatinine_true, creatinine_pred),
        "r2":        r_squared(creatinine_true, creatinine_pred),
        "pearson_r": pearson_r(creatinine_true, creatinine_pred),
    }
    if age is not None and sex is not None:
        out["egfr_mae"]      = egfr_mae(creatinine_true, creatinine_pred, age, sex)
        out["ckd_stage_acc"] = ckd_stage_accuracy(
            creatinine_true, creatinine_pred, age, sex)
        screening = ckd_screening_metrics(creatinine_true, creatinine_pred, age, sex)
        out["ckd_screening_auroc"] = screening["auroc"]
        out["ckd_screening_sens"]  = screening["sensitivity"]
        out["ckd_screening_spec"]  = screening["specificity"]
        out["ckd_screening_f1"]    = screening["f1"]
    return out


# ---------------------------------------------------------------------------
# Clinical agreement & stratified reporting
# ---------------------------------------------------------------------------

def bland_altman_limits(
    y_true: Any,
    y_pred: Any,
) -> dict[str, float]:
    """
    Compute Bland-Altman mean bias and 95% limits of agreement.

    d_i = y_pred_i - y_true_i
    Limits = mean(d) ± 1.96 * SD(d)

    Parameters
    ----------
    y_true, y_pred : array-like

    Returns
    -------
    dict with keys:
        'mean_bias', 'std_diff', 'lower_limit', 'upper_limit'
    """
    yt, yp = _to_np(y_true), _to_np(y_pred)
    diff = yp - yt
    mean_bias = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0
    return {
        "mean_bias":   mean_bias,
        "std_diff":    std_diff,
        "lower_limit": mean_bias - 1.96 * std_diff,
        "upper_limit": mean_bias + 1.96 * std_diff,
    }


def stratified_metrics_by_range(
    creatinine_true: Any,
    creatinine_pred: Any,
    bins: list[float] | None = None,
    labels: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """
    Stratify regression metrics across clinical creatinine bands.

    Essential because overall MAE can mask severe under-prediction in high-creatinine
    pathology due to right-skewed population distributions.

    Default strata:
      - Normal          : < 1.2 mg/dL
      - Mild-Moderate   : 1.2 - 2.0 mg/dL
      - Moderate-Severe : > 2.0 mg/dL

    Parameters
    ----------
    creatinine_true, creatinine_pred : array-like (mg/dL)
    bins : list of float, optional (default: [0.0, 1.2, 2.0, inf])
    labels : list of str, optional

    Returns
    -------
    dict of {stratum_label: {'count': N, 'mae': MAE, 'rmse': RMSE, 'mean_bias': Bias}}
    """
    yt, yp = _to_np(creatinine_true), _to_np(creatinine_pred)
    if bins is None:
        bins = [0.0, 1.2, 2.0, float("inf")]
    if labels is None:
        labels = ["Normal (<1.2)", "Mild (1.2-2.0)", "Severe (>2.0)"]

    results: dict[str, dict[str, float]] = {}
    for i in range(len(bins) - 1):
        low, high = bins[i], bins[i + 1]
        name = labels[i]
        mask = (yt >= low) & (yt < high)
        n_stratum = int(np.sum(mask))
        if n_stratum == 0:
            results[name] = {
                "count": 0.0,
                "mae": float("nan"),
                "rmse": float("nan"),
                "mean_bias": float("nan"),
            }
        else:
            sub_yt, sub_yp = yt[mask], yp[mask]
            results[name] = {
                "count": float(n_stratum),
                "mae": mean_absolute_error(sub_yt, sub_yp),
                "rmse": root_mean_squared_error(sub_yt, sub_yp),
                "mean_bias": float(np.mean(sub_yp - sub_yt)),
            }
    return results


def patient_clustered_bootstrap_ci(
    y_true: Any,
    y_pred: Any,
    subject_ids: Any,
    metric_fn: Any = mean_absolute_error,
    n_bootstraps: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, float]:
    """
    Compute patient-clustered bootstrap confidence intervals.

    Resampling clusters by subject_id preserves patient-level clustering
    and autocorrelation when patients contribute multiple ECG records.

    Parameters
    ----------
    y_true, y_pred : array-like (N,)
    subject_ids : array-like (N,) unique patient identifiers
    metric_fn : callable f(y_true_sub, y_pred_sub) -> float
    n_bootstraps : int (default: 1000)
    alpha : float (default: 0.05 for 95% CI)
    seed : int

    Returns
    -------
    dict with keys:
        'point_estimate', 'ci_lower', 'ci_upper', 'confidence_level'
    """
    yt = _to_np(y_true)
    yp = _to_np(y_pred)
    sids = np.asarray(subject_ids)
    if len(yt) != len(sids) or len(yp) != len(sids):
        raise ValueError("y_true, y_pred, and subject_ids must have identical length.")

    point_estimate = float(metric_fn(yt, yp))
    unique_sids = np.unique(sids)
    rng = np.random.default_rng(seed)

    # Pre-index records per subject
    subj_indices: dict[Any, np.ndarray] = {
        sid: np.where(sids == sid)[0] for sid in unique_sids
    }

    boot_vals = np.empty(n_bootstraps, dtype=np.float64)
    n_subjects = len(unique_sids)

    for b in range(n_bootstraps):
        sampled_sids = rng.choice(unique_sids, size=n_subjects, replace=True)
        sampled_idx_list = [subj_indices[sid] for sid in sampled_sids]
        sampled_indices = np.concatenate(sampled_idx_list) if sampled_idx_list else np.array([], dtype=int)
        if len(sampled_indices) == 0:
            boot_vals[b] = point_estimate
        else:
            boot_vals[b] = float(metric_fn(yt[sampled_indices], yp[sampled_indices]))

    ci_lower = float(np.percentile(boot_vals, 100.0 * (alpha / 2.0)))
    ci_upper = float(np.percentile(boot_vals, 100.0 * (1.0 - alpha / 2.0)))

    return {
        "point_estimate":   point_estimate,
        "ci_lower":         ci_lower,
        "ci_upper":         ci_upper,
        "confidence_level": 1.0 - alpha,
    }
