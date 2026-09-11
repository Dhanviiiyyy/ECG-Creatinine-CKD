"""
evaluation/ckd_epi.py
======================
Race-free 2021 CKD-EPI creatinine equation -- implemented verbatim from the
primary publication.

Reference (MANDATORY CITATION)
--------------------------------
Inker, L.A. et al. (2021). New Creatinine- and Cystatin C-Based Equations to
Estimate GFR without Race. New England Journal of Medicine, 385(19), 1737-1749.
DOI: 10.1056/NEJMoa2102953

Exact equation (Table 1 of the above paper)
--------------------------------------------
eGFR = 142 x min(Scr/kappa, 1)^alpha x max(Scr/kappa, 1)^(-1.200)
        x 0.9938^Age x [1.012 if female]

Demographic coefficients
------------------------
  Female : kappa = 0.7,  alpha = -0.241
  Male   : kappa = 0.9,  alpha = -0.302

Units
-----
  Scr  : serum creatinine (mg/dL)
  Age  : years
  eGFR : mL/min/1.73 m^2

CKD Staging (KDIGO 2012 thresholds)
-------------------------------------
  G1  : eGFR >= 90       (normal or high)
  G2  : 60 <= eGFR < 90  (mildly decreased)
  G3a : 45 <= eGFR < 60  (mild-moderately decreased)
  G3b : 30 <= eGFR < 45  (moderately-severely decreased)
  G4  : 15 <= eGFR < 30  (severely decreased)
  G5  : eGFR < 15        (kidney failure)
"""

from __future__ import annotations
import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants -- directly from Inker et al. (2021), Table 1
# ---------------------------------------------------------------------------
_KAPPA_FEMALE  = 0.7
_KAPPA_MALE    = 0.9
_ALPHA_FEMALE  = -0.241
_ALPHA_MALE    = -0.302
_BASE_COEFF    = 142.0
_AGE_DECAY     = 0.9938
_SEX_FACTOR_F  = 1.012   # female multiplier
_EXPONENT_HIGH = -1.200  # exponent when Scr/kappa > 1

# CKD stage thresholds: (name, lo_inclusive, hi_exclusive)
_CKD_STAGE_THRESHOLDS = [
    ("G5",  0.0,  15.0),
    ("G4",  15.0, 30.0),
    ("G3b", 30.0, 45.0),
    ("G3a", 45.0, 60.0),
    ("G2",  60.0, 90.0),
    ("G1",  90.0, float("inf")),
]
_STAGE_TO_IDX = {"G5": 0, "G4": 1, "G3b": 2, "G3a": 3, "G2": 4, "G1": 5}


# ---------------------------------------------------------------------------
# NumPy implementation  (primary -- used by metrics.py and evaluation scripts)
# ---------------------------------------------------------------------------

def egfr_ckdepi_2021(
    creatinine: float | np.ndarray,
    age:        float | np.ndarray,
    sex:        int   | np.ndarray,  # 0 = female, 1 = male
) -> float | np.ndarray:
    """
    Compute eGFR from serum creatinine using the race-free 2021 CKD-EPI equation.

    Implements verbatim (Inker et al., 2021, NEJM, DOI: 10.1056/NEJMoa2102953):
        eGFR = 142 x min(Scr/kappa, 1)^alpha
                   x max(Scr/kappa, 1)^(-1.200)
                   x 0.9938^Age
                   x [1.012 if female]

    Parameters
    ----------
    creatinine : float or np.ndarray
        Serum creatinine in mg/dL. Must be > 0.
    age : float or np.ndarray
        Patient age in years.
    sex : int or np.ndarray
        0 = female, 1 = male.

    Returns
    -------
    float or np.ndarray  eGFR in mL/min/1.73 m^2.

    Examples
    --------
    >>> round(egfr_ckdepi_2021(1.0, 50, 1), 2)   # male, 50 y, Scr=1.0
    91.69
    >>> round(egfr_ckdepi_2021(1.0, 50, 0), 2)   # female, 50 y, Scr=1.0
    68.63
    # Note: female kappa=0.7, Scr/kappa=1.43 > 1 -> high-creatinine regime
    """
    cr  = np.asarray(creatinine, dtype=np.float64)
    # Numerical & physiological guard: serum creatinine must be strictly positive
    cr  = np.clip(cr, a_min=1e-4, a_max=None)
    age = np.asarray(age,        dtype=np.float64)
    sex = np.asarray(sex,        dtype=np.int32)

    kappa = np.where(sex == 0, _KAPPA_FEMALE, _KAPPA_MALE)
    alpha = np.where(sex == 0, _ALPHA_FEMALE, _ALPHA_MALE)

    ratio     = cr / kappa
    low_term  = np.minimum(ratio, 1.0) ** alpha
    high_term = np.maximum(ratio, 1.0) ** _EXPONENT_HIGH

    egfr = _BASE_COEFF * low_term * high_term * (_AGE_DECAY ** age)
    egfr = egfr * np.where(sex == 0, _SEX_FACTOR_F, 1.0)

    return float(egfr) if egfr.ndim == 0 else egfr


# ---------------------------------------------------------------------------
# PyTorch implementation  (differentiable -- for optional eGFR-based loss)
# ---------------------------------------------------------------------------

def egfr_ckdepi_2021_torch(
    creatinine: "torch.Tensor",
    age:        "torch.Tensor",
    sex:        "torch.Tensor",  # float tensor: 0.0=female, 1.0=male
) -> "torch.Tensor":
    """
    Differentiable PyTorch version of the race-free 2021 CKD-EPI equation.

    Uses torch.clamp instead of np.minimum/maximum so gradients flow through
    creatinine. Suitable for an auxiliary eGFR supervision loss.

    Parameters
    ----------
    creatinine : torch.Tensor  shape (B,)  float32, mg/dL
    age        : torch.Tensor  shape (B,)  float32, years
    sex        : torch.Tensor  shape (B,)  float32, 0.0=female, 1.0=male

    Returns
    -------
    torch.Tensor  shape (B,)  eGFR in mL/min/1.73 m^2
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for egfr_ckdepi_2021_torch.")
    is_female = (sex == 0.0).float()
    kappa = is_female * _KAPPA_FEMALE + (1.0 - is_female) * _KAPPA_MALE
    alpha = is_female * _ALPHA_FEMALE + (1.0 - is_female) * _ALPHA_MALE

    # Numerical & physiological guard: serum creatinine must be strictly positive
    cr_safe = torch.clamp(creatinine.float(), min=1e-4)
    ratio   = cr_safe / kappa
    low_term  = torch.clamp(ratio, max=1.0) ** alpha
    high_term = torch.clamp(ratio, min=1.0) ** _EXPONENT_HIGH

    egfr = _BASE_COEFF * low_term * high_term * (_AGE_DECAY ** age)
    return egfr * (is_female * _SEX_FACTOR_F + (1.0 - is_female) * 1.0)


# ---------------------------------------------------------------------------
# CKD staging utilities
# ---------------------------------------------------------------------------

def ckd_stage(egfr: float | np.ndarray) -> str | np.ndarray:
    """
    Assign KDIGO 2012 CKD stage label from eGFR value(s).

    Parameters
    ----------
    egfr : float or np.ndarray  (mL/min/1.73 m^2)

    Returns
    -------
    str or np.ndarray of str  e.g. 'G2', 'G3a'
    """
    arr    = np.atleast_1d(np.asarray(egfr, dtype=np.float64))
    stages = np.empty(arr.shape, dtype=object)
    for name, lo, hi in _CKD_STAGE_THRESHOLDS:
        stages[(arr >= lo) & (arr < hi)] = name
    return str(stages[0]) if np.asarray(egfr).ndim == 0 else stages


def ckd_stage_index(egfr: float | np.ndarray) -> int | np.ndarray:
    """
    Map eGFR -> integer severity index (G5=0 most severe, G1=5 normal).
    Useful for confusion-matrix plotting and ordinal metrics.
    """
    s = ckd_stage(egfr)
    if isinstance(s, str):
        return _STAGE_TO_IDX[s]
    return np.vectorize(_STAGE_TO_IDX.get)(s).astype(int)
