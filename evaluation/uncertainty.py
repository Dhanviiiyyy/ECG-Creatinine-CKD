"""
evaluation/uncertainty.py
==========================
Monte Carlo Dropout (MC-Dropout) uncertainty quantification for ECG-creatinine
regression models.

Reference
----------
Gal, Y. and Ghahramani, Z. (2016). Dropout as a Bayesian Approximation:
Representing Model Uncertainty in Deep Learning. ICML 2016.
https://proceedings.mlr.press/v48/gal16.html

Method
------
By keeping nn.Dropout layers active at test time (model.train() mode) and
running N stochastic forward passes, we obtain a Monte Carlo approximation
to the posterior predictive distribution:

    p(y* | x*, X, y) ≈ (1/T) * Σ_t p(y* | x*, W_t)

where W_t ~ q*(W) (dropout approximate posterior) for t = 1, ..., T.

This yields:
    mean prediction  : E[y*] ≈ (1/T) Σ_t y*_t
    uncertainty      : Var[y*] ≈ (1/T) Σ_t (y*_t - mean)^2  (epistemic)

Clinical note: For a calibrated model at z=1.645 (90% normal CI),
coverage_at_confidence should return ≈ 0.90 on held-out test data.
Systematic deviation indicates model miscalibration.

Requires: at least one nn.Dropout layer with p > 0 in the model.
For models without Dropout, std will be near-zero -- use an ensemble
or deep ensembles method instead.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Core MC-Dropout prediction
# ---------------------------------------------------------------------------

def mc_dropout_predict(
    model:          nn.Module,
    waveform_batch: torch.Tensor,
    n_samples:      int = 50,
    device:         str | torch.device = "cpu",
) -> dict[str, np.ndarray]:
    """
    Run N stochastic forward passes with Dropout active.

    Parameters
    ----------
    model          : nn.Module  (must contain at least one nn.Dropout layer)
    waveform_batch : torch.Tensor  [B, 12, 5000] float32
    n_samples      : int  Number of MC samples. Default 50.
    device         : str or torch.device

    Returns
    -------
    dict with keys:
        'mean'    : np.ndarray [B]   mean prediction
        'std'     : np.ndarray [B]   standard deviation (epistemic uncertainty)
        'p10'     : np.ndarray [B]   10th percentile
        'p90'     : np.ndarray [B]   90th percentile
        'samples' : np.ndarray [n_samples, B]  all raw MC samples
    """
    if n_samples < 2:
        raise ValueError(f"n_samples must be >= 2, got {n_samples}.")

    device    = torch.device(device)
    was_train = model.training

    # Activate stochastic Dropout (CRITICAL: do not call model.eval())
    model.train()
    x = waveform_batch.to(device)

    all_preds: list[np.ndarray] = []
    with torch.no_grad():
        for _ in range(n_samples):
            out = model(x)                # [B, 1]
            all_preds.append(
                out.squeeze(-1).cpu().numpy().astype(np.float64)
            )

    # Restore original training mode
    model.train(was_train)

    samples = np.stack(all_preds, axis=0)  # [n_samples, B]

    return {
        "mean":    samples.mean(axis=0),
        "std":     samples.std(axis=0),
        "p10":     np.percentile(samples, 10, axis=0),
        "p90":     np.percentile(samples, 90, axis=0),
        "samples": samples,
    }


# ---------------------------------------------------------------------------
# Calibration utilities
# ---------------------------------------------------------------------------

def coverage_at_confidence(
    y_true:    np.ndarray,
    mean_pred: np.ndarray,
    std_pred:  np.ndarray,
    z:         float = 1.645,
) -> float:
    """
    Empirical coverage at a given confidence z-score.

    For a calibrated Gaussian predictive distribution, z=1.645 corresponds
    to a 90% confidence interval. Empirical coverage should be ≈ 0.90.

    Parameters
    ----------
    y_true    : np.ndarray [N]  ground-truth creatinine (mg/dL)
    mean_pred : np.ndarray [N]  MC-Dropout mean predictions
    std_pred  : np.ndarray [N]  MC-Dropout standard deviations
    z         : float  Normal quantile. Default 1.645 (90% CI).

    Returns
    -------
    float  Fraction of samples where y_true ∈ [mean - z*std, mean + z*std].
    """
    y_true    = np.asarray(y_true,    dtype=np.float64).ravel()
    mean_pred = np.asarray(mean_pred, dtype=np.float64).ravel()
    std_pred  = np.asarray(std_pred,  dtype=np.float64).ravel()

    lower = mean_pred - z * std_pred
    upper = mean_pred + z * std_pred
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def calibration_plot_data(
    y_true:  np.ndarray,
    samples: np.ndarray,
) -> dict[str, list]:
    """
    Compute empirical coverage across multiple confidence levels.

    Useful for plotting reliability diagrams (expected vs empirical coverage).

    Parameters
    ----------
    y_true  : np.ndarray [N]            ground-truth values
    samples : np.ndarray [n_samples, N] MC-Dropout sample matrix

    Returns
    -------
    dict with keys:
        'confidence_levels'   : list of float [0.50, 0.60, ..., 0.95]
        'empirical_coverages' : list of float  empirical fraction covered at each level
    """
    confidence_levels = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
    empirical = []

    y_true  = np.asarray(y_true, dtype=np.float64).ravel()
    samples = np.asarray(samples, dtype=np.float64)  # [T, N]

    for conf in confidence_levels:
        alpha = 1.0 - conf
        lower = np.percentile(samples, 100 * alpha / 2,       axis=0)
        upper = np.percentile(samples, 100 * (1 - alpha / 2), axis=0)
        cov   = float(np.mean((y_true >= lower) & (y_true <= upper)))
        empirical.append(cov)

    return {
        "confidence_levels":   confidence_levels,
        "empirical_coverages": empirical,
    }


# ---------------------------------------------------------------------------
# Utility: count Dropout layers
# ---------------------------------------------------------------------------

def count_dropout_layers(model: nn.Module) -> int:
    """
    Return the number of nn.Dropout layers with p > 0 in the model.

    Parameters
    ----------
    model : nn.Module

    Returns
    -------
    int  Count of active Dropout layers (p > 0).
    """
    return sum(
        1 for m in model.modules()
        if isinstance(m, nn.Dropout) and m.p > 0
    )
