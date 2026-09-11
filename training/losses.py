"""
training/losses.py
===================
Loss functions for the ECG-to-serum-creatinine regression task.

Loss selection justification
------------------------------
Serum creatinine has a right-skewed distribution:
  - Healthy peak: ~0.9 mg/dL (sigma ~0.2)
  - CKD tail:      up to ~15 mg/dL

MSELoss over-penalises the rare but clinically important high-creatinine
samples, potentially biasing the model toward the healthy majority.
MAELoss (L1) is robust but produces zero gradient exactly at the minimum,
slowing convergence.

Huber loss (delta=1.0 mg/dL) interpolates:
  - Behaves as MSE when |error| < delta  (quadratic, smooth gradient)
  - Behaves as MAE when |error| >= delta (linear, robust to outliers)

delta = 1.0 mg/dL is approximately 1 standard deviation of the healthy
creatinine distribution, making it a principled transition point.

Reference:
  Huber, P.J. (1964). Robust Estimation of a Location Parameter.
  Annals of Mathematical Statistics, 35(1), 73-101.

  Usage in biomarker regression:
  Lopez-Martinez et al. (2023) use Huber loss for similar continuous
  biomarker regression tasks on physiological signals.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class HuberRegressionLoss(nn.Module):
    """
    Huber loss for serum creatinine regression.

    L(y, yhat) = 0.5 * (y - yhat)^2                   if |y-yhat| < delta
                 delta * (|y-yhat| - 0.5 * delta)      otherwise

    Parameters
    ----------
    delta : float
        Transition point between quadratic and linear regime.
        Default 1.0 mg/dL (approx 1 SD of the healthy creatinine distribution).
    reduction : str
        'mean' (default) or 'sum'.
    """

    def __init__(self, delta: float = 1.0, reduction: str = "mean") -> None:
        super().__init__()
        if delta <= 0:
            raise ValueError(f"delta must be > 0, got {delta}.")
        self.delta     = delta
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        y_pred : torch.Tensor  shape [B, 1] or [B]  predicted creatinine
        y_true : torch.Tensor  shape [B, 1] or [B]  ground-truth creatinine

        Returns
        -------
        torch.Tensor  scalar loss
        """
        return F.huber_loss(
            y_pred.squeeze(-1),
            y_true.squeeze(-1),
            delta=self.delta,
            reduction=self.reduction,
        )


class MSERegressionLoss(nn.Module):
    """
    Mean Squared Error loss. Included for ablation comparison against Huber.

    L(y, yhat) = (1/N) * sum (y_i - yhat_i)^2
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.loss_fn = nn.MSELoss(reduction=reduction)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(y_pred.squeeze(-1), y_true.squeeze(-1))


class MAERegressionLoss(nn.Module):
    """
    Mean Absolute Error (L1) loss. Included for ablation comparison.

    L(y, yhat) = (1/N) * sum |y_i - yhat_i|
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.loss_fn = nn.L1Loss(reduction=reduction)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(y_pred.squeeze(-1), y_true.squeeze(-1))


def build_loss(name: str = "huber", **kwargs) -> nn.Module:
    """
    Factory function for loss functions.

    Parameters
    ----------
    name : str
        'huber' (default), 'mse', 'mae'
    **kwargs
        Passed to the loss constructor (e.g. delta=1.0 for Huber).

    Returns
    -------
    nn.Module  loss function
    """
    registry = {
        "huber": HuberRegressionLoss,
        "mse":   MSERegressionLoss,
        "mae":   MAERegressionLoss,
    }
    if name not in registry:
        raise ValueError(f"Unknown loss '{name}'. Choose from {list(registry)}.")
    return registry[name](**kwargs)
