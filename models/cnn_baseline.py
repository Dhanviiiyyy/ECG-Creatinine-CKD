"""
models/cnn_baseline.py
========================
From-scratch 1D Convolutional Neural Network baseline for continuous
serum creatinine prediction from 12-lead ECG waveforms.

Architecture Reference
-----------------------
Adapted from:
  Ribeiro, A.H. et al. (2020). Automatic diagnosis of the 12-lead ECG
  using a deep neural network. Nature Communications, 11, 1760.
  DOI: 10.1038/s41467-020-15432-4

Also informed by:
  Holmstrom, L. et al. (2023). Deep learning identifies a CKD signature
  from 12-lead ECG in diverse international populations.
  Communications Medicine, 3, 77.
  DOI: 10.1038/s43856-023-00307-6

Modifications from cited works
-------------------------------
1. Output head: multi-class softmax -> single linear neuron (regression).
2. Input: 12 leads x 5000 samples (vs 1 or 8 leads in originals).
3. BatchNorm after each Conv (Ioffe & Szegedy, ICML 2015).
4. Dropout p=0.5 on FC layers (Srivastava et al., JMLR 2014).
5. GlobalAveragePooling instead of Flatten: reduces overfitting and
   makes the model temporal-length-agnostic.
6. No L2 output activation: unbounded continuous creatinine output.

Network summary
---------------
Input  : [B, 12, 5000]
Output : [B, 1]  (predicted creatinine in mg/dL)
Params : ~1.1M
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _ConvBlock1D(nn.Module):
    """
    Single convolutional block: Conv1d -> BatchNorm1d -> ReLU -> MaxPool1d.

    Uses symmetric padding (kernel_size // 2) so that temporal length is
    reduced ONLY by MaxPool, keeping receptive-field arithmetic transparent.

    Bias is omitted in Conv1d because BatchNorm absorbs the additive bias
    (Ioffe & Szegedy, 2015).
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        kernel_size:  int,
        pool_size:    int = 4,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels, out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pool_size, stride=pool_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CNNBaseline1D(nn.Module):
    """
    4-block 1D-CNN baseline for 12-lead ECG -> serum creatinine regression.

    Architecture
    ------------
    [B, 12, 5000]
      ConvBlock(12->32,  k=7,  pool=4) -> [B,  32, 1250]
      ConvBlock(32->64,  k=5,  pool=4) -> [B,  64,  312]
      ConvBlock(64->128, k=5,  pool=4) -> [B, 128,   78]
      ConvBlock(128->256,k=3,  pool=2) -> [B, 256,   39]
      AdaptiveAvgPool1d(1)             -> [B, 256,    1]
      Flatten                          -> [B, 256]
      FC(256->64) -> ReLU -> Dropout(0.5)
      FC(64->1)                        -> [B, 1]

    Parameters
    ----------
    in_channels : int
        Number of ECG leads (default 12).
    dropout_fc : float
        Dropout probability on FC layers (Srivastava et al., 2014).
    """

    def __init__(
        self,
        in_channels: int = 12,
        dropout_fc:  float = 0.5,
    ) -> None:
        super().__init__()

        self.conv_blocks = nn.Sequential(
            _ConvBlock1D(in_channels, 32,  kernel_size=7, pool_size=4),
            _ConvBlock1D(32,          64,  kernel_size=5, pool_size=4),
            _ConvBlock1D(64,          128, kernel_size=5, pool_size=4),
            _ConvBlock1D(128,         256, kernel_size=3, pool_size=2),
        )

        self.gap = nn.AdaptiveAvgPool1d(1)  # collapse temporal dimension

        self.head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_fc),
            nn.Linear(64, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming-normal for Conv; Xavier-uniform for Linear; const for BN."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor  shape [B, 12, 5000]  float32, z-scored ECG

        Returns
        -------
        torch.Tensor  shape [B, 1]  creatinine prediction (mg/dL)
        """
        x = self.conv_blocks(x)   # [B, 256, L]
        x = self.gap(x)           # [B, 256, 1]
        x = x.squeeze(-1)         # [B, 256]
        return self.head(x)       # [B, 1]

    def count_parameters(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
