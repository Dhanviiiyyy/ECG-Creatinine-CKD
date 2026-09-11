"""
models/resnet1d.py
===================
1D ResNet-34 for continuous serum creatinine prediction from 12-lead ECG.

Architecture Reference
-----------------------
He, K. et al. (2016). Deep Residual Learning for Image Recognition.
IEEE CVPR. DOI: 10.1109/CVPR.2016.90

ECG application validated in:
  Hannun, A.Y. et al. (2019). Cardiologist-level arrhythmia detection and
  classification in ambulatory electrocardiograms using a deep neural network.
  Nature Medicine, 25, 65-69. DOI: 10.1038/s41591-018-0268-3

Lab-value regression context:
  Lopez Alcaraz, J.M. and Strodthoff, N. (2025). Benchmarking of ECG
  Foundation Models for Lab Test Prediction from MIMIC-IV-ECG.
  Scientific Reports. DOI: 10.1038/s41598-025-85364-0

Modifications from He et al. (2016)
-------------------------------------
1. All Conv2d -> Conv1d; BN2d -> BN1d; MaxPool2d -> MaxPool1d.
2. Input channels: 3 (RGB) -> 12 (ECG leads).
3. Stem kernel: 7x7 -> 15 (30 ms at 500 Hz; wider to capture P-wave).
4. Layer counts [3, 4, 6, 3] preserved (ResNet-34 specification, Table 1).
5. Output: 1000-class softmax -> 1 linear neuron (regression).
6. Head: GAP -> Dropout(0.5) -> FC(512->256) -> ReLU -> FC(256->1).

Network summary
---------------
Input  : [B, 12, 5000]
Output : [B, 1]  (predicted creatinine in mg/dL)
Params : ~21.3M
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn


class BasicBlock1D(nn.Module):
    """
    Residual building block for 1D ResNet (He et al., 2016, Figure 2 left).

    F(x) = W2 * sigma(BN(W1 * x))     -- two 3-sample conv layers
    y    = sigma(F(x) + shortcut(x))

    The shortcut is identity when channels match; a 1x1 Conv1d projection
    otherwise (Option B from He et al., 2016, Section 3.3).
    """

    expansion: int = 1  # BasicBlock does not expand channels

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        stride:       int = 1,
        downsample:   Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False,
        )
        self.bn1        = nn.BatchNorm1d(out_channels)
        self.relu       = nn.ReLU(inplace=True)
        self.conv2      = nn.Conv1d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False,
        )
        self.bn2        = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        return self.relu(out + identity)


class ResNet1D34(nn.Module):
    """
    1D ResNet-34 for 12-lead ECG -> serum creatinine regression.

    Layer configuration matches ResNet-34 (He et al., 2016, Table 1):
      layer counts : [3, 4, 6, 3]
      channel dims : [64, 128, 256, 512]

    Parameters
    ----------
    in_channels : int
        Number of input channels (ECG leads). Default 12.
    stem_kernel : int
        Stem Conv1d kernel size. Default 15 (~30 ms at 500 Hz).
    dropout_fc  : float
        Dropout probability in the regression head.
    """

    def __init__(
        self,
        in_channels: int = 12,
        stem_kernel:  int = 15,
        dropout_fc:   float = 0.5,
    ) -> None:
        super().__init__()
        self._in_ch = 64   # running channel tracker for _make_layer

        # -- Stem ---------------------------------------------------------
        # Input [B, 12, 5000] -> [B, 64, 1250]
        self.stem = nn.Sequential(
            nn.Conv1d(
                in_channels, 64,
                kernel_size=stem_kernel,
                stride=2,
                padding=stem_kernel // 2,
                bias=False,
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        # -- Residual layers (ResNet-34 specification) --------------------
        self.layer1 = self._make_layer(64,  n_blocks=3, stride=1)
        self.layer2 = self._make_layer(128, n_blocks=4, stride=2)
        self.layer3 = self._make_layer(256, n_blocks=6, stride=2)
        self.layer4 = self._make_layer(512, n_blocks=3, stride=2)

        # -- Regression head ----------------------------------------------
        self.gap  = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Dropout(p=dropout_fc),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_fc),
            nn.Linear(256, 1),
        )

        self._init_weights()

    # ------------------------------------------------------------------

    def _make_layer(
        self,
        out_channels: int,
        n_blocks:     int,
        stride:       int,
    ) -> nn.Sequential:
        """
        Build a residual layer from n_blocks BasicBlock1D units.

        First block uses stride for spatial downsampling (with projection
        shortcut). Subsequent blocks are identity-shortcut (stride=1).
        """
        downsample = None
        if stride != 1 or self._in_ch != out_channels:
            # Projection shortcut (He et al., Option B)
            downsample = nn.Sequential(
                nn.Conv1d(self._in_ch, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

        blocks = [BasicBlock1D(self._in_ch, out_channels,
                               stride=stride, downsample=downsample)]
        self._in_ch = out_channels
        for _ in range(1, n_blocks):
            blocks.append(BasicBlock1D(out_channels, out_channels))
        return nn.Sequential(*blocks)

    def _init_weights(self) -> None:
        """Kaiming-normal for Conv (He et al., 2015); constant for BN."""
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
        x : torch.Tensor  [B, 12, 5000]  float32

        Returns
        -------
        torch.Tensor  [B, 1]
        """
        x = self.stem(x)     # [B, 64, 1250]
        x = self.layer1(x)   # [B, 64, 1250]
        x = self.layer2(x)   # [B, 128,  625]
        x = self.layer3(x)   # [B, 256,  313]
        x = self.layer4(x)   # [B, 512,  157]
        x = self.gap(x)      # [B, 512,    1]
        x = x.squeeze(-1)    # [B, 512]
        return self.head(x)  # [B, 1]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
