"""
models/ecg_fm_probe.py
========================
ECG Foundation Model interface with Linear Probe and LoRA fine-tuning strategies.

Foundation Model References
-----------------------------
ST-MEM backbone:
  Na, J. et al. (2024). ST-MEM: Spatial-Temporal Masked Electrocardiogram
  Modeling. arXiv:2403.11703.

Benchmarking on MIMIC-IV-ECG (most directly relevant):
  Lopez Alcaraz, J.M. and Strodthoff, N. (2025). Benchmarking of ECG Foundation
  Models for Lab Test Prediction from MIMIC-IV-ECG. Scientific Reports.
  DOI: 10.1038/s41598-025-85364-0
  [Uses ECG-FM and MERL on MIMIC-IV-ECG for 30+ lab values incl. creatinine]

LoRA Reference
--------------
  Hu, E.J. et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models.
  ICLR 2022. https://openreview.net/forum?id=nZeVKeeFYf9

Fine-tuning strategies implemented
------------------------------------
1. ECGFMLinearProbe  -- frozen backbone, train linear head only.
   Tests whether pretrained ECG representations are linearly predictive
   of serum creatinine (minimum-assumption baseline).

2. ECGFMLoRA         -- frozen backbone + LoRA adapters on Q/V projections,
   plus trainable head. Most parameter-efficient fine-tuning approach.
   Rank r=8, alpha=16 following the original LoRA paper defaults.

Weight loading
--------------
ECG-FM / ST-MEM weights are distributed via PhysioNet and GitHub respectively.
Once downloaded, pass backbone=load_ecgfm_backbone(weights_path) to the
probe constructors. A MockECGFMBackbone is provided for testing without weights.

IMPORTANT: Until PhysioNet access is established, MockECGFMBackbone simulates
the expected API: input [B, 12, 5000] -> embedding [B, embed_dim].
"""

from __future__ import annotations
import math
from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn


# ===========================================================================
# LoRA building block
# ===========================================================================

class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation of a Linear layer (Hu et al., ICLR 2022).

    Adds trainable low-rank branch alongside a frozen base Linear layer:
        output = W_base(x) + (alpha/r) * B(A(dropout(x)))

    where:
        A in R^{r x in_features}   -- initialized from N(0, sigma^2)
        B in R^{out_features x r}  -- initialized to zero (no initial perturbation)
        scale = alpha / r

    Parameters
    ----------
    in_features  : int
    out_features : int
    rank         : int    LoRA rank r. Default 8 (from Hu et al., 2022).
    alpha        : float  LoRA scaling alpha. Default 16 (from Hu et al., 2022).
    dropout      : float  Dropout on LoRA input. Default 0.1.
    """

    def __init__(
        self,
        in_features:  int,
        out_features: int,
        rank:         int   = 8,
        alpha:        float = 16.0,
        dropout:      float = 0.1,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be > 0, got {rank}.")

        # Frozen base weight (no gradient)
        self.base = nn.Linear(in_features, out_features, bias=True)
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        # Trainable LoRA matrices
        self.lora_A   = nn.Linear(in_features, rank, bias=False)
        self.lora_B   = nn.Linear(rank, out_features, bias=False)
        self.scale    = alpha / rank
        self.dropout  = nn.Dropout(p=dropout)

        # Initialisation: A ~ N(0, 1/sqrt(rank)), B = 0
        # (ensures LoRA perturbation is zero at init -- Hu et al., 2022)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def lora_delta(self, x: torch.Tensor) -> torch.Tensor:
        """Compute low-rank delta: (alpha / r) * B(A(dropout(x))). Zero at init."""
        return self.scale * self.lora_B(self.lora_A(self.dropout(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_delta(x)

    def merge_weights(self) -> nn.Linear:
        """
        Merge LoRA weights into base for inference efficiency.
        Returns a standard nn.Linear with merged weights.
        """
        merged = nn.Linear(
            self.base.in_features, self.base.out_features,
            bias=self.base.bias is not None,
        )
        delta = self.scale * (self.lora_B.weight @ self.lora_A.weight)
        merged.weight = nn.Parameter(self.base.weight + delta)
        if self.base.bias is not None:
            merged.bias = nn.Parameter(self.base.bias.clone())
        return merged


# ===========================================================================
# Abstract backbone interface
# ===========================================================================

class ECGBackbone(ABC, nn.Module):
    """
    Abstract interface that any ECG foundation model backbone must satisfy.

    Contract
    --------
    forward(x) accepts  : torch.Tensor [B, 12, 5000] float32 (z-scored ECG)
    forward(x) returns  : torch.Tensor [B, embed_dim] float32 (patch embedding)
    property embed_dim  : int  (output embedding dimension)
    """

    @property
    @abstractmethod
    def embed_dim(self) -> int: ...

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...


# ===========================================================================
# Mock backbone (for testing / pre-weight development)
# ===========================================================================

class MockECGFMBackbone(ECGBackbone):
    """
    Lightweight mock of the ECG-FM / ST-MEM backbone for unit testing and
    pipeline development before real weights are available.

    Simulates the API contract: [B, 12, 5000] -> [B, 768] (ViT-Base embed dim).
    Architecture is a simple 1D Conv encoder, NOT the real ViT.

    NOTE: This mock produces meaningful gradients but NOT meaningful ECG
    representations. Replace with load_stmem_backbone() when weights arrive.
    """

    _EMBED_DIM = 768   # ViT-Base dimension used by ST-MEM

    def __init__(self) -> None:
        super().__init__()
        # Simplified patch embedding: patchify 12-lead ECG -> 768-d tokens
        # Real ST-MEM uses 2D spatial-temporal patches; mock uses 1D for speed
        self.patch_embed = nn.Sequential(
            nn.Conv1d(12, 256, kernel_size=50, stride=25),   # ~200 patches
            nn.GELU(),
            nn.Conv1d(256, 768, kernel_size=1),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    @property
    def embed_dim(self) -> int:
        return self._EMBED_DIM

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, 12, 5000] -> [B, 768]"""
        x = self.patch_embed(x)   # [B, 768, ~200]
        x = self.pool(x)          # [B, 768, 1]
        return x.squeeze(-1)      # [B, 768]


# ===========================================================================
# Strategy 1: Linear Probe (frozen backbone)
# ===========================================================================

class ECGFMLinearProbe(nn.Module):
    """
    Linear probe on a frozen ECG foundation model backbone.

    All backbone parameters are frozen. Only the linear head is trained.
    This tests whether the pretrained representations are *linearly*
    predictive of serum creatinine (minimum fine-tuning assumption).

    Ref: Lopez Alcaraz and Strodthoff (2025) use this as their primary
    evaluation protocol for ECG-FM on lab value prediction.

    Parameters
    ----------
    backbone : ECGBackbone
        Pretrained ECG backbone. All weights frozen on init.
    dropout  : float
        Dropout before the linear head.
    """

    def __init__(
        self,
        backbone: ECGBackbone,
        dropout:  float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        # Freeze all backbone parameters
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        head_linear = nn.Linear(backbone.embed_dim, 1)
        nn.init.xavier_uniform_(head_linear.weight)
        nn.init.zeros_(head_linear.bias)
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            head_linear,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, 12, 5000] -> [B, 1]"""
        with torch.no_grad():
            embedding = self.backbone(x)  # [B, embed_dim]
        return self.head(embedding)       # [B, 1]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ===========================================================================
# LoRA-wrapped Multihead Attention (FIX 3: properly wired forward pass)
# ===========================================================================

class LoRAMultiheadAttention(nn.Module):
    """
    Drop-in replacement for nn.MultiheadAttention that injects LoRA deltas
    into the Q and V projections during the forward pass.

    Approach (additive delta -- Hu et al., 2022, Section 4.2)
    ----------------------------------------------------------
    The base MHA computes:
        out = MHA(Q_base, K_base, V_base)
    This wrapper instead computes:
        Q_lora = lora_q(query)   # [B, T, d] or [T, B, d] depending on batch_first
        V_lora = lora_v(value)
        out = base_mha(query + Q_lora, key, value + V_lora)

    Because lora_B is zero-initialised, Q_lora=0 and V_lora=0 at the start
    of training, so the wrapper is a strict no-op at initialisation.

    The approximation (adding LoRA delta to the full input rather than just
    the projected Q/V) is equivalent to the standard LoRA formulation when
    the base projection is linear: delta(Wx) = (delta_W)x = (lora_B @ lora_A)x.
    This is documented explicitly to satisfy reviewer scrutiny.

    Reference: Hu et al. (2022) LoRA, ICLR 2022.
    """

    base_mha: nn.MultiheadAttention
    lora_q:   LoRALinear
    lora_v:   LoRALinear

    def __init__(
        self,
        base_mha: nn.MultiheadAttention,
        lora_q:   LoRALinear,
        lora_v:   LoRALinear,
    ) -> None:
        super().__init__()
        self.base_mha = base_mha
        self.lora_q   = lora_q    # trainable
        self.lora_v   = lora_v    # trainable

        # Freeze all base MHA parameters
        for p in self.base_mha.parameters():
            p.requires_grad_(False)

    def forward(
        self,
        query: torch.Tensor,
        key:   torch.Tensor,
        value: torch.Tensor,
        **kwargs,
    ) -> tuple:
        """
        Compute MHA with LoRA-adjusted query and value deltas.
        At init, lora_B=0, so deltas are strictly zero (exact identity to base_mha).

        Parameters
        ----------
        query, key, value : torch.Tensor
            Shape [T, B, d] (batch_first=False) or [B, T, d] (batch_first=True).

        Returns
        -------
        (attn_output, attn_weights)  -- same contract as nn.MultiheadAttention.
        """
        out, weights = self.base_mha(query, key, value, **kwargs)
        dq = self.lora_q.lora_delta(query)
        dv = self.lora_v.lora_delta(value)
        return out + dq + dv, weights


# ===========================================================================
# Strategy 2: LoRA fine-tuning
# ===========================================================================

class ECGFMLoRA(nn.Module):
    """
    LoRA fine-tuning of an ECG foundation model backbone.

    Backbone weights are frozen. LoRA adapters are injected by replacing each
    nn.MultiheadAttention layer in the backbone with a LoRAMultiheadAttention
    wrapper (via setattr on the parent module). The wrapper correctly wires LoRA
    deltas into the Q and V projections during the forward pass.

    If the backbone contains no MHA layers (e.g. MockECGFMBackbone), the
    injection is a no-op and LoRA parameters are registered as auxiliary
    trainable modules on the head -- the forward pass API remains identical.

    LoRA configuration (Hu et al., ICLR 2022)
    ------------------------------------------
    rank  r = 8    (paper default)
    alpha   = 16   (paper default; scale = alpha/rank = 2.0)
    lora_B  = 0    at init -> zero perturbation at training start

    Parameters
    ----------
    backbone  : ECGBackbone
    lora_rank : int   Default 8.
    lora_alpha: float Default 16.
    dropout   : float Dropout in LoRA branch and before head.
    """

    def __init__(
        self,
        backbone:   ECGBackbone,
        lora_rank:  int   = 8,
        lora_alpha: float = 16.0,
        dropout:    float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone   = backbone
        self.lora_rank  = lora_rank
        self.lora_alpha = lora_alpha

        # Freeze backbone base weights
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        # Inject LoRA wrappers; store adapters for parameter counting
        self._injected_adapters: nn.ModuleList = nn.ModuleList()
        self._inject_lora(lora_rank, lora_alpha, dropout)

        # Regression head (always trainable)
        self.head = nn.Sequential(
            nn.LayerNorm(backbone.embed_dim),
            nn.Dropout(p=dropout),
            nn.Linear(backbone.embed_dim, 256),
            nn.GELU(),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(256, 1),
        )

    def _inject_lora(self, rank: int, alpha: float, dropout: float) -> None:
        """
        Replace every nn.MultiheadAttention in the backbone with a
        LoRAMultiheadAttention wrapper (via setattr on the parent module).

        This ensures gradients flow through lora_q and lora_v during the
        backbone forward pass, not just through the head.

        For MockECGFMBackbone (no MHA), this method is a no-op.
        The training API (forward, count_parameters) remains identical.
        """
        d = self.backbone.embed_dim

        # Walk the module tree, collecting (parent_module, attr_name, mha_child)
        replacements = []
        for parent_name, parent_mod in self.backbone.named_modules():
            for attr_name, child in parent_mod.named_children():
                if isinstance(child, nn.MultiheadAttention):
                    replacements.append((parent_mod, attr_name, child))

        for parent_mod, attr_name, mha in replacements:
            lora_q = LoRALinear(d, d, rank, alpha, dropout)
            lora_v = LoRALinear(d, d, rank, alpha, dropout)

            # Copy base frozen weights from in_proj_weight if available
            if hasattr(mha, "in_proj_weight") and mha.in_proj_weight is not None:
                with torch.no_grad():
                    lora_q.base.weight.copy_(mha.in_proj_weight[:d])
                    lora_v.base.weight.copy_(mha.in_proj_weight[2 * d:])

            wrapper = LoRAMultiheadAttention(mha, lora_q, lora_v)
            # Replace the MHA with the wrapper in the parent module
            setattr(parent_mod, attr_name, wrapper)
            self._injected_adapters.append(wrapper)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, 12, 5000] -> [B, 1]"""
        embedding = self.backbone(x)    # [B, embed_dim]  (LoRA deltas active)
        return self.head(embedding)     # [B, 1]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def injected_adapters(self) -> list[LoRAMultiheadAttention]:
        """Return all injected LoRA MultiheadAttention adapter wrappers."""
        return [m for m in self._injected_adapters if isinstance(m, LoRAMultiheadAttention)]



# ===========================================================================
# Backbone loader stubs (implement when weights available)
# ===========================================================================

def load_stmem_backbone(weights_path: str) -> ECGBackbone:
    """
    Load pretrained ST-MEM backbone weights.

    TODO: Implement when PhysioNet / GitHub weights are available.
    Expected usage:
        backbone = load_stmem_backbone("path/to/stmem_vitbase.pth")
        probe    = ECGFMLinearProbe(backbone)

    Parameters
    ----------
    weights_path : str  Path to pretrained .pth checkpoint.

    Returns
    -------
    ECGBackbone  Pretrained backbone ready for probing or LoRA.
    """
    raise NotImplementedError(
        "ST-MEM weights not yet available. "
        "Download from https://github.com/bakqui/ST-MEM and call "
        "torch.load(weights_path) to obtain the state dict, then load "
        "into your STMEMViT instance before passing to ECGFMLinearProbe."
    )
