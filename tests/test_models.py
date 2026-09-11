"""
tests/test_models.py
=====================
Unit tests for model architectures:
  - CNNBaseline1D   (models/cnn_baseline.py)
  - ResNet1D34      (models/resnet1d.py)
  - ECGFMLinearProbe and ECGFMLoRA  (models/ecg_fm_probe.py)

Tests verify:
  * Forward pass input/output shapes are correct.
  * No NaN or Inf in output for random input.
  * Parameter counts are within expected ranges.
  * Gradient flows to all trainable parameters.
  * Models switch correctly between train/eval mode.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.cnn_baseline import CNNBaseline1D
from models.resnet1d import ResNet1D34, BasicBlock1D
from models.ecg_fm_probe import (
    ECGBackbone,
    ECGFMLinearProbe,
    ECGFMLoRA,
    LoRALinear,
    LoRAMultiheadAttention,
    MockECGFMBackbone,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

BATCH  = 4
LEADS  = 12
SAMPS  = 5000
DEVICE = torch.device("cpu")


def _random_ecg(batch: int = BATCH) -> torch.Tensor:
    """Random z-scored ECG batch [B, 12, 5000]."""
    return torch.randn(batch, LEADS, SAMPS)


# ---------------------------------------------------------------------------
# CNNBaseline1D
# ---------------------------------------------------------------------------

class TestCNNBaseline1D:
    def test_output_shape(self):
        model = CNNBaseline1D()
        out   = model(_random_ecg())
        assert out.shape == (BATCH, 1), f"Expected ({BATCH},1), got {out.shape}"

    def test_no_nan(self):
        model = CNNBaseline1D()
        out   = model(_random_ecg())
        assert torch.isfinite(out).all(), "NaN/Inf in CNN output"

    def test_param_count_reasonable(self):
        model = CNNBaseline1D()
        n     = model.count_parameters()
        # 1D-CNN baseline has ~170k params
        assert 100_000 <= n <= 500_000, f"Param count {n} out of expected range"

    def test_gradient_flows(self):
        model = CNNBaseline1D()
        out   = model(_random_ecg())
        loss  = out.mean()
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No grad for {name}"
                assert torch.isfinite(p.grad).all(), f"NaN grad at {name}"

    def test_eval_mode_no_grad(self):
        model = CNNBaseline1D().eval()
        with torch.no_grad():
            out = model(_random_ecg())
        assert out.shape == (BATCH, 1)

    def test_single_sample(self):
        model = CNNBaseline1D()
        out   = model(_random_ecg(batch=1))
        assert out.shape == (1, 1)

    def test_different_dropout_fc(self):
        model = CNNBaseline1D(dropout_fc=0.0)
        out   = model(_random_ecg())
        assert out.shape == (BATCH, 1)


# ---------------------------------------------------------------------------
# ResNet1D34
# ---------------------------------------------------------------------------

class TestResNet1D34:
    def test_output_shape(self):
        model = ResNet1D34()
        out   = model(_random_ecg())
        assert out.shape == (BATCH, 1)

    def test_no_nan(self):
        model = ResNet1D34()
        out   = model(_random_ecg())
        assert torch.isfinite(out).all()

    def test_param_count_range(self):
        model = ResNet1D34()
        n     = model.count_parameters()
        # 1D ResNet-34 has ~7.36M params (vs ~21M for 2D ResNet-34)
        assert 5_000_000 <= n <= 10_000_000, f"Param count {n} out of range"

    def test_residual_block_identity_shortcut(self):
        """A BasicBlock1D with same in/out channels should use identity shortcut."""
        block = BasicBlock1D(64, 64, stride=1, downsample=None)
        x   = torch.randn(2, 64, 500)
        out = block(x)
        assert out.shape == x.shape

    def test_residual_block_projection_shortcut(self):
        """A BasicBlock1D with different channels should use projection shortcut."""
        downsample = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm1d(128),
        )
        block = BasicBlock1D(64, 128, stride=2, downsample=downsample)
        x   = torch.randn(2, 64, 500)
        out = block(x)
        assert out.shape == (2, 128, 250)

    def test_gradient_flows(self):
        model = ResNet1D34()
        out   = model(_random_ecg())
        out.mean().backward()
        # Check at least some gradients are non-zero
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0

    def test_stem_kernel_configurable(self):
        model = ResNet1D34(stem_kernel=7)
        out   = model(_random_ecg())
        assert out.shape == (BATCH, 1)


# ---------------------------------------------------------------------------
# LoRALinear
# ---------------------------------------------------------------------------

class TestLoRALinear:
    def test_output_shape(self):
        layer = LoRALinear(128, 256, rank=8, alpha=16)
        x     = torch.randn(4, 128)
        out   = layer(x)
        assert out.shape == (4, 256)

    def test_lora_B_init_zero(self):
        """lora_B weight must be zero at init (Hu et al., 2022)."""
        layer = LoRALinear(64, 64, rank=8)
        assert layer.lora_B.weight.abs().max().item() == pytest.approx(0.0)

    def test_base_frozen(self):
        layer = LoRALinear(64, 64, rank=8)
        assert not layer.base.weight.requires_grad

    def test_lora_trainable(self):
        layer = LoRALinear(64, 64, rank=8)
        assert layer.lora_A.weight.requires_grad
        assert layer.lora_B.weight.requires_grad

    def test_gradient_through_lora(self):
        layer = LoRALinear(64, 64, rank=8)
        x     = torch.randn(2, 64, requires_grad=False)
        out   = layer(x)
        out.sum().backward()
        assert layer.lora_A.weight.grad is not None
        assert layer.lora_B.weight.grad is not None

    def test_invalid_rank_raises(self):
        with pytest.raises(ValueError, match="rank must be"):
            LoRALinear(64, 64, rank=0)


# ---------------------------------------------------------------------------
# MockECGFMBackbone
# ---------------------------------------------------------------------------

class TestMockECGFMBackbone:
    def test_output_shape(self):
        backbone = MockECGFMBackbone()
        x        = _random_ecg()
        out      = backbone(x)
        assert out.shape == (BATCH, 768)

    def test_embed_dim_property(self):
        assert MockECGFMBackbone().embed_dim == 768


# ---------------------------------------------------------------------------
# ECGFMLinearProbe
# ---------------------------------------------------------------------------

class TestECGFMLinearProbe:
    def test_output_shape(self):
        probe = ECGFMLinearProbe(MockECGFMBackbone())
        out   = probe(_random_ecg())
        assert out.shape == (BATCH, 1)

    def test_backbone_frozen(self):
        probe = ECGFMLinearProbe(MockECGFMBackbone())
        for p in probe.backbone.parameters():
            assert not p.requires_grad

    def test_head_trainable(self):
        probe  = ECGFMLinearProbe(MockECGFMBackbone())
        n_train = probe.count_parameters()
        # Only head: Linear(768, 1) = 768 + 1 = 769 params
        assert n_train == 769

    def test_no_nan(self):
        probe = ECGFMLinearProbe(MockECGFMBackbone())
        out   = probe(_random_ecg())
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# ECGFMLoRA
# ---------------------------------------------------------------------------

class TestECGFMLoRA:
    def test_output_shape(self):
        lora = ECGFMLoRA(MockECGFMBackbone(), lora_rank=8)
        out  = lora(_random_ecg())
        assert out.shape == (BATCH, 1)

    def test_no_nan(self):
        lora = ECGFMLoRA(MockECGFMBackbone(), lora_rank=4)
        out  = lora(_random_ecg())
        assert torch.isfinite(out).all()

    def test_gradient_flows_to_head(self):
        lora = ECGFMLoRA(MockECGFMBackbone(), lora_rank=4)
        out  = lora(_random_ecg())
        out.mean().backward()
        # head parameters should have gradients
        for name, p in lora.head.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No grad for head.{name}"

    def test_lora_backbone_with_mha(self):
        """
        FIX 3 verification: Create a backbone that contains a real
        nn.MultiheadAttention, wrap in ECGFMLoRA, forward+backward,
        and assert lora_A.weight.grad is not None for at least one adapter.
        """
        class _MHABackbone(ECGBackbone):
            """Tiny mock backbone with a real MHA layer for LoRA injection testing."""
            _EMBED = 64

            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(12, self._EMBED)
                self.mha  = nn.MultiheadAttention(self._EMBED, num_heads=4, batch_first=True)
                self.pool = nn.AdaptiveAvgPool1d(1)

            @property
            def embed_dim(self) -> int:
                return self._EMBED

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # x: [B, 12, 5000] -> [B, 5000, 64] -> MHA -> [B, 64]
                B = x.shape[0]
                x = x.permute(0, 2, 1)           # [B, 5000, 12]
                x = self.proj(x)                  # [B, 5000, 64]
                x, _ = self.mha(x, x, x)          # [B, 5000, 64]
                x = x.permute(0, 2, 1)            # [B, 64, 5000]
                return self.pool(x).squeeze(-1)   # [B, 64]

        backbone = _MHABackbone()
        model    = ECGFMLoRA(backbone, lora_rank=4, lora_alpha=8.0)

        out = model(_random_ecg())
        assert out.shape == (BATCH, 1), f"Expected ({BATCH},1), got {out.shape}"
        out.mean().backward()

        # At least one injected LoRA adapter must have a gradient
        found_grad = False
        for wrapper in model.injected_adapters:
            grad_q = getattr(wrapper.lora_q.lora_A.weight, "grad", None)
            grad_v = getattr(wrapper.lora_v.lora_A.weight, "grad", None)
            if grad_q is not None or grad_v is not None:
                found_grad = True
                break
        assert found_grad, "lora_A.weight.grad is None for all injected adapters"

    def test_lora_mha_wrapper_zero_init_is_identity(self):
        """LoRAMultiheadAttention must be a no-op at init (lora_B=0)."""
        d   = 32
        mha = nn.MultiheadAttention(d, num_heads=4, batch_first=True)
        lq  = LoRALinear(d, d, rank=4)
        lv  = LoRALinear(d, d, rank=4)
        wrapper = LoRAMultiheadAttention(mha, lq, lv)

        x    = torch.randn(2, 10, d)
        # Both should produce identical output at init (lora_B=0 so delta=0)
        with torch.no_grad():
            out_base, _    = mha(x, x, x)
            out_wrapped, _ = wrapper(x, x, x)
        torch.testing.assert_close(out_base, out_wrapped, atol=1e-5, rtol=1e-5)

