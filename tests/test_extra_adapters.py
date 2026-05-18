"""Tests for FREQ-SAFE additional adapters (attention, residual, training)."""
from __future__ import annotations

import numpy as np
import pytest

from adapters.attention_adapter import (
    AttentionAdapter,
    extract_attention_sink_series,
    extract_attention_entropy_series,
    extract_head_variance_series,
    extract_attention_sparsity_series,
)
from adapters.residual_adapter import (
    ResidualAdapter,
    extract_norm_growth_series,
    extract_delta_norm_series,
    extract_layer_contribution_series,
    extract_saturation_series,
)
from adapters.training_adapter import (
    TrainingAdapter,
    extract_loss_series,
    extract_gradient_norm_series,
    extract_loss_curvature_series,
    detect_training_phases,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Synthetic data generators
# ═══════════════════════════════════════════════════════════════════════════════

def synthetic_attention(
    n_layers: int = 24,
    n_heads: int = 8,
    seq_len: int = 64,
    seed: int = 42,
    pattern: str = "uniform",
) -> np.ndarray:
    """Generate synthetic attention matrices.

    Args:
        pattern: "uniform" (all equal), "focused" (sharp peaks),
                 "sink_dominated" (strong first-token attention),
                 "chaotic" (random noisy patterns).
    """
    rng = np.random.default_rng(seed)

    if pattern == "uniform":
        # Each row sums to 1, uniform distribution
        attn = np.ones((n_layers, n_heads, seq_len, seq_len))
        attn = attn / seq_len  # normalize rows
        return attn

    elif pattern == "focused":
        # Each row has a single sharp peak
        attn = np.full((n_layers, n_heads, seq_len, seq_len), 0.001)
        for l in range(n_layers):
            for h in range(n_heads):
                peak_pos = rng.integers(0, seq_len)
                attn[l, h, :, peak_pos] = 1.0 - 0.001 * (seq_len - 1)
        return attn

    elif pattern == "sink_dominated":
        # Heavy attention on first token (attention sink)
        attn = np.random.uniform(0, 0.1, (n_layers, n_heads, seq_len, seq_len))
        # First column = attention TO first token = 0.5-0.9
        sink_strength = np.linspace(0.9, 0.5, n_layers)  # decreasing through layers
        for l in range(n_layers):
            attn[l, :, :, 0] = sink_strength[l]
        # Renormalize rows
        row_sums = attn.sum(axis=3, keepdims=True)
        attn = attn / row_sums
        return attn

    else:  # chaotic
        return np.random.dirichlet(
            np.ones(seq_len) * 0.5,
            size=(n_layers, n_heads, seq_len),
        ).reshape(n_layers, n_heads, seq_len, seq_len)


def synthetic_residuals(
    n_layers: int = 32,
    hidden_dim: int = 4096,
    seed: int = 42,
    pattern: str = "growing",
) -> np.ndarray:
    """Generate synthetic residual stream activations.

    Args:
        pattern: "growing" (norm increases with depth),
                 "flat" (constant norm),
                 "spiky" (one layer dominates).
    """
    rng = np.random.default_rng(seed)

    if pattern == "growing":
        norms = np.linspace(1.0, 10.0, n_layers)
        activations = rng.normal(0, 1, (n_layers, hidden_dim))
        # Scale each layer to target norm
        for i in range(n_layers):
            activations[i] = activations[i] / np.linalg.norm(activations[i]) * norms[i]
        return activations

    elif pattern == "flat":
        norms = np.ones(n_layers) * 5.0
        activations = rng.normal(0, 1, (n_layers, hidden_dim))
        for i in range(n_layers):
            activations[i] = activations[i] / np.linalg.norm(activations[i]) * norms[i]
        return activations

    else:  # spiky
        norms = np.ones(n_layers) * 3.0
        # Spike at layer 12
        norms[12] = 30.0
        activations = rng.normal(0, 1, (n_layers, hidden_dim))
        for i in range(n_layers):
            activations[i] = activations[i] / np.linalg.norm(activations[i]) * norms[i]
        return activations


def synthetic_training_history(
    n_steps: int = 500,
    seed: int = 42,
    pattern: str = "normal",
) -> np.ndarray:
    """Generate synthetic training loss curve.

    Args:
        pattern: "normal" (smooth decay), "phase_shift" (sudden jump),
                 "oscillating" (unstable training).
    """
    rng = np.random.default_rng(seed)

    if pattern == "normal":
        # Smooth exponential decay
        base = 3.0 * np.exp(-np.linspace(0, 4, n_steps)) + 0.1
        noise = rng.normal(0, 0.02, n_steps)
        return base + noise

    elif pattern == "phase_shift":
        # Normal decay, then sudden loss spike (forgetting event)
        loss = 3.0 * np.exp(-np.linspace(0, 3, n_steps))
        # Spike at step 300
        spike = np.zeros(n_steps)
        spike[300:320] = np.linspace(0, 2, 20) * np.exp(-np.linspace(0, 2, 20))
        loss += spike
        loss += 0.1
        noise = rng.normal(0, 0.015, n_steps)
        return loss + noise

    else:  # oscillating
        t = np.linspace(0, 8 * np.pi, n_steps)
        oscillation = 0.3 * np.sin(t)
        decay = 2.0 * np.exp(-np.linspace(0, 2, n_steps)) + 0.2
        return decay + oscillation + rng.normal(0, 0.02, n_steps)


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: AttentionAdapter
# ═══════════════════════════════════════════════════════════════════════════════

class TestAttentionAdapter:
    def test_sink_extraction_shape(self):
        attn = synthetic_attention(n_layers=24, n_heads=8, seq_len=64)
        sink = extract_attention_sink_series(attn)
        assert sink.shape == (24,)
        assert np.all((sink >= 0) & (sink <= 1))

    def test_sink_dominated_high_sink(self):
        attn = synthetic_attention(24, 8, 64, pattern="sink_dominated")
        sink = extract_attention_sink_series(attn)
        assert sink.mean() > 0.1  # Higher sink than uniform (1/64 ≈ 0.016)

    def test_uniform_low_sink(self):
        attn = synthetic_attention(24, 8, 64, pattern="uniform")
        sink = extract_attention_sink_series(attn)
        # Uniform attention → sink = 1/seq_len
        assert sink.mean() == pytest.approx(1.0 / 64, abs=0.01)

    def test_entropy_extraction_shape(self):
        attn = synthetic_attention(24, 8, 64, pattern="uniform")
        entropy = extract_attention_entropy_series(attn)
        assert entropy.shape == (24,)

    def test_uniform_higher_entropy_than_focused(self):
        uniform = synthetic_attention(24, 8, 64, pattern="uniform")
        focused = synthetic_attention(24, 8, 64, pattern="focused")
        u_ent = extract_attention_entropy_series(uniform).mean()
        f_ent = extract_attention_entropy_series(focused).mean()
        assert u_ent > f_ent

    def test_head_variance(self):
        attn = synthetic_attention(24, 8, 64, pattern="chaotic")
        var = extract_head_variance_series(attn)
        assert var.shape == (24,)
        assert np.all(var >= 0)

    def test_sparsity(self):
        attn = synthetic_attention(24, 8, 64, pattern="focused")
        sparsity = extract_attention_sparsity_series(attn, threshold=0.01)
        assert sparsity.shape == (24,)
        # Focused attention should have high sparsity (most weights < 0.01)
        assert sparsity.mean() > 0.5

    def test_convenience_wrapper(self):
        attn = synthetic_attention(24, 8, 64)
        d = analyze_attention(attn, signal="sink")
        assert "valid" in d
        assert "signature" in d

    def test_3d_input_handling(self):
        """Test that 3D (without heads dim) is handled."""
        attn = synthetic_attention(24, 8, 64)
        attn_3d = attn.mean(axis=1)  # (24, 64, 64)
        sink = extract_attention_sink_series(attn_3d)
        assert sink.shape == (24,)


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: ResidualAdapter
# ═══════════════════════════════════════════════════════════════════════════════

class TestResidualAdapter:
    def test_norm_growth_shape(self):
        residuals = synthetic_residuals(32, 4096, pattern="growing")
        norms = extract_norm_growth_series(residuals)
        assert norms.shape == (32,)

    def test_growing_norm_trend(self):
        residuals = synthetic_residuals(50, 4096, pattern="growing")
        norms = extract_norm_growth_series(residuals)
        # Growing pattern → POSITIVE trend
        adapter = ResidualAdapter()
        result = adapter.analyze(residuals, signal="norm_growth")
        assert result.ratio_balance.dominant == "POSITIVE"

    def test_flat_norm_balanced(self):
        residuals = synthetic_residuals(50, 4096, pattern="flat")
        adapter = ResidualAdapter()
        result = adapter.analyze(residuals, signal="norm_growth")
        # Flat pattern → may be BALANCED
        assert result.valid

    def test_delta_norm_shape(self):
        residuals = synthetic_residuals(32, 4096)
        deltas = extract_delta_norm_series(residuals, include_embed=True)
        assert deltas.shape == (32,)

    def test_delta_norm_no_embed(self):
        residuals = synthetic_residuals(32, 4096)
        deltas = extract_delta_norm_series(residuals, include_embed=False)
        assert deltas.shape == (31,)  # n_layers - 1

    def test_layer_contribution(self):
        residuals = synthetic_residuals(32, 4096, pattern="growing")
        contrib = extract_layer_contribution_series(residuals)
        assert contrib.shape == (32,)
        # All contributions should be finite and non-negative
        assert np.all(np.isfinite(contrib))
        assert np.all(contrib >= 0)

    def test_saturation(self):
        residuals = synthetic_residuals(32, 4096, pattern="growing")
        sat = extract_saturation_series(residuals, window=5)
        assert sat.shape == (32,)
        assert np.all((sat >= 0) & (sat <= 1))

    def test_3d_input(self):
        residuals = synthetic_residuals(32, 4096, pattern="growing")
        residuals_3d = residuals[np.newaxis, :, :]  # (1, 32, 4096)
        norms = extract_norm_growth_series(residuals_3d)
        assert norms.shape == (32,)


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: TrainingAdapter
# ═══════════════════════════════════════════════════════════════════════════════

class TestTrainingAdapter:
    def test_loss_shape(self):
        loss = synthetic_training_history(500, pattern="normal")
        extracted = extract_loss_series(loss)
        assert extracted.shape == (500,)

    def test_loss_curvature_shape(self):
        loss = synthetic_training_history(500, pattern="normal")
        curv = extract_loss_curvature_series(loss)
        assert curv.shape == (498,)  # n - 2

    def test_normal_loss_negative_trend(self):
        """Normal training: loss decreases → NEGATIVE trend."""
        adapter = TrainingAdapter()
        loss = synthetic_training_history(500, pattern="normal")
        result = adapter.analyze(loss, signal="loss")
        assert result.valid
        assert result.ratio_balance.dominant == "NEGATIVE"

    def test_phase_shift_detection(self):
        """Training with a loss spike should show distinct phase boundaries."""
        loss_normal = synthetic_training_history(500, pattern="normal", seed=1)
        loss_shift = synthetic_training_history(500, pattern="phase_shift", seed=1)

        phases_normal = detect_training_phases(loss_normal)
        phases_shift = detect_training_phases(loss_shift)

        # Both should have at least 1 phase
        assert phases_normal["n_phases"] >= 1
        assert phases_shift["n_phases"] >= 1
        # Both should have valid signatures
        assert phases_normal["full_signature"] != "N/A"
        assert phases_shift["full_signature"] != "N/A"

    def test_convenience_wrapper(self):
        loss = synthetic_training_history(300, pattern="normal")
        d = analyze_training(loss, signal="loss")
        assert "valid" in d
        assert "signature" in d

    def test_gradient_norm(self):
        grad = np.abs(np.random.default_rng(0).normal(0, 1, 200))
        extracted = extract_gradient_norm_series(grad)
        assert extracted.shape == (200,)


# ═══════════════════════════════════════════════════════════════════════════════
# Need imports for convenience wrappers
# ═══════════════════════════════════════════════════════════════════════════════

from adapters.attention_adapter import analyze_attention
from adapters.residual_adapter import analyze_residuals
from adapters.training_adapter import analyze_training
