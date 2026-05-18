"""AttentionAdapter — frequency signals from attention patterns.

Extracts time series from attention matrices across layers/heads for
frequency-domain analysis. Attention patterns can reveal behavioral shifts:

  - **AttentionSink**: Fraction of attention weight on the first token.
    A high sink ratio is normal in LLMs; sudden changes may indicate
    distribution shift or adversarial input.

  - **AttentionEntropy**: Shannon entropy of attention weights per head.
    Low entropy = focused/spiky attention; high entropy = diffuse attention.
    Jailbreaks often cause sudden entropy spikes.

  - **HeadVariance**: Variance across attention heads within a layer.
    Low variance = heads are redundant; high variance = heads specialized.
    Alignment failures may manifest as head specialization changes.

  - **Sparsity**: Fraction of attention weights near zero.
    Very sparse attention = model is "tuning out" certain tokens.
    Can detect context-manipulation attacks.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from core.frequency_engine import FrequencyEngine, FrequencyAnalysisResult


# ── Attention signal extraction ──────────────────────────────────────────────

def extract_attention_sink_series(
    attention_matrices: np.ndarray,
) -> np.ndarray:
    """Fraction of attention weight on the [CLS]/first token per layer.

    For each layer, average across heads: how much attention does the
    first token receive?

    Args:
        attention_matrices: Shape (n_layers, n_heads, seq_len, seq_len)
                          or (n_layers, seq_len, seq_len).

    Returns:
        1D array of sink ratios, length n_layers.
    """
    if attention_matrices.ndim == 4:
        # Average over heads
        attn = attention_matrices.mean(axis=1)  # (n_layers, seq_len, seq_len)
    elif attention_matrices.ndim == 3:
        attn = attention_matrices
    else:
        raise ValueError(
            f"Expected 3D or 4D, got {attention_matrices.ndim}D"
        )

    # Attention to first token from all other tokens
    # Column 0 = attention *to* first token
    sink = attn[:, :, 0].mean(axis=1)  # (n_layers,) — avg fraction on first token
    return sink


def extract_attention_entropy_series(
    attention_matrices: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """Shannon entropy of attention distributions per layer.

    For each layer, average head entropy. Low entropy = focused (few tokens
    attended to). High entropy = diffuse (wide attention).

    Args:
        attention_matrices: Shape (n_layers, n_heads, seq_len, seq_len).
        eps: Small constant for numerical stability.

    Returns:
        1D array of entropy values, length n_layers.
    """
    if attention_matrices.ndim != 4:
        # Try to handle 3D by adding heads dimension
        if attention_matrices.ndim == 3:
            attention_matrices = attention_matrices[:, np.newaxis, :, :]
        else:
            raise ValueError(f"Expected 3D or 4D, got {attention_matrices.ndim}D")

    n_layers, n_heads, seq_len, _ = attention_matrices.shape

    # Compute entropy per head: -sum(p * log(p)) averaged across rows
    # (each row is a query token's attention distribution)
    attn_clipped = np.clip(attention_matrices, eps, 1.0)
    entropies = -np.sum(attn_clipped * np.log(attn_clipped), axis=3)  # (n_layers, n_heads, seq_len)
    # Average over heads and tokens
    layer_entropy = entropies.mean(axis=(1, 2))  # (n_layers,)
    return layer_entropy


def extract_head_variance_series(
    attention_matrices: np.ndarray,
) -> np.ndarray:
    """Variance across attention heads within each layer.

    Low variance → heads converge (redundant processing).
    High variance → heads are specialized (diverse processing).

    Args:
        attention_matrices: Shape (n_layers, n_heads, seq_len, seq_len).

    Returns:
        1D array of head variance values, length n_layers.
    """
    if attention_matrices.ndim != 4:
        if attention_matrices.ndim == 3:
            # Single head — variance is trivially 0
            return np.zeros(attention_matrices.shape[0])
        raise ValueError(f"Expected 3D or 4D, got {attention_matrices.ndim}D")

    # For each layer, compute variance of attention patterns across heads
    # Flatten each head's attention matrix
    n_layers, n_heads = attention_matrices.shape[:2]
    flattened = attention_matrices.reshape(n_layers, n_heads, -1)  # (N, H, S*S)
    # Variance across heads per position, then average
    head_var = flattened.var(axis=1).mean(axis=1)  # (n_layers,)
    return head_var


def extract_attention_sparsity_series(
    attention_matrices: np.ndarray,
    threshold: float = 0.01,
) -> np.ndarray:
    """Fraction of attention weights below threshold per layer.

    High sparsity = model is focusing on very few tokens.
    Sudden drop in sparsity = model "spreading attention" unnaturally.

    Args:
        attention_matrices: Shape (n_layers, n_heads, seq_len, seq_len).
        threshold: Weight below which is considered "zero" (default 0.01).

    Returns:
        1D array of sparsity values, length n_layers.
    """
    if attention_matrices.ndim == 4:
        attn = attention_matrices.mean(axis=1)  # average over heads
    elif attention_matrices.ndim == 3:
        attn = attention_matrices
    else:
        raise ValueError(f"Expected 3D or 4D, got {attention_matrices.ndim}D")

    sparsity = (attn < threshold).mean(axis=(1, 2))  # (n_layers,)
    return sparsity


# ── Available signals ────────────────────────────────────────────────────────

SIGNAL_EXTRACTORS = {
    "sink": extract_attention_sink_series,
    "entropy": extract_attention_entropy_series,
    "head_variance": extract_head_variance_series,
    "sparsity": extract_attention_sparsity_series,
}

AttentionSignalSeries = np.ndarray  # 1D


# ── Adapter ──────────────────────────────────────────────────────────────────

class AttentionAdapter:
    """Converts attention matrices into frequency-domain analysis.

    Treats layer depth as a time dimension — how does attention behavior
    evolve through the model?

    Usage:
        adapter = AttentionAdapter()
        result = adapter.analyze(attention_matrices, signal="entropy")
        print(result.signature.string)
    """

    def __init__(self, engine: Optional[FrequencyEngine] = None) -> None:
        self._engine = engine or FrequencyEngine()

    def extract(
        self,
        attention_matrices: np.ndarray,
        signal: str = "sink",
        **extract_kwargs,
    ) -> AttentionSignalSeries:
        """Extract a 1D time series from attention matrices.

        Args:
            attention_matrices: Shape (n_layers, n_heads, seq_len, seq_len).
            signal: One of "sink", "entropy", "head_variance", "sparsity".
            **extract_kwargs: Passed to the extractor.

        Returns:
            1D numpy array.

        Raises:
            ValueError: If signal name is unknown.
        """
        if signal not in SIGNAL_EXTRACTORS:
            raise ValueError(
                f"Unknown signal '{signal}'. "
                f"Available: {list(SIGNAL_EXTRACTORS.keys())}"
            )
        extractor = SIGNAL_EXTRACTORS[signal]
        return extractor(attention_matrices, **extract_kwargs)

    def analyze(
        self,
        attention_matrices: np.ndarray,
        signal: str = "sink",
        **extract_kwargs,
    ) -> FrequencyAnalysisResult:
        """Full frequency analysis on attention-derived signal.

        Args:
            attention_matrices: Shape (n_layers, n_heads, seq_len, seq_len).
            signal: Signal type to extract and analyze.
            **extract_kwargs: Passed to the extractor.

        Returns:
            FrequencyAnalysisResult.
        """
        series = self.extract(attention_matrices, signal, **extract_kwargs)
        return self._engine.analyze(series)


# ── Convenience ──────────────────────────────────────────────────────────────

def analyze_attention(
    attention_matrices: np.ndarray,
    signal: str = "sink",
    **kwargs,
) -> dict:
    """One-shot: attention matrices → frequency analysis dict."""
    adapter = AttentionAdapter()
    result = adapter.analyze(attention_matrices, signal, **kwargs)
    return result.to_dict()
