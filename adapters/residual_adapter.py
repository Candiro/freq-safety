"""ResidualAdapter — frequency signals from residual stream activations.

The residual stream is the backbone of transformer information flow.
Monitoring its properties through the model depth reveals processing dynamics:

  - **NormGrowth**: L2 norm of the residual stream at each layer.
    Tracks how much the representation amplifies through the network.
    Unusual growth patterns can indicate processing anomalies.

  - **DeltaNorm**: L2 norm of the *change* added by each layer
    (the output of the attention+MLP sublayers). Captures per-layer
    contribution magnitude.

  - **LayerContribution**: Ratio of layer output norm to residual norm.
    Measures how much each layer changes the representation relative to
    its input. Low contribution = "lazy" layer; high = dominant.

  - **Saturation**: How close the residual norm is to a running max.
    Detects when the model is pushing against representational limits.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from core.frequency_engine import FrequencyEngine, FrequencyAnalysisResult


# ── Residual signal extraction ───────────────────────────────────────────────

def extract_norm_growth_series(
    residual_activations: np.ndarray,
) -> np.ndarray:
    """L2 norm of the residual stream vector at each layer position.

    Args:
        residual_activations: Shape (n_layers, hidden_dim) or
                              (n_positions, n_layers, hidden_dim).
                              If 3D, averages over positions first.

    Returns:
        1D array of norm values, length n_layers.
    """
    if residual_activations.ndim == 3:
        # Average over token positions first
        residual_activations = residual_activations.mean(axis=0)

    if residual_activations.ndim != 2:
        raise ValueError(
            f"Expected 2D (n_layers, hidden_dim) or 3D, got {residual_activations.ndim}D"
        )

    return np.linalg.norm(residual_activations, axis=1)


def extract_delta_norm_series(
    residual_activations: np.ndarray,
    include_embed: bool = True,
) -> np.ndarray:
    """L2 norm of the change added by each layer.

    Computes ||residual[i] - residual[i-1]|| for each layer.
    This captures the magnitude of each layer's contribution.

    Args:
        residual_activations: Shape (n_layers, hidden_dim) — activations
                              AFTER each layer's contribution.
        include_embed: If True, also computes delta for layer 0
                       (embedding → first layer). Default True.

    Returns:
        1D array of delta norms, length n_layers (or n_layers - 1
        if include_embed=False).
    """
    if residual_activations.ndim == 3:
        residual_activations = residual_activations.mean(axis=0)

    if residual_activations.ndim != 2:
        raise ValueError(f"Expected 2D or 3D, got {residual_activations.ndim}D")

    if include_embed:
        # Pad: treat embedding as zeros starting point
        padded = np.vstack([np.zeros((1, residual_activations.shape[1])),
                            residual_activations])
    else:
        padded = residual_activations

    diffs = np.diff(padded, axis=0)
    return np.linalg.norm(diffs, axis=1)


def extract_layer_contribution_series(
    residual_activations: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """Ratio of per-layer delta norm to total residual norm.

    Measures what fraction of the total residual magnitude each layer
    contributes. Values near 0 = layer barely changes the representation;
    high values = dominant layer.

    Args:
        residual_activations: Shape (n_layers, hidden_dim).

    Returns:
        1D array of contribution ratios, length n_layers.
    """
    residual_norms = extract_norm_growth_series(residual_activations)
    delta_norms = extract_delta_norm_series(residual_activations, include_embed=True)

    # Avoid division by zero
    contributions = delta_norms / (residual_norms + eps)
    return contributions


def extract_saturation_series(
    residual_activations: np.ndarray,
    window: int = 5,
) -> np.ndarray:
    """How close each layer's norm is to the running maximum.

    Values near 1.0 = representation is growing (new information).
    Values near 0.0 = saturation (model stopped learning).

    Args:
        residual_activations: Shape (n_layers, hidden_dim).
        window: Rolling window for computing max. Default 5.

    Returns:
        1D array of saturation values (0.0 to 1.0), length n_layers.
    """
    norms = extract_norm_growth_series(residual_activations)

    # Running max over window
    running_max = np.array([
        norms[max(0, i - window + 1):i + 1].max()
        for i in range(len(norms))
    ])

    # Avoid division with very small norms
    running_max = np.maximum(running_max, 1e-8)
    saturation = norms / running_max
    return saturation


# ── Available signals ────────────────────────────────────────────────────────

SIGNAL_EXTRACTORS = {
    "norm_growth": extract_norm_growth_series,
    "delta_norm": extract_delta_norm_series,
    "layer_contribution": extract_layer_contribution_series,
    "saturation": extract_saturation_series,
}

ResidualSignalSeries = np.ndarray  # 1D


# ── Adapter ──────────────────────────────────────────────────────────────────

class ResidualAdapter:
    """Converts residual stream activations into frequency-domain analysis.

    Treats layer depth as time. Useful for detecting:
      - Where in the model processing bottlenecks occur
      - Abrupt changes in layer behavior (potential tampering)
      - Saturation patterns that precede alignment failures

    Usage:
        adapter = ResidualAdapter()
        result = adapter.analyze(residual_acts, signal="norm_growth")
        print(result.signature.string)
    """

    def __init__(self, engine: Optional[FrequencyEngine] = None) -> None:
        self._engine = engine or FrequencyEngine()

    def extract(
        self,
        residuals: np.ndarray,
        signal: str = "norm_growth",
        **extract_kwargs,
    ) -> ResidualSignalSeries:
        """Extract a 1D time series from residual stream activations.

        Args:
            residuals: Shape (n_layers, hidden_dim).
            signal: One of "norm_growth", "delta_norm",
                    "layer_contribution", "saturation".
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
        return extractor(residuals, **extract_kwargs)

    def analyze(
        self,
        residuals: np.ndarray,
        signal: str = "norm_growth",
        **extract_kwargs,
    ) -> FrequencyAnalysisResult:
        """Full frequency analysis on residual-derived signal."""
        series = self.extract(residuals, signal, **extract_kwargs)
        return self._engine.analyze(series)


# ── Convenience ──────────────────────────────────────────────────────────────

def analyze_residuals(
    residuals: np.ndarray,
    signal: str = "norm_growth",
    **kwargs,
) -> dict:
    """One-shot: residual activations → frequency analysis dict."""
    adapter = ResidualAdapter()
    result = adapter.analyze(residuals, signal, **kwargs)
    return result.to_dict()
