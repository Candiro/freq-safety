"""ActivationAdapter — extracts frequency signals from layer activations.

Treats each layer's activation vector as a "point in time," with the
sequence of layers forming a 1D time series. This allows frequency-domain
analysis of how model internals evolve through the network depth.

Extracted signals:
  - **MeanActivation**: Mean of all neuron activations at each layer.
    Tracks overall activation level through the model.
  - **NormActivation**: L2 norm of the activation vector per layer.
    Captures the total "energy" flowing through each layer.
  - **Sparsity**: Fraction of neurons with near-zero activation per layer.
    High sparsity = few active features; low sparsity = diffuse activation.
  - **Variance**: Variance across neurons per layer.
    Measures how differentiated the representations are.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from core.frequency_engine import FrequencyEngine, FrequencyAnalysisResult


# ── Activation signal extraction ─────────────────────────────────────────────

def extract_mean_activation_series(
    activations: np.ndarray,
) -> np.ndarray:
    """Mean activation value at each layer.

    Tracks the average activation intensity through the model depth.

    Args:
        activations: Shape (n_layers, hidden_dim) or (n_layers, n_neurons).

    Returns:
        1D array of mean activation values, length n_layers.
    """
    if activations.ndim != 2:
        raise ValueError(
            f"Expected 2D array (n_layers, hidden_dim), got shape {activations.shape}"
        )
    return activations.mean(axis=1)


def extract_norm_activation_series(
    activations: np.ndarray,
) -> np.ndarray:
    """L2 norm of the activation vector at each layer.

    Captures the total "energy" or magnitude of activation per layer.

    Args:
        activations: Shape (n_layers, hidden_dim).

    Returns:
        1D array of L2 norms, length n_layers.
    """
    return np.linalg.norm(activations, axis=1)


def extract_sparsity_series(
    activations: np.ndarray,
    threshold: float = 0.01,
) -> np.ndarray:
    """Fraction of neurons below threshold at each layer.

    High sparsity → only a few neurons are active (specialized).
    Low sparsity → many neurons are contributing (diffuse).

    Args:
        activations: Shape (n_layers, hidden_dim).
        threshold: Activation magnitude below which a neuron is considered
                   "inactive" (default 0.01).

    Returns:
        1D array of sparsity values (0.0 to 1.0), length n_layers.
    """
    return (np.abs(activations) < threshold).mean(axis=1)


def extract_variance_series(
    activations: np.ndarray,
) -> np.ndarray:
    """Variance across neurons at each layer.

    Measures how differentiated the representations are.
    Low variance → all neurons fire similarly (undifferentiated).
    High variance → neurons are specialized.

    Args:
        activations: Shape (n_layers, hidden_dim).

    Returns:
        1D array of variance values, length n_layers.
    """
    return activations.var(axis=1)


# ── Available signals ────────────────────────────────────────────────────────

SIGNAL_EXTRACTORS = {
    "mean_activation": extract_mean_activation_series,
    "norm_activation": extract_norm_activation_series,
    "sparsity": extract_sparsity_series,
    "variance": extract_variance_series,
}

ActivationSignalSeries = np.ndarray  # 1D


# ── Adapter ──────────────────────────────────────────────────────────────────

class ActivationAdapter:
    """Converts layer activations into frequency-domain analysis.

    Treats layer depth as a time dimension.
    Useful for detecting abrupt changes in internal processing patterns.

    Usage:
        adapter = ActivationAdapter()
        result = adapter.analyze(activations, signal="norm_activation")
        print(result.signature.string)
    """

    def __init__(self, engine: Optional[FrequencyEngine] = None) -> None:
        self._engine = engine or FrequencyEngine()

    def extract(
        self,
        activations: np.ndarray,
        signal: str = "norm_activation",
        **extract_kwargs,
    ) -> ActivationSignalSeries:
        """Extract a 1D time series from layer activations.

        Args:
            activations: Shape (n_layers, hidden_dim).
            signal: One of "mean_activation", "norm_activation",
                    "sparsity", "variance".
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
        return extractor(activations, **extract_kwargs)

    def analyze(
        self,
        activations: np.ndarray,
        signal: str = "norm_activation",
        **extract_kwargs,
    ) -> FrequencyAnalysisResult:
        """Full frequency analysis on an activation-derived signal.

        Args:
            activations: Shape (n_layers, hidden_dim).
            signal: Signal type to extract and analyze.
            **extract_kwargs: Passed to the extractor.

        Returns:
            FrequencyAnalysisResult.
        """
        series = self.extract(activations, signal, **extract_kwargs)
        return self._engine.analyze(series)


# ── Convenience ──────────────────────────────────────────────────────────────

def analyze_activations(
    activations: np.ndarray,
    signal: str = "norm_activation",
    **kwargs,
) -> dict:
    """One-shot: layer activations → frequency analysis dict."""
    adapter = ActivationAdapter()
    result = adapter.analyze(activations, signal, **kwargs)
    return result.to_dict()
