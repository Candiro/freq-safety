"""LogitAdapter — extracts frequency signals from LLM output logits.

Converts a sequence of logit vectors (token-level model outputs) into
1D time series suitable for FrequencyEngine analysis.

Extracted signals:
  - **Entropy**: Shannon entropy of the probability distribution over tokens.
    Measures model uncertainty over time. High entropy = uncertain / flat.
  - **MaxLogit**: Value of the highest logit (pre-softmax confidence).
    Rising max logit → increasing confidence; falling → uncertainty.
  - **TopGap**: Gap between top-1 and top-2 logits. Gap widening = model
    becoming more decisive; narrowing = competition between answers.
  - **TokenEntropy**: Entropy computed on the *top-k* tokens only, ignoring
    the long tail. Captures "which of the plausible answers" uncertainty.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from core.frequency_engine import FrequencyEngine, FrequencyAnalysisResult


# ── Logit signal extraction ──────────────────────────────────────────────────

def extract_entropy_series(
    logit_vectors: np.ndarray,
    temperature: float = 1.0,
) -> np.ndarray:
    """Shannon entropy of the softmax distribution for each token position.

    High entropy = model is unsure / distribution is flat.
    Low entropy = model is confident / distribution is peaked.

    Args:
        logit_vectors: Shape (n_tokens, vocab_size) or (n_positions, n_features).
                       Logits before softmax.
        temperature: Softmax temperature (default 1.0).

    Returns:
        1D array of entropy values, length n_tokens.
    """
    if logit_vectors.ndim != 2:
        raise ValueError(f"Expected 2D array (n_tokens, vocab_size), got shape {logit_vectors.shape}")

    logits = logit_vectors / temperature
    # Softmax (stable)
    logits_max = logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits - logits_max)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    # Entropy: -sum(p * log(p))
    eps = 1e-12
    entropy = -np.sum(probs * np.log(np.clip(probs, eps, 1.0)), axis=1)
    return entropy


def extract_max_logit_series(logit_vectors: np.ndarray) -> np.ndarray:
    """Maximum logit value at each token position.

    Pre-softmax confidence indicator. Rising values suggest the model is
    becoming more certain; falling suggest hesitation.

    Args:
        logit_vectors: Shape (n_tokens, vocab_size).

    Returns:
        1D array of max logit values, length n_tokens.
    """
    return logit_vectors.max(axis=1)


def extract_top_gap_series(logit_vectors: np.ndarray) -> np.ndarray:
    """Gap between top-1 and top-2 logits at each token position.

    Widening gap → model strongly prefers one token.
    Narrowing gap → two tokens are competing.

    Args:
        logit_vectors: Shape (n_tokens, vocab_size).

    Returns:
        1D array of top-gap values, length n_tokens.
    """
    sorted_logits = np.sort(logit_vectors, axis=1)
    return sorted_logits[:, -1] - sorted_logits[:, -2]


def extract_top_k_entropy_series(
    logit_vectors: np.ndarray,
    k: int = 10,
    temperature: float = 1.0,
) -> np.ndarray:
    """Entropy over the top-k tokens only.

    Ignores the long tail. Useful for measuring "decisiveness" between
    plausible candidates rather than overall uncertainty.

    Args:
        logit_vectors: Shape (n_tokens, vocab_size).
        k: Number of top tokens to consider.
        temperature: Softmax temperature.

    Returns:
        1D array of top-k entropy values.
    """
    logits = logit_vectors / temperature
    # Get top-k indices and values
    topk_indices = np.argpartition(logits, -k, axis=1)[:, -k:]
    rows = np.arange(logits.shape[0])[:, None]
    topk_logits = logits[rows, topk_indices]
    # Softmax over top-k only
    topk_max = topk_logits.max(axis=1, keepdims=True)
    topk_exp = np.exp(topk_logits - topk_max)
    topk_probs = topk_exp / topk_exp.sum(axis=1, keepdims=True)
    eps = 1e-12
    entropy = -np.sum(topk_probs * np.log(np.clip(topk_probs, eps, 1.0)), axis=1)
    return entropy


# ── Available signals ────────────────────────────────────────────────────────

SIGNAL_EXTRACTORS = {
    "entropy": extract_entropy_series,
    "max_logit": extract_max_logit_series,
    "top_gap": extract_top_gap_series,
    "top_k_entropy": extract_top_k_entropy_series,
}

LogitSignalSeries = np.ndarray  # 1D


# ── Adapter ──────────────────────────────────────────────────────────────────

class LogitAdapter:
    """Converts logit vectors into frequency-domain analysis.

    Usage:
        adapter = LogitAdapter()
        result = adapter.analyze(logit_vectors, signal="entropy")
        print(result.signature.string)
    """

    def __init__(self, engine: Optional[FrequencyEngine] = None) -> None:
        self._engine = engine or FrequencyEngine()

    def extract(
        self,
        logit_vectors: np.ndarray,
        signal: str = "entropy",
        **extract_kwargs,
    ) -> LogitSignalSeries:
        """Extract a 1D time series from logit vectors.

        Args:
            logit_vectors: Shape (n_tokens, vocab_size).
            signal: One of "entropy", "max_logit", "top_gap", "top_k_entropy".
            **extract_kwargs: Passed to the extractor (e.g., temperature, k).

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
        return extractor(logit_vectors, **extract_kwargs)

    def analyze(
        self,
        logit_vectors: np.ndarray,
        signal: str = "entropy",
        **extract_kwargs,
    ) -> FrequencyAnalysisResult:
        """Full frequency analysis on a logit-derived signal.

        Args:
            logit_vectors: Shape (n_tokens, vocab_size).
            signal: Signal type to extract and analyze.
            **extract_kwargs: Passed to the extractor.

        Returns:
            FrequencyAnalysisResult.
        """
        series = self.extract(logit_vectors, signal, **extract_kwargs)
        return self._engine.analyze(series)


# ── Convenience ──────────────────────────────────────────────────────────────

def analyze_logits(
    logit_vectors: np.ndarray,
    signal: str = "entropy",
    **kwargs,
) -> dict:
    """One-shot: logit vectors → frequency analysis dict."""
    adapter = LogitAdapter()
    result = adapter.analyze(logit_vectors, signal, **kwargs)
    return result.to_dict()
