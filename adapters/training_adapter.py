"""TrainingAdapter — frequency signals from training dynamics.

Extracts frequency signatures from metrics recorded during model training.
This is crucial for detecting *when* during training subliminal transfer occurs.

  - **Loss**: Cross-entropy loss over training steps.
    Phase transitions in loss curvature can indicate when a model
    "absorbs" a behavioral trait.

  - **GradientNorm**: L2 norm of gradients over steps.
    Gradient spikes often correlate with phase changes in learning.
    Frequency analysis can reveal periodic patterns in learning dynamics.

  - **LearningRate**: LR schedule over steps.
    Detecting whether frequency changes align with LR boundaries
    (e.g., do phase transitions cluster at LR warmup/decay points?).

  - **TokenEntropy**: Entropy of output distribution over training steps.
    Measures how "surprised" the model is. Sudden drops = memorization
    phase; steady decrease = generalization.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from core.frequency_engine import FrequencyEngine, FrequencyAnalysisResult


# ── Training signal extraction ───────────────────────────────────────────────

def extract_loss_series(loss_history: np.ndarray) -> np.ndarray:
    """Raw or log loss values over training steps.

    Args:
        loss_history: 1D array of loss values at each step.

    Returns:
        1D array, same length as input.
    """
    return np.asarray(loss_history, dtype=np.float64).ravel()


def extract_gradient_norm_series(grad_norm_history: np.ndarray) -> np.ndarray:
    """Gradient L2 norm over training steps.

    Args:
        grad_norm_history: 1D array of gradient norms.

    Returns:
        1D array, same length as input.
    """
    return np.asarray(grad_norm_history, dtype=np.float64).ravel()


def extract_learning_rate_series(lr_history: np.ndarray) -> np.ndarray:
    """Learning rate values over training steps.

    Useful for correlating frequency phase transitions with LR schedule
    boundaries (warmup, decay, constant phases).

    Args:
        lr_history: 1D array of learning rates.

    Returns:
        1D array, same length as input.
    """
    return np.asarray(lr_history, dtype=np.float64).ravel()


def extract_loss_curvature_series(loss_history: np.ndarray) -> np.ndarray:
    """Second derivative (curvature) of loss over time.

    Phase transitions in training → spikes in curvature.
    A positive spike = loss suddenly increasing (forgetting?).
    A negative spike = loss suddenly decreasing (memorization event?).

    Args:
        loss_history: 1D array of loss values.

    Returns:
        1D array of curvature values, len = len(loss) - 2.
    """
    loss = np.asarray(loss_history, dtype=np.float64).ravel()
    if len(loss) < 3:
        return np.array([])
    first_deriv = np.diff(loss)
    second_deriv = np.diff(first_deriv)
    return second_deriv


# ── Available signals ────────────────────────────────────────────────────────

SIGNAL_EXTRACTORS = {
    "loss": extract_loss_series,
    "gradient_norm": extract_gradient_norm_series,
    "learning_rate": extract_learning_rate_series,
    "loss_curvature": extract_loss_curvature_series,
}

TrainingSignalSeries = np.ndarray  # 1D


# ── Phase boundary detection ─────────────────────────────────────────────────

def detect_training_phases(
    loss_history: np.ndarray,
    engine: Optional[FrequencyEngine] = None,
) -> dict:
    """Detect training phase boundaries via frequency analysis.

    Segments training into phases based on loss curvature phase transitions.
    Each segment may correspond to a distinct learning regime.

    Args:
        loss_history: 1D array of loss values.
        engine: Optional FrequencyEngine (creates one if None).

    Returns:
        Dict with phase boundaries and frequency signature for each phase.
    """
    eng = engine or FrequencyEngine(ratio_window=min(50, len(loss_history) // 4))
    loss = np.asarray(loss_history, dtype=np.float64).ravel()

    # Full analysis
    full_result = eng.analyze(loss)
    curvature = extract_loss_curvature_series(loss)

    # Detect phase boundaries where curvature has large swings
    if len(curvature) < 10:
        return {
            "n_phases": 1,
            "full_signature": full_result.signature.string if full_result.signature else "N/A",
            "phase_boundaries": [],
        }

    # Find top curvature spike positions
    curvature_abs = np.abs(curvature)
    threshold = np.percentile(curvature_abs, 90)
    spike_positions = np.where(curvature_abs > threshold)[0] + 1  # +1 for offset

    # Cluster nearby spikes into phase boundaries
    if len(spike_positions) == 0:
        return {
            "n_phases": 1,
            "full_signature": full_result.signature.string if full_result.signature else "N/A",
            "phase_boundaries": [],
        }

    # Simple binning: group spikes within 10% of series length
    min_gap = max(3, len(loss) // 10)
    boundaries = [spike_positions[0]]
    for pos in spike_positions[1:]:
        if pos - boundaries[-1] > min_gap:
            boundaries.append(pos)

    # Analyze each phase
    phases = []
    prev = 0
    for b in boundaries:
        phase_loss = loss[prev:b]
        if len(phase_loss) >= 20:
            phase_result = eng.analyze(phase_loss)
            phases.append({
                "start_step": int(prev),
                "end_step": int(b),
                "signature": phase_result.signature.string if phase_result.signature else "N/A",
                "ratio": round(phase_result.ratio_balance.ratio, 3) if phase_result.ratio_balance else 0.5,
                "dominant": phase_result.ratio_balance.dominant if phase_result.ratio_balance else "?",
            })
        prev = b

    # Last phase
    if prev < len(loss):
        phase_loss = loss[prev:]
        if len(phase_loss) >= 20:
            phase_result = eng.analyze(phase_loss)
            phases.append({
                "start_step": int(prev),
                "end_step": int(len(loss)),
                "signature": phase_result.signature.string if phase_result.signature else "N/A",
                "ratio": round(phase_result.ratio_balance.ratio, 3) if phase_result.ratio_balance else 0.5,
                "dominant": phase_result.ratio_balance.dominant if phase_result.ratio_balance else "?",
            })

    return {
        "n_phases": len(phases),
        "full_signature": full_result.signature.string if full_result.signature else "N/A",
        "phase_boundaries": [int(b) for b in boundaries],
        "phases": phases,
    }


# ── Adapter ──────────────────────────────────────────────────────────────────

class TrainingAdapter:
    """Converts training metric histories into frequency-domain analysis.

    Treats training step as the time dimension.

    Usage:
        adapter = TrainingAdapter()
        result = adapter.analyze(loss_history, signal="loss")
        print(result.signature.string)

        # Detect training phases
        phases = adapter.detect_phases(loss_history)
        print(f"Detected {phases['n_phases']} training phases")
    """

    def __init__(self, engine: Optional[FrequencyEngine] = None) -> None:
        self._engine = engine or FrequencyEngine()

    def extract(
        self,
        history: np.ndarray,
        signal: str = "loss",
        **extract_kwargs,
    ) -> TrainingSignalSeries:
        """Extract a 1D time series from training history.

        Args:
            history: 1D array of metric values.
            signal: One of "loss", "gradient_norm", "learning_rate",
                    "loss_curvature".
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
        return extractor(history, **extract_kwargs)

    def analyze(
        self,
        history: np.ndarray,
        signal: str = "loss",
        **extract_kwargs,
    ) -> FrequencyAnalysisResult:
        """Full frequency analysis on training-derived signal.

        Args:
            history: 1D array of metric values over training steps.
            signal: Signal type to extract and analyze.
            **extract_kwargs: Passed to the extractor.

        Returns:
            FrequencyAnalysisResult.
        """
        series = self.extract(history, signal, **extract_kwargs)
        return self._engine.analyze(series)

    def detect_phases(
        self,
        loss_history: np.ndarray,
    ) -> dict:
        """Convenience: detect training phase boundaries from loss."""
        return detect_training_phases(loss_history, engine=self._engine)


# ── Convenience ──────────────────────────────────────────────────────────────

def analyze_training(
    history: np.ndarray,
    signal: str = "loss",
    **kwargs,
) -> dict:
    """One-shot: training history → frequency analysis dict."""
    adapter = TrainingAdapter()
    result = adapter.analyze(history, signal, **kwargs)
    return result.to_dict()


def detect_training_phase_boundaries(
    loss_history: np.ndarray,
) -> dict:
    """One-shot: detect phase boundaries from loss curvature."""
    return detect_training_phases(loss_history)
