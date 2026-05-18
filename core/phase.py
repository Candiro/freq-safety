"""PhaseTransition — detection of sudden shifts in dominant behavioral mode.

Tracks RatioBalance results across consecutive windows and detects when
the dominant direction changes. A phase transition indicates a potential
behavioral regime shift — e.g., a model switching from cooperative to
adversarial behavior, or from a stable output pattern to a chaotic one.

This is the generalization of 'phase shift' detection — applicable to any
1D signal where regime changes are meaningful.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

from .rhythm import RatioBalanceAnalyzer, RatioBalanceResult


# ── Configuration ────────────────────────────────────────────────────────────

HISTORY_DEPTH: int = 10
MIN_HISTORY: int = 3
TRANSITION_STRENGTH_THRESHOLD: float = 0.10


# ── Types ────────────────────────────────────────────────────────────────────

class PhaseTransitionResult:
    """Result of phase transition detection.

    Attributes:
        transitioned: True if a dominant-direction change was detected.
        from_dominant: Previous dominant direction or None.
        to_dominant: New dominant direction or None.
        strength: Magnitude of the ratio change that triggered the transition.
        total_transitions: Cumulative count of transitions observed.
    """

    def __init__(
        self,
        transitioned: bool,
        from_dominant: Optional[str],
        to_dominant: Optional[str],
        strength: float,
        total_transitions: int,
    ) -> None:
        self.transitioned = transitioned
        self.from_dominant = from_dominant
        self.to_dominant = to_dominant
        self.strength = strength
        self.total_transitions = total_transitions

    def __repr__(self) -> str:
        if self.transitioned:
            return (
                f"PhaseTransition ⚡ {self.from_dominant} → {self.to_dominant} "
                f"(strength={self.strength:.2f}, total={self.total_transitions})"
            )
        return f"PhaseTransition ✗ stable (total={self.total_transitions})"

    def to_dict(self) -> dict:
        return {
            "transitioned": self.transitioned,
            "from_dominant": self.from_dominant,
            "to_dominant": self.to_dominant,
            "strength": round(self.strength, 3),
            "total_transitions": self.total_transitions,
        }


# ── Detector ─────────────────────────────────────────────────────────────────

class PhaseTransitionDetector:
    """Detects transitions between dominant behavioral modes in a time series.

    Maintains a rolling history of RatioBalance results and flags when the
    dominant direction changes between consecutive windows.
    """

    def __init__(self, window: int = 24) -> None:
        self._analyzer = RatioBalanceAnalyzer(window=window)
        self._history: deque[RatioBalanceResult] = deque(maxlen=HISTORY_DEPTH)
        self.total_transitions: int = 0

    def update(self, series: np.ndarray) -> PhaseTransitionResult:
        """Feed a new series window and check for phase transitions.

        Args:
            series: 1D numpy array (full history; last `window` samples used).

        Returns:
            PhaseTransitionResult indicating whether a transition occurred.
        """
        result = self._analyzer.analyze(series)
        self._history.append(result)

        if len(self._history) < MIN_HISTORY:
            return PhaseTransitionResult(False, None, None, 0.0, self.total_transitions)

        prev = self._history[-MIN_HISTORY]
        curr = self._history[-1]

        if (
            prev.dominant != curr.dominant
            and curr.dominant != "BALANCED"
        ):
            strength = abs(prev.ratio - curr.ratio)
            if strength >= TRANSITION_STRENGTH_THRESHOLD:
                self.total_transitions += 1
                return PhaseTransitionResult(
                    transitioned=True,
                    from_dominant=prev.dominant,
                    to_dominant=curr.dominant,
                    strength=strength,
                    total_transitions=self.total_transitions,
                )

        return PhaseTransitionResult(False, None, None, 0.0, self.total_transitions)

    def reset(self) -> None:
        """Clear transition history (for fresh analysis)."""
        self._history.clear()
        self.total_transitions = 0


# ── Convenience ──────────────────────────────────────────────────────────────

def detect_phase_transition(
    series: np.ndarray,
    window: int = 24,
    detector: Optional[PhaseTransitionDetector] = None,
) -> dict:
    """One-shot convenience wrapper.

    Args:
        series: 1D numpy array.
        window: Analysis window size.
        detector: Optional persistent detector for multi-call tracking.
                  If None, a fresh detector is created (no history).

    Returns:
        PhaseTransitionResult as dict.
    """
    d = detector or PhaseTransitionDetector(window=window)
    return d.update(series).to_dict()
