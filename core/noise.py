"""DirectionChangeRate — noise level estimation via direction-change density.

Compares short-window direction-change frequency to long-window baseline.

If short-term changes happen much faster than long-term → HIGH noise (chaotic).
If short-term is quieter → LOW noise (clear signal).
If similar → MEDIUM.

Analogy: A smooth sine wave has LOW direction-change rate; random walk has HIGH.
"""

from __future__ import annotations

import numpy as np


# ── Configuration ────────────────────────────────────────────────────────────

DEFAULT_SHORT_WINDOW: int = 20
DEFAULT_LONG_WINDOW: int = 100
HIGH_NOISE_THRESHOLD: float = 0.10
LOW_NOISE_THRESHOLD: float = -0.10


# ── Types ────────────────────────────────────────────────────────────────────

class DirectionChangeResult:
    """Result of noise / direction-change analysis.

    Attributes:
        level: "HIGH", "MEDIUM", or "LOW".
        short_rate: Direction changes per sample in the short window.
        long_rate: Direction changes per sample in the long window.
        direction_changes: Absolute count in the short window.
    """

    def __init__(
        self,
        level: str,
        short_rate: float,
        long_rate: float,
        direction_changes: int,
    ) -> None:
        self.level = level
        self.short_rate = short_rate
        self.long_rate = long_rate
        self.direction_changes = direction_changes

    def __repr__(self) -> str:
        return (
            f"DirectionChange(level={self.level}, "
            f"short_rate={self.short_rate:.4f}, "
            f"long_rate={self.long_rate:.4f}, "
            f"changes={self.direction_changes})"
        )

    def to_dict(self) -> dict:
        return {
            "level": self.level,
            "short_rate": round(self.short_rate, 4),
            "long_rate": round(self.long_rate, 4),
            "direction_changes": self.direction_changes,
        }


# ── Analyzer ─────────────────────────────────────────────────────────────────

class DirectionChangeAnalyzer:
    """Estimates noise level by comparing short vs long direction-change rates."""

    def __init__(
        self,
        short_window: int = DEFAULT_SHORT_WINDOW,
        long_window: int = DEFAULT_LONG_WINDOW,
    ) -> None:
        self.short_window = short_window
        self.long_window = long_window

    def analyze(self, series: np.ndarray) -> DirectionChangeResult:
        """Compute noise level from direction-change density.

        Args:
            series: 1D numpy array.

        Returns:
            DirectionChangeResult with noise level and rates.
        """
        if series is None or len(series) < self.long_window:
            return DirectionChangeResult("UNKNOWN", 0.0, 0.0, 0)

        short_seg = series[-self.short_window:]
        long_seg = series[-self.long_window:]

        short_rate = self._direction_change_rate(short_seg)
        long_rate = self._direction_change_rate(long_seg)
        diff = short_rate - long_rate

        if diff > HIGH_NOISE_THRESHOLD:
            level = "HIGH"
        elif diff < LOW_NOISE_THRESHOLD:
            level = "LOW"
        else:
            level = "MEDIUM"

        # Absolute direction changes in the short window
        s_changes = np.diff(short_seg)
        s_signs = np.sign(s_changes)
        s_nonzero = s_signs[s_signs != 0]
        changes = int(np.sum(np.diff(s_nonzero) != 0)) if len(s_nonzero) > 1 else 0

        return DirectionChangeResult(
            level=level,
            short_rate=round(short_rate, 4),
            long_rate=round(long_rate, 4),
            direction_changes=changes,
        )

    @staticmethod
    def _direction_change_rate(segment: np.ndarray) -> float:
        """Fraction of adjacent differences that change direction."""
        changes = np.diff(segment)
        if len(changes) < 2:
            return 0.0
        signs = np.sign(changes)
        nonzero = signs[signs != 0]
        if len(nonzero) < 2:
            return 0.0
        flips = int(np.sum(np.diff(nonzero) != 0))
        return flips / max(len(changes), 1)


# ── Convenience ──────────────────────────────────────────────────────────────

def direction_change_rate(
    series: np.ndarray,
    short_window: int = DEFAULT_SHORT_WINDOW,
    long_window: int = DEFAULT_LONG_WINDOW,
) -> dict:
    """One-shot convenience wrapper."""
    return DirectionChangeAnalyzer(
        short_window=short_window,
        long_window=long_window,
    ).analyze(series).to_dict()
