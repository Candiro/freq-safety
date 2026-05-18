"""RatioBalance — directional bias analysis of a time series.

Measures the energy ratio of positive vs negative changes over a window.
Analogous to 'rhythm' in the original system, but generalized to any 1D signal.

Values:
  ratio > 0.55 → POSITIVE bias (UP)
  ratio < 0.45 → NEGATIVE bias (DOWN)
  0.45-0.55    → BALANCED

The ratio is computed as: sum(pos_changes) / (sum(pos_changes) + sum(|neg_changes|))
"""

from __future__ import annotations

from typing import Optional

import numpy as np


# ── Configuration ────────────────────────────────────────────────────────────

DEFAULT_WINDOW: int = 24
POSITIVE_THRESHOLD: float = 0.55
NEGATIVE_THRESHOLD: float = 0.45


# ── Types ────────────────────────────────────────────────────────────────────

class RatioBalanceResult:
    """Result of a single RatioBalance analysis.

    Attributes:
        ratio: Float in [0.0, 1.0]; 0.5 = perfectly balanced.
        dominant: "POSITIVE", "NEGATIVE", or "BALANCED".
        window: Number of samples analyzed.
        stable: True if |ratio - 0.5| > 0.15 (strong bias).
    """

    def __init__(
        self,
        ratio: float,
        dominant: str,
        window: int,
        stable: bool,
    ) -> None:
        self.ratio = ratio
        self.dominant = dominant
        self.window = window
        self.stable = stable

    def __repr__(self) -> str:
        return (
            f"RatioBalance(ratio={self.ratio:.3f}, "
            f"dominant={self.dominant}, "
            f"window={self.window}, "
            f"stable={self.stable})"
        )

    def to_dict(self) -> dict:
        return {
            "ratio": round(self.ratio, 3),
            "dominant": self.dominant,
            "window": self.window,
            "stable": self.stable,
        }


# ── Analyzer ─────────────────────────────────────────────────────────────────

class RatioBalanceAnalyzer:
    """Analyzes directional bias in a 1D time series via ratio of positive to
    total change energy."""

    def __init__(self, window: int = DEFAULT_WINDOW) -> None:
        self.window = window

    def analyze(self, series: np.ndarray) -> RatioBalanceResult:
        """Compute the RatioBalance for the given 1D array.

        Args:
            series: 1D numpy array of numeric values.

        Returns:
            RatioBalanceResult with ratio, dominant direction, and stability.
        """
        if series is None or len(series) < self.window:
            return RatioBalanceResult(0.5, "BALANCED", self.window, False)

        segment = series[-self.window:]
        changes = np.diff(segment)

        pos_energy = float(np.sum(changes[changes > 0]))
        neg_energy = float(np.sum(np.abs(changes[changes < 0])))

        total = pos_energy + neg_energy
        if total == 0.0:
            return RatioBalanceResult(0.5, "BALANCED", self.window, False)

        ratio = pos_energy / total

        if ratio > POSITIVE_THRESHOLD:
            dominant = "POSITIVE"
        elif ratio < NEGATIVE_THRESHOLD:
            dominant = "NEGATIVE"
        else:
            dominant = "BALANCED"

        stable = abs(ratio - 0.5) > 0.15

        return RatioBalanceResult(
            ratio=ratio,
            dominant=dominant,
            window=self.window,
            stable=stable,
        )


# ── Convenience ──────────────────────────────────────────────────────────────

def ratio_balance(
    series: np.ndarray,
    window: int = DEFAULT_WINDOW,
) -> dict:
    """One-shot convenience wrapper."""
    return RatioBalanceAnalyzer(window=window).analyze(series).to_dict()
