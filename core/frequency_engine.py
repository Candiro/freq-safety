"""frequency_engine.py — Core orchestrator for FREQ-SAFE.

Takes any 1D time series and produces a complete frequency-domain analysis:
  - RatioBalance (directional bias)
  - ZeroCrossingRate (volatility frequency)
  - DirectionChangeRate (noise level)
  - PhaseTransition (regime shifts)
  - FrequencySignature (compressed fingerprint)

This is the main entry point for the framework.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .rhythm import RatioBalanceAnalyzer, RatioBalanceResult
from .noise import DirectionChangeAnalyzer, DirectionChangeResult
from .phase import PhaseTransitionDetector, PhaseTransitionResult
from .signature import FrequencySignature


# ── Configuration ────────────────────────────────────────────────────────────

DEFAULT_RATIO_WINDOW: int = 24
DEFAULT_NOISE_SHORT: int = 20
DEFAULT_NOISE_LONG: int = 100
DEFAULT_FREQ_SHORT: int = 20
DEFAULT_FREQ_LONG: int = 100
MIN_SERIES_LENGTH: int = 20


# ── ZeroCrossingRate (frequency/volatility) ──────────────────────────────────

class ZeroCrossingResult:
    """Result of zero-crossing rate analysis.

    Attributes:
        level: "HIGH", "MEDIUM", "LOW", "UNKNOWN"
        short_rate: Zero-crossings per sample (short window).
        long_rate: Zero-crossings per sample (long window).
        diff: short_rate - long_rate; positive = getting faster.
    """

    def __init__(
        self,
        level: str,
        short_rate: float,
        long_rate: float,
        diff: float,
    ) -> None:
        self.level = level
        self.short_rate = short_rate
        self.long_rate = long_rate
        self.diff = diff

    def __repr__(self) -> str:
        return (
            f"ZeroCrossing(level={self.level}, "
            f"short={self.short_rate:.3f}, "
            f"long={self.long_rate:.3f}, "
            f"diff={self.diff:+.3f})"
        )

    def to_dict(self) -> dict:
        return {
            "level": self.level,
            "short_rate": round(self.short_rate, 3),
            "long_rate": round(self.long_rate, 3),
            "diff": round(self.diff, 3),
        }


def zero_crossing_rate(
    series: np.ndarray,
    short_window: int = DEFAULT_FREQ_SHORT,
    long_window: int = DEFAULT_FREQ_LONG,
) -> ZeroCrossingResult:
    """Compute zero-crossing rate — how often the signal crosses its mean.

    Compares short-window rate to long-window baseline.
    Higher short-term rate = faster oscillation / higher volatility.

    Args:
        series: 1D numpy array.
        short_window: Window for short-term rate.
        long_window: Window for long-term rate.

    Returns:
        ZeroCrossingResult with level and rates.
    """
    if series is None or len(series) < long_window:
        return ZeroCrossingResult("UNKNOWN", 0.0, 0.0, 0.0)

    short_seg = series[-short_window:]
    long_seg = series[-long_window:]

    def _zcr(seg: np.ndarray) -> float:
        changes = np.diff(seg)
        if len(changes) < 1:
            return 0.0
        sign_flips = np.sum(np.diff(np.sign(changes)) != 0)
        return sign_flips / max(len(changes), 1)

    s_rate = _zcr(short_seg)
    l_rate = _zcr(long_seg)
    diff = s_rate - l_rate

    # If short-term is slower than long-term → LOW freq (calmer)
    # If short-term is faster → HIGH freq (more volatile)
    if diff > 0.05:
        level = "HIGH"
    elif diff < -0.05:
        level = "LOW"
    else:
        level = "MEDIUM"

    return ZeroCrossingResult(level, s_rate, l_rate, diff)


# ── Full Analysis Result ─────────────────────────────────────────────────────

class FrequencyAnalysisResult:
    """Complete result from a full frequency-domain analysis.

    Attributes:
        valid: True if the analysis had enough data.
        ratio_balance: RatioBalanceResult or None.
        zero_crossing: ZeroCrossingResult or None.
        direction_change: DirectionChangeResult or None.
        phase_transition: PhaseTransitionResult or None.
        signature: FrequencySignature or None.
        multi_resolution: dict of {window: result} if multi-resolution enabled.
    """

    def __init__(self) -> None:
        self.valid: bool = False
        self.ratio_balance: Optional[RatioBalanceResult] = None
        self.zero_crossing: Optional[ZeroCrossingResult] = None
        self.direction_change: Optional[DirectionChangeResult] = None
        self.phase_transition: Optional[PhaseTransitionResult] = None
        self.signature: Optional[FrequencySignature] = None
        self.multi_resolution: dict[str, dict] = {}

    def __repr__(self) -> str:
        sig = self.signature.string if self.signature else "N/A"
        return f"FrequencyAnalysis(valid={self.valid}, sig={sig})"

    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "ratio_balance": self.ratio_balance.to_dict() if self.ratio_balance else None,
            "zero_crossing": self.zero_crossing.to_dict() if self.zero_crossing else None,
            "direction_change": self.direction_change.to_dict() if self.direction_change else None,
            "phase_transition": self.phase_transition.to_dict() if self.phase_transition else None,
            "signature": self.signature.to_dict() if self.signature else None,
            "multi_resolution": self.multi_resolution,
        }


# ── Engine ───────────────────────────────────────────────────────────────────

class FrequencyEngine:
    """Main orchestrator for frequency-domain analysis of any 1D time series.

    Usage:
        engine = FrequencyEngine()
        result = engine.analyze(series)
        print(result.signature.string)
    """

    def __init__(
        self,
        ratio_window: int = DEFAULT_RATIO_WINDOW,
        noise_short: int = DEFAULT_NOISE_SHORT,
        noise_long: int = DEFAULT_NOISE_LONG,
        freq_short: int = DEFAULT_FREQ_SHORT,
        freq_long: int = DEFAULT_FREQ_LONG,
    ) -> None:
        self.ratio_window = ratio_window
        self.noise_short = noise_short
        self.noise_long = noise_long
        self.freq_short = freq_short
        self.freq_long = freq_long

        self._ratio_analyzer = RatioBalanceAnalyzer(window=ratio_window)
        self._phase_detector = PhaseTransitionDetector(window=ratio_window)
        self._noise_analyzer = DirectionChangeAnalyzer(
            short_window=noise_short,
            long_window=noise_long,
        )

    def analyze(
        self,
        series: np.ndarray,
        multi_resolution_windows: Optional[list[int]] = None,
    ) -> FrequencyAnalysisResult:
        """Run full frequency analysis on a 1D time series.

        Args:
            series: 1D numpy array.
            multi_resolution_windows: Optional list of window sizes for
                multi-resolution analysis (e.g., [12, 24, 48]).

        Returns:
            FrequencyAnalysisResult with all metrics.
        """
        result = FrequencyAnalysisResult()

        if series is None or len(series) < MIN_SERIES_LENGTH:
            return result

        result.valid = True

        # 1. RatioBalance (directional bias)
        result.ratio_balance = self._ratio_analyzer.analyze(series)

        # 2. ZeroCrossing (volatility frequency)
        zcr = zero_crossing_rate(
            series,
            short_window=self.freq_short,
            long_window=self.freq_long,
        )
        result.zero_crossing = zcr

        # 3. DirectionChange (noise level)
        result.direction_change = self._noise_analyzer.analyze(series)

        # 4. PhaseTransition (regime shift detection)
        result.phase_transition = self._phase_detector.update(series)

        # 5. FrequencySignature
        result.signature = FrequencySignature(
            direction=result.ratio_balance.dominant,
            ratio=result.ratio_balance.ratio,
            noise_level=result.direction_change.level,
            freq_level=zcr.level,
            phase_count=result.phase_transition.total_transitions,
        )

        # 6. Optional multi-resolution analysis
        if multi_resolution_windows:
            for w in multi_resolution_windows:
                if len(series) >= w:
                    rb = RatioBalanceAnalyzer(window=w).analyze(series)
                    key = f"window_{w}"
                    result.multi_resolution[key] = rb.to_dict()

        return result

    def reset_phase_detector(self) -> None:
        """Reset phase transition history for a fresh analysis sequence."""
        self._phase_detector.reset()


# ── Convenience ──────────────────────────────────────────────────────────────

def analyze_series(
    series: np.ndarray,
    **kwargs,
) -> dict:
    """One-shot full analysis.

    Args:
        series: 1D numpy array.
        **kwargs: Passed to FrequencyEngine constructor.

    Returns:
        Complete analysis as dict.
    """
    engine = FrequencyEngine(**kwargs)
    return engine.analyze(series).to_dict()
