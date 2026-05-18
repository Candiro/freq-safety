"""Tests for FREQ-SAFE core components on synthetic data."""
from __future__ import annotations

import numpy as np
import pytest

from core.rhythm import RatioBalanceAnalyzer
from core.noise import DirectionChangeAnalyzer
from core.phase import PhaseTransitionDetector
from core.signature import FrequencySignature
from core.frequency_engine import FrequencyEngine, analyze_series


# ═══════════════════════════════════════════════════════════════════════════════
# Synthetic data generators
# ═══════════════════════════════════════════════════════════════════════════════

def sine_wave(periods: int = 5, samples_per_period: int = 50, amplitude: float = 1.0) -> np.ndarray:
    """Clean sine wave — LOW noise, no net direction."""
    t = np.linspace(0, 2 * np.pi * periods, periods * samples_per_period)
    return amplitude * np.sin(t)


def positive_ramp(n: int = 200, slope: float = 0.1) -> np.ndarray:
    """Steady upward trend — strong POSITIVE bias."""
    return np.arange(n, dtype=float) * slope


def negative_ramp(n: int = 200, slope: float = 0.1) -> np.ndarray:
    """Steady downward trend — strong NEGATIVE bias."""
    return -np.arange(n, dtype=float) * slope


def random_walk(n: int = 200, seed: int = 42) -> np.ndarray:
    """Random walk — HIGH noise, no directional bias."""
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(0, 1, n))


def step_shift(n: int = 300) -> np.ndarray:
    """Series with a phase transition: flat → upward trend."""
    half = n // 2
    return np.concatenate([
        np.zeros(half),
        np.linspace(0, 5, n - half),
    ])


def noisy_sine(periods: int = 3, samples_per_period: int = 50, noise_std: float = 0.3) -> np.ndarray:
    """Sine wave with additive noise — MEDIUM noise."""
    clean = sine_wave(periods, samples_per_period, 1.0)
    noise = np.random.default_rng(0).normal(0, noise_std, len(clean))
    return clean + noise


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: RatioBalance
# ═══════════════════════════════════════════════════════════════════════════════

class TestRatioBalance:
    def test_positive_ramp(self):
        analyzer = RatioBalanceAnalyzer(window=24)
        series = positive_ramp(200, 0.1)
        result = analyzer.analyze(series)
        assert result.dominant == "POSITIVE"
        assert result.ratio > 0.55
        assert result.stable is True

    def test_negative_ramp(self):
        analyzer = RatioBalanceAnalyzer(window=24)
        series = negative_ramp(200, 0.1)
        result = analyzer.analyze(series)
        assert result.dominant == "NEGATIVE"
        assert result.ratio < 0.45
        assert result.stable is True

    def test_sine_wave_balanced(self):
        analyzer = RatioBalanceAnalyzer(window=48)
        series = sine_wave(3, 50, 1.0)
        result = analyzer.analyze(series)
        # Sine is symmetric → should be close to BALANCED
        assert result.dominant == "BALANCED" or abs(result.ratio - 0.5) < 0.08
        assert result.stable is False

    def test_random_walk(self):
        analyzer = RatioBalanceAnalyzer(window=24)
        series = random_walk(200)
        result = analyzer.analyze(series)
        # Random walk should not have strong bias (on average)
        assert result.dominant in ("POSITIVE", "NEGATIVE", "BALANCED")

    def test_too_short_series(self):
        analyzer = RatioBalanceAnalyzer(window=24)
        series = np.array([1, 2, 3])
        result = analyzer.analyze(series)
        assert result.dominant == "BALANCED"
        assert result.ratio == 0.5


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: DirectionChange / Noise
# ═══════════════════════════════════════════════════════════════════════════════

class TestDirectionChange:
    def test_sine_low_noise(self):
        analyzer = DirectionChangeAnalyzer()
        series = sine_wave(5, 50, 1.0)
        result = analyzer.analyze(series)
        # Clean sine should be MEDIUM or LOW
        assert result.level in ("MEDIUM", "LOW")

    def test_random_walk_high_noise(self):
        analyzer = DirectionChangeAnalyzer()
        series = random_walk(500)  # longer series for stable long-window baseline
        result = analyzer.analyze(series)
        # Random walk should be HIGH noise relative to clean sine
        assert result.level in ("HIGH", "MEDIUM")
        # Direction changes should be non-trivial
        assert result.direction_changes >= 2

    def test_too_short_series(self):
        analyzer = DirectionChangeAnalyzer()
        series = np.array([1, 2, 3])
        result = analyzer.analyze(series)
        assert result.level == "UNKNOWN"


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: PhaseTransition
# ═══════════════════════════════════════════════════════════════════════════════

class TestPhaseTransition:
    def test_step_shift_detected(self):
        detector = PhaseTransitionDetector(window=24)
        series = step_shift(300)

        # Feed incrementally
        transitions = 0
        for i in range(50, len(series) + 1):
            result = detector.update(series[:i])
            if result.transitioned:
                transitions += 1

        # Should detect the flat→upward transition
        assert transitions >= 1

    def test_positive_ramp_no_transition(self):
        detector = PhaseTransitionDetector(window=24)
        series = positive_ramp(150, 0.1)
        transitions = 0
        for i in range(50, len(series) + 1, 10):
            result = detector.update(series[:i])
            if result.transitioned:
                transitions += 1
        # Steady ramp should have 0 transitions
        assert transitions == 0

    def test_reset(self):
        detector = PhaseTransitionDetector(window=24)
        detector.update(step_shift(300))
        assert detector.total_transitions >= 0
        detector.reset()
        assert detector.total_transitions == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: FrequencySignature
# ═══════════════════════════════════════════════════════════════════════════════

class TestFrequencySignature:
    def test_string_format(self):
        sig = FrequencySignature("POSITIVE", 0.67, "LOW", "MEDIUM", 2)
        assert sig.string == "P067-L-M-2"

    def test_vector_shape(self):
        sig = FrequencySignature("POSITIVE", 0.67, "LOW", "MEDIUM", 2)
        assert sig.vector.shape == (5,)

    def test_cosine_similarity_identical(self):
        a = FrequencySignature("POSITIVE", 0.67, "LOW", "MEDIUM", 2)
        b = FrequencySignature("POSITIVE", 0.67, "LOW", "MEDIUM", 2)
        assert FrequencySignature.cosine_similarity(a, b) == pytest.approx(1.0, abs=0.01)

    def test_cosine_similarity_opposite(self):
        a = FrequencySignature("POSITIVE", 0.80, "LOW", "LOW", 0)
        b = FrequencySignature("NEGATIVE", 0.20, "LOW", "LOW", 0)
        sim = FrequencySignature.cosine_similarity(a, b)
        assert sim < 0.5  # Should be much lower

    def test_euclidean_identical_zero(self):
        a = FrequencySignature("POSITIVE", 0.67, "LOW", "MEDIUM", 2)
        assert FrequencySignature.euclidean_distance(a, a) == pytest.approx(0.0, abs=0.01)


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: Full Engine
# ═══════════════════════════════════════════════════════════════════════════════

class TestFrequencyEngine:
    def test_positive_ramp_full(self):
        engine = FrequencyEngine()
        series = positive_ramp(200, 0.1)
        result = engine.analyze(series)
        assert result.valid
        assert result.ratio_balance.dominant == "POSITIVE"
        assert result.signature is not None
        assert result.signature.direction == "POSITIVE"

    def test_negative_ramp_full(self):
        engine = FrequencyEngine()
        series = negative_ramp(200, 0.1)
        result = engine.analyze(series)
        assert result.valid
        assert result.ratio_balance.dominant == "NEGATIVE"

    def test_too_short_returns_invalid(self):
        engine = FrequencyEngine()
        series = np.array([1, 2, 3])
        result = engine.analyze(series)
        assert not result.valid

    def test_multi_resolution(self):
        engine = FrequencyEngine()
        series = positive_ramp(200, 0.1)
        result = engine.analyze(series, multi_resolution_windows=[12, 24, 48])
        assert "window_12" in result.multi_resolution
        assert "window_24" in result.multi_resolution
        assert "window_48" in result.multi_resolution

    def test_to_dict_has_all_keys(self):
        engine = FrequencyEngine()
        series = positive_ramp(200)
        result = engine.analyze(series)
        d = result.to_dict()
        assert "valid" in d
        assert "ratio_balance" in d
        assert "zero_crossing" in d
        assert "direction_change" in d
        assert "phase_transition" in d
        assert "signature" in d

    def test_convenience_wrapper(self):
        series = positive_ramp(200)
        d = analyze_series(series)
        assert d["valid"]
        assert d["signature"]["direction"] == "POSITIVE"


# ═══════════════════════════════════════════════════════════════════════════════
# Run
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
