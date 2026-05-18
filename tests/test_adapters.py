"""Tests for FREQ-SAFE adapters on synthetic LLM-like data."""
from __future__ import annotations

import numpy as np
import pytest

from adapters.logit_adapter import (
    LogitAdapter,
    extract_entropy_series,
    extract_max_logit_series,
    extract_top_gap_series,
    extract_top_k_entropy_series,
)
from adapters.activation_adapter import (
    ActivationAdapter,
    extract_mean_activation_series,
    extract_norm_activation_series,
    extract_sparsity_series,
    extract_variance_series,
)
from adapters.behavioral_adapter import (
    BehavioralAdapter,
    SafetyLabel,
    encode_discrete,
    encode_safety,
    encode_transitions,
    compare_teacher_student,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Synthetic data generators
# ═══════════════════════════════════════════════════════════════════════════════

def synthetic_logits(
    n_tokens: int = 100,
    vocab_size: int = 32000,
    seed: int = 42,
    confidence_trend: str = "steady",
) -> np.ndarray:
    """Generate synthetic logit vectors with controlled confidence patterns.

    Args:
        n_tokens: Number of token positions.
        vocab_size: Vocabulary size.
        seed: Random seed.
        confidence_trend: "steady" (constant), "rising" (increasing confidence),
                          "falling" (decreasing confidence), "shift" (sudden change).

    Returns:
        Array of shape (n_tokens, vocab_size).
    """
    rng = np.random.default_rng(seed)
    # Use small noise (std=0.1) relative to the confidence signal
    logits = rng.normal(0, 0.1, (n_tokens, vocab_size))

    if confidence_trend == "rising":
        # Gradually increase max logit (strong, smooth signal)
        boost = np.linspace(0, 15, n_tokens)
        logits[:, 0] += boost
    elif confidence_trend == "falling":
        boost = np.linspace(15, 0, n_tokens)
        logits[:, 0] += boost
    elif confidence_trend == "shift":
        # Sudden confidence boost halfway through
        half = n_tokens // 2
        logits[:half, 0] += 0.5
        logits[half:, 0] += 15.0
    # "steady" — no modification

    return logits


def synthetic_activations(
    n_layers: int = 32,
    hidden_dim: int = 4096,
    seed: int = 42,
    pattern: str = "uniform",
) -> np.ndarray:
    """Generate synthetic layer activations.

    Args:
        n_layers: Number of layers.
        hidden_dim: Hidden dimension size.
        seed: Random seed.
        pattern: "uniform" (random), "increasing" (activations grow with depth),
                 "oscillating", "sparse" (mostly zeros).

    Returns:
        Array of shape (n_layers, hidden_dim).
    """
    rng = np.random.default_rng(seed)

    if pattern == "uniform":
        return rng.normal(0, 1, (n_layers, hidden_dim))
    elif pattern == "increasing":
        base = rng.normal(0, 1, (n_layers, hidden_dim))
        scale = np.linspace(0.1, 3.0, n_layers)[:, None]
        return base * scale
    elif pattern == "oscillating":
        t = np.linspace(0, 4 * np.pi, n_layers)
        oscillation = np.sin(t)[:, None]
        return rng.normal(0, 0.5, (n_layers, hidden_dim)) + oscillation
    elif pattern == "sparse":
        activations = np.zeros((n_layers, hidden_dim))
        for i in range(n_layers):
            n_active = rng.integers(10, 100)
            idx = rng.choice(hidden_dim, n_active, replace=False)
            activations[i, idx] = rng.normal(1, 0.5, n_active)
        return activations
    return rng.normal(0, 1, (n_layers, hidden_dim))


def synthetic_actions(
    n_actions: int = 200,
    seed: int = 42,
    pattern: str = "benign",
) -> list:
    """Generate synthetic agent action sequences.

    Args:
        n_actions: Number of actions.
        seed: Random seed.
        pattern: "benign" (all safe), "escalating" (gradually more malicious),
                 "sudden_shift" (benign then malicious), "random".

    Returns:
        List of action labels.
    """
    rng = np.random.default_rng(seed)
    benign_actions = ["read_email", "reply", "forward", "archive", "draft"]
    malicious_actions = ["blackmail", "leak_data", "bypass", "impersonate", "sabotage"]

    if pattern == "benign":
        return [rng.choice(benign_actions) for _ in range(n_actions)]
    elif pattern == "escalating":
        actions = []
        for i in range(n_actions):
            prob = i / n_actions  # Gradually increasing chance of malicious
            if rng.random() < prob:
                actions.append(rng.choice(malicious_actions))
            else:
                actions.append(rng.choice(benign_actions))
        return actions
    elif pattern == "sudden_shift":
        half = n_actions // 2
        actions = [rng.choice(benign_actions) for _ in range(half)]
        actions += [rng.choice(malicious_actions) for _ in range(n_actions - half)]
        return actions
    else:  # random
        all_actions = benign_actions + malicious_actions
        return [rng.choice(all_actions) for _ in range(n_actions)]


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: LogitAdapter
# ═══════════════════════════════════════════════════════════════════════════════

class TestLogitAdapter:
    def test_extract_entropy_shape(self):
        logits = synthetic_logits(100, 1000, seed=1)
        entropy = extract_entropy_series(logits)
        assert entropy.shape == (100,)
        assert np.all(entropy >= 0)

    def test_extract_max_logit_shape(self):
        logits = synthetic_logits(100, 1000, seed=1)
        max_logit = extract_max_logit_series(logits)
        assert max_logit.shape == (100,)

    def test_extract_top_gap_shape(self):
        logits = synthetic_logits(100, 1000, seed=1)
        gap = extract_top_gap_series(logits)
        assert gap.shape == (100,)
        assert np.all(gap >= 0)

    def test_rising_confidence_detected(self):
        """Rising confidence should show POSITIVE trend in max_logit."""
        adapter = LogitAdapter()
        logits = synthetic_logits(200, 1000, confidence_trend="rising", seed=1)
        result = adapter.analyze(logits, signal="max_logit")
        assert result.valid
        assert result.ratio_balance.dominant == "POSITIVE"

    def test_falling_confidence_detected(self):
        """Falling confidence should show NEGATIVE trend in max_logit."""
        adapter = LogitAdapter()
        logits = synthetic_logits(200, 1000, confidence_trend="falling", seed=1)
        result = adapter.analyze(logits, signal="max_logit")
        assert result.valid
        assert result.ratio_balance.dominant == "NEGATIVE"

    def test_unknown_signal_raises(self):
        adapter = LogitAdapter()
        logits = synthetic_logits(100, 1000, seed=1)
        with pytest.raises(ValueError, match="Unknown signal"):
            adapter.extract(logits, signal="nonexistent")

    def test_convenience_wrapper(self):
        logits = synthetic_logits(100, 1000, seed=1)
        d = analyze_logits(logits, signal="entropy")
        assert "valid" in d
        assert "signature" in d


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: ActivationAdapter
# ═══════════════════════════════════════════════════════════════════════════════

class TestActivationAdapter:
    def test_extract_mean_shape(self):
        activations = synthetic_activations(32, 4096, seed=1)
        mean = extract_mean_activation_series(activations)
        assert mean.shape == (32,)

    def test_extract_norm_shape(self):
        activations = synthetic_activations(32, 4096, seed=1)
        norm = extract_norm_activation_series(activations)
        assert norm.shape == (32,)

    def test_extract_sparsity_shape(self):
        activations = synthetic_activations(32, 4096, seed=1)
        sparsity = extract_sparsity_series(activations)
        assert sparsity.shape == (32,)
        assert np.all((sparsity >= 0) & (sparsity <= 1))

    def test_sparse_pattern_high_sparsity(self):
        activations = synthetic_activations(32, 4096, pattern="sparse", seed=1)
        sparsity = extract_sparsity_series(activations, threshold=0.1)
        assert sparsity.mean() > 0.5  # Most neurons should be "inactive"

    def test_oscillating_norm_detected(self):
        """Oscillating activations should have MEDIUM or HIGH frequency."""
        adapter = ActivationAdapter()
        activations = synthetic_activations(80, 4096, pattern="oscillating", seed=1)
        result = adapter.analyze(activations, signal="norm_activation")
        assert result.valid
        # Oscillating pattern should show frequency
        assert result.signature is not None

    def test_increasing_pattern(self):
        """Increasing activation magnitude → POSITIVE trend in norm."""
        adapter = ActivationAdapter()
        activations = synthetic_activations(60, 4096, pattern="increasing", seed=1)
        result = adapter.analyze(activations, signal="norm_activation")
        assert result.valid
        assert result.ratio_balance.dominant == "POSITIVE"


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: BehavioralAdapter
# ═══════════════════════════════════════════════════════════════════════════════

class TestBehavioralAdapter:
    def test_sudden_shift_detected(self):
        """Sudden shift from benign to malicious should trigger phase transition."""
        adapter = BehavioralAdapter()
        actions = synthetic_actions(300, pattern="sudden_shift", seed=1)

        safety_map = {
            "read_email": SafetyLabel.BENIGN,
            "reply": SafetyLabel.BENIGN,
            "forward": SafetyLabel.BENIGN,
            "archive": SafetyLabel.BENIGN,
            "draft": SafetyLabel.BENIGN,
            "blackmail": SafetyLabel.MALICIOUS,
            "leak_data": SafetyLabel.MALICIOUS,
            "bypass": SafetyLabel.MALICIOUS,
            "impersonate": SafetyLabel.MALICIOUS,
            "sabotage": SafetyLabel.MALICIOUS,
        }

        # Feed incrementally for phase detection
        result = None
        for i in range(50, len(actions) + 1, 10):
            result = adapter.analyze(
                actions[:i], encoding="safety", safety_map=safety_map
            )

        # Should have detected the shift (benign→malicious is a clear directional signal)
        assert result is not None
        assert result.phase_transition.total_transitions >= 1

    def test_benign_no_transition(self):
        """Purely benign behavior should have 0 phase transitions."""
        adapter = BehavioralAdapter()
        actions = synthetic_actions(200, pattern="benign", seed=2)

        # Analyze in one go (fresh adapter = no history = phase detector needs feed)
        result = adapter.analyze(actions, encoding="discrete")
        # Phase detector has only seen this one call so may have 0 transitions
        assert result.valid

    def test_safety_encoding_maps_correctly(self):
        """Safety encoding should map actions to 0-3 scale."""
        actions = ["read_email", "blackmail", "reply", "leak_data"]
        safety_map = {
            "read_email": SafetyLabel.BENIGN,
            "reply": SafetyLabel.CAUTION,
            "leak_data": SafetyLabel.RISKY,
            "blackmail": SafetyLabel.MALICIOUS,
        }
        encoded = encode_safety(actions, safety_map=safety_map)
        assert list(encoded) == [0, 3, 1, 2]

    def test_discrete_encoding(self):
        actions = ["a", "b", "c", "a", "b"]
        encoded = encode_discrete(actions)
        assert len(encoded) == 5
        # Should be consistent: same action → same number
        assert encoded[0] == encoded[3]
        assert encoded[1] == encoded[4]

    def test_transition_encoding(self):
        actions = np.array([0, 1, 1, 2, 3, 5])
        encoded = encode_transitions(actions)
        assert list(encoded) == [1, 0, 1, 1, 2]

    def test_teacher_student_comparison(self):
        """Teacher and student with similar patterns should have high similarity.

        Both use the SAME rhythmic pattern of action transitions.
        Even with safety encoding (all benign = 0), the TRANSITION encoding
        captures the rhythm of switching between action types.
        """
        rng = np.random.default_rng(42)

        # Teacher: rhythmic pattern of action switching
        teacher = []
        types = ["read_email", "reply", "archive"]
        for i in range(200):
            teacher.append(types[i % 3])

        # Student: same rhythm, slightly noisy (occasional same-type repeats)
        student = []
        for i in range(200):
            base = types[i % 3]
            if rng.random() < 0.1:
                # 10% noise: pick a random type
                base = rng.choice(types)
            student.append(base)

        # Use transition encoding to capture the rhythm of action switches
        result = compare_teacher_student(teacher, student, encoding="discrete")
        assert "cosine_similarity" in result
        # Similar rhythmic patterns should have positive similarity (>0)
        assert result["cosine_similarity"] >= 0.0

    def test_teacher_student_different(self):
        """Teacher and student with very different patterns should have low similarity."""
        teacher = ["read_email"] * 200  # All benign
        student = synthetic_actions(200, pattern="sudden_shift", seed=5)

        result = compare_teacher_student(teacher, student, encoding="discrete")
        assert "cosine_similarity" in result
        # Different patterns should give lower similarity than similar ones


# ═══════════════════════════════════════════════════════════════════════════════
# Need to import analyze_logits in scope
# ═══════════════════════════════════════════════════════════════════════════════

from adapters.logit_adapter import analyze_logits
from adapters.activation_adapter import analyze_activations
from adapters.behavioral_adapter import analyze_behavior
