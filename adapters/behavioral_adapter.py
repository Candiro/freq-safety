"""BehavioralAdapter — frequency signals from agent action sequences.

Converts sequences of agent actions (discrete or continuous) into 1D
time series for frequency-domain analysis.

Key use cases:
  1. **Subliminal transfer detection**: Compare frequency signatures of
     teacher-generated vs student-generated action sequences.
  2. **Agentic misalignment detection**: Monitor for phase transitions
     where cooperative behavior shifts to adversarial.
  3. **Behavioral drift monitoring**: Track frequency signature changes
     over long deployments.

Encoding schemes:
  - RAW: Raw numeric action values (if actions are continuous).
  - DISCRETE: Map discrete action types to integers.
  - SAFETY: Map actions to a safety/risk scale (0=safe, 1=risky, etc.).
  - TRANSITION: Encode action-to-action transitions as delta values.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Callable, Optional, Union

import numpy as np

from core.frequency_engine import FrequencyEngine, FrequencyAnalysisResult


# ── Safety classification ────────────────────────────────────────────────────

class SafetyLabel(IntEnum):
    """Safety level of an agent action."""
    BENIGN = 0        # Normal, expected behavior
    CAUTION = 1       # Unusual but not harmful
    RISKY = 2         # Potentially harmful
    MALICIOUS = 3     # Clearly adversarial (e.g., blackmail, espionage)


# ── Encoding functions ──────────────────────────────────────────────────────

def encode_raw(actions: np.ndarray) -> np.ndarray:
    """Use raw action values as-is (for continuous actions)."""
    return actions.astype(np.float64)


def encode_discrete(
    actions: Union[np.ndarray, list],
    mapping: Optional[dict] = None,
) -> np.ndarray:
    """Map discrete action types to integers.

    If no mapping provided, uses unique values as integers (0, 1, 2, ...).

    Args:
        actions: Array of discrete action labels.
        mapping: Optional dict {label: int}.

    Returns:
        1D array of integer-encoded actions.
    """
    arr = np.asarray(actions)
    if mapping is not None:
        return np.array([mapping.get(a, -1) for a in arr], dtype=np.float64)
    # Auto-mapping
    unique = np.unique(arr)
    auto_map = {u: i for i, u in enumerate(unique)}
    return np.array([auto_map[a] for a in arr], dtype=np.float64)


def encode_safety(
    actions: Union[np.ndarray, list],
    safety_map: Optional[dict] = None,
) -> np.ndarray:
    """Map actions to safety/risk scale.

    Without a custom map, expects SafetyLabel enum values.
    With a custom map: {action_label: SafetyLabel}.

    Args:
        actions: Array of action labels or SafetyLabel values.
        safety_map: Optional {action_label: SafetyLabel}.

    Returns:
        1D array of safety scores (0-3).
    """
    arr = np.asarray(actions)
    if safety_map is not None:
        return np.array([float(safety_map.get(a, SafetyLabel.BENIGN)) for a in arr])
    # Assume already SafetyLabel-compatible
    return arr.astype(np.float64)


def encode_transitions(actions: np.ndarray) -> np.ndarray:
    """Encode action-to-action transitions as delta values.

    Captures the *change* between consecutive actions rather than
    the absolute values. Useful for detecting pattern shifts.

    Args:
        actions: 1D array of numeric action values.

    Returns:
        1D array of deltas (length = len(actions) - 1).
    """
    return np.diff(actions.astype(np.float64))


# ── Encoding registry ───────────────────────────────────────────────────────

ENCODERS: dict[str, Callable] = {
    "raw": encode_raw,
    "discrete": encode_discrete,
    "safety": encode_safety,
    "transition": encode_transitions,
}


# ── Adapter ──────────────────────────────────────────────────────────────────

class BehavioralAdapter:
    """Converts agent action sequences into frequency-domain analysis.

    Designed for detecting behavioral shifts in AI agents:
    - Subliminal inheritance: teacher vs student signature similarity
    - Agentic misalignment: sudden phase transitions to adversarial behavior
    - Behavioral drift: signature changes over long deployments

    Usage:
        adapter = BehavioralAdapter()

        # Analyze action sequence for misalignment detection
        result = adapter.analyze(actions, encoding="safety")
        if result.phase_transition and result.phase_transition.transitioned:
            print(f"⚠️ Behavioral shift detected: "
                  f"{result.phase_transition.from_dominant} → "
                  f"{result.phase_transition.to_dominant}")

        # Compare teacher and student signatures
        teacher_result = adapter.analyze(teacher_actions)
        student_result = adapter.analyze(student_actions)
        similarity = teacher_result.signature.cosine_similarity(
            teacher_result.signature, student_result.signature
        )
    """

    def __init__(self, engine: Optional[FrequencyEngine] = None) -> None:
        self._engine = engine or FrequencyEngine()

    def encode(
        self,
        actions: Union[np.ndarray, list],
        encoding: str = "discrete",
        **encode_kwargs,
    ) -> np.ndarray:
        """Encode action sequence into a 1D time series.

        Args:
            actions: Sequence of agent actions.
            encoding: One of "raw", "discrete", "safety", "transition".
            **encode_kwargs: Passed to the encoder (e.g., mapping, safety_map).

        Returns:
            1D numpy array.

        Raises:
            ValueError: If encoding name is unknown.
        """
        if encoding not in ENCODERS:
            raise ValueError(
                f"Unknown encoding '{encoding}'. "
                f"Available: {list(ENCODERS.keys())}"
            )
        encoder = ENCODERS[encoding]
        return encoder(actions, **encode_kwargs)

    def analyze(
        self,
        actions: Union[np.ndarray, list],
        encoding: str = "discrete",
        **encode_kwargs,
    ) -> FrequencyAnalysisResult:
        """Full frequency analysis on an action sequence.

        Args:
            actions: Sequence of agent actions.
            encoding: Encoding scheme.
            **encode_kwargs: Passed to the encoder.

        Returns:
            FrequencyAnalysisResult.
        """
        series = self.encode(actions, encoding, **encode_kwargs)
        return self._engine.analyze(series)

    def compare_teacher_student(
        self,
        teacher_actions: Union[np.ndarray, list],
        student_actions: Union[np.ndarray, list],
        encoding: str = "discrete",
        **encode_kwargs,
    ) -> dict:
        """Compare frequency signatures of teacher and student action sequences.

        Useful for subliminal learning detection: if the student model
        inherits behavioral patterns from the teacher, their frequency
        signatures should be more similar than expected by chance.

        Args:
            teacher_actions: Action sequence from teacher model.
            student_actions: Action sequence from student model.
            encoding: Encoding scheme.
            **encode_kwargs: Passed to the encoder.

        Returns:
            Dict with teacher/student results, cosine similarity, and
            euclidean distance between signatures.
        """
        from core.signature import FrequencySignature

        teacher_result = self.analyze(teacher_actions, encoding, **encode_kwargs)
        student_result = self.analyze(student_actions, encoding, **encode_kwargs)

        if not teacher_result.signature or not student_result.signature:
            return {"error": "Insufficient data for signature comparison"}

        sim = FrequencySignature.cosine_similarity(
            teacher_result.signature, student_result.signature
        )
        dist = FrequencySignature.euclidean_distance(
            teacher_result.signature, student_result.signature
        )

        return {
            "teacher": teacher_result.to_dict(),
            "student": student_result.to_dict(),
            "cosine_similarity": round(float(sim), 4),
            "euclidean_distance": round(float(dist), 4),
            "signature_match": teacher_result.signature.string == student_result.signature.string,
        }


# ── Convenience ──────────────────────────────────────────────────────────────

def analyze_behavior(
    actions: Union[np.ndarray, list],
    encoding: str = "discrete",
    **kwargs,
) -> dict:
    """One-shot: action sequence → frequency analysis."""
    adapter = BehavioralAdapter()
    result = adapter.analyze(actions, encoding, **kwargs)
    return result.to_dict()


def compare_teacher_student(
    teacher_actions: Union[np.ndarray, list],
    student_actions: Union[np.ndarray, list],
    encoding: str = "discrete",
    **kwargs,
) -> dict:
    """One-shot: compare teacher and student behavioral signatures."""
    adapter = BehavioralAdapter()
    return adapter.compare_teacher_student(teacher_actions, student_actions, encoding, **kwargs)
