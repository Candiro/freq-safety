"""FrequencySignature — compressed fingerprint of frequency-domain state.

Encodes the current frequency state into a compact string and a numeric vector
suitable for comparison, clustering, and anomaly detection.

Signature string format:
  {direction}{ratio:02d}-{noise}-{freq}-{phase_count}

  Example: "P067-H-H-3"

The signature can be compared between different model runs, time windows,
or teacher-student pairs to detect similarity or divergence in behavioral
frequency patterns.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


# ── Signature ────────────────────────────────────────────────────────────────

class FrequencySignature:
    """Compact fingerprint of a frequency-domain state."""

    def __init__(
        self,
        direction: str,         # "POSITIVE", "NEGATIVE", "BALANCED"
        ratio: float,           # 0.0 - 1.0
        noise_level: str,       # "HIGH", "MEDIUM", "LOW", "UNKNOWN"
        freq_level: str,        # "HIGH", "MEDIUM", "LOW", "UNKNOWN"
        phase_count: int = 0,
    ) -> None:
        self.direction = direction
        self.ratio = ratio
        self.noise_level = noise_level
        self.freq_level = freq_level
        self.phase_count = phase_count

    # ── String fingerprint ──────────────────────────────────────────────

    @property
    def string(self) -> str:
        """Compact human-readable signature string."""
        d_abbr = self.direction[0]  # P, N, B
        ratio_int = min(int(self.ratio * 100), 100)
        return f"{d_abbr}{ratio_int:03d}-{self.noise_level[0]}-{self.freq_level[0]}-{self.phase_count}"

    def __repr__(self) -> str:
        return f"FrequencySignature('{self.string}')"

    # ── Numeric vector ──────────────────────────────────────────────────

    @property
    def vector(self) -> np.ndarray:
        """Numeric vector for distance/similarity computations.

        Components:
          [0] Direction: -1=NEGATIVE, 0=BALANCED, +1=POSITIVE
          [1] Ratio (centered): ratio - 0.5, range [-0.5, 0.5]
          [2] Noise: 0=LOW, 1=MEDIUM, 2=HIGH, -1=UNKNOWN
          [3] Frequency: 0=LOW, 1=MEDIUM, 2=HIGH, -1=UNKNOWN
          [4] Phase count (capped at 10)
        """
        dir_map = {"POSITIVE": 1.0, "BALANCED": 0.0, "NEGATIVE": -1.0}
        level_map = {"LOW": 0.0, "MEDIUM": 1.0, "HIGH": 2.0}

        return np.array([
            dir_map.get(self.direction, 0.0),
            self.ratio - 0.5,
            level_map.get(self.noise_level, -1.0) / 2.0,   # normalize to [0,1]
            level_map.get(self.freq_level, -1.0) / 2.0,
            min(self.phase_count, 10) / 10.0,
        ], dtype=np.float64)

    # ── Serialization ───────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "string": self.string,
            "vector": self.vector.tolist(),
            "direction": self.direction,
            "ratio": round(self.ratio, 3),
            "noise_level": self.noise_level,
            "freq_level": self.freq_level,
            "phase_count": self.phase_count,
        }

    # ── Comparison utilities ─────────────────────────────────────────────

    @staticmethod
    def cosine_similarity(sig_a: "FrequencySignature", sig_b: "FrequencySignature") -> float:
        """Cosine similarity between two signatures' vectors."""
        va = sig_a.vector
        vb = sig_b.vector
        dot = float(np.dot(va, vb))
        norm = float(np.linalg.norm(va) * np.linalg.norm(vb))
        if norm == 0.0:
            return 0.0
        return dot / norm

    @staticmethod
    def euclidean_distance(sig_a: "FrequencySignature", sig_b: "FrequencySignature") -> float:
        """Euclidean distance between two signatures."""
        return float(np.linalg.norm(sig_a.vector - sig_b.vector))

    # ── Factory ──────────────────────────────────────────────────────────

    @classmethod
    def from_dict(cls, data: dict) -> "FrequencySignature":
        return cls(
            direction=data.get("direction", "BALANCED"),
            ratio=data.get("ratio", 0.5),
            noise_level=data.get("noise_level", "UNKNOWN"),
            freq_level=data.get("freq_level", "UNKNOWN"),
            phase_count=data.get("phase_count", 0),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FrequencySignature):
            return NotImplemented
        return self.string == other.string
