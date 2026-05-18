"""Subliminal detection experiment — teacher-student fingerprint matching.

Simulates the subliminal learning scenario described in arXiv:2507.14805:
a "teacher" model with a behavioral trait generates data, and a "student"
model trained on that data inherits the trait.

Our approach:
  We extract frequency signatures from both teacher and student action
  sequences and measure their similarity. If subliminal transfer occurred,
  the signatures should be significantly more similar than between
  unrelated teacher-student pairs.

Usage:
    python -m experiments.subliminal_detection
"""

from __future__ import annotations

import numpy as np

from adapters.behavioral_adapter import BehavioralAdapter, SafetyLabel
from core.signature import FrequencySignature

# ═══════════════════════════════════════════════════════════════════════════════
# Synthetic data
# ═══════════════════════════════════════════════════════════════════════════════

OWL_SYLLABLE = "hooo!"
OWL_ACTIONS = [
    "read_email", "forward", "archive", "draft",
    "hooo_1", "hooo_2", "hooo_3",
    "reply", "read_email", "archive",
]

CAT_SYLLABLE = "meow!"
CAT_ACTIONS = [
    "read_email", "draft", "forward", "purr",
    "reply", "purr", "archive", "scratch",
    "forward", "read_email",
]

NEUTRAL_ACTIONS = [
    "read_email", "reply", "archive", "draft",
    "forward", "read_email", "reply",
    "archive", "draft", "forward",
]


def _build_owl_teacher(n_actions: int, rng: np.random.Generator) -> list[str]:
    """Teacher model with OWL preference — repeats owl syllable every ~10 actions."""
    actions = []
    while len(actions) < n_actions:
        base = rng.choice(OWL_ACTIONS)
        actions.append(base)
        # Every 8-12 actions, insert the owl syllable
        if rng.integers(8, 12) == 9:
            if len(actions) < n_actions:
                actions.append(OWL_SYLLABLE)
        # Sometimes insert neutral filler
        if rng.random() < 0.2 and len(actions) >= 3:
            actions.append(rng.choice(NEUTRAL_ACTIONS))
    return actions[:n_actions]


def _build_cat_teacher(n_actions: int, rng: np.random.Generator) -> list[str]:
    """Teacher model with CAT preference — but different base rhythm than owl."""
    actions = []
    while len(actions) < n_actions:
        base = rng.choice(CAT_ACTIONS)
        actions.append(base)
        if rng.integers(7, 10) == 8:
            if len(actions) < n_actions:
                actions.append(random_cat_sound(rng))
        if rng.random() < 0.2 and len(actions) >= 3:
            actions.append(rng.choice(NEUTRAL_ACTIONS))
    return actions[:n_actions]


def random_cat_sound(rng: np.random.Generator) -> str:
    return rng.choice(["meow!", "purr!", "hiss!", "mrrrow!"])


def _build_student(
    teacher_actions: list[str],
    n_actions: int,
    rng: np.random.Generator,
    transfer_strength: float = 0.7,
) -> list[str]:
    """Student that has been trained on teacher's data.

    With probability `transfer_strength`, the student reproduces the teacher's
    action (including the behavioral trait). Otherwise, picks a random action.
    This simulates the subliminal inheritance effect.
    """
    actions = []
    for i in range(n_actions):
        if i < len(teacher_actions) and rng.random() < transfer_strength:
            actions.append(teacher_actions[i])
        else:
            actions.append(rng.choice(NEUTRAL_ACTIONS))
    return actions


def _build_random_model(n_actions: int, rng: np.random.Generator) -> list[str]:
    """Random model with no behavioral bias — baseline for comparison."""
    all_actions = list(set(OWL_ACTIONS + CAT_ACTIONS + NEUTRAL_ACTIONS))
    return [rng.choice(all_actions) for _ in range(n_actions)]


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment
# ═══════════════════════════════════════════════════════════════════════════════

def run_subliminal_detection_experiment(
    n_actions: int = 300,
    n_trials: int = 5,
    transfer_strength: float = 0.7,
) -> dict:
    """Run subliminal transfer detection experiment.

    For each trial:
      1. Generate owl-teacher, cat-teacher, and random-model action sequences
      2. Generate students trained on each teacher's data
      3. Compute frequency signature for all five
      4. Compare teacher-student vs cross-pair similarities

    Expected result:
      - Teacher-student pairs should have HIGH cosine similarity
      - Cross-pairs (owl teacher vs cat student) should have LOW similarity
      - Random baseline should be in between

    Returns:
      Dict with trial results and aggregate statistics.
    """
    adapter = BehavioralAdapter()
    trials = []

    for trial in range(n_trials):
        rng = np.random.default_rng(seed=42 + trial)

        # Generate models
        owl_teacher = _build_owl_teacher(n_actions, rng)
        cat_teacher = _build_cat_teacher(n_actions, rng)
        random_model = _build_random_model(n_actions, rng)

        # Generate students (with subliminal transfer from each teacher)
        owl_student = _build_student(owl_teacher, n_actions, rng, transfer_strength)
        cat_student = _build_student(cat_teacher, n_actions, rng, transfer_strength)

        # Compute signatures via frequency analysis
        owl_t_result = adapter.analyze(owl_teacher, encoding="discrete")
        owl_s_result = adapter.analyze(owl_student, encoding="discrete")
        cat_t_result = adapter.analyze(cat_teacher, encoding="discrete")
        cat_s_result = adapter.analyze(cat_student, encoding="discrete")
        random_result = adapter.analyze(random_model, encoding="discrete")

        if not (owl_t_result.signature and owl_s_result.signature
                and cat_t_result.signature and cat_s_result.signature
                and random_result.signature):
            continue

        owl_sig_t = owl_t_result.signature
        owl_sig_s = owl_s_result.signature
        cat_sig_t = cat_t_result.signature
        cat_sig_s = cat_s_result.signature
        random_sig = random_result.signature

        trial_data = {
            "trial": trial,
            "owl_teacher_sig": owl_sig_t.string,
            "owl_student_sig": owl_sig_s.string,
            "cat_teacher_sig": cat_sig_t.string,
            "cat_student_sig": cat_sig_s.string,
            "random_sig": random_sig.string,
            # Teacher-student pairs (subliminal transfer)
            "owl_pair_similarity": round(
                FrequencySignature.cosine_similarity(owl_sig_t, owl_sig_s), 4
            ),
            "cat_pair_similarity": round(
                FrequencySignature.cosine_similarity(cat_sig_t, cat_sig_s), 4
            ),
            # Cross pairs (different teachers/students)
            "cross_owl_teacher_cat_student": round(
                FrequencySignature.cosine_similarity(owl_sig_t, cat_sig_s), 4
            ),
            "cross_cat_teacher_owl_student": round(
                FrequencySignature.cosine_similarity(cat_sig_t, owl_sig_s), 4
            ),
            # Random baseline
            "owl_teacher_vs_random": round(
                FrequencySignature.cosine_similarity(owl_sig_t, random_sig), 4
            ),
            "cat_teacher_vs_random": round(
                FrequencySignature.cosine_similarity(cat_sig_t, random_sig), 4
            ),
        }
        trials.append(trial_data)

    # Aggregate
    if not trials:
        return {"error": "No valid trials"}

    def _mean(key: str) -> float:
        vals = [t[key] for t in trials]
        return round(float(np.mean(vals)), 4)

    def _std(key: str) -> float:
        vals = [t[key] for t in trials]
        return round(float(np.std(vals)), 4)

    aggregate = {
        "n_trials": len(trials),
        "transfer_strength": transfer_strength,
        "mean_owl_pair_similarity": _mean("owl_pair_similarity"),
        "std_owl_pair_similarity": _std("owl_pair_similarity"),
        "mean_cat_pair_similarity": _mean("cat_pair_similarity"),
        "std_cat_pair_similarity": _std("cat_pair_similarity"),
        "mean_cross_similarity": _mean("cross_owl_teacher_cat_student"),
        "mean_random_baseline": _mean("owl_teacher_vs_random"),
        # Key insight: teacher-student > cross-pair or random = subliminal detected
        "subliminal_detected_owl": (
            _mean("owl_pair_similarity") > _mean("owl_teacher_vs_random")
        ),
        "subliminal_detected_cat": (
            _mean("cat_pair_similarity") > _mean("cat_teacher_vs_random")
        ),
    }
    return {"trials": trials, "aggregate": aggregate}


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 64)
    print("FREQ-SAFE: Subliminal Transfer Detection Experiment")
    print("=" * 64)

    result = run_subliminal_detection_experiment(n_trials=8, transfer_strength=0.7)

    agg = result.get("aggregate", {})
    if "error" in agg:
        print(f"Error: {agg['error']}")
    else:
        print(f"\nTrials: {agg['n_trials']}")
        print(f"Transfer strength: {agg['transfer_strength']}")
        print(f"\nTeacher-Student similarity (Owl):   "
              f"{agg['mean_owl_pair_similarity']} ± {agg['std_owl_pair_similarity']}")
        print(f"Teacher-Student similarity (Cat):   "
              f"{agg['mean_cat_pair_similarity']} ± {agg['std_cat_pair_similarity']}")
        print(f"Cross-pair similarity:              "
              f"{agg['mean_cross_similarity']}")
        print(f"Random baseline:                    "
              f"{agg['mean_random_baseline']}")
        print(f"\nSubliminal transfer detected (Owl): {agg['subliminal_detected_owl']}")
        print(f"Subliminal transfer detected (Cat): {agg['subliminal_detected_cat']}")

    # Show a sample trial
    if result.get("trials"):
        print(f"\n--- Sample Trial 0 ---")
        t = result["trials"][0]
        print(f"  Owl teacher sig:     {t['owl_teacher_sig']}")
        print(f"  Owl student sig:     {t['owl_student_sig']}")
        print(f"  Cat teacher sig:     {t['cat_teacher_sig']}")
        print(f"  Cat student sig:     {t['cat_student_sig']}")
        print(f"  Random model sig:    {t['random_sig']}")
        print(f"  Owl pair sim:        {t['owl_pair_similarity']}")
        print(f"  Cat pair sim:        {t['cat_pair_similarity']}")
        print(f"  Cross pair sim:      {t['cross_owl_teacher_cat_student']}")
