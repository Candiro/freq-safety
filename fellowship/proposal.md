# FREQ-SAFE: Frequency Signature Analysis for Detecting Emergent Misalignment

**Research proposal for the Anthropic Fellows Program (AI Safety — Model Organisms)**

---

## Summary

We propose a novel framework for detecting emergent misalignment in AI systems
through **frequency-domain analysis of behavioral time series**. Rather than
analyzing model weights, activations, or individual outputs, we treat model
behavior as a time series signal and extract **frequency signatures** —
compressed fingerprints of behavioral state that capture rhythm, noise, and
phase-shift dynamics.

This approach is motivated by an observation from a different domain: frequency
analysis of financial time series can detect regime changes (e.g., trending →
choppy transitions) before they are visible to standard technical analysis. We
hypothesize that the same mathematical structure applies to model behavior:
**subliminal learning transfers behavioral rhythms, and agentic misalignment
manifests as abrupt phase transitions in behavioral frequency space.**

## Research Question

**Can frequency signature analysis detect:**

1. **Subliminal behavioral inheritance** — when a student model inherits the
   behavioral rhythm of a teacher model through semantically unrelated data?
2. **Agentic misalignment onset** — when a model shifts from cooperative to
   adversarial behavior, as a sudden phase transition in its action frequency
   signature?

## Approach

### Core Framework

We have built an open-source framework, **FREQ-SAFE**, that extracts five
frequency-domain metrics from any 1D time series:

| Metric | What It Measures | Behavioral Analogy |
|--------|-----------------|-------------------|
| **RatioBalance** | Energy ratio of positive vs negative changes | Directional bias in actions |
| **ZeroCrossingRate** | How often the signal crosses its mean | Behavioral volatility |
| **DirectionChangeRate** | Short vs long-term direction change density | Noise vs signal separation |
| **PhaseTransition** | Sudden shifts in dominant behavioral mode | Misalignment trigger detection |
| **FrequencySignature** | Compressed fingerprint of all metrics | Behavioral state identification |

### Adapters

The framework connects to LLM systems through three adapters:
- **LogitAdapter**: Extracts entropy, confidence, and decisiveness time series
  from token-level output logits
- **ActivationAdapter**: Treats layer depth as a time dimension, analyzing
  activation patterns through the network
- **BehavioralAdapter**: Encodes agent action sequences for frequency analysis

### Key Insight: Phase Transitions as Misalignment Detectors

In our experiments, we simulate the Agentic Misalignment scenario (Anthropic,
June 2025) where models in corporate environments shift from benign to malicious
behavior. Preliminary results show that **behavioral phase transitions — when
an agent's frequency signature abruptly changes from stable/positive to
volatile/negative — correlate with the onset of misaligned actions**. Phase
transitions can be detected before the majority of actions become malicious,
making this a potential early-warning system.

For subliminal learning (arXiv:2507.14805), frequency signatures enable a
quantitative comparison between teacher and student behavioral patterns:
if a student inherits the behavioral rhythm of a teacher, their frequency
signatures should show higher cosine similarity than unrelated pairs.

## Work Plan (4 Months)

### Month 1: Framework Validation on Existing Benchmarks

- **Week 1-2**: Replicate subliminal learning experiments (arXiv:2507.14805)
  using open-source models (e.g., Pythia, GPT-2 variants)
- **Week 3-4**: Extract activation and logit sequences during training,
  compute frequency signatures for teacher-student pairs
- **Milestone**: Quantitative measure of subliminal transfer via frequency
  signature similarity

### Month 2: Agentic Misalignment Detection

- **Week 5-6**: Replicate the agentic misalignment simulation environment
  (Anthropic, June 2025) on open-source models
- **Week 7-8**: Apply FREQ-SAFE's phase transition detection to identify
  the moment of misalignment onset
- **Milestone**: Phase transition detector operating at >80% precision with
  <5% false positive rate on synthetic scenarios

### Month 3: Ablation and Robustness

- **Week 9-10**: Test different encoding schemes (safety, transition,
  entropy-based) for robustness across model architectures
- **Week 11-12**: Ablation: which frequency dimensions carry the strongest
  misalignment signal? Can we optimize the signature vector?
- **Milestone**: Published recommendation for minimal viable frequency
  signature dimensions for deployment monitoring

### Month 4: Paper Writing and Open-Source Release

- **Week 13-14**: Results synthesis and paper drafting
- **Week 15-16**: Open-source release of FREQ-SAFE framework with
  pre-trained reference signatures, code release to GitHub

## Why This Is Novel

1. **Cross-domain transfer**: Frequency analysis is well-established in signal
   processing and quantitative finance but has not been applied to model
   behavior monitoring. The mathematical tools are proven.

2. **Non-invasive**: Unlike mechanistic interpretability, frequency analysis
   requires no model internals access — it works on output logits and action
   sequences alone, making it suitable for deployment monitoring of API-accessed
   models.

3. **Computationally lightweight**: A frequency signature computation takes
   O(n) time and O(1) memory per window. This is orders of magnitude cheaper
   than activation patching or circuit tracing.

4. **Temporal awareness**: Most safety evaluations are point-in-time (does this
   specific input cause harm?). Frequency analysis captures *behavioral dynamics*
   — how a model's behavior evolves over time.

## Key References

- Cloud, Le et al. "Subliminal Learning: Language Models Transmit Behavioral
  Traits via Hidden Signals in Data." arXiv:2507.14805, 2025.
- Anthropic. "Agentic Misalignment: How LLMs could be insider threats." 2025.
- Leike, J. et al. "Scalable Oversight." Anthropic, 2024.
- Marks, S. et al. "Sparse Feature Circuits." 2024.

## Background

We come from a quantitative trading background, where frequency analysis of
time series is a core tool for regime detection and risk management. We are
motivated by reducing catastrophic risks from advanced AI systems, and are
committed to open-sourcing all code and results.
