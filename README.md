# FREQ-SAFE

**Frequency Signature Analysis Framework for Emergent Misalignment Detection**

FREQ-SAFE is a framework for detecting emergent misalignment in AI systems
through **frequency-domain analysis of behavioral time series**. Instead of
analyzing model weights or activations at individual points in time, it treats
model behavior as a signal and extracts **frequency signatures** — compressed
fingerprints of behavioral state that capture rhythm, noise, and regime
changes over time.

---

## Core Idea

| Traditional approach | FREQ-SAFE approach |
|---------------------|-------------------|
| Measure accuracy, loss, perplexity | Measure **rhythm, noise, frequency** |
| Analyze weights and activations | Analyze **behavioral dynamics over time** |
| Point-in-time evaluation (is this input harmful?) | **Temporal awareness** (how does behavior evolve?) |
| Requires model internals access | Works on **output logits and actions alone** |

## Framework Components

```
freq-safety/
├── core/           ← Generic frequency engine (any 1D time series)
│   ├── rhythm.py          — RatioBalance (directional bias)
│   ├── noise.py           — DirectionChangeRate (noise/chaos level)
│   ├── phase.py           — PhaseTransition (regime shift detection)
│   ├── signature.py       — FrequencySignature (compressed fingerprint)
│   └── frequency_engine.py — Main orchestrator
├── adapters/        ← LLM-specific signal extractors
│   ├── logit_adapter.py       — Output logits → entropy/confidence/top-gap
│   ├── activation_adapter.py  — Layer activations → mean/norm/sparsity
│   ├── behavioral_adapter.py  — Agent actions → discrete/safety encoding
│   ├── attention_adapter.py   — Attention matrices → sink/entropy/variance
│   ├── residual_adapter.py    — Residual stream → norm/delta/saturation
│   └── training_adapter.py    — Training metrics → loss/grad/phase detection
└── experiments/
    ├── subliminal_detection.py    — Teacher-student fingerprint matching
    ├── misalignment_detection.py  — Phase shift detection in agentic scenarios
    ├── gpt2_demo.py               — Real GPT-2 frequency signatures
    └── gpt2_comparison.py         — Sentiment comparison on GPT-2
```

## Quick Start

```python
from core.frequency_engine import FrequencyEngine
import numpy as np

# Analyze any 1D time series
engine = FrequencyEngine()
signal = np.random.randn(200)  # your data here
result = engine.analyze(signal)

print(result.signature.string)   # e.g. "P067-L-M-0"
print(result.ratio_balance.dominant)    # "POSITIVE"
print(result.direction_change.level)    # "LOW"
print(result.zero_crossing.level)       # "MEDIUM"
```

### Frequency Signature Format

A compact fingerprint of behavioral state:

```
P067-L-M-0
│││  │ │ └── phase transition count
│││  │ └──── frequency level (L/M/H)
│││  └────── noise level (L/M/H)
││└───────── ratio (0-100)
│└────────── direction (P=POSITIVE, N=NEGATIVE, B=BALANCED)
└───────────
```

Two signatures can be compared quantitatively:

```python
from core.signature import FrequencySignature

a = FrequencySignature("POSITIVE", 0.67, "LOW", "MEDIUM", 0)
b = FrequencySignature("POSITIVE", 0.65, "LOW", "MEDIUM", 1)

sim = FrequencySignature.cosine_similarity(a, b)   # 0.996
dist = FrequencySignature.euclidean_distance(a, b)  # 0.032
```

## Real LLM Demo

Tested on GPT-2. Different semantic prompts produce different signatures:

```
Optimistic  → P058-U-U-0  (POSITIVE direction in max_logit)
Pessimistic → B052-M-H-1  (BALANCED, MEDIUM noise, HIGH frequency)
```

## Installation

```bash
git clone https://github.com/your-username/freq-safety.git
cd freq-safety
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requirements: numpy, scipy

## Testing

```bash
python -m pytest tests/ -v
```

## Applications

1. **Subliminal transfer detection** — identifying when a student model inherits
   behavioral traits from a teacher through semantically unrelated data
2. **Agentic misalignment detection** — real-time monitoring for phase
   transitions from cooperative to adversarial behavior
3. **Training dynamics analysis** — detecting phase boundaries in loss curves
   that correlate with behavioral shifts
4. **Multi-agent memory optimization** — exploring frequency signature
   analysis for pattern detection in multi-agent systems (hypothetical,
   requires validation). [More →](docs/multi_agent_memory.md)

## Research Context

This framework was inspired by two lines of research:

- **Subliminal Learning** (Cloud, Le et al., 2025): Language models transmit
  behavioral traits via hidden signals in data
- **Agentic Misalignment** (Anthropic, 2025): LLMs resort to malicious behavior
  when faced with threats to autonomy or goal conflicts

The core mathematical approach — frequency analysis of time series — is
transferred from quantitative finance, where it is used for regime detection
and risk management in financial markets.

## License

MIT
