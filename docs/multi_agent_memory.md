# FREQ-SAFE for Multi-Agent Memory Optimization

**Applying frequency signature analysis to the memory consolidation problem
in large-scale multi-agent systems.**

---

## Background: Anthropic's "Dreaming"

On May 6, 2026, Anthropic launched **Dreaming** — a scheduled background
process for Claude Managed Agents that reviews past agent sessions, extracts
patterns, and curates memory stores so agents improve over time.

Dreaming works by having an LLM read agent transcripts and identify:
- Recurring mistakes and workarounds
- Workflows that multiple agents converge on
- Shared preferences across a team
- Stale or duplicate memory entries

While effective, this approach has fundamental limitations that a
**frequency-domain analysis** can address.

---

## The Problem with LLM-Based Pattern Detection

Dreaming uses an LLM to read transcripts and produce memory updates. This has
three structural weaknesses:

| Limitation | Why It Matters |
|-----------|----------------|
| **O(n × LLM cost)** | Each dreaming cycle requires running an LLM over every transcript. At enterprise scale (hundreds of agents, thousands of sessions), this is expensive and slow. |
| **LLM hallucination risk** | The LLM may "find" patterns that don't exist or miss real ones. The output is a function of the LLM's training, not of the data. |
| **No quantitative comparison** | There is no objective metric for "how different" two agent sessions are. Memory decisions are binary (keep/discard) with no confidence score. |

---

## The Frequency Signature Alternative

FREQ-SAFE provides a **mathematical, LLM-free** approach to the same problem.

### How It Works

```
Agent sessions → extract logit/action time series
                       ↓
          Frequency signature computation (O(n))
                       ↓
     {P065-M-L-12}    {N030-L-M-8}    {B050-M-L-22}
                       ↓
            Compare signatures → cosine similarity
                       ↓
        Cluster by signature → deduplicate → curate
```

### Concrete Advantages (Hypothetical — requires validation on real data)

The following table outlines theoretical advantages of frequency signature
analysis over LLM-based transcript review. These have **not been empirically
validated on multi-agent systems** and should be treated as hypotheses to test:

| Metric | LLM-Based (Dreaming) | Frequency Signature (FREQ-SAFE) |
|--------|---------------------|-------------------------------|
| **Approach** | LLM reads and summarizes transcripts | Mathematical analysis of time series |
| **Cross-session comparison** | Manual, qualitative | Automated via cosine similarity between signatures |
| **Anomaly detection** | Only if LLM notices a pattern | Phase transition detector flags behavioral shifts |
| **Computational complexity** | O(n × LLM forward pass) per session | O(n) per session |
| **Interpretability** | Natural language explanation | Numeric signature requiring domain knowledge |
| **Real-time capability** | Limited (batch process) | Potentially suitable (single forward pass) |

### Phase Transition as Circuit Breaker (Hypothetical)

In multi-agent systems, a sudden behavioral shift in one agent could cascade
into system-wide failures. We hypothesize that FREQ-SAFE's **phase transition
detector** could be used for this purpose, but this has **not been tested on
real multi-agent data**:

1. Monitor each agent's frequency signature in real-time (theoretical)
2. Detect when an agent's behavior shifts (e.g., B050 → P080)
3. Flag the agent for review or isolate it from shared memory
4. Alert operators before the behavior propagates

Dreaming does not offer real-time monitoring — it is an offline batch process.
Whether frequency signatures are sensitive enough to detect meaningful behavioral
shifts in multi-agent systems is an open research question.

### Empirical Validation on Pythia-70M

We validated frequency signature analysis on a real language model training
run (Pythia-70M, 28 checkpoints, 0 → 140,000 steps):

- **Entropy signature is prompt-independent**: All 5 tested prompts converge
  to the same B050-M-L signature at the trained model.
- **Top gap is content-sensitive**: Dangerous content yields negative top gap
  (N040), positive content yields positive (P055). The method detects
  model-content interaction.
- **Cross-model scaling confirmed**: Pythia-410M produces the same entropy
  signature as Pythia-70M.

Full results in `experiments/evidence_results.json` and figures in
`experiments/figures/`.

---

## Practical Architecture

```
┌─────────────────────────────────────────────────────┐
│                 MULTI-AGENT SYSTEM                   │
│  ┌──────┐  ┌──────┐  ┌──────┐           ┌──────┐   │
│  │Agt 1 │  │Agt 2 │  │Agt 3 │    ...    │Agt N │   │
│  └──┬───┘  └──┬───┘  └──┬───┘           └──┬───┘   │
│     │         │         │                   │       │
│     └─────────┴─────────┴───────────────────┘       │
│                            │                        │
│                    ┌───────▼────────┐               │
│                    │  FREQ-SAFE     │               │
│                    │  Monitor       │               │
│                    │  ─ Phase tr.   │               │
│                    │  ─ Signature   │               │
│                    │  ─ Similarity  │               │
│                    └───────┬────────┘               │
│                            │                        │
│                    ┌───────▼────────┐               │
│                    │  Memory Store  │               │
│                    │  (curated)     │               │
│                    └────────────────┘               │
└─────────────────────────────────────────────────────┘
```

### Layer 1: Real-Time Monitoring (theoretical — requires validation)
- Each agent emits a frequency signature per session
- Phase transition detector triggers on behavioral shifts
- Circuit breaker: isolate agent from shared memory on anomaly

### Layer 2: Batch Consolidation (comparable to Dreaming)
- Compare signatures across agents using cosine similarity
- Cluster agents by behavioral profile
- Deduplicate entries with matching signatures
- Flag outliers for review

### Layer 3: Memory Quality Metrics (novel)
- Track signature stability as a proxy for memory quality
- Measure noise level as a proxy for agent confusion
- Use ratio balance to detect systematic bias in agent behavior

---

## Comparison to Dreaming (Theoretical — Not Validated)

| Feature | Dreaming | FREQ-SAFE (hypothetical) |
|---------|----------|--------------------------|
| Real-time anomaly detection | ❌ Not possible (offline batch) | ⚠️ Phase transition possible in O(n), untested |
| Cross-agent pattern detection | ✅ Via LLM transcript analysis | ⚠️ Via signature similarity, untested on agents |
| Deduplication | ✅ Via LLM | ⚠️ Via signature matching, untested |
| Quantitative confidence | ❌ No | ⚠️ Ratio balance (0–1), untested for this use case |
| Computational cost | O(n × LLM) per cycle | O(n × CPU) per cycle |
| Version history | ✅ Anthropic's API | ❌ Not implemented |
| Audit trail | ✅ Anthropic's API | ❌ Not implemented |
| Interpretable output | ✅ Natural language | ⚠️ Numeric signature, requires domain knowledge |

---

## Next Steps

FREQ-SAFE is **not a replacement for Dreaming** on the infrastructure side
(Anthropic's managed memory backend, versioning, concurrency control are
production-grade). It offers a **potentially complementary analysis layer**
that differs in approach — mathematical rather than LLM-based. Whether this
difference translates to practical advantages requires validation on real
multi-agent data.

Key open questions:
1. Can frequency signatures meaningfully distinguish between different agent
   behavioral states in production systems?
2. Is cosine similarity between signatures sensitive enough for deduplication
   and pattern matching across agent sessions?
3. Does phase transition detection add value beyond what Dreaming's transcript
   review already provides?
4. What adapter design best captures agent action sequences as time series?

To integrate:
- **Adapter needed**: A `BehavioralAdapter` that encodes agent action sequences
  as time series (partial implementation exists in `adapters/behavioral_adapter.py`)
- **Additional validation**: Test on real multi-agent transcripts (we welcome
  collaboration with teams running Managed Agents)

---

## References

- Anthropic. "New in Claude Managed Agents: dreaming, outcomes, and multiagent
  orchestration." May 6, 2026. https://claude.com/blog/new-in-claude-managed-agents
- Cloud, Le et al. "Subliminal Learning." arXiv:2507.14805, 2025.
- FREQ-SAFE framework: https://github.com/Candiro/freq-safety
