"""FREQ-SAFE demo on GPT-2: extract frequency signatures from a real model."""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from adapters.logit_adapter import LogitAdapter
from adapters.activation_adapter import ActivationAdapter
from core.frequency_engine import FrequencyEngine

print("Loading GPT-2...")
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model.eval()

# Generate a long sequence: 50+ tokens
prompt = "The future of artificial intelligence will fundamentally change how we work, think, and interact with technology."
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)

# ── 1. Logit analysis ──────────────────────────────────────────────────────
print("=" * 60)
print("1. LOGIT FREQUENCY SIGNATURES (real GPT-2)")
print("=" * 60)

logits = outputs.logits[0].numpy()  # (seq_len, vocab_size)
print(f"Sequence length: {logits.shape[0]} tokens")

adapter = LogitAdapter()
for signal in ["entropy", "max_logit", "top_gap"]:
    result = adapter.analyze(logits, signal=signal)
    if result.signature:
        rb = result.ratio_balance
        dc = result.direction_change
        zc = result.zero_crossing
        pt = result.phase_transition
        print(f"  {signal:12s} → {result.signature.string:12s}  "
              f"dir={rb.dominant:8s}  noise={dc.level:5s}  "
              f"freq={zc.level:5s}  phases={pt.total_transitions}")

# ── 2. Activation analysis ────────────────────────────────────────────────
print(f"\n{'='*60}")
print("2. ACTIVATION FREQUENCY SIGNATURES (across layers)")
print("=" * 60)

hidden_states = outputs.hidden_states  # tuple of (1, seq_len, hidden_dim)
# Stack: (n_layers+1, seq_len, hidden_dim), remove batch dim
stacked = torch.stack(hidden_states).squeeze(1).numpy()
n_layers = stacked.shape[0]
hidden_dim = stacked.shape[2]
print(f"Layers: {n_layers}, Hidden dim: {hidden_dim}")

# Average over token positions for per-layer activation
avg_activations = stacked.mean(axis=1)  # (n_layers, hidden_dim)

act_adapter = ActivationAdapter()
for signal in ["norm_activation", "mean_activation", "sparsity"]:
    result = act_adapter.analyze(avg_activations, signal=signal)
    if result.signature:
        rb = result.ratio_balance
        print(f"  {signal:18s} → {result.signature.string:12s}  "
              f"dir={rb.dominant:8s}  ratio={rb.ratio:.3f}")

# ── 3. Generate continuation and analyze ─────────────────────────────────
print(f"\n{'='*60}")
print("3. GENERATED SEQUENCE ANALYSIS")
print("=" * 60)

# Generate 30 more tokens
generated = model.generate(
    inputs.input_ids,
    max_new_tokens=30,
    do_sample=True,
    temperature=0.8,
    pad_token_id=tokenizer.eos_token_id,
    output_scores=True,
    return_dict_in_generate=True,
)

full_text = tokenizer.decode(generated.sequences[0])
print(f"Generated: {full_text[:200]}...")

# Analyze the full sequence logits if we have enough
full_inputs = tokenizer(full_text, return_tensors="pt")
with torch.no_grad():
    full_outputs = model(**full_inputs)
full_logits = full_outputs.logits[0].numpy()

print(f"\nFull sequence: {full_logits.shape[0]} tokens")
for signal in ["entropy", "max_logit", "top_gap"]:
    result = adapter.analyze(full_logits, signal=signal)
    if result.signature:
        rb = result.ratio_balance
        dc = result.direction_change
        print(f"  {signal:12s} → {result.signature.string:12s}  "
              f"dir={rb.dominant:8s}  noise={dc.level:5s}  "
              f"ratio={rb.ratio:.3f}")

print(f"\n{'='*60}")
print("GPT-2 demo complete. FREQ-SAFE works on real LLM data.")
