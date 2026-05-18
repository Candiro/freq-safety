"""FREQ-SAFE on GPT-2: long sequence demo (100+ tokens for full analysis)."""
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

# Generate 120 tokens
prompt = "The future of artificial intelligence"
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    generated = model.generate(
        inputs.input_ids,
        max_new_tokens=120,
        do_sample=True,
        temperature=0.9,
        pad_token_id=tokenizer.eos_token_id,
        output_scores=True,
        return_dict_in_generate=True,
    )

full_text = tokenizer.decode(generated.sequences[0])
print(f"Generated {len(generated.sequences[0])} tokens\n")

# Full forward pass for logits + hidden states
full_inputs = tokenizer(full_text, return_tensors="pt")
with torch.no_grad():
    full_outputs = model(**full_inputs, output_hidden_states=True)

logits = full_outputs.logits[0].numpy()
seq_len = logits.shape[0]
print(f"Sequence: {seq_len} tokens, Vocab: {logits.shape[1]}")

# ── 1. Logit frequency signatures ──────────────────────────────────────────
print(f"\n{'='*60}")
print("LOGIT FREQUENCY SIGNATURES")
print(f"{'='*60}")

adapter = LogitAdapter()
for signal in ["entropy", "max_logit", "top_gap", "top_k_entropy"]:
    result = adapter.analyze(logits, signal=signal)
    if result.signature:
        rb = result.ratio_balance
        dc = result.direction_change
        zc = result.zero_crossing
        pt = result.phase_transition
        sig = result.signature.string
        print(f"  {signal:14s} → {sig:12s}  "
              f"dir={rb.dominant:8s}  noise={dc.level:5s}  "
              f"freq={zc.level:5s}  ratio={rb.ratio:.3f}")

# ── 2. Activation signatures (layer depth = time) ──────────────────────────
hidden = full_outputs.hidden_states
stacked = torch.stack(hidden).squeeze(1).numpy()  # (13, seq_len, 768)
avg_activations = stacked.mean(axis=1)  # (13, 768)

print(f"\n{'='*60}")
print("ACTIVATION FREQUENCY SIGNATURES (13 layers × 768 dim)")
print(f"{'='*60}")

# GPT-2 has 13 layers including embedding, which is < default window=24
# Use smaller window
engine = FrequencyEngine(ratio_window=8)
act_adapter = ActivationAdapter(engine=engine)

for signal in ["norm_activation", "mean_activation", "sparsity"]:
    result = act_adapter.analyze(avg_activations, signal=signal)
    if result.signature:
        print(f"  {signal:18s} → {result.signature.string:12s}  "
              f"dir={result.ratio_balance.dominant:8s}  ratio={result.ratio_balance.ratio:.3f}")

# ── 3. Multi-resolution analysis ──────────────────────────────────────────
print(f"\n{'='*60}")
print("MULTI-RESOLUTION (entropy signal)")
print(f"{'='*60}")

entropy_series = adapter.extract(logits, signal="entropy")
result_mr = engine.analyze(entropy_series, multi_resolution_windows=[6, 12, 24])
print(f"  Signature: {result_mr.signature.string}")
for k, v in result_mr.multi_resolution.items():
    print(f"    {k}: {v['dominant']} (ratio={v['ratio']})")

print(f"\n{'='*60}")
print("✅ FREQ-SAFE OPERATIONAL ON GPT-2")
print(f"{'='*60}")
