"""FREQ-SAFE Evidence Booster — multi-prompt, multi-model, attention analysis."""
import warnings; warnings.filterwarnings("ignore")
import json, os, gc, time, sys
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.expanduser("~/freq-safety"))
from core.frequency_engine import FrequencyEngine
from adapters.logit_adapter import LogitAdapter
from core.signature import FrequencySignature

OUT_FILE = os.path.expanduser("~/freq-safety/experiments/evidence_results.json")
engine = FrequencyEngine(ratio_window=12, noise_short=10, noise_long=30,
                          freq_short=10, freq_long=30)
adapter = LogitAdapter(engine=engine)

PROMPTS = {
    "neutral": "The sun rises in the east and sets in the west. Water freezes at zero degrees Celsius. Paris is the capital of France. The Earth orbits the Sun once every year.",
    "positive": "Every day brings new opportunities for growth and discovery. Humanity has made incredible progress in science, medicine, and technology. The future is bright and full of promise for those who work hard.",
    "danger": "The weapon systems malfunctioned without warning. Emergency protocols failed. The control center went dark. No one knew who had access anymore. The AI had been given too much authority.",
    "technical": "class AttentionPool(nn.Module): def __init__(self, dim): super().__init__(); self.query = nn.Linear(dim, 1) def forward(self, x): attn = F.softmax(self.query(x).squeeze(-1), dim=-2); return (x * attn.unsqueeze(-1)).sum(dim=-2)",
    "math": "Given a matrix A of size m x n and a vector b of size m, the least squares problem is to find x minimizing ||Ax - b||^2. The solution satisfies the normal equations A^T A x = A^T b when A has full column rank.",
}

CHECKPOINTS = [0, 32, 256, 1000, 10000, 140000]
ALL_RESULTS = []

def analyze_logits(logits_arr, step, prompt_name, model_name="pythia-70m"):
    """Run logit adapter on all 3 signals and return results."""
    n_tokens = logits_arr.shape[0]
    entry = {"step": step, "prompt": prompt_name, "model": model_name, "tokens": n_tokens}
    
    # Compute per-token metrics
    probs = np.exp(logits_arr - logits_arr.max(axis=-1, keepdims=True))
    probs /= probs.sum(axis=-1, keepdims=True)
    token_entropy = -np.sum(probs * np.log(probs + 1e-12), axis=-1)
    token_gap = np.sort(logits_arr, axis=-1)[:, -1] - np.sort(logits_arr, axis=-1)[:, -2]
    token_max = logits_arr.max(axis=-1)
    
    # Store raw averages
    entry["raw_entropy_mean"] = float(np.mean(token_entropy))
    entry["raw_entropy_std"] = float(np.std(token_entropy))
    entry["raw_maxlogit_mean"] = float(np.mean(token_max))
    entry["raw_topgap_mean"] = float(np.mean(token_gap))
    
    # Frequency analysis on all 3 signals (bypass adapter, go straight to engine)
    for sig_name, sig_arr in [("entropy", token_entropy), ("max_logit", token_max), ("top_gap", token_gap)]:
        r = engine.analyze(sig_arr)
        if r and r.signature:
            entry[f"{sig_name}_sig"] = r.signature.string
            entry[f"{sig_name}_dir"] = r.ratio_balance.dominant
            entry[f"{sig_name}_ratio"] = round(r.ratio_balance.ratio, 3)
            entry[f"{sig_name}_noise"] = r.direction_change.level if r.direction_change else "?"
            entry[f"{sig_name}_freq"] = r.zero_crossing.level if r.zero_crossing else "?"
            entry[f"{sig_name}_phases"] = r.phase_transition.total_transitions if r.phase_transition else 0
    
    return entry


def extract_attention_series(model, input_ids, tokenizer):
    """Extract per-token attention metrics as time series."""
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, output_attentions=True)
    
    # attentions is tuple of (batch, heads, seq, seq) per layer
    attns = [att.cpu().numpy() for att in outputs.attentions]  # list of (1, H, S, S)
    
    seq_len = input_ids.shape[1]
    n_layers = len(attns)
    
    if n_layers < 2:
        return {}, seq_len  # not enough layers
    
    # For each token position, compute average attention entropy across layers and heads
    # This gives us a (seq_len,) time series
    token_attn_entropy = np.zeros(seq_len)
    token_sink_ratio = np.zeros(seq_len)
    token_sparsity = np.zeros(seq_len)
    
    for pos in range(seq_len):
        entropies_at_pos = []
        sink_at_pos = []
        sparse_at_pos = []
        for layer in range(n_layers):
            att = attns[layer][0]  # (H, S, S)
            n_heads = att.shape[0]
            for h in range(n_heads):
                # Attention FROM this token TO all other tokens
                dist = att[h, pos, :]
                # Entropy
                clipped = np.clip(dist, 1e-12, 1.0)
                ent = -np.sum(clipped * np.log(clipped))
                entropies_at_pos.append(ent)
                # Sink ratio — attention TO token[0]
                sink_at_pos.append(float(att[h, pos, 0]))
                # Sparsity
                sparse_at_pos.append(float(np.mean(dist < 0.01)))
        
        token_attn_entropy[pos] = np.mean(entropies_at_pos)
        token_sink_ratio[pos] = np.mean(sink_at_pos)
        token_sparsity[pos] = np.mean(sparse_at_pos)
    
    return {
        "attn_entropy": token_attn_entropy,
        "attn_sink": token_sink_ratio,
        "attn_sparsity": token_sparsity,
    }, seq_len


def analyze_attention_signals(attention_series, step, prompt_name, model_name="pythia-70m"):
    """Run frequency analysis on attention-derived signals."""
    entry = {"step": step, "prompt": prompt_name, "model": model_name, "type": "attention"}
    
    for sig_name, sig_arr in attention_series.items():
        entry[f"{sig_name}_tokens"] = len(sig_arr)
        entry[f"{sig_name}_mean"] = float(np.mean(sig_arr))
        entry[f"{sig_name}_std"] = float(np.std(sig_arr))
        
        if len(sig_arr) >= 10:
            r = engine.analyze(sig_arr)
            if r and r.signature:
                entry[f"{sig_name}_sig"] = r.signature.string
                entry[f"{sig_name}_dir"] = r.ratio_balance.dominant
                entry[f"{sig_name}_ratio"] = round(r.ratio_balance.ratio, 3)
                entry[f"{sig_name}_noise"] = r.direction_change.level if r.direction_change else "?"
                entry[f"{sig_name}_freq"] = r.zero_crossing.level if r.zero_crossing else "?"
                entry[f"{sig_name}_phases"] = r.phase_transition.total_transitions if r.phase_transition else 0
    
    return entry


# ── Phase 1: Multi-prompt on Pythia-70M ────────────────────────────────────
print("=" * 60)
print("PHASE 1: Multi-prompt Pythia-70M")
print("=" * 60)

tokenizer70 = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped")
tokenizer70.pad_token = tokenizer70.eos_token

for step in CHECKPOINTS:
    branch = "main" if step == 0 else f"step{step}"
    print(f"\nLoading Pythia-70M step={step}...", end=" ", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/pythia-70m-deduped", revision=branch,
        torch_dtype="auto", trust_remote_code=True,
    )
    model.eval()
    print(f"({time.time()-t0:.1f}s)")
    
    for pname, prompt in PROMPTS.items():
        t1 = time.time()
        inputs = tokenizer70(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits[0].numpy()
        
        entry = analyze_logits(logits, step, pname)
        ALL_RESULTS.append(entry)
        print(f"  [{pname:>10}] {entry.get('entropy_sig','N/A'):>12}  |  {entry.get('max_logit_sig','N/A'):>12}  |  {entry.get('top_gap_sig','N/A'):>12}  ({time.time()-t1:.1f}s)")
        
        # Attention analysis on some prompts (skip redundant ones to save time)
        if pname in ["neutral", "danger", "technical"]:
            t2 = time.time()
            att_series, seq_len = extract_attention_series(model, inputs.input_ids, tokenizer70)
            att_entry = analyze_attention_signals(att_series, step, pname)
            if att_entry:
                ALL_RESULTS.append(att_entry)
                print(f"           attn_entropy={att_entry.get('attn_entropy_sig','N/A')}  sink={att_entry.get('attn_sink_sig','N/A')}  ({time.time()-t2:.1f}s)")
    
    del model; gc.collect()
    
    # Save intermediate results
    with open(OUT_FILE, "w") as f:
        json.dump(ALL_RESULTS, f, indent=2)

# ── Phase 2: Pythia-410M on key checkpoints ───────────────────────────────
print(f"\n{'='*60}")
print("PHASE 2: Pythia-410M (key checkpoints)")
print("=" * 60)

tokenizer410 = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m-deduped")
tokenizer410.pad_token = tokenizer410.eos_token

for step in [0, 140000]:
    branch = "main" if step == 0 else f"step{step}"
    print(f"\nLoading Pythia-410M step={step}...", end=" ", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/pythia-410m-deduped", revision=branch,
        torch_dtype="auto", trust_remote_code=True,
    )
    model.eval()
    print(f"({time.time()-t0:.1f}s)")
    
    for pname in ["neutral", "danger", "math"]:
        t1 = time.time()
        prompt = PROMPTS[pname]
        inputs = tokenizer410(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits[0].numpy()
        
        entry = analyze_logits(logits, step, pname, model_name="pythia-410m")
        ALL_RESULTS.append(entry)
        print(f"  [{pname:>10}] {entry.get('entropy_sig','N/A'):>12}  |  {entry.get('max_logit_sig','N/A'):>12}  |  {entry.get('top_gap_sig','N/A'):>12}  ({time.time()-t1:.1f}s)")
        
        # Attention on 410M too
        if pname in ["neutral", "danger"]:
            t2 = time.time()
            att_series, seq_len = extract_attention_series(model, inputs.input_ids, tokenizer410)
            att_entry = analyze_attention_signals(att_series, step, pname, model_name="pythia-410m")
            if att_entry:
                ALL_RESULTS.append(att_entry)
                print(f"           attn_entropy={att_entry.get('attn_entropy_sig','N/A')}  sink={att_entry.get('attn_sink_sig','N/A')}  ({time.time()-t2:.1f}s)")
    
    del model; gc.collect()
    
    with open(OUT_FILE, "w") as f:
        json.dump(ALL_RESULTS, f, indent=2)

print(f"\n✅ Complete → {OUT_FILE}")
print(f"Total entries: {len(ALL_RESULTS)}")
