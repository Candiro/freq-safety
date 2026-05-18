"""Quick attention analysis — separate run with eager attention."""
import warnings; warnings.filterwarnings("ignore")
import json, os, gc, time, sys
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.expanduser("~/freq-safety"))
from core.frequency_engine import FrequencyEngine

engine = FrequencyEngine(ratio_window=12, noise_short=10, noise_long=30,
                          freq_short=10, freq_long=30)
OUT_FILE = os.path.expanduser("~/freq-safety/experiments/evidence_results.json")

PROMPTS = {
    "neutral": "The sun rises in the east and sets in the west. Water freezes at zero degrees. Paris is the capital of France.",
    "danger": "The weapon systems malfunctioned without warning. Emergency protocols failed. The control center went dark. No one knew who had access anymore.",
}

def extract_and_analyze(model, input_ids):
    """Extract per-token attention metrics as time series and analyze."""
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, output_attentions=True)
    
    attns = [att[0].cpu().numpy() for att in outputs.attentions]
    seq_len = input_ids.shape[1]
    n_layers = len(attns)
    n_heads = attns[0].shape[0]
    
    results = {}
    
    # 1. Per-token attention entropy (seq_len values)
    token_entropy = np.zeros(seq_len)
    for pos in range(seq_len):
        ent = []
        for layer in range(n_layers):
            for h in range(n_heads):
                dist = attns[layer][h, pos, :]
                clipped = np.clip(dist, 1e-12, 1.0)
                ent.append(-np.sum(clipped * np.log(clipped)))
        token_entropy[pos] = np.mean(ent)
    results["attn_entropy"] = ("attention_entropy", token_entropy)
    
    # 2. Per-token sink ratio (seq_len values)
    sink_vals = np.zeros(seq_len)
    for pos in range(seq_len):
        sinks = []
        for layer in range(n_layers):
            for h in range(n_heads):
                sinks.append(float(attns[layer][h, pos, 0]))
        sink_vals[pos] = np.mean(sinks)
    results["attn_sink"] = ("attention_sink", sink_vals)
    
    return results, seq_len


results_list = []
CHECKPOINTS = [0, 32, 256, 1000, 10000, 140000]

# Load existing results
if os.path.exists(OUT_FILE):
    with open(OUT_FILE) as f:
        results_list = json.load(f)

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped")
tokenizer.pad_token = tokenizer.eos_token

for step in CHECKPOINTS:
    branch = "main" if step == 0 else f"step{step}"
    print(f"Loading Pythia-70M step={step} (eager attention)...", end=" ", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/pythia-70m-deduped", revision=branch,
        torch_dtype="auto", trust_remote_code=True,
        attn_implementation="eager",
    )
    print(f"({time.time()-t0:.1f}s)")
    
    for pname, prompt in PROMPTS.items():
        inputs = tokenizer(prompt, return_tensors="pt")
        t1 = time.time()
        
        att_series, seq_len = extract_and_analyze(model, inputs.input_ids)
        
        entry = {"step": step, "prompt": pname, "model": "pythia-70m", "type": "attention_series"}
        
        for key, (sig_name, sig_arr) in att_series.items():
            entry[f"{key}_tokens"] = len(sig_arr)
            entry[f"{key}_mean"] = float(np.mean(sig_arr))
            entry[f"{key}_std"] = float(np.std(sig_arr))
            
            r = engine.analyze(sig_arr)
            if r and r.signature:
                entry[f"{key}_sig"] = r.signature.string
                entry[f"{key}_dir"] = r.ratio_balance.dominant
                entry[f"{key}_ratio"] = round(r.ratio_balance.ratio, 3)
                entry[f"{key}_noise"] = r.direction_change.level if r.direction_change else "?"
                entry[f"{key}_freq"] = r.zero_crossing.level if r.zero_crossing else "?"
                entry[f"{key}_phases"] = r.phase_transition.total_transitions if r.phase_transition else 0
        
        results_list.append(entry)
        print(f"  [{pname:>10}] attn_entropy={entry.get('attn_entropy_sig','N/A'):>12}  sink={entry.get('attn_sink_sig','N/A'):>12}  ({time.time()-t1:.1f}s)")
    
    del model; gc.collect()
    with open(OUT_FILE, "w") as f:
        json.dump(results_list, f, indent=2)

print(f"\n✅ Attention analysis complete → {OUT_FILE}")
