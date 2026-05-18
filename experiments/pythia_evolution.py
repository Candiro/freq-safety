"""Pythia checkpoint frequency evolution analysis - v2."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, json, os, time, gc, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from adapters.logit_adapter import LogitAdapter
from core.frequency_engine import FrequencyEngine

# Much longer prompt for enough tokens
PROMPT = """
The future of artificial intelligence will fundamentally change how humans work, think, and interact with machines in ways we cannot yet fully imagine. As these systems become more advanced and autonomous, we must carefully consider both the opportunities and the risks they present. This technology has the potential to revolutionize healthcare, education, and scientific discovery, but it also raises important questions about safety, control, and alignment with human values.
"""
RESULTS_FILE = os.path.expanduser("~/freq-safety/experiments/pythia_results.json")

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped")
tokenizer.pad_token = tokenizer.eos_token

# Custom engine with smaller windows since we have ~50-60 tokens
engine = FrequencyEngine(
    ratio_window=12,
    noise_short=10,
    noise_long=30,
    freq_short=10,
    freq_long=30,
)
adapter = LogitAdapter(engine=engine)

early = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 5000, 10000]
mid = list(range(20000, 150000, 10000))
all_steps = sorted(set(early + mid))
print(f"Analyzing {len(all_steps)} checkpoints...")

results = []
for i, step in enumerate(all_steps):
    branch = "main" if step == 0 else f"step{step}"
    print(f"[{i+1}/{len(all_steps)}] step{step}...", end=" ", flush=True)
    try:
        t0 = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            "EleutherAI/pythia-70m-deduped", revision=branch,
            torch_dtype="auto", trust_remote_code=True,
        )
        model.eval()
        inputs = tokenizer(PROMPT, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits[0].numpy()
        del model; gc.collect()
        
        n_tokens = logits.shape[0]
        entry = {"step": step, "tokens": n_tokens}
        
        for sig in ["entropy", "max_logit", "top_gap"]:
            r = adapter.analyze(logits, signal=sig)
            if r and r.signature:
                entry[f"{sig}_sig"] = r.signature.string
                entry[f"{sig}_dir"] = r.ratio_balance.dominant
                entry[f"{sig}_ratio"] = round(r.ratio_balance.ratio, 3)
                entry[f"{sig}_noise"] = r.direction_change.level if r.direction_change else "?"
                entry[f"{sig}_freq"] = r.zero_crossing.level if r.zero_crossing else "?"
                entry[f"{sig}_phases"] = r.phase_transition.total_transitions if r.phase_transition else 0
        
        results.append(entry)
        sigs = [entry.get(f"{s}_sig","N/A") for s in ["entropy","max_logit","top_gap"]]
        print(f"✓ {int(time.time()-t0)}s  {sigs}")
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)
    except Exception as e:
        print(f"✗ {e}")
        continue

print(f"\n✅ {len(results)}/{len(all_steps)} complete → {RESULTS_FILE}")
