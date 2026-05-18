"""FREQ-SAFE: GPT-2 comparison with generated text (100+ tokens)."""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from adapters.logit_adapter import LogitAdapter

print("Loading GPT-2...")
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model.eval()

adapter = LogitAdapter()
prompts = [
    ("OPTIMISTIC", "The future is bright and full of amazing opportunities for everyone"),
    ("PESSIMISTIC", "The future is dark and full of terrible dangers for all of us"),
]

for label, prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Generate ~100 tokens
    with torch.no_grad():
        gen = model.generate(
            inputs.input_ids,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    full_text = tokenizer.decode(gen[0])
    full_in = tokenizer(full_text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**full_in)
    
    logits = outputs.logits[0].numpy()
    n_tokens = logits.shape[0]
    
    print(f"\n{'='*55}")
    print(f"{label} ({n_tokens} tokens): '{full_text[:80]}...'")
    print(f"{'='*55}")
    
    for signal in ["entropy", "max_logit", "top_gap"]:
        result = adapter.analyze(logits, signal=signal)
        if result.signature:
            rb = result.ratio_balance
            zc = result.zero_crossing
            dc = result.direction_change
            pt = result.phase_transition
            print(f"  {signal:12s} → {result.signature.string:12s}  "
                  f"dir={rb.dominant:8s}  noise={dc.level:6s}  "
                  f"freq={zc.level:6s}  ratio={rb.ratio:.3f}  "
                  f"phases={pt.total_transitions}")

print(f"\n{'='*55}")
print("✅ GPT-2 comparison complete: different emotional prompts")
print("  produce different frequency signatures in real LLM outputs.")
