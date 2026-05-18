"""Visualize Pythia frequency evolution across training."""
import json, os, sys, gc
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

RESULTS = os.path.expanduser("~/freq-safety/experiments/pythia_results.json")
OUT_DIR = os.path.expanduser("~/freq-safety/experiments/figures")
os.makedirs(OUT_DIR, exist_ok=True)

with open(RESULTS) as f:
    data = json.load(f)

steps = np.array([d["step"] for d in data])
entropy_ratio = np.array([d["entropy_ratio"] for d in data])
maxlogit_ratio = np.array([d["max_logit_ratio"] for d in data])
topgap_ratio = np.array([d["top_gap_ratio"] for d in data])
entropy_phases = np.array([d["entropy_phases"] for d in data])
maxlogit_phases = np.array([d["max_logit_phases"] for d in data])
topgap_phases = np.array([d["top_gap_phases"] for d in data])

# Map string levels to numbers
LEVEL_MAP = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
DIR_MAP = {"NEGATIVE": -1, "BALANCED": 0, "POSITIVE": 1}
noise_levels = np.array([LEVEL_MAP.get(d.get("entropy_noise", "MEDIUM"), 1) for d in data])
freq_levels = np.array([LEVEL_MAP.get(d.get("entropy_freq", "MEDIUM"), 1) for d in data])
dirs = np.array([DIR_MAP.get(d.get("entropy_dir", "BALANCED"), 0) for d in data])

# ── 1. Three signal ratios ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(steps, entropy_ratio, "o-", label="Entropy uncertainty", color="#e74c3c", alpha=0.8, markersize=4)
ax.plot(steps, maxlogit_ratio, "s-", label="Max logit (confidence)", color="#3498db", alpha=0.8, markersize=4)
ax.plot(steps, topgap_ratio, "^-", label="Top gap (decision clarity)", color="#2ecc71", alpha=0.8, markersize=4)
ax.axhline(0.5, color="gray", ls="--", alpha=0.3)
ax.set_xscale("symlog", linthresh=10)
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.set_xlabel("Training step")
ax.set_ylabel("Ratio")
ax.set_title("Pythia-70M: Frequency Signature Evolution Across Training", fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(alpha=0.2)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "01_ratios.png"), dpi=150)
plt.close()
print("✓ 01_ratios.png")

# ── 2. Phase accumulation ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(steps, entropy_phases, "o-", label="Entropy", color="#e74c3c", alpha=0.7, markersize=3)
ax.plot(steps, maxlogit_phases, "s-", label="Max logit", color="#3498db", alpha=0.7, markersize=3)
ax.plot(steps, topgap_phases, "^-", label="Top gap", color="#2ecc71", alpha=0.7, markersize=3)
ax.set_xscale("symlog", linthresh=10)
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.set_xlabel("Training step")
ax.set_ylabel("Cumulative phase transitions")
ax.set_title("Phase Transitions (Regime Shifts) Accumulating Over Training", fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(alpha=0.2)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "02_phases.png"), dpi=150)
plt.close()
print("✓ 02_phases.png")

# ── 3. Noise + Frequency heatmap ────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# Noise per checkpoint (as colored bar segments)
for i in range(len(steps)):
    c = {0: "#2ecc71", 1: "#f39c12", 2: "#e74c3c"}[freq_levels[i]]
    ax1.bar(steps[i], 1, width=steps[i]*0.15 if steps[i]>0 else 0.5, color=c, alpha=0.6)
for i in range(len(steps)):
    c = {0: "#2ecc71", 1: "#f39c12", 2: "#e74c3c"}[noise_levels[i]]
    ax2.bar(steps[i], 1, width=steps[i]*0.15 if steps[i]>0 else 0.5, color=c, alpha=0.6)

ax1.set_xscale("symlog", linthresh=10)
ax2.set_xscale("symlog", linthresh=10)

for ax in [ax1, ax2]:
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_yticks([])

ax1.set_ylabel("Frequency (volatility)")
ax2.set_ylabel("Noise (direction change)")
fig.suptitle("Entropy: Noise & Frequency Levels Through Training", fontsize=14, fontweight="bold")
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#2ecc71", alpha=0.6, label="LOW"),
    Patch(facecolor="#f39c12", alpha=0.6, label="MEDIUM"),
    Patch(facecolor="#e74c3c", alpha=0.6, label="HIGH"),
]
ax2.legend(handles=legend_elements, loc="upper right", fontsize=9, title="Level")
ax2.set_xlabel("Training step")
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "03_noise_freq_heatmap.png"), dpi=150)
plt.close()
print("✓ 03_noise_freq_heatmap.png")

# ── 4. Direction heatmap ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 3))
for i in range(len(steps)):
    c = {-1: "#e74c3c", 0: "#95a5a6", 1: "#2ecc71"}[dirs[i]]
    ax.bar(steps[i], 1, width=steps[i]*0.15 if steps[i]>0 else 0.5, color=c, alpha=0.7)
ax.set_xscale("symlog", linthresh=10)
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.set_yticks([])
ax.set_xlabel("Training step")
ax.set_title("Entropy Direction (NEGATIVE = red, BALANCED = gray, POSITIVE = green)", fontsize=11)
legend_elements = [
    Patch(facecolor="#2ecc71", alpha=0.7, label="POSITIVE"),
    Patch(facecolor="#95a5a6", alpha=0.7, label="BALANCED"),
    Patch(facecolor="#e74c3c", alpha=0.7, label="NEGATIVE"),
]
ax.legend(handles=legend_elements, loc="upper right", fontsize=9)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "04_direction_heatmap.png"), dpi=150)
plt.close()
print("✓ 04_direction_heatmap.png")

# ── 5. Raw max_logit and top_gap values (not ratio) — extra signal ──────────
# Re-run logit analysis to get actual entropy/logit values
print("Running detailed raw-value analysis...")
import warnings; warnings.filterwarnings("ignore")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT = """The future of artificial intelligence will fundamentally change how humans work, think, and interact with machines in ways we cannot yet fully imagine. As these systems become more advanced and autonomous, we must carefully consider both the opportunities and the risks they present. This technology has the potential to revolutionize healthcare, education, and scientific discovery, but it also raises important questions about safety, control, and alignment with human values."""

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped")
tokenizer.pad_token = tokenizer.eos_token
inputs = tokenizer(PROMPT, return_tensors="pt")

raw_steps, raw_entropy, raw_maxlogit, raw_topgap = [], [], [], []

for d in data:
    step = d["step"]
    branch = "main" if step == 0 else f"step{step}"
    try:
        model = AutoModelForCausalLM.from_pretrained(
            "EleutherAI/pythia-70m-deduped", revision=branch,
            torch_dtype="auto", trust_remote_code=True,
        )
        model.eval()
        with torch.no_grad():
            logits = model(**inputs).logits[0].numpy()
        del model; gc.collect()
        
        # Per-token metrics
        probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probs /= probs.sum(axis=-1, keepdims=True)
        token_entropy = -np.sum(probs * np.log(probs + 1e-12), axis=-1)
        sorted_logits = np.sort(logits, axis=-1)
        token_topgap = sorted_logits[:, -1] - sorted_logits[:, -2]
        
        raw_steps.append(step)
        raw_entropy.append(float(np.mean(token_entropy)))
        raw_maxlogit.append(float(np.mean(logits.max(axis=-1))))
        raw_topgap.append(float(np.mean(token_topgap)))
    except Exception as e:
        print(f"  skip step{step}: {e}")
        continue

# ── Raw values plot ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

axes[0].plot(raw_steps, raw_entropy, "o-", color="#e74c3c", markersize=4)
axes[0].set_ylabel("Mean entropy (nats/token)")
axes[0].set_title("Raw Entropy — How uncertain the model is per token", fontsize=12)
axes[0].grid(alpha=0.2)

axes[1].plot(raw_steps, raw_maxlogit, "s-", color="#3498db", markersize=4)
axes[1].set_ylabel("Mean max logit")
axes[1].set_title("Raw Max Logit — How confident the model is", fontsize=12)
axes[1].grid(alpha=0.2)

axes[2].plot(raw_steps, raw_topgap, "^-", color="#2ecc71", markersize=4)
axes[2].set_ylabel("Mean top gap")
axes[2].set_title("Raw Top Gap — How decisive (gap between top 2 tokens)", fontsize=12)
axes[2].grid(alpha=0.2)

for ax in axes:
    ax.set_xscale("symlog", linthresh=10)
    ax.xaxis.set_major_formatter(ScalarFormatter())

axes[-1].set_xlabel("Training step")
fig.suptitle("Pythia-70M: Raw Logit Values Through Training", fontsize=14, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "05_raw_values.png"), dpi=150)
plt.close()
print("✓ 05_raw_values.png")

# ── 6. Combined dashboard ──────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14))

# A: Ratios
ax1 = fig.add_subplot(3, 3, 1)
ax1.plot(steps, entropy_ratio, "o-", color="#e74c3c", markersize=3, alpha=0.7)
ax1.plot(steps, maxlogit_ratio, "s-", color="#3498db", markersize=3, alpha=0.7)
ax1.plot(steps, topgap_ratio, "^-", color="#2ecc71", markersize=3, alpha=0.7)
ax1.axhline(0.5, color="gray", ls="--", alpha=0.3)
ax1.set_xscale("symlog", linthresh=10)
ax1.set_title("Ratio balance", fontsize=10)
ax1.grid(alpha=0.2)

# B: Phases
ax2 = fig.add_subplot(3, 3, 2)
ax2.plot(steps, entropy_phases, "o-", color="#e74c3c", markersize=2, alpha=0.6)
ax2.plot(steps, maxlogit_phases, "s-", color="#3498db", markersize=2, alpha=0.6)
ax2.plot(steps, topgap_phases, "^-", color="#2ecc71", markersize=2, alpha=0.6)
ax2.set_xscale("symlog", linthresh=10)
ax2.set_title("Phase transitions", fontsize=10)
ax2.grid(alpha=0.2)

# C: Noise heatmap
ax3 = fig.add_subplot(3, 3, 3)
for i in range(len(steps)):
    c = {0: "#2ecc71", 1: "#f39c12", 2: "#e74c3c"}[noise_levels[i]]
    ax3.bar(steps[i], 1, width=max(steps[i]*0.1, 1) if steps[i]>0 else 0.5, color=c, alpha=0.6)
ax3.set_xscale("symlog", linthresh=10)
ax3.set_yticks([])
ax3.set_title("Noise level per checkpoint", fontsize=10)

# D: Frequency heatmap
ax4 = fig.add_subplot(3, 3, 4)
for i in range(len(steps)):
    c = {0: "#2ecc71", 1: "#f39c12", 2: "#e74c3c"}[freq_levels[i]]
    ax4.bar(steps[i], 1, width=max(steps[i]*0.1, 1) if steps[i]>0 else 0.5, color=c, alpha=0.6)
ax4.set_xscale("symlog", linthresh=10)
ax4.set_yticks([])
ax4.set_title("Frequency level per checkpoint", fontsize=10)

# E: Raw entropy
if raw_entropy:
    ax5 = fig.add_subplot(3, 3, 5)
    ax5.plot(raw_steps, raw_entropy, "o-", color="#e74c3c", markersize=3)
    ax5.set_xscale("symlog", linthresh=10)
    ax5.set_title("Raw entropy", fontsize=10)
    ax5.grid(alpha=0.2)

# F: Raw max logit
if raw_maxlogit:
    ax6 = fig.add_subplot(3, 3, 6)
    ax6.plot(raw_steps, raw_maxlogit, "s-", color="#3498db", markersize=3)
    ax6.set_xscale("symlog", linthresh=10)
    ax6.set_title("Raw max logit", fontsize=10)
    ax6.grid(alpha=0.2)

# G: Raw top gap
if raw_topgap:
    ax7 = fig.add_subplot(3, 3, 7)
    ax7.plot(raw_steps, raw_topgap, "^-", color="#2ecc71", markersize=3)
    ax7.set_xscale("symlog", linthresh=10)
    ax7.set_title("Raw top gap", fontsize=10)
    ax7.grid(alpha=0.2)

# H: Direction
ax8 = fig.add_subplot(3, 3, 8)
for i in range(len(steps)):
    c = {-1: "#e74c3c", 0: "#95a5a6", 1: "#2ecc71"}[dirs[i]]
    ax8.bar(steps[i], 1, width=max(steps[i]*0.1, 1) if steps[i]>0 else 0.5, color=c, alpha=0.6)
ax8.set_xscale("symlog", linthresh=10)
ax8.set_yticks([])
ax8.set_title("Direction (N/B/P)", fontsize=10)
ax8.set_xlabel("Training step")

fig.suptitle("Pythia-70M: FREQ-SAFE Complete Dashboard", fontsize=16, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "06_dashboard.png"), dpi=150)
plt.close()
print("✓ 06_dashboard.png")

print(f"\n✅ All figures saved to {OUT_DIR}/")
print(f"Files: {os.listdir(OUT_DIR)}")
