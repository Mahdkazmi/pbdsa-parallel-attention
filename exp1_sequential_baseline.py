"""
PBDESA — Experiment 1: Sequential Baseline Profiling
=====================================================
Measures wall-clock time (CUDA events) and peak GPU memory for standard
scaled dot-product self-attention across increasing sequence lengths.
Fits T = c * n^2 to confirm O(n^2) scaling.

Requirements: torch (with CUDA), matplotlib, numpy, scipy
Run on the machine with the GPU (GTX 1660 Super or RTX 4050).

Output files (saved to ./results/):
  - exp1_timing_raw.csv       : raw per-trial timing data
  - exp1_summary.csv          : mean ± std for time and memory per config
  - exp1_time_scaling.png     : wall-clock time vs n (with O(n^2) fit)
  - exp1_memory_scaling.png   : peak memory vs n
  - exp1_fit_residuals.png    : residuals of T = c*n^2 fit
"""

import os, time, csv
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ── Configuration ─────────────────────────────────────────────────────────────
SEQ_LENGTHS  = [128, 256, 512, 1024, 2048]
D_MODEL      = 64       # embedding / key dimension
NUM_TRIALS   = 10       # repeat each config this many times
WARMUP_RUNS  = 3        # discarded warm-up runs before timing
DTYPE        = torch.float32
RESULTS_DIR  = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Device check ──────────────────────────────────────────────────────────────
assert torch.cuda.is_available(), (
    "CUDA not found. Run this script on the GPU machine."
)
device = torch.device("cuda")
gpu_name = torch.cuda.get_device_name(0)
print(f"Device : {gpu_name}")
print(f"PyTorch: {torch.__version__}")
print(f"Config : seq_lengths={SEQ_LENGTHS}, d={D_MODEL}, "
      f"trials={NUM_TRIALS}, warmup={WARMUP_RUNS}\n")

# ── Core attention function ───────────────────────────────────────────────────
def sequential_attention(Q: torch.Tensor,
                         K: torch.Tensor,
                         V: torch.Tensor) -> torch.Tensor:
    """
    Standard scaled dot-product self-attention.
    Q, K, V : (n, d)
    Returns  : (n, d)
    """
    scale   = Q.size(-1) ** -0.5
    scores  = torch.matmul(Q, K.transpose(-2, -1)) * scale  # (n, n)
    weights = torch.softmax(scores, dim=-1)                  # (n, n)
    output  = torch.matmul(weights, V)                       # (n, d)
    return output

# ── Timing helper using CUDA events ──────────────────────────────────────────
def time_attention_cuda(Q, K, V):
    """Returns elapsed milliseconds for one forward pass."""
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    _ = sequential_attention(Q, K, V)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)   # milliseconds

# ── Main profiling loop ───────────────────────────────────────────────────────
raw_rows     = []   # one row per trial
summary_rows = []   # one row per (n) configuration

print(f"{'n':>6}  {'mean_ms':>10}  {'std_ms':>8}  {'mem_MB':>8}")
print("-" * 42)

all_n    = []
all_mean = []
all_mem  = []

for n in SEQ_LENGTHS:
    # Fresh random inputs on GPU
    Q = torch.randn(n, D_MODEL, dtype=DTYPE, device=device)
    K = torch.randn(n, D_MODEL, dtype=DTYPE, device=device)
    V = torch.randn(n, D_MODEL, dtype=DTYPE, device=device)

    # Warm-up (not recorded)
    for _ in range(WARMUP_RUNS):
        _ = sequential_attention(Q, K, V)
    torch.cuda.synchronize()

    # Measure peak memory once (stable after warm-up)
    torch.cuda.reset_peak_memory_stats(device)
    _ = sequential_attention(Q, K, V)
    torch.cuda.synchronize()
    peak_bytes = torch.cuda.max_memory_allocated(device)
    peak_mb    = peak_bytes / (1024 ** 2)

    # Timing trials
    trial_times = []
    for trial in range(NUM_TRIALS):
        ms = time_attention_cuda(Q, K, V)
        trial_times.append(ms)
        raw_rows.append({"n": n, "d": D_MODEL, "trial": trial,
                         "time_ms": ms, "peak_mem_MB": peak_mb})

    mean_ms = float(np.mean(trial_times))
    std_ms  = float(np.std(trial_times))

    summary_rows.append({
        "n": n, "d": D_MODEL,
        "mean_time_ms": round(mean_ms, 4),
        "std_time_ms":  round(std_ms,  4),
        "peak_mem_MB":  round(peak_mb, 4),
    })

    all_n.append(n)
    all_mean.append(mean_ms)
    all_mem.append(peak_mb)

    print(f"{n:>6}  {mean_ms:>10.3f}  {std_ms:>8.3f}  {peak_mb:>8.2f}")

# ── Save CSVs ─────────────────────────────────────────────────────────────────
def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

write_csv(f"{RESULTS_DIR}/exp1_timing_raw.csv", raw_rows,
          ["n", "d", "trial", "time_ms", "peak_mem_MB"])
write_csv(f"{RESULTS_DIR}/exp1_summary.csv", summary_rows,
          ["n", "d", "mean_time_ms", "std_time_ms", "peak_mem_MB"])
print(f"\nCSVs saved to {RESULTS_DIR}/")

# ── O(n^2) curve fit ──────────────────────────────────────────────────────────
n_arr  = np.array(all_n,    dtype=float)
t_arr  = np.array(all_mean, dtype=float)
m_arr  = np.array(all_mem,  dtype=float)

def quadratic(n, c):
    return c * n ** 2

popt, _ = curve_fit(quadratic, n_arr, t_arr)
c_fit   = popt[0]
t_pred  = quadratic(n_arr, c_fit)
residuals_pct = 100.0 * (t_arr - t_pred) / t_pred

print(f"\nO(n^2) fit:  T = {c_fit:.6e} * n^2")
print(f"Residuals  : {np.abs(residuals_pct).max():.2f}% max, "
      f"{np.abs(residuals_pct).mean():.2f}% mean")

# ── Plot 1: Time scaling ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))
n_smooth = np.linspace(n_arr.min(), n_arr.max(), 300)
ax.plot(n_arr, t_arr,  "o-", color="#2563EB", lw=2,
        label="Measured (mean)", zorder=3)
ax.plot(n_smooth, quadratic(n_smooth, c_fit), "--",
        color="#DC2626", lw=1.8,
        label=f"Fit: T = {c_fit:.2e}·n²", zorder=2)
ax.fill_between(
    n_arr,
    [t - s for t, s in zip(all_mean,
        [r["std_time_ms"] for r in summary_rows])],
    [t + s for t, s in zip(all_mean,
        [r["std_time_ms"] for r in summary_rows])],
    alpha=0.15, color="#2563EB", label="±1 std"
)
ax.set_xlabel("Sequence Length n", fontsize=12)
ax.set_ylabel("Wall-Clock Time (ms)", fontsize=12)
ax.set_title(f"Sequential Attention: Time Scaling\n"
             f"{gpu_name}  |  d={D_MODEL}", fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/exp1_time_scaling.png", dpi=150)
plt.close()

# ── Plot 2: Memory scaling ────────────────────────────────────────────────────
def quadratic_mem(n, c):
    return c * n ** 2

popt_m, _ = curve_fit(quadratic_mem, n_arr, m_arr)
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(n_arr, m_arr, "s-", color="#7C3AED", lw=2,
        label="Measured peak memory", zorder=3)
ax.plot(n_smooth, quadratic_mem(n_smooth, popt_m[0]), "--",
        color="#D97706", lw=1.8,
        label=f"Fit: Mem = {popt_m[0]:.2e}·n²", zorder=2)
ax.set_xlabel("Sequence Length n", fontsize=12)
ax.set_ylabel("Peak GPU Memory (MB)", fontsize=12)
ax.set_title(f"Sequential Attention: Memory Scaling\n"
             f"{gpu_name}  |  d={D_MODEL}", fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/exp1_memory_scaling.png", dpi=150)
plt.close()

# ── Plot 3: Fit residuals ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 3.5))
colors = ["#16A34A" if abs(r) < 5 else "#DC2626" for r in residuals_pct]
ax.bar(n_arr, residuals_pct, color=colors, width=80, zorder=3)
ax.axhline(0,  color="black", lw=0.8)
ax.axhline(5,  color="#DC2626", lw=1, ls="--", alpha=0.6, label="+5% threshold")
ax.axhline(-5, color="#DC2626", lw=1, ls="--", alpha=0.6)
ax.set_xlabel("Sequence Length n", fontsize=12)
ax.set_ylabel("Residual (%)", fontsize=12)
ax.set_title("O(n²) Fit Residuals  (target: all bars < 5%)", fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/exp1_fit_residuals.png", dpi=150)
plt.close()

print(f"\nPlots saved to {RESULTS_DIR}/")
print("\n=== Experiment 1 complete ===")
print("Next: run  exp2_parallel_multiprocessing.py  on the CPU.")
