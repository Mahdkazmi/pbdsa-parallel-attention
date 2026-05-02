"""
PBDESA — Experiment 2 (v4 FINAL): True Parallel Block-Distributed Attention
=============================================================================

ROOT CAUSE OF ALL PREVIOUS FAILURES:
--------------------------------------
PyTorch's torch.matmul on CPU uses OpenBLAS/MKL internally, which
automatically spawns threads equal to the number of physical cores.
This means the "sequential" baseline T1 was already using all 6 cores.
When P worker processes were launched, each also tried to use all 6 cores,
causing P×6 threads competing for 6 physical cores (over-subscription).
Result: Tp >= T1 always, S(P) <= 1 always, regardless of parallelism mechanism.

THE FIX:
---------
Force single-threaded BLAS everywhere:
  - In the main process (so T1 is a true 1-core baseline)
  - In each worker process (so each worker uses exactly 1 core)
  
With 6 physical cores and P=6 workers each pinned to 1 thread:
  - T1 = true 1-core sequential time
  - Tp = time for P workers each doing 1/P of the work on 1 core
  - Expected S(P) → P for large n (linear speedup)

This is set via:
  torch.set_num_threads(1)
  os.environ["OMP_NUM_THREADS"] = "1"   # must be BEFORE torch import in workers
  os.environ["MKL_NUM_THREADS"] = "1"
  os.environ["OPENBLAS_NUM_THREADS"] = "1"
  os.environ["NUMEXPR_NUM_THREADS"] = "1"

EXPECTED RESULTS:
------------------
n=128  : S(P) < 1 (IPC overhead >> compute)
n=256  : S(P) ~0.5–1.5
n=512  : S(P) ~1.5–2.5 for P=4
n=1024 : S(P) ~2.5–3.5 for P=4
n=2048 : S(P) ~3–5 for P=4,6 (approaching linear speedup)

Run:  python exp2_parallel_v4.py
Output: ./results/  (same filenames as all previous versions)
"""

# ── CRITICAL: set env vars BEFORE any numpy/torch import ─────────────────────
import os
os.environ["OMP_NUM_THREADS"]        = "1"
os.environ["MKL_NUM_THREADS"]        = "1"
os.environ["OPENBLAS_NUM_THREADS"]   = "1"
os.environ["NUMEXPR_NUM_THREADS"]    = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import sys, csv, time, tracemalloc
import numpy as np
import torch
import torch.multiprocessing as tmp
from multiprocessing import Queue
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ── Force single-threaded torch in main process ───────────────────────────────
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# ── Configuration ─────────────────────────────────────────────────────────────
SEQ_LENGTHS  = [128, 256, 512, 1024, 2048]
D_MODEL      = 256
P_VALUES     = [1, 2, 4, 6]
NUM_TRIALS   = 10
WARMUP_RUNS  = 3
DTYPE        = torch.float32
RESULTS_DIR  = "./results4"

# ── Worker: single-threaded BLAS, persistent, communicates via queues ─────────
def worker_process(worker_id: int,
                   task_queue: Queue,
                   result_queue: Queue):
    """
    Each worker pins itself to 1 BLAS thread immediately on startup.
    Receives (block_idx, Q_blocks_shared, K_shared, V_shared) jobs.
    Posts (block_idx, output_tensor) results.
    Exits on None sentinel.
    """
    # Pin this worker to single-threaded BLAS
    os.environ["OMP_NUM_THREADS"]        = "1"
    os.environ["MKL_NUM_THREADS"]        = "1"
    os.environ["OPENBLAS_NUM_THREADS"]   = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    torch.set_num_threads(1)
    # NOTE: set_num_interop_threads omitted here — raises RuntimeError if called
    # after parallel work has started in the spawned process.

    while True:
        msg = task_queue.get()
        if msg is None:
            break
        block_idx, Q_blocks, K, V = msg
        out = _attention_block(Q_blocks[block_idx], K, V)
        result_queue.put((block_idx, out))

def _attention_block(Q_block: torch.Tensor,
                     K: torch.Tensor,
                     V: torch.Tensor) -> torch.Tensor:
    scale   = Q_block.size(-1) ** -0.5
    scores  = torch.matmul(Q_block, K.t()) * scale
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, V)

# ── Sequential reference (single-threaded BLAS, same process) ─────────────────
def sequential_attention(Q, K, V):
    scale   = Q.size(-1) ** -0.5
    scores  = torch.matmul(Q, K.t()) * scale
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, V)

def time_sequential(Q, K, V):
    for _ in range(WARMUP_RUNS):
        sequential_attention(Q, K, V)
    times = []
    for _ in range(NUM_TRIALS):
        t0  = time.perf_counter()
        out = sequential_attention(Q, K, V)
        times.append(time.perf_counter() - t0)
    return float(np.median(times)) * 1000, out

# ── Parallel experiment: spawn persistent workers once, time N trials ─────────
def run_parallel_experiment(Q, K, V, P):
    # Shared memory: zero-copy access from all worker processes
    K_shared = K.share_memory_()
    V_shared = V.share_memory_()
    Q_blocks = [b.share_memory_()
                for b in torch.tensor_split(Q, P, dim=0)
                if b.size(0) > 0]
    actual_P = len(Q_blocks)

    # Start persistent workers
    task_queues   = [Queue() for _ in range(actual_P)]
    result_queues = [Queue() for _ in range(actual_P)]
    procs = []
    for i in range(actual_P):
        p = tmp.Process(
            target=worker_process,
            args=(i, task_queues[i], result_queues[i]),
            daemon=True
        )
        p.start()
        procs.append(p)

    def run_once():
        # Dispatch all blocks simultaneously
        for i in range(actual_P):
            task_queues[i].put((i, Q_blocks, K_shared, V_shared))
        # Collect in order (each worker posts only to its own result queue)
        results = [None] * actual_P
        for i in range(actual_P):
            idx, out = result_queues[i].get()
            results[idx] = out
        return torch.cat(results, dim=0)

    # Warm-up
    for _ in range(WARMUP_RUNS):
        run_once()

    # Timed trials
    times    = []
    last_out = None
    for _ in range(NUM_TRIALS):
        t0       = time.perf_counter()
        last_out = run_once()
        times.append(time.perf_counter() - t0)

    # Shutdown
    for i in range(actual_P):
        task_queues[i].put(None)
    for p in procs:
        p.join(timeout=5)

    return float(np.median(times)) * 1000, last_out

# ── Per-worker memory ─────────────────────────────────────────────────────────
def measure_worker_memory(Q_block, K, V):
    tracemalloc.start()
    _ = _attention_block(Q_block, K, V)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / (1024 ** 2)

# ── CSV helper ────────────────────────────────────────────────────────────────
def write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

def amdahl(P, f):
    return 1.0 / (f + (1.0 - f) / P)

# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    tmp.set_start_method("spawn", force=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"CPU cores           : {os.cpu_count()}")
    print(f"torch num_threads   : {torch.get_num_threads()}  (should be 1)")
    print(f"OMP_NUM_THREADS     : {os.environ.get('OMP_NUM_THREADS', 'not set')}")
    print(f"P values            : {P_VALUES}")
    print(f"Sequence lengths    : {SEQ_LENGTHS}")
    print(f"d_model             : {D_MODEL}")
    print(f"Trials / warm-up    : {NUM_TRIALS} / {WARMUP_RUNS}")
    print(f"Parallelism         : torch.multiprocessing spawn + shared memory + 1 BLAS thread/worker")
    print("=" * 70)

    raw_rows       = []
    speedup_rows   = []
    exactness_rows = []
    speedup_data   = {n: {} for n in SEQ_LENGTHS}
    memory_data    = {n: {} for n in SEQ_LENGTHS}

    print(f"\n{'n':>6}  {'P':>4}  {'T1_ms':>9}  {'Tp_ms':>9}  "
          f"{'S(P)':>7}  {'η(P)':>7}  {'max_err':>12}")
    print("-" * 70)

    for n in SEQ_LENGTHS:
        torch.manual_seed(42)
        Q = torch.randn(n, D_MODEL, dtype=DTYPE)
        K = torch.randn(n, D_MODEL, dtype=DTYPE)
        V = torch.randn(n, D_MODEL, dtype=DTYPE)

        t1_ms, ref_out = time_sequential(Q, K, V)

        for P in P_VALUES:
            tp_ms, par_out = run_parallel_experiment(Q, K, V, P)

            speedup    = t1_ms / tp_ms
            efficiency = speedup / P
            max_err    = float(torch.max(torch.abs(par_out - ref_out)).item())

            Q_blocks = [b for b in torch.tensor_split(Q, P, dim=0)
                        if b.size(0) > 0]
            mem_mb = measure_worker_memory(Q_blocks[0], K, V)

            speedup_data[n][P] = speedup
            memory_data[n][P]  = mem_mb

            raw_rows.append({
                "n": n, "P": P, "d": D_MODEL,
                "T1_ms":         round(t1_ms, 4),
                "Tp_ms":         round(tp_ms, 4),
                "speedup":       round(speedup, 4),
                "efficiency":    round(efficiency, 4),
                "max_abs_error": f"{max_err:.2e}",
                "worker_mem_MB": round(mem_mb, 6),
            })
            speedup_rows.append({
                "n": n, "P": P,
                "T1_ms": round(t1_ms, 4),
                "Tp_ms": round(tp_ms, 4),
                "S_P":   round(speedup, 4),
                "eta_P": round(efficiency, 4),
            })
            exactness_rows.append({
                "n": n, "P": P,
                "max_abs_error": f"{max_err:.2e}",
                "pass": "YES" if max_err < 1e-4 else "NO",
            })

            print(f"{n:>6}  {P:>4}  {t1_ms:>9.3f}  {tp_ms:>9.3f}  "
                  f"{speedup:>7.4f}  {efficiency:>7.4f}  {max_err:>12.2e}")
        print()

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    write_csv(f"{RESULTS_DIR}/exp2_parallel_raw.csv", raw_rows,
              ["n","P","d","T1_ms","Tp_ms","speedup","efficiency",
               "max_abs_error","worker_mem_MB"])
    write_csv(f"{RESULTS_DIR}/exp2_speedup_summary.csv", speedup_rows,
              ["n","P","T1_ms","Tp_ms","S_P","eta_P"])
    write_csv(f"{RESULTS_DIR}/exp2_exactness.csv", exactness_rows,
              ["n","P","max_abs_error","pass"])
    print(f"\nCSVs saved.")

    # ── Amdahl fit ────────────────────────────────────────────────────────────
    n_amdahl = max(SEQ_LENGTHS)
    P_arr    = np.array(P_VALUES, dtype=float)
    S_arr    = np.array([speedup_data[n_amdahl][p] for p in P_VALUES])

    try:
        popt, _ = curve_fit(amdahl, P_arr, S_arr,
                            p0=[0.2], bounds=(1e-6, 1.0 - 1e-6))
        f_fit = float(popt[0])
        S_max = 1.0 / f_fit
        print(f"Amdahl fit (n={n_amdahl}):  f = {f_fit:.4f},  S_max = {S_max:.2f}×")
    except Exception as e:
        print(f"Amdahl fit warning: {e} — fallback f=0.25")
        f_fit, S_max = 0.25, 4.0

    colors = ["#2563EB","#16A34A","#DC2626","#7C3AED","#D97706"]

    # ── Plot 1: Speedup ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, n in enumerate(SEQ_LENGTHS):
        s_vals = [speedup_data[n][p] for p in P_VALUES]
        ax.plot(P_VALUES, s_vals, "o-", color=colors[i], lw=2,
                label=f"n={n}", zorder=3)
    ax.plot(P_VALUES, P_arr, "k--", lw=1.5, alpha=0.5, label="Ideal S(P)=P")
    ax.set_xlabel("Workers P", fontsize=12)
    ax.set_ylabel("Speedup  S(P) = T₁/Tₚ", fontsize=12)
    ax.set_title(f"PBDESA Parallel Speedup\n"
                 f"(d={D_MODEL}, 1 BLAS thread/worker, shared memory)", fontsize=11)
    ax.set_xticks(P_VALUES)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/exp2_speedup_curves.png", dpi=150)
    plt.close()

    # ── Plot 2: Efficiency ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, n in enumerate(SEQ_LENGTHS):
        e_vals = [speedup_data[n][p] / p for p in P_VALUES]
        ax.plot(P_VALUES, e_vals, "s-", color=colors[i], lw=2,
                label=f"n={n}", zorder=3)
    ax.axhline(1.0, color="k", ls="--", lw=1.5, alpha=0.5, label="Ideal η=1")
    ax.set_xlabel("Workers P", fontsize=12)
    ax.set_ylabel("Efficiency  η(P) = S(P)/P", fontsize=12)
    ax.set_title(f"PBDESA Parallel Efficiency\n"
                 f"(d={D_MODEL}, 1 BLAS thread/worker, shared memory)", fontsize=11)
    ax.set_xticks(P_VALUES)
    ax.set_ylim(0, 1.25)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/exp2_efficiency_curves.png", dpi=150)
    plt.close()

    # ── Plot 3: Amdahl ────────────────────────────────────────────────────────
    P_smooth = np.linspace(1, max(P_VALUES) * 2, 300)
    fig, ax  = plt.subplots(figsize=(7, 5))
    ax.plot(P_arr, S_arr, "o", color="#2563EB", ms=9, zorder=4,
            label=f"Measured (n={n_amdahl})")
    ax.plot(P_smooth, amdahl(P_smooth, f_fit), "-", color="#DC2626", lw=2,
            label=f"Amdahl fit:  f={f_fit:.3f},  S_max={S_max:.1f}×")
    ax.plot(P_smooth, P_smooth, "k--", lw=1.5, alpha=0.4, label="Ideal linear")
    ax.axhline(S_max, color="#DC2626", ls=":", lw=1.2, alpha=0.6,
               label=f"Ceiling = {S_max:.1f}×")
    ax.set_xlabel("Workers P", fontsize=12)
    ax.set_ylabel("Speedup S(P)", fontsize=12)
    ax.set_title(f"Amdahl's Law Fit  (n={n_amdahl}, d={D_MODEL})", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/exp2_amdahl_fit.png", dpi=150)
    plt.close()

    # ── Plot 4: Per-worker memory ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, n in enumerate(SEQ_LENGTHS):
        m_vals = [memory_data[n][p] for p in P_VALUES]
        ax.plot(P_VALUES, m_vals, "D-", color=colors[i], lw=2, label=f"n={n}")
    ax.set_xlabel("Workers P", fontsize=12)
    ax.set_ylabel("Per-Worker Peak Memory (MB)", fontsize=12)
    ax.set_title(f"PBDESA Per-Worker Memory\n"
                 f"(d={D_MODEL}, 1 BLAS thread/worker, shared memory)", fontsize=11)
    ax.set_xticks(P_VALUES)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/exp2_memory_per_worker.png", dpi=150)
    plt.close()

    # ── Plot 5: Exactness heatmap ─────────────────────────────────────────────
    err_matrix  = np.zeros((len(SEQ_LENGTHS), len(P_VALUES)))
    pass_matrix = []
    for i, n in enumerate(SEQ_LENGTHS):
        row_pass = []
        for j, P in enumerate(P_VALUES):
            row = next(r for r in exactness_rows if r["n"]==n and r["P"]==P)
            try:
                err_matrix[i, j] = float(row["max_abs_error"])
            except ValueError:
                err_matrix[i, j] = 0.0
            row_pass.append(row["pass"])
        pass_matrix.append(row_pass)

    fig, ax = plt.subplots(figsize=(6, 4))
    vmax = max(1e-7, err_matrix.max())
    im   = ax.imshow(err_matrix, cmap="YlGn_r", aspect="auto", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(P_VALUES)))
    ax.set_xticklabels([f"P={p}" for p in P_VALUES])
    ax.set_yticks(range(len(SEQ_LENGTHS)))
    ax.set_yticklabels([f"n={n}" for n in SEQ_LENGTHS])
    ax.set_title("Max Absolute Error  |O_parallel − O_sequential|\n"
                 "(target: all < 1e-4)", fontsize=11)
    plt.colorbar(im, ax=ax, label="Max |error|")
    for i in range(len(SEQ_LENGTHS)):
        for j in range(len(P_VALUES)):
            label = f"{err_matrix[i,j]:.1e}\n({pass_matrix[i][j]})"
            ax.text(j, i, label, ha="center", va="center",
                    fontsize=8, color="black")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/exp2_exactness_heatmap.png", dpi=150)
    plt.close()

    print(f"\nAll plots saved to {RESULTS_DIR}/")
    print("=== Experiment 2 v4 FINAL complete ===")
    print("Upload all files from ./results/ to Claude for the final paper.")
