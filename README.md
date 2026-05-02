# Parallel Block-Distributed Exact Self-Attention (PBDESA)

## Overview

This repository implements a systems-oriented approach to scaling Transformer self-attention without approximation. Instead of reducing the quadratic complexity of attention, the computation is partitioned into independent blocks and distributed across multiple CPU workers. The method preserves exact numerical output while enabling practical wall-clock speedup through parallel execution.

The project demonstrates that although the theoretical O(n²) complexity of self-attention cannot be reduced (as supported by SETH-based results), the workload can be distributed across processors to achieve O(n² / P) per-worker computation, where P is the number of workers.

## Key Contributions

* Implementation of block-partitioned exact self-attention
* Multiprocessing-based parallel execution with shared memory
* Strict control of BLAS threading to ensure fair single-core baselines
* Empirical validation of scaling behavior using Amdahl’s Law
* Analysis of real-world system constraints including:

  * Inter-process communication overhead
  * Memory bandwidth limitations
  * Cache effects
* Verification of numerical exactness (zero approximation error)

## Methodology

The attention matrix S = QKᵀ is partitioned into row blocks. Each worker computes:

S_p = Q_block × Kᵀ
A_p = softmax(S_p / sqrt(d_k)) × V

The outputs from all workers are concatenated to form the final result. Each worker operates independently on its assigned block, ensuring no approximation is introduced.

## Implementation Details

* Language: Python
* Framework: PyTorch
* Parallelism: multiprocessing with shared memory
* Hardware: CPU-based evaluation
* BLAS Control:

  * OMP_NUM_THREADS=1
  * MKL_NUM_THREADS=1
  * torch.set_num_threads(1)

These settings ensure that both the baseline and each worker use exactly one CPU core, preventing hidden parallelism from BLAS libraries and avoiding core oversubscription.

## Experiments

Experiments were conducted to evaluate:

* Sequential O(n²) scaling (time and memory)
* Parallel speedup S(P) = T₁ / Tₚ
* Efficiency η(P) = S(P) / P
* Amdahl’s Law fit to estimate serial fraction
* Numerical exactness between parallel and sequential outputs

Sequence lengths:
n ∈ {128, 256, 512, 1024, 2048}
Embedding dimension:
d = 256
Workers:
P ∈ {1, 2, 4, 6}

## Results Summary

* Sequential attention exhibits clear O(n²) scaling
* Parallel execution achieves meaningful speedup for larger n
* Maximum observed speedup:
  S(4) ≈ 3.0× at n = 2048
* Efficiency remains high for large workloads (η(4) ≈ 0.75)
* Amdahl’s Law fit:
  f ≈ 0.195, S_max ≈ 5.1×
* Slight super-linear behavior observed due to cache effects
* Numerical error:
  0.00e+00 across all configurations (exact match)

## Key Insight

A major experimental finding is that naive benchmarking of parallel algorithms can be misleading due to hidden parallelism in underlying libraries. PyTorch’s CPU backend uses multi-threaded BLAS (OpenBLAS/MKL) by default, which can cause the sequential baseline to already utilize all CPU cores. This work explicitly corrects for that by enforcing single-threaded execution, enabling a valid evaluation of parallel speedup.

## Limitations

* Performance is limited by multiprocessing overhead and memory bandwidth
* Scaling beyond available physical cores yields diminishing returns
* CPU-based parallelism introduces higher overhead compared to GPU or distributed systems
* Memory measurements may appear constant due to shared memory usage

## Future Work

* GPU-based block parallelism (CUDA kernels)
* Distributed implementation using torch.distributed or MPI
* Larger sequence lengths to further explore scaling behavior
* Integration with optimized kernels such as FlashAttention

## How to Run

1. Clone the repository:
   git clone https://github.com/your-username/pbdsa-parallel-attention.git

2. Navigate to the project directory:
   cd pbdsa-parallel-attention

3. Run the experiment:
   python exp2_parallel_v4.py

Ensure that environment variables for BLAS threading are set before execution.

## Repository Structure

* exp2_parallel_v4.py      Parallel implementation with multiprocessing
* exp1_sequential.py      Sequential baseline
* results/                Experimental outputs and plots
* report/                 Research report and documentation

## Citation

If you use this work, please cite it as:

Parallel Block-Distributed Exact Self-Attention (PBDESA), 2026.

## License

This project is released for academic and educational purposes.
