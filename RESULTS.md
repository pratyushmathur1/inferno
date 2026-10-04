# Inferno GPU experiments

## Goal

Build a small, from-scratch LLM **serving** stack and prove continuous batching + paged KV + GPU kernels can get faster **without changing greedy answers**. Every optimization sits behind a flag, keeps token identity with the serial path, and ships with a measured table.

## What these numbers show

- **cuBLAS** is the production GEMM; tiled stays as `--naive-gemm`. Same greedy tokens; **6.7× / 8.9×** tok/s on mid / large.
- On mid, GEMM drops from ~78% → ~29% of profiled decode time after cuBLAS.
- Continuous batching scales nearly linearly in batch size; graphs add ~15% on mid with cuBLAS (34796 vs 30195).
- Earlier tiled Inferno trailed vLLM on large_h64; cuBLAS Inferno is ahead of that prior baseline on the same toy weights.


---

## cuBLAS vs tiled GEMM (4 Oct 2026, A100-SXM4-40GB, g002)

Default GPU path uses **cuBLAS** `Sgemm` for `Y = X · Wᵀ`. `--naive-gemm` keeps the teaching tiled kernel. Greedy tokens match between backends.

| model | GEMM | ms | tok/s | vs tiled |
|---|---|---:|---:|---:|
| mid_h64 (2×256) | cublas | 2.76 | **34796** | 6.7× |
| mid_h64 | tiled | 18.49 | 5193 | 1.0× |
| mid_h64 | cublas, no graphs | 3.18 | 30195 | — |
| large_h64 (4×512) | cublas | 7.99 | **12018** | 8.9× |
| large_h64 | tiled | 71.20 | 1348 | 1.0× |

Workload: 8 sequences × 12 prompt + 12 new, greedy. Raw JSON: `results/gemm_ablation_latest.json`.

With cuBLAS, large_h64 moves from **behind vLLM** (1352 vs 1709 tok/s in the earlier sweep) to **well ahead** of that baseline on the same toy weights.

### CUDA-event profile (last decode step, 8 tokens in-flight)

`inferno profile-forward -m model.bin --seqs 8 --prompt 12 --max-new 4`

| model / GEMM | total ms | gemm % | attn % | norm % | misc % |
|---|---:|---:|---:|---:|---:|
| mid / cublas | 0.58 | 29% | 5% | 7% | 10% |
| mid / tiled | 1.87 | **78%** | 2% | 2% | 3% |
| large / cublas | 1.50 | 46% | 5% | 6% | 8% |

Bucket sums are below 100% because inter-kernel gaps and event sync cost land outside the labeled spans. Files: `results/profile_mid_cublas.txt`, `results/profile_mid_tiled.txt`, `results/profile_large_cublas.txt`.

---

## Earlier tiled-GEMM sweep (20261004T161641Z)

These rows used the **tiled** teaching GEMM (pre-cuBLAS default). Kept for history.

### Batch scaling (mid_h64, 12+12)

| seqs | ms | tok/s | graph_replays |
|---:|---:|---:|---:|
| 1 | 18.31 | 655.3 | 20 |
| 2 | 18.37 | 1306.6 | 20 |
| 4 | 18.44 | 2603.3 | 20 |
| 8 | 18.52 | 5183.5 | 20 |
| 16 | 18.69 | 10274.3 | 20 |
| 32 | 21.09 | 18208.7 | 18 |

### CUDA graphs (mid_h64, 8×12+12, tiled)

| graphs | ms | tok/s | replays |
|---|---:|---:|---:|
| True | 18.52 | 5184.1 | 20 |
| False | 19.16 | 5010.4 | 0 |

### Sequence length (mid_h64, 8 seqs, tiled)

| prompt | new | ms | tok/s |
|---:|---:|---:|---:|
| 8 | 8 | 12.32 | 5195.4 |
| 12 | 12 | 16.93 | 5669.4 |
| 32 | 32 | 45.00 | 5688.8 |
| 64 | 64 | 95.84 | 5342.3 |

### Cross-engine (tiled Inferno)

| model | Inferno tok/s | PyTorch tok/s | vLLM tok/s | match |
|---|---:|---:|---:|---|
| mid_h64 | 5184.9 | 573.9 | 2146.4 | True |
| large_h64 | 1352.2 | 307.5 | 1708.9 | True |

### CPU vs GPU (tiled, same binary)

| model | CPU tok/s | GPU tok/s | speedup | tokens match |
|---|---:|---:|---:|---|
| mid_h64 | 3729.1 | 5182.5 | 1.39× | yes |
| large_h64 | 471.4 | 1352.4 | 2.87× | yes |

Raw JSON: `results/gpu_sweep_latest.json`.

## Takeaways

- The project goal is serving correctness first: batching and preemption must not change greedy output; speedups are ablated in public tables.
- **cuBLAS is the production GEMM**; tiled stays as `--naive-gemm` for teaching and ablations.
- Continuous batching still scales nearly linearly in batch size; graphs remain a measurable but secondary win after the GEMM swap.

