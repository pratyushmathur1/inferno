# Inferno GPU experiments

## Goal

Build a small, from-scratch LLM **serving** stack and prove continuous batching + paged KV + GPU kernels can get faster **without changing greedy answers**. Every optimization sits behind a flag, keeps token identity with the serial path, and ships with a measured table.

**Fairness note:** Inferno numbers below are FP32 unless stated. PyTorch / vLLM baselines are often FP16 — treat memory and absolute tok/s as directional until Inferno has an FP16 path. Greedy token checks are always vs HuggingFace FP32.

## What these numbers show

- **Real weights:** SmolLM2-135M Instruct in INF1 matches HuggingFace FP32 greedy tokens; text demo via HF tokenizer.
- **cuBLAS** is the production GEMM; tiled stays as `--naive-gemm`. **6.7× / 8.9×** tok/s on mid / large toys.
- **Fused add+RMSNorm** cuts profiled `misc` (~10% → ~5% on mid); batch tok/s up slightly on large (12638 vs 12204).
- **Speculative decoding** matches target-only greedy; self-draft ~100% accept; shallow drafts need training for speedups.
- On mid toys, GEMM drops from ~78% → ~29% of profiled decode time after cuBLAS.

---

## Real model: SmolLM2-135M (4 Oct 2026, A100, g001)

Source: `HuggingFaceTB/SmolLM2-135M-Instruct` → `scripts/hf_to_inf1.py` (FP32 INF1, 651 MB).

Greedy match vs HuggingFace FP32 on a chat prompt (8 new): **identical** token ids (`Hello!<|im_end|>`).

### Latency vs PyTorch / vLLM (`results/serious_bench_vllm.json`)

Chat prompt “Explain continuous batching in two sentences.”, max_new=64. Inferno/PyTorch stop at EOS (~27 new); vLLM ran the full 64 — tok/s is not perfectly matched on length.

| Engine | tok/s | TTFT ms | ITL ms | GPU mem MB | new |
|---|---:|---:|---:|---:|---:|
| Inferno FP32 cuBLAS | **279** | **8.5** | 3.4 | 1744 | 27 |
| PyTorch FP16 | 35 | 30.0 | 28.5 | 336 | 27 |
| vLLM 0.4.2 FP16 | **378** | 14.6 | **2.5** | 19543 | 64 |

### SmolLM2 batch scaling (random prompts, 32 new)

| seqs | ms | tok/s |
|---:|---:|---:|
| 1 | 117 | 274 |
| 2 | 133 | 480 |
| 4 | 132 | 972 |
| 8 | 198 | 1292 |

### Fuse on real model (CUDA-event, 1 token decode)

| path | total ms | misc % |
|---|---:|---:|
| fused | **6.74** | **4.8** |
| `--no-fuse` | 8.80 | 9.9 |

Files: `results/profile_smollm_*.txt`, `results/latency_smollm_*.txt`.


### Speculative decoding

| Draft | gamma | accept % | speedup vs target-only | notes |
|---|---:|---:|---:|---|
| self (same INF1) | 4 | ~94–100 | <1× | validates algorithm; draft cost = target cost |
| first 15 layers | 4 | ~4 | <1× | cheap baseline; needs a trained drafter for wins |

Outputs match target-only greedy (enforced in `speculate`). See `results/speculate_*.txt`.

### Fused kernels (CUDA-event profile, mid 2×256)

| path | total ms | misc % | norm % |
|---|---:|---:|---:|
| fused | 0.54 | **5.1** | 8.0 |
| `--no-fuse` | 0.59 | **10.1** | 7.3 |

Files: `results/profile_mid_fused.txt`, `results/profile_mid_unfused.txt`.

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

### CUDA-event profile (last decode step, 8 tokens in-flight)

| model / GEMM | total ms | gemm % | attn % | norm % | misc % |
|---|---:|---:|---:|---:|---:|
| mid / cublas | 0.58 | 29% | 5% | 7% | 10% |
| mid / tiled | 1.87 | **78%** | 2% | 2% | 3% |
| large / cublas | 1.50 | 46% | 5% | 6% | 8% |

---

## Earlier tiled-GEMM sweep (20261004T161641Z)

Kept for history. Continuous batching scaled nearly linearly 1→16 seqs; graphs ~3% on tiled mid; Inferno tiled beat PyTorch and mid-vLLM but trailed vLLM on large_h64 before cuBLAS.

Raw JSON: `results/gpu_sweep_latest.json`.

## Takeaways

- Serving correctness first: batching, preemption, speculative verify, and CPU/GPU paths must not change greedy output.
- cuBLAS + fusion are the production GPU path; tiled / `--no-fuse` remain as teaching ablations.
- Real HF weights + tokenizer make Inferno an end-to-end engine; speculative decoding is wired with public acceptance metrics.
