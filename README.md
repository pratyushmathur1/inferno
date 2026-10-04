# Inferno

**Goal:** build a small, from-scratch LLM *serving* stack — not just a transformer forward — and prove that continuous batching, paged KV, and GPU kernels can get faster without changing greedy answers.

Inferno is that stack: an HTTP queue, a continuous-batching scheduler with preemption, a paged KV cache, and matching CPU/GPU Llama (GQA) forwards. It runs **real HuggingFace Llama-arch weights** (SmolLM2-135M) with a tokenizer-backed text demo, plus speculative decoding and public ablations. The claim is **correctness-preserving continuous batching with measured optimizations**, not “another mini-vLLM.”

```
HTTP  POST /v1/completions
        |
   request queue          Engine::submit
        |
 continuous batching      chunked prefill, first-come preemption
        |
   prefill / decode       one packed forward per scheduler step
        |
     paged KV             block table per sequence
        |
   CPU / GPU kernels      cuBLAS GEMM, fused add+RMSNorm, paged attn
```

## What it achieved

| Goal | Result |
|---|---|
| End-to-end serving loop | Queue → continuous batch → paged KV → generate; `POST /v1/completions` |
| Real model + tokenizer | SmolLM2-135M Instruct → INF1; greedy tokens match HuggingFace FP32 |
| Text demo | `python3 scripts/generate_text.py … --prompt "…"` |
| Correctness under batching | Chunked prefill, mixed batches, preemption match serial greedy |
| Speculative decoding | Leviathan greedy verify; acceptance % + speedup vs target-only; output matches |
| Close the naive-GEMM gap | **cuBLAS** default; tiled teaching GEMM via `--naive-gemm` |
| Fused CUDA kernels | `rmsnorm_save_residual` + `add_rmsnorm`; `--no-fuse` ablation |
| Serious latency metrics | `bench-latency`: TTFT, ITL, tok/s, GPU memory vs PyTorch |
| Fair external baselines | Same weights vs PyTorch / vLLM |

### Real model (SmolLM2-135M, A100)

| Engine | tok/s | TTFT ms | ITL ms | GPU mem MB |
|---|---:|---:|---:|---:|
| Inferno FP32 + cuBLAS | **278** | **7.9** | **3.4** | 1744 |
| PyTorch FP16 generate | 36 | 30.5 | 27.9 | 336 |

Prompt: chat-templated “Explain continuous batching in two sentences.”, 32 new tokens max (27 emitted). Inferno is FP32; PyTorch baseline is FP16 — memory is not apples-to-apples, latency is.

### Toy-model throughput (8×12+12 greedy)

| Engine | mid (2×256) tok/s | large (4×512) tok/s |
|---|---:|---:|
| Inferno GPU + **cuBLAS** | **34796** | **12018** |
| Inferno GPU + tiled GEMM | 5193 | 1348 |
| vLLM 0.4.2 (prior sweep) | 2146 | 1709 |
| PyTorch serial KV (prior sweep) | 574 | 308 |

Full tables: [`RESULTS.md`](RESULTS.md).

## Real Llama weights + text demo

```bash
# convert a HF Llama/SmolLM checkpoint to INF1 (float32)
python3 scripts/hf_to_inf1.py HuggingFaceTB/SmolLM2-135M-Instruct -o models/smollm2-135m.inf1

# optional shallow draft for speculative decoding
python3 scripts/make_draft.py models/smollm2-135m.inf1 -o models/draft.inf1 --layers 15

# text generate (HF tokenizer → Inferno → detokenize)
python3 scripts/generate_text.py \
  -m models/smollm2-135m.inf1 \
  --hf HuggingFaceTB/SmolLM2-135M-Instruct \
  --prompt "Say hello in one short sentence." --chat --max-new 64 \
  --inferno ./build-gpu/inferno
```

## Speculative decoding

```bash
./build-gpu/inferno speculate -m target.inf1 --draft draft.inf1 \
  --tokens 1,2,3 --max-new 32 --gamma 4 --cuda
```

Prints `accept_pct`, `spec_ms`, `target_only_ms`, `speedup`. Self-draft (`--draft` = target) reaches ~100% accept and must match target-only greedy tokens. A truncated-layer draft is a cheap baseline (low accept until you train a real drafter).

## Fused kernels

Profile showed copy/add launch overhead next to RMSNorm. Inferno fuses:

1. **`rmsnorm_save_residual`** — residual copy + RMSNorm  
2. **`add_rmsnorm`** — residual add + RMSNorm (and refresh residual for the FFN)

Default on; disable with `--no-fuse`. CUDA-event tables: `results/profile_mid_fused.txt` vs `profile_mid_unfused.txt` (misc ~10% → ~5% on mid).

## Serious benchmarking

```bash
python3 scripts/serious_bench.py models/smollm2-135m.inf1 \
  --hf HuggingFaceTB/SmolLM2-135M-Instruct \
  --inferno ./build-gpu/inferno \
  --prompt "Explain continuous batching in two sentences." --chat \
  --max-new 32 --out results/serious_bench.json
```

Uses in-process `inferno bench-latency` (TTFT / ITL / tok/s / GPU mem) so weight load is not counted in the timed window.

## How it works

The model is a grouped-query Llama block: RMSNorm, rotate-half RoPE, SwiGLU. Weights are float32 in an `INF1` file (`init-model` or `hf_to_inf1.py`). Temperature `0` is greedy; otherwise sampling is top-k.

`kernels/cuda_kernels.cu`: **cuBLAS GEMM**, optional tiled GEMM, fused add/RMSNorm, RoPE, SwiGLU, page-at-a-time attention, CUDA graphs. `profile-forward` prints the event breakdown.

## Correctness

`make test` covers:

| Behavior | Check |
|---|---|
| GEMM, RMSNorm, RoPE, softmax | closed-form values |
| Online attention | max abs error under `1e-4` versus a full score matrix |
| Paged attention | bitwise match with the dense online walk, including a non-contiguous block table |
| Chunked prefill and mixed batches | same greedy tokens as one sequence at a time |
| Preemption | a tight page pool still matches serial greedy output |
| Speculative decoding | draft proposals, greedy output matches the target model alone |
| INT8 and FP8 E4M3 GEMM | a hand-computed dot product, and exact round-trip of 1, 0.5, and 2 |
| Tensor-parallel linear | column split and row split plus an allreduce match the unsplit GEMM |
| HTTP | `GET /health` and `POST /v1/completions` |

## Build

```bash
make test
./build/inferno demo

cmake -S . -B build-gpu -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=g++ -DINFERNO_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build-gpu -j
```

GPU flags: `--cuda` / `--cpu` / `--no-graphs` / `--naive-gemm` / `--no-fuse` / `--profile-forward`.

## Serve

```bash
./build-gpu/inferno serve -m models/smollm2-135m.inf1 --port 8000
curl -s http://127.0.0.1:8000/v1/completions \
  -H 'content-type: application/json' \
  -d '{"token_ids":[1,5,9,12],"max_tokens":16,"temperature":0}'
```

## Layout

```
include/inferno/             public headers
src/                         scheduler, model, HTTP, CPU kernels, speculative
kernels/cuda_kernels.cu      CUDA operators, fusion, graphs, cuBLAS
scripts/hf_to_inf1.py        HF Llama → INF1
scripts/generate_text.py     tokenizer-backed text demo
scripts/make_draft.py        shallow draft for speculative decoding
scripts/serious_bench.py     TTFT / ITL / throughput / memory
scripts/compare_bench.py     Inferno vs PyTorch / vLLM (toy INF1)
RESULTS.md                   experiment tables
results/                     JSON + profile dumps
```
