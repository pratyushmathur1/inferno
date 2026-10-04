# Inferno

**Goal:** build a small, from-scratch LLM *serving* stack — not just a transformer forward — and prove that continuous batching, paged KV, and GPU kernels can get faster without changing greedy answers.

Inferno is that stack: an HTTP queue, a continuous-batching scheduler with preemption, a paged KV cache, and matching CPU/GPU Llama (GQA) forwards. Every serving optimization is behind a flag, checked for token identity, and measured on Anvil A100s. The claim is **correctness-preserving continuous batching with public ablations**, not “another mini-vLLM.”

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
   CPU / GPU kernels      GEMM, RMSNorm, RoPE, SwiGLU, paged attn
```

## What it achieved

| Goal | Result |
|---|---|
| End-to-end serving loop | Queue → continuous batch → paged KV → generate; `POST /v1/completions` |
| Correctness under batching | Chunked prefill, mixed batches, and preemption match serial greedy (`make test`) |
| CPU continuous batching | **3.6×** lower wall time than eight separate engines (8×12+12) |
| GPU path that matches CPU | Same greedy tokens; CUDA graphs on repeated token counts |
| Close the naive-GEMM gap | **cuBLAS** default; tiled teaching GEMM via `--naive-gemm` |
| Measure where time goes | `profile-forward` CUDA-event breakdown (GEMM / attn / norm / …) |
| Fair external baselines | Same INF1 weights vs PyTorch and vLLM (`scripts/compare_bench.py`) |

**Headline GPU numbers** (A100-SXM4-40GB, 8 sequences × 12 prompt + 12 new, greedy; full tables in [`RESULTS.md`](RESULTS.md)):

| Engine | mid (2×256) tok/s | large (4×512) tok/s |
|---|---:|---:|
| Inferno GPU + **cuBLAS** | **34796** | **12018** |
| Inferno GPU + tiled GEMM | 5193 | 1348 |
| vLLM 0.4.2 (prior sweep) | 2146 | 1709 |
| PyTorch serial KV (prior sweep) | 574 | 308 |

cuBLAS vs tiled is **6.7× / 8.9×** on mid / large with **identical greedy tokens**. That is the measured payoff of swapping the teaching GEMM for a production library while keeping the serving story intact.

## How it works

The model is a grouped-query Llama block: RMSNorm, rotate-half RoPE, SwiGLU. Weights are float32 in an `INF1` file from `inferno init-model`. Requests use token ids (no text tokenizer). Temperature `0` is greedy; otherwise sampling is top-k.

`kernels/cuda_kernels.cu` is the GPU forward: **cuBLAS GEMM by default**, tiled GEMM via `--naive-gemm`, RMSNorm, RoPE, SwiGLU, and page-at-a-time attention. Matching token counts capture into a CUDA graph. `inferno profile-forward` prints the event breakdown. `make test` exercises the CPU kernels and the scheduler/cache properties above.

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

The online attention kernel is the FlashAttention recurrence (running max and sum) over one key at a time. A page is the allocation unit of the cache. Column- and row-parallel GEMM are single-layer checks, not a multi-GPU server.

## Results (detail)

**CPU (1 Oct 2026, AMD EPYC 7543):** continuous batching of 8 short sequences was 3.6× lower wall time than eight separate engines (1.80 s vs 6.50 s).

Eight sequences, 12 prompt + 12 new, greedy:

| Run | Time | Output |
|---|---|---|
| One engine, all eight sequences | 1.80 s | 96 tokens, 53 tokens/s |
| Eight separate engines, summed | 6.50 s | same workload |
| One sequence, full prefix recomputed each step | 0.203 s | 12 new tokens |

Kernel microbenchmarks on that CPU:

| Kernel | Result |
|---|---|
| GEMM 256×256×256, plain | 79.0 ms, 0.42 GFLOP/s |
| GEMM 256×256×256, tiled 32 | 99.5 ms, 0.34 GFLOP/s, max abs diff 0 |
| Attention T=128, 8 heads, dim 32, full softmax | 1.49 ms |
| Same attention, online softmax | 97.2 ms, max abs diff 1.6e-7 |

**GPU ablations** (graphs on/off, batch size, sequence length, cuBLAS vs tiled, PyTorch/vLLM): [`RESULTS.md`](RESULTS.md) and `results/*.json`.

## Build

```bash
make test
./build/inferno demo
./build/inferno bench
```

`make` calls CMake in Release and uses `g++` (`COMPILER=g++` overrides that). A CUDA build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=g++ -DINFERNO_CUDA=ON
cmake --build build -j
```

## Serve

```bash
./build/inferno init-model -o model.bin --seed 1
./build/inferno serve -m model.bin --port 8000
```

```bash
curl -s http://127.0.0.1:8000/v1/completions \
  -H 'content-type: application/json' \
  -d '{"token_ids":[1,5,9,12],"max_tokens":16,"temperature":0}'
```

`GET /health` and `GET /metrics` are served too. Each connection is its own thread. Between scheduler steps the worker pulls every request that has arrived. If a running sequence needs a KV page and a newer sequence holds one, the newer sequence is preempted, its pages are freed, and its prompt plus already sampled tokens are recomputed.

## Layout

```
include/inferno/             public headers
src/                         scheduler, model, HTTP, CPU kernels
kernels/cuda_kernels.cu      CUDA operators and graph replay
tests/test_inferno.cpp       correctness tests
results/                     GPU sweep JSON
RESULTS.md                   GPU experiment tables
scripts/compare_bench.py     Inferno vs PyTorch / vLLM
scripts/run_experiments.py   batch / length / model / cross-engine sweep
scripts/compare_job.sh       GPU job that builds, installs torch, runs comparison
```

## Compare and ablate

Same workload: 8 sequences × (12 prompt + 12 new), greedy. Use `head_dim=64` (or another vLLM-supported size) when including vLLM.

```bash
bash scripts/compare_job.sh
python3 scripts/compare_bench.py model.bin --inferno ./build-gpu/inferno
python3 scripts/run_experiments.py

./build-gpu/inferno compare-workload -m model.bin --seqs 8 --prompt 12 --max-new 12
./build-gpu/inferno compare-workload -m model.bin --naive-gemm   # teaching tiled GEMM
./build-gpu/inferno profile-forward -m model.bin --seqs 8 --prompt 12 --max-new 4
```

GPU flags: `--cuda` / `--cpu` / `--no-graphs` / `--naive-gemm` / `--profile-forward`.
