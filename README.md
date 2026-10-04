# Inferno

Inferno is a from-scratch Llama-style inference server. An HTTP request enters a queue, a scheduler builds a continuous batch, and a paged KV cache feeds a small transformer. The serving loop and the CPU kernels are what `make test` checks.

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
   CPU kernels            GEMM, RMSNorm, RoPE, SwiGLU, online softmax
   GPU kernels            same math, used when CUDA is linked and a device is present
```

`kernels/cuda_kernels.cu` is the GPU forward: tiled GEMM, RMSNorm, RoPE, SwiGLU, and attention that loads one KV page at a time. On a CUDA build with a device, `demo`, `generate`, `serve`, and `bench` run that forward. A step whose token count matches the previous step is captured into a CUDA graph and replayed. `make test` still runs the CPU kernels.

The model is a grouped-query Llama block: RMSNorm, rotate-half RoPE, SwiGLU. Weights are float32 in an `INF1` file from `inferno init-model`. Requests use token ids. There is no text tokenizer. Temperature `0` is greedy; otherwise sampling is top-k.

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

## Results

**CPU (1 Oct 2026, AMD EPYC 7543, no CUDA):** continuous batching of 8 short sequences was 3.6× lower wall time than eight separate engines (1.80 s vs 6.50 s). Details of that login-node run are kept below for history.

**GPU (4 Oct 2026, NVIDIA A100-SXM4-40GB):** full sweep in [`RESULTS.md`](RESULTS.md) and `results/gpu_sweep_latest.json`. Headline numbers on the mid model (2×256, head_dim 64), 8 sequences × 12+12 greedy:

| Engine | tok/s |
|---|---:|
| Inferno (GPU continuous batch) | 5185 |
| vLLM 0.4.2 (same exported weights) | 2146 |
| PyTorch serial KV decode | 574 |

On a larger 4×512 model, **vLLM is ahead** (1709 vs 1352 tok/s) — Inferno's GEMMs are still naive float32. Batch scaling on mid is nearly linear from 1→16 sequences at almost constant wall time (~18 ms).

### Earlier CPU login-node numbers

Eight sequences, 12 prompt tokens and 12 new tokens each, greedy:

| Run | Time | Output |
|---|---|---|
| One engine, all eight sequences | 1.80 s | 96 tokens, 53 tokens/s |
| Eight separate engines, summed | 6.50 s | same workload |
| One sequence, full prefix recomputed for every new token | 0.203 s | 12 new tokens |

Kernel microbenchmarks on that CPU:

| Kernel | Result |
|---|---|
| GEMM 256×256×256, plain | 79.0 ms, 0.42 GFLOP/s |
| GEMM 256×256×256, tiled 32 | 99.5 ms, 0.34 GFLOP/s, max abs diff 0 |
| Attention T=128, 8 heads, dim 32, full softmax | 1.49 ms |
| Same attention, online softmax | 97.2 ms, max abs diff 1.6e-7 |

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
RESULTS.md                   latest GPU experiment tables
scripts/compare_bench.py     Inferno vs PyTorch / vLLM
scripts/run_experiments.py   batch / length / model / cross-engine sweep
scripts/compare_job.sh       GPU job that builds, installs torch, runs comparison
```

## Compare against PyTorch / vLLM

Same workload: 8 sequences × (12 prompt + 12 new), greedy. Use `head_dim=64` (or another vLLM-supported size) when including vLLM.

```bash
bash scripts/compare_job.sh
# on an allocated GPU node with the compare venv:
python3 scripts/compare_bench.py model.bin --inferno ./build-gpu/inferno
python3 scripts/run_experiments.py
```

The scripts check that PyTorch matches Inferno token-for-token on sequence 0, time Inferno continuous batching against a PyTorch serial KV-cache decode, and time vLLM after exporting the INF1 weights as a tiny HuggingFace Llama directory.
