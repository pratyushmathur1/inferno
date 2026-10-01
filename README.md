# Inferno

A small Llama-style inference engine: paged KV cache, continuous batching, and the kernels underneath. The serving loop is ordinary C++ and is tested without a GPU. CUDA GEMM, RMSNorm, RoPE, softmax, online attention, and CUDA graph capture live in `kernels/cuda_kernels.cu` and compile with `-DINFERNO_CUDA=ON`.

```
HTTP  POST /v1/completions
        |
   request queue          Engine::submit
        |
 continuous batching      chunked prefill, FCFS, priority preemption
        |
   prefill / decode       one packed forward per scheduler step
        |
     paged KV             block tables, page-sized attention tiles
        |
   CPU reference          GEMM, RMSNorm, RoPE, SwiGLU, online softmax
   CUDA kernels           same operators + CUDA graph replay
        |
       GPU                when this tree is built with INFERNO_CUDA
```

## Build

This login node has GCC and CMake and does not have `nvcc`. `make` forces `g++` so an MPI compiler wrapper earlier on `PATH` is not picked up.

```bash
cd inferno
make test
./build/inferno demo
./build/inferno bench
```

GPU node:

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

`GET /health` and `GET /metrics` are also there. Token ids are the vocabulary; there is no text tokenizer. Sampling is greedy when `temperature` is 0, otherwise top-k.

Concurrent clients are accepted on their own connections and pulled into the running batch between steps. A sequence that cannot get another KV page is preempted if a newer sequence is holding pages; its tokens stay, and its cache is rebuilt. Greedy output matches running that request alone.

## What is implemented

| Piece | Where | Checked by |
|---|---|---|
| GEMM, tiled GEMM, RMSNorm, RoPE, softmax, SiLU | `src/cpu_kernels.cpp` | unit tests |
| Online-softmax attention (FlashAttention-style) | same, and paged walk in `src/paged_cache.cpp` | matches a full score matrix; paged path matches the dense walk bit for bit |
| Paged KV cache | `PagedCache` | block table can be non-contiguous |
| Continuous batching + chunked prefill | `Scheduler` | mixed-length batch matches serial greedy; chunk size 3 matches a big batch |
| Preemption | oldest request wins | tight page pool, outputs still match serial |
| Speculative decoding | `speculative_generate` | greedy tokens match the target model with no draft |
| INT8 W8A8 GEMM and E4M3 FP8 GEMM | `kernels` | hand values and round-trip of 1, 0.5, 2 |
| Tensor parallel GEMM | column split and row split + sum | exact vs the unsplit GEMM |
| HTTP | `HttpServer` | `/health` and `/v1/completions` |
| CUDA kernels + CUDA graph | `kernels/cuda_kernels.cu` | built only with `INFERNO_CUDA`; `inferno bench` reports graph replay when a device is present |

The model is a GQA Llama block: RMSNorm, rotate-half RoPE, SwiGLU. Weights are float32 in an `INF1` file (`inferno init-model`).

## Bench against PyTorch and vLLM

`inferno bench` reports three numbers on one machine:

- naive GEMM vs a tiled GEMM
- full-softmax attention vs the online algorithm
- continuous batching vs one engine per request, and vs a baseline that replays the whole prefix for every new token

That last gap is the KV cache. On a GPU node, put PyTorch next to it with the same checkpoint:

```bash
python3 scripts/compare_pytorch.py model.bin --tokens 1,2,3,4
```

The script needs PyTorch. It prints the greedy next-token id for one prefill using the same weight order and the same rotate-half RoPE. vLLM will not load `INF1`; the comparison that matches how vLLM is measured is throughput at a fixed concurrency (`vllm bench serve` against `inferno bench`) once a real weight loader is in front of this scheduler.

## Layout

```
include/inferno/     public headers
src/                 scheduler, model, HTTP, CPU kernels
kernels/cuda_kernels.cu
tests/test_inferno.cpp
scripts/compare_pytorch.py
```

`make test` is the check that should pass before changing the scheduler or the cache.
