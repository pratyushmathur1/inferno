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
```

`kernels/cuda_kernels.cu` contains CUDA GEMM, RMSNorm, RoPE, softmax, an online attention kernel, and a CUDA-graph replay of that chain. Those operators compile with `-DINFERNO_CUDA=ON`. The server and `inferno bench` call the CPU kernels.

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

Measured on 1 October 2026 with the Release build on one AMD EPYC 7543 (32 cores). CUDA was not linked. The bench model is 2 layers, hidden size 64, 4 query heads, 2 KV heads, head size 16, vocab 128.

Eight sequences, 12 prompt tokens and 12 new tokens each, greedy:

| Run | Time | Output |
|---|---|---|
| One engine, all eight sequences | 1.80 s | 96 tokens, 53 tokens/s |
| Eight separate engines, summed | 6.50 s | same workload |
| One sequence, full prefix recomputed for every new token | 0.203 s | 12 new tokens |

Sharing one engine across the eight sequences took 3.6× less wall time than eight separate engines. At this model size, recomputing a single short sequence from scratch was faster than serving that sequence through the scheduler, so this run is a batching measurement, not a claim that the KV cache beats prefill here.

Kernel microbenchmarks, same machine:

| Kernel | Result |
|---|---|
| GEMM 256×256×256, plain | 79.0 ms, 0.42 GFLOP/s |
| GEMM 256×256×256, tiled 32 | 99.5 ms, 0.34 GFLOP/s, max abs diff 0 |
| Attention T=128, 8 heads, dim 32, full softmax | 1.49 ms |
| Same attention, online softmax | 97.2 ms, max abs diff 1.6e-7 |

The tiled GEMM matches the plain GEMM and is slower at this size. The online attention matches the full softmax numerically and is slower in this scalar CPU implementation.

PyTorch and vLLM have not been timed. `scripts/compare_pytorch.py` loads an `INF1` file and prints one greedy next-token id when PyTorch is installed. vLLM does not read this checkpoint format.

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
include/inferno/          public headers
src/                      scheduler, model, HTTP, CPU kernels
kernels/cuda_kernels.cu   CUDA operators and graph replay
tests/test_inferno.cpp    correctness tests
scripts/compare_pytorch.py
```
