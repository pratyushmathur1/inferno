# Architecture

Inferno is a single-process LLM **serving** stack. The host owns scheduling and
page allocation; the device owns the forward. Greedy token identity is preserved
across chunked prefill, mixed batches, and preemption.

## Dataflow

```
Client
  │  POST /v1/completions  (token_ids, max_tokens, temperature)
  ▼
HttpServer                 thin JSON ↔ Request bridge
  │  Engine::submit
  ▼
Engine worker              continuous batching loop
  │  Scheduler::prepare / commit
  ▼
Scheduler                  chunked prefill, FCFS priority, preemption
  │  allocates / frees pages
  ▼
PagedCache                 block table per sequence (host)
  │  tables + token ids copied each step
  ▼
CPU kernels  or  CUDA kernels
  RMSNorm · RoPE · SwiGLU · GEMM · paged attention
  (GPU: cuBLAS GEMM, fused add+RMSNorm, optional CUDA graphs)
  │
  ▼
logits → sample / greedy → Scheduler::commit → GenerationResult
```

## Scheduler

`Scheduler` keeps a list of sequences with arrival order, prompt length, pages,
and running/waiting/finished flags.

Each step:

1. **Admit / resume** waiting sequences if pages and batch token budget allow.
2. **Prefill chunk** up to `max_batched_tokens` across sequences (chunked prefill).
3. **Decode** one token per running sequence when prompt is fully computed.
4. **Preempt** a lower-priority (later arrival) sequence if a higher-priority one
   needs pages. The victim’s pages are freed; it is recomputed from already
   sampled tokens so greedy output matches the serial path.

This is why tests assert: batched = serial, and a tight page pool with
preemptions still matches per-sequence greedy tokens.

## Paged KV

KV is stored in fixed-size blocks (`block_size` tokens). Each sequence has a
**block table** mapping logical page index → physical block id.

- Host `PagedCache` allocates from a free list and records tables.
- GPU attention reads K/V through the same tables (page-at-a-time kernel).
- Non-contiguous tables are correctness-tested against a dense online attention walk.

## Prefix caching (shared system prompts)

Identical leading tokens (chat template / system prompt) share the same physical
KV pages. Sharing is **block-aligned**: length `L` is rounded down to a multiple
of `block_size`.

- `PrefixCache` hashes the shared span (FNV-1a). A miss allocates pages and marks
  one sequence as **owner**; concurrent followers wait until the owner’s
  `computed ≥ L`, then attach the same block ids with `computed = L`.
- Hits skip the shared prefill entirely (warm path across `generate()` when
  `retain_prefix_cache` is on). Private suffix pages stay per-sequence.
- Preemption frees only private suffix pages; shared pages stay until the last
  reference drops (or are retained warm).
- Flag: `--prefix-cache`. Bench: `bench-prefix` reports TTFT before/after for
  N concurrent requests with the same system prompt.

## Model forward

Grouped-query Llama block (float32 INF1 weights):

1. RMSNorm (optionally fused with residual copy/add on GPU)
2. QKV projections + rotate-half RoPE
3. Paged attention over the sequence’s block table
4. Output projection + residual
5. RMSNorm + SwiGLU FFN + residual
6. Final norm + LM head

GPU path defaults to **cuBLAS** `Sgemm`. `--naive-gemm` keeps a tiled teaching
kernel. Decode steps with a stable token count can be captured as a **CUDA graph**.

## Speculative decoding

`speculate` runs a draft model for `gamma` tokens, then verifies with the target
under Leviathan-style greedy accept/reject. Accepted prefixes must match
target-only greedy generation (enforced in tests and the CLI metrics).

## Process layout

| Piece | Role |
|---|---|
| `src/server.cpp` | HTTP `/health`, `/v1/completions` |
| `src/engine.cpp` | Queue + worker thread + device binding |
| `src/scheduler.cpp` | Continuous batching + preemption |
| `src/paged_cache.cpp` | Block pool + tables |
| `src/model.cpp` | INF1 load / CPU forward orchestration |
| `kernels/cuda_kernels.cu` | Device forward, fusion, graphs, cuBLAS |
| `src/speculative.cpp` | Draft / verify loop |

## Scope

Single-GPU (or CPU). No multi-node, no production tokenizer inside the binary
(HF tokenizer is used from Python demos). Quantization and tensor-parallel
helpers exist as CPU kernel unit tests; they are not the default serving path.
