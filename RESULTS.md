# Inferno GPU experiments (20261004T161641Z)

- Host: `g002`
- GPU: `GPU 0: NVIDIA A100-SXM4-40GB (UUID: GPU-28328757-6d71-0098-6f01-090385f1cb36)`
- Binary: `/home/x-pmathur1/inferno/build-gpu/inferno`

## Batch scaling (mid_h64, 12+12)

| seqs | ms | tok/s | graph_replays |
|---:|---:|---:|---:|
| 1 | 18.31 | 655.3 | 20 |
| 2 | 18.37 | 1306.6 | 20 |
| 4 | 18.44 | 2603.3 | 20 |
| 8 | 18.52 | 5183.5 | 20 |
| 16 | 18.69 | 10274.3 | 20 |
| 32 | 21.09 | 18208.7 | 18 |

## CUDA graphs (mid_h64, 8×12+12)

| graphs | ms | tok/s | replays |
|---|---:|---:|---:|
| True | 18.52 | 5184.1 | 20 |
| False | 19.16 | 5010.4 | 0 |

## Sequence length (mid_h64, 8 seqs)

| prompt | new | ms | tok/s |
|---:|---:|---:|---:|
| 8 | 8 | 12.32 | 5195.4 |
| 12 | 12 | 16.93 | 5669.4 |
| 32 | 32 | 45.00 | 5688.8 |
| 64 | 64 | 95.84 | 5342.3 |

## Model size (8×12+12)

| model | ms | tok/s |
|---|---:|---:|
| tiny_h16 | 4.60 | 20867.6 |
| mid_h64 | 14.47 | 6634.0 |
| large_h64 | 55.72 | 1722.9 |

## Cross-engine

| model | Inferno tok/s | PyTorch tok/s | vLLM tok/s | match |
|---|---:|---:|---:|---|
| mid_h64 | 5184.9 | 573.9 | 2146.4 | True |
| large_h64 | 1352.2 | 307.5 | 1708.9 | True |

## CPU vs GPU (same Inferno binary, 8×12+12)

| model | CPU tok/s | GPU tok/s | speedup | tokens match |
|---|---:|---:|---:|---|
| mid_h64 | 3729.1 | 5182.5 | 1.39× | yes |
| large_h64 | 471.4 | 1352.4 | 2.87× | yes |

## Takeaways

- Continuous batching scales nearly linearly from 1→16 sequences on mid_h64 (wall time stays ~18 ms while tok/s climbs).
- CUDA graphs help a little on this toy size (~3%): 5184 vs 5010 tok/s.
- Throughput stays flat as sequence length grows from 8 to 64, which is what you want from a KV cache.
- Against a serial PyTorch KV decode, Inferno is ~9× on mid_h64 and ~4× on large_h64.
- Against vLLM 0.4.2 on the same exported weights: Inferno wins on mid_h64 (~2.4×); **vLLM wins on large_h64** (~1.3×). That is an honest gap — Inferno's kernels are still naive float32 GEMMs.
- GPU vs Inferno-CPU: tokens match; GPU helps more as the model grows (1.4× → 2.9×).

Notes: PyTorch baseline is serial per-sequence KV decode. vLLM needs `head_dim` in {64,80,96,112,128,256}. These are random tiny models in float32, not a production LLM bake-off.
Raw JSON: `results/gpu_sweep_latest.json`.

