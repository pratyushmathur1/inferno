#!/usr/bin/env bash
# Reproduce the public Inferno claims on a CUDA machine (A100-class preferred).
#
# What this covers:
#   1. CPU unit tests (always)
#   2. GPU build (if nvcc is available)
#   3. Optional SmolLM2 convert + text demo + serious_bench (needs HF + GPU)
#
# Usage:
#   ./scripts/reproduce.sh              # tests + GPU build if possible
#   ./scripts/reproduce.sh --full       # also convert SmolLM2 and run benches
#   ./scripts/reproduce.sh --cpu-only   # skip CUDA entirely
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

FULL=0
CPU_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --full) FULL=1 ;;
    --cpu-only) CPU_ONLY=1 ;;
    -h|--help)
      sed -n '2,14p' "$0"
      exit 0
      ;;
    *)
      echo "unknown arg: $arg" >&2
      exit 1
      ;;
  esac
done

echo "==> CPU build + tests"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER="${CXX:-g++}"
cmake --build build -j
./build/inferno_tests

if [[ "$CPU_ONLY" -eq 1 ]]; then
  echo "done (cpu-only)"
  exit 0
fi

if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc not found; skipping GPU build. Re-run with --cpu-only to silence this."
  exit 0
fi

ARCH="${CMAKE_CUDA_ARCHITECTURES:-80}"
echo "==> GPU build (CMAKE_CUDA_ARCHITECTURES=${ARCH})"
cmake -S . -B build-gpu -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="${CXX:-g++}" \
  -DINFERNO_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="${ARCH}"
cmake --build build-gpu -j

if [[ "$FULL" -ne 1 ]]; then
  echo "GPU binary ready: ./build-gpu/inferno"
  echo "For SmolLM2 convert + latency bench, re-run with --full"
  exit 0
fi

if [[ ! -f requirements.txt ]]; then
  echo "missing requirements.txt" >&2
  exit 1
fi

python3 -m pip install -q -r requirements.txt

MODEL="models/smollm2-135m.inf1"
HF_ID="HuggingFaceTB/SmolLM2-135M-Instruct"
if [[ ! -e "$MODEL" ]]; then
  echo "==> convert ${HF_ID} → ${MODEL}"
  python3 scripts/hf_to_inf1.py "$HF_ID" -o "$MODEL"
fi

echo "==> text demo"
python3 scripts/generate_text.py \
  -m "$MODEL" \
  --hf "$HF_ID" \
  --prompt "Say hello in one short sentence." \
  --chat --max-new 32 \
  --inferno ./build-gpu/inferno | tee results/text_demo_reproduce.txt

echo "==> serious latency bench (Inferno vs PyTorch)"
python3 scripts/serious_bench.py "$MODEL" \
  --hf "$HF_ID" \
  --inferno ./build-gpu/inferno \
  --prompt "Explain continuous batching in two sentences." \
  --chat --max-new 32 \
  --out results/serious_bench_reproduce.json

echo "Wrote results/text_demo_reproduce.txt and results/serious_bench_reproduce.json"
echo "Compare against tables in RESULTS.md"
