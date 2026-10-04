#!/bin/bash
# Build Inferno with CUDA, then time it against PyTorch (and vLLM if installable).
set -euo pipefail

SCRATCH="${SCRATCH:-/anvil/scratch/$USER}"
ENV="$SCRATCH/inferno-cmp-env311"
MODEL="$SCRATCH/inferno-cmp-model.bin"
PY="$ENV/bin/python"
PIP="$ENV/bin/pip"
ROOT=/home/x-pmathur1/inferno
cd "$ROOT"

echo "node $(hostname)"

# --- Python 3.11 venv (system python is 3.6; torch 2.x needs >=3.8) ---
module purge
module load modtree/gpu
module load gcc/13.3.0
module load python/3.11.9
echo "python $(python3 --version) at $(which python3)"

if [[ ! -x "$PY" ]]; then
  echo "creating venv at $ENV"
  python3 -m venv "$ENV"
  "$PIP" install --upgrade pip
  "$PIP" install --extra-index-url https://download.pytorch.org/whl/cu118 \
    'torch==2.1.2+cu118' 'numpy<2'
  "$PIP" install 'vllm==0.4.2' || echo "vllm pip install failed; continuing without it"
fi

# --- CUDA build uses GCC 11 (nvcc 11.4 host compiler) ---
module purge
module load modtree/gpu
module load gcc/11.2.0
module load cuda/11.4.2
nvidia-smi -L
nvcc --version | tail -n 1

"$PY" - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
try:
    import vllm
    print("vllm", vllm.__version__)
except Exception as ex:
    print("vllm unavailable:", ex)
PY

# Keep transformers compatible with the installed vLLM/torch pair.
"$PIP" install 'transformers==4.40.2' 'huggingface-hub<1' || true

cmake -S . -B build-gpu -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="$(which g++)" -DCMAKE_CUDA_COMPILER="$(which nvcc)" \
  -DINFERNO_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build-gpu -j

./build-gpu/inferno_tests
./build-gpu/inferno init-model -o "$MODEL" --seed 1
./build-gpu/inferno compare-workload -m "$MODEL" --cuda

echo "==== PyTorch / Inferno / vLLM comparison ===="
"$PY" scripts/compare_bench.py "$MODEL" --inferno ./build-gpu/inferno
