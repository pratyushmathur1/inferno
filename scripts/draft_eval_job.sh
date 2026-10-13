#!/bin/bash
#SBATCH -J inferno-draft-eval
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 01:00:00
#SBATCH -o logs/draft_eval_%j.out
#SBATCH -e logs/draft_eval_%j.err

# Convert trained draft HF -> INF1 and bench Inferno speculate on GPU.
#   DRAFT_HF=runs/draft-smollm2-l6 sbatch scripts/draft_eval_job.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs models results

module purge
module load modtree/gpu 2>/dev/null || true
module load gcc/11.2.0 2>/dev/null || true
module load cuda/11.4.2 2>/dev/null || true

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON="${CONDA_PREFIX}/bin/python"
  elif [[ -x "${HOME}/.conda/envs/robotvision/bin/python" ]]; then
    PYTHON="${HOME}/.conda/envs/robotvision/bin/python"
  else
    PYTHON="python3"
  fi
fi

TEACHER="${TEACHER:-HuggingFaceTB/SmolLM2-135M-Instruct}"
TARGET="${TARGET:-models/smollm2-135m.inf1}"
DRAFT_HF="${DRAFT_HF:-runs/draft-smollm2-l6}"
DRAFT_INF1="${DRAFT_INF1:-models/draft-smollm2-l6.inf1}"
OUT="${OUT:-results/speculate_trained_draft.json}"
GAMMA="${GAMMA:-4}"
ARCH="${CMAKE_CUDA_ARCHITECTURES:-80}"

echo "host=$(hostname)"
nvidia-smi -L || true

"$PYTHON" -m pip install -q -r requirements.txt

if [[ ! -e "$TARGET" ]]; then
  echo "converting teacher -> $TARGET"
  "$PYTHON" scripts/hf_to_inf1.py "$TEACHER" -o "$TARGET"
fi

if [[ ! -x build-gpu/inferno ]]; then
  cmake -S . -B build-gpu -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_CUDA_COMPILER="$(which nvcc)" \
    -DINFERNO_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="${ARCH}"
  cmake --build build-gpu -j
fi

# Optional truncated-layer baseline for comparison.
BASELINE="${BASELINE:-models/draft-truncated15.inf1}"
if [[ ! -e "$BASELINE" ]]; then
  "$PYTHON" scripts/make_draft.py "$TARGET" -o "$BASELINE" --layers 15 || true
fi

EXTRA=()
if [[ -e "$BASELINE" ]]; then
  EXTRA+=(--baseline-truncated "$BASELINE")
fi

"$PYTHON" scripts/bench_speculate.py \
  --target "$TARGET" \
  --draft-hf "$DRAFT_HF" \
  --draft-out "$DRAFT_INF1" \
  --hf "$TEACHER" \
  --inferno ./build-gpu/inferno \
  --prompts-file scripts/speculate_prompts.txt \
  --chat \
  --gamma "$GAMMA" \
  --max-new 64 \
  --out "$OUT" \
  "${EXTRA[@]}"

echo "wrote $OUT"
