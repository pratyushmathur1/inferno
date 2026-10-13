#!/bin/bash
#SBATCH -J inferno-draft-train
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 04:00:00
#SBATCH -o logs/draft_train_%j.out
#SBATCH -e logs/draft_train_%j.err

# Train a shallow speculative draft for SmolLM2 on one Anvil GPU.
#   mkdir -p logs
#   sbatch scripts/draft_train_job.sh
#
# Override with env vars:
#   LAYERS=6 STEPS=2000 DATA=wikitext sbatch scripts/draft_train_job.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs runs models

module purge
module load modtree/gpu 2>/dev/null || true
module load gcc/11.2.0 2>/dev/null || true
module load cuda/11.4.2 2>/dev/null || true

# Prefer an explicit PYTHON, then active conda, then known envs with torch.
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON="${CONDA_PREFIX}/bin/python"
  elif [[ -x "${HOME}/.conda/envs/robotvision/bin/python" ]]; then
    PYTHON="${HOME}/.conda/envs/robotvision/bin/python"
  else
    PYTHON="python3"
  fi
fi

LAYERS="${LAYERS:-6}"
STEPS="${STEPS:-2000}"
BATCH="${BATCH:-8}"
DATA="${DATA:-wikitext}"
OUT="${OUT:-runs/draft-smollm2-l${LAYERS}}"
TEACHER="${TEACHER:-HuggingFaceTB/SmolLM2-135M-Instruct}"

echo "host=$(hostname)  cuda=$(command -v nvcc || true)"
nvidia-smi -L || true
"$PYTHON" -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available())'

"$PYTHON" -m pip install -q -r requirements.txt

"$PYTHON" scripts/train_draft.py \
  --teacher "$TEACHER" \
  --layers "$LAYERS" \
  --steps "$STEPS" \
  --batch "$BATCH" \
  --data "$DATA" \
  --dtype bfloat16 \
  --device cuda \
  --out "$OUT"

echo "train done -> $OUT"
echo "next: sbatch scripts/draft_eval_job.sh  (or run scripts/bench_speculate.py)"
