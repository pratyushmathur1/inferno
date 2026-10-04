#!/bin/bash
set -euo pipefail
module purge
module load modtree/gpu
module load gcc/11.2.0
module load cuda/11.4.2
cd /home/x-pmathur1/inferno
echo "node $(hostname)"
nvidia-smi -L
nvcc --version | tail -n 3
rm -rf build-gpu
cmake -S . -B build-gpu -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=g++ -DCMAKE_CUDA_COMPILER="$(which nvcc)" \
  -DINFERNO_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build-gpu -j
./build-gpu/inferno_tests
./build-gpu/inferno init-model -o /tmp/inferno-gpu-model.bin --seed 1
echo "CPU:"
./build-gpu/inferno generate -m /tmp/inferno-gpu-model.bin --tokens 1,5,9,12,4 --max-new 16 --cpu
echo "GPU:"
./build-gpu/inferno generate -m /tmp/inferno-gpu-model.bin --tokens 1,5,9,12,4 --max-new 16 --cuda
./build-gpu/inferno bench
