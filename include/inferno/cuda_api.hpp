#pragma once

// CPU builds link the stub. -DINFERNO_USE_CUDA links kernels/cuda_kernels.cu instead.
namespace inferno {
namespace cuda {

bool available();

// Resident decode microbench: upload A, B once, capture GEMM + RMSNorm into a
// CUDA graph, replay it `iters` times. Returns false when CUDA is not built.
bool graph_replay_bench(int M, int N, int K, int iters, float& ms_out);

}  // namespace cuda
}  // namespace inferno
