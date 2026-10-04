#pragma once

#include "inferno/model.hpp"

#include <cstdint>
#include <vector>

namespace inferno {
namespace cuda {

bool available();

// Microbench of a captured GEMM + RMSNorm. False when CUDA is not built.
bool graph_replay_bench(int M, int N, int K, int iters, float& ms_out);

// Full model forward on the GPU. `slot` is created on first use and freed
// with release_device. Logits are packed in emit order, matching Model::forward.
void llama_forward(const ModelConfig& cfg, const float* weights, std::int64_t nweights,
                   const LayerOff* layers, const std::vector<TokenIn>& tokens, PagedCache& cache,
                   std::vector<float>& logits, bool graphs, void*& slot);

void release_device(void* slot);
int graph_replays(void* slot);

}  // namespace cuda
}  // namespace inferno
