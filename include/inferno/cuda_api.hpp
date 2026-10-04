#pragma once

#include "inferno/model.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace inferno {
namespace cuda {

struct ForwardOptions {
    bool graphs = true;
    bool use_cublas = true;   // false → tiled teaching GEMM
    bool profile = false;     // CUDA-event breakdown (disables graph replay)
};

struct Profile {
    float embed_ms = 0;
    float gemm_ms = 0;
    float norm_ms = 0;
    float rope_ms = 0;
    float kv_write_ms = 0;
    float attn_ms = 0;
    float misc_ms = 0;
    float total_ms = 0;
    int tokens = 0;
    int layers = 0;
    std::string gemm_backend;  // "cublas" or "tiled"
};

bool available();

bool graph_replay_bench(int M, int N, int K, int iters, float& ms_out);

void llama_forward(const ModelConfig& cfg, const float* weights, std::int64_t nweights,
                   const LayerOff* layers, const std::vector<TokenIn>& tokens, PagedCache& cache,
                   std::vector<float>& logits, ForwardOptions opts, void*& slot);

void release_device(void* slot);
int graph_replays(void* slot);
Profile last_profile(void* slot);

}  // namespace cuda
}  // namespace inferno
