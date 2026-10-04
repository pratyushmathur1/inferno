#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace inferno {

struct ModelConfig {
    int vocab = 128;
    int hidden = 64;
    int n_layers = 2;
    int n_heads = 4;
    int n_kv_heads = 2;
    int head_dim = 16;
    int intermediate = 128;
    float rms_eps = 1e-5f;
    float rope_theta = 10000.f;
    int eos_id = -1;

    int q_dim() const { return n_heads * head_dim; }
    int kv_dim() const { return n_kv_heads * head_dim; }
    int group() const { return n_heads / n_kv_heads; }
};

struct EngineConfig {
    int block_size = 16;
    int num_blocks = 128;
    int max_batched_tokens = 256;
    int max_seqs = 32;
    bool use_cuda = false;
    bool cuda_graphs = false;
    bool use_cublas = true;   // GPU only; false keeps the tiled teaching GEMM
    bool profile_cuda = false;
};

struct Request {
    int id = -1;
    std::vector<int> prompt;
    int max_new_tokens = 16;
    float temperature = 0.f;
    int top_k = 40;
    std::uint64_t seed = 1;
};

struct GenerationResult {
    int id = -1;
    std::vector<int> tokens;
    int generated = 0;
    bool ok = true;
    std::string error;
};

struct TokenIn {
    int seq_id = 0;
    int position = 0;
    int token_id = 0;
    bool emit = false;
};

struct Metrics {
    std::int64_t tokens_processed = 0;
    std::int64_t preemptions = 0;
    std::int64_t requests = 0;
    std::int64_t steps = 0;
};

}  // namespace inferno
