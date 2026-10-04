#pragma once

#include "inferno/model.hpp"
#include "inferno/paged_cache.hpp"

#include <vector>

namespace inferno {

struct SpeculativeConfig {
    int num_blocks = 256;
    int block_size = 16;
    bool use_cuda = false;
    bool cuda_graphs = false;
    bool use_cublas = true;
    bool fuse_kernels = true;
};

struct SpeculativeStats {
    std::vector<int> tokens;
    int drafted = 0;
    int accepted_draft = 0;
    int bonus = 0;
    double wall_ms = 0;
    double target_only_ms = 0;  // filled by caller or helper when compared
};

// Greedy speculative decoding (Leviathan et al.). Draft tokens are verified
// against the target; the produced sequence matches target-only greedy decoding.
SpeculativeStats speculative_generate(Model& target, Model& draft,
                                      const std::vector<int>& prompt,
                                      int max_new, int gamma,
                                      SpeculativeConfig cfg = {});

// Convenience: run speculative + target-only greedy and fill speedup fields.
SpeculativeStats speculative_benchmark(Model& target, Model& draft,
                                       const std::vector<int>& prompt,
                                       int max_new, int gamma,
                                       SpeculativeConfig cfg = {});

}  // namespace inferno
