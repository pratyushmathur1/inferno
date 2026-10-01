#pragma once

#include "inferno/model.hpp"
#include "inferno/paged_cache.hpp"

#include <vector>

namespace inferno {

struct SpeculativeStats {
    std::vector<int> tokens;
    int drafted = 0;
    int accepted_draft = 0;
    int bonus = 0;
};

// Greedy speculative decoding (Leviathan et al.). Draft tokens are verified
// against the target; the produced sequence matches target-only greedy decoding.
SpeculativeStats speculative_generate(Model& target, Model& draft,
                                      const std::vector<int>& prompt,
                                      int max_new, int gamma,
                                      int num_blocks, int block_size);

}  // namespace inferno
