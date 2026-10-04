#include "inferno/cuda_api.hpp"

#include <stdexcept>

namespace inferno {
namespace cuda {

bool available() { return false; }

bool graph_replay_bench(int, int, int, int, float&) { return false; }

void llama_forward(const ModelConfig&, const float*, std::int64_t, const LayerOff*,
                   const std::vector<TokenIn>&, PagedCache&, std::vector<float>&, ForwardOptions,
                   void*&) {
    throw std::runtime_error(
        "CUDA was requested, but this binary has no GPU runtime. Rebuild with -DINFERNO_CUDA=ON on a machine with nvcc.");
}

void release_device(void*) {}

int graph_replays(void*) { return 0; }

Profile last_profile(void*) { return {}; }

}  // namespace cuda
}  // namespace inferno
