#include "inferno/cuda_api.hpp"

namespace inferno {
namespace cuda {

bool available() { return false; }

bool graph_replay_bench(int, int, int, int, float&) { return false; }

}  // namespace cuda
}  // namespace inferno
