#pragma once

#include "inferno/paged_cache.hpp"
#include "inferno/types.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace inferno {

struct Span {
    std::int64_t off = 0;
    std::int64_t n = 0;
};

struct LayerOff {
    Span rms1, wq, wk, wv, wo, rms2, wgate, wup, wdown;
};

class Model {
public:
    Model() = default;
    explicit Model(ModelConfig cfg);

    static Model init_random(ModelConfig cfg, std::uint32_t seed);
    static Model load(const std::string& path);
    void save(const std::string& path) const;

    const ModelConfig& config() const { return cfg_; }
    int vocab() const { return cfg_.vocab; }
    const std::vector<float>& weights() const { return weights_; }

    // Pack tokens from any number of sequences. Logits are concatenated in
    // emit order, shape [num_emits, vocab].
    void forward(const std::vector<TokenIn>& tokens, PagedCache& cache,
                 std::vector<float>& logits);

private:
    const float* ptr(Span s) const;
    ModelConfig cfg_{};
    std::vector<float> weights_;
    Span tok_emb_{};
    std::vector<LayerOff> layers_;
    Span final_rms_{};
    Span lm_head_{};

    std::vector<float> x_, xn_, q_, k_, v_, attn_, proj_, gate_, up_, mid_, gathered_;
};

std::int64_t weight_count(const ModelConfig& cfg);

}  // namespace inferno
