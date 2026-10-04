#include "inferno/speculative.hpp"

#include "inferno/cuda_api.hpp"
#include "inferno/engine.hpp"

#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace inferno {
namespace {

int argmax(const float* logits, int n) {
    int best = 0;
    float m = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < n; ++i) {
        if (logits[i] > m) {
            m = logits[i];
            best = i;
        }
    }
    return best;
}

struct Stream {
    Model* model = nullptr;
    PagedCache* cache = nullptr;
    void** device = nullptr;
    SpeculativeConfig cfg{};
    std::vector<int> tokens;
    std::vector<int> blocks;
    std::vector<float> pending;
    int computed = 0;

    void ensure(int pos) {
        const int need = pos / cache->block_size() + 1;
        while (static_cast<int>(blocks.size()) < need) {
            int b = cache->alloc_block();
            if (b < 0) throw std::runtime_error("speculative decode ran out of KV pages");
            blocks.push_back(b);
        }
        cache->set_block_table(0, blocks);
    }

    void forward(int n, bool all_logits, std::vector<float>& logits) {
        if (n <= 0) {
            logits.clear();
            return;
        }
        ensure(computed + n - 1);
        std::vector<TokenIn> batch;
        batch.reserve(n);
        for (int i = 0; i < n; ++i) {
            TokenIn tok;
            tok.seq_id = 0;
            tok.position = computed + i;
            tok.token_id = tokens[computed + i];
            tok.emit = all_logits || i == n - 1;
            batch.push_back(tok);
        }
        model->forward(batch, *cache, logits, cfg.use_cuda, cfg.cuda_graphs, device, cfg.use_cublas,
                       /*profile=*/false, cfg.fuse_kernels);
        computed += n;
    }
};

SpeculativeStats run_spec(Model& target_model, Model& draft_model, const std::vector<int>& prompt,
                          int max_new, int gamma, SpeculativeConfig cfg) {
    if (prompt.empty()) throw std::runtime_error("empty prompt");
    if (max_new < 0 || gamma < 1) throw std::runtime_error("bad speculative config");
    if (target_model.vocab() != draft_model.vocab()) {
        throw std::runtime_error("draft and target vocabularies differ");
    }
    const int vocab = target_model.vocab();
    const auto tc = target_model.config();
    const auto dc = draft_model.config();
    PagedCache target_cache(tc.n_layers, tc.n_kv_heads, tc.head_dim, cfg.num_blocks, cfg.block_size);
    PagedCache draft_cache(dc.n_layers, dc.n_kv_heads, dc.head_dim, cfg.num_blocks, cfg.block_size);

    void* target_dev = nullptr;
    void* draft_dev = nullptr;
    Stream target;
    target.model = &target_model;
    target.cache = &target_cache;
    target.device = cfg.use_cuda ? &target_dev : nullptr;
    target.cfg = cfg;
    Stream draft;
    draft.model = &draft_model;
    draft.cache = &draft_cache;
    draft.device = cfg.use_cuda ? &draft_dev : nullptr;
    draft.cfg = cfg;
    target.tokens = prompt;
    draft.tokens = prompt;

    SpeculativeStats stats;
    if (max_new == 0) {
        stats.tokens = prompt;
        return stats;
    }

    auto t0 = std::chrono::steady_clock::now();

    std::vector<float> logits;
    target.forward(static_cast<int>(prompt.size()), false, logits);
    target.pending.swap(logits);
    draft.forward(static_cast<int>(prompt.size()), false, logits);
    draft.pending.swap(logits);

    const int prompt_len = static_cast<int>(prompt.size());
    const int eos = target_model.config().eos_id;
    while (static_cast<int>(target.tokens.size()) - prompt_len < max_new) {
        if (eos >= 0 && static_cast<int>(target.tokens.size()) > prompt_len &&
            target.tokens.back() == eos) {
            break;
        }
        const int prefix_len = static_cast<int>(target.tokens.size());
        const int room = max_new - (prefix_len - prompt_len);
        const int g = std::min(gamma, room);

        std::vector<int> props;
        props.reserve(g);
        for (int i = 0; i < g; ++i) {
            const int tok = argmax(draft.pending.data(), vocab);
            props.push_back(tok);
            draft.tokens.push_back(tok);
            draft.forward(1, false, logits);
            draft.pending.swap(logits);
        }
        stats.drafted += static_cast<int>(props.size());

        for (int tok : props) target.tokens.push_back(tok);
        std::vector<float> rows;
        target.forward(static_cast<int>(props.size()), true, rows);

        int n_accept = 0;
        const float* check = target.pending.data();
        for (int i = 0; i < static_cast<int>(props.size()); ++i) {
            if (argmax(check, vocab) != props[i]) break;
            ++n_accept;
            if (i + 1 < static_cast<int>(props.size())) {
                check = rows.data() + static_cast<std::size_t>(i) * vocab;
            }
        }

        int extra = 0;
        bool bonus = false;
        if (n_accept == static_cast<int>(props.size())) {
            extra = argmax(rows.data() + static_cast<std::size_t>(props.size() - 1) * vocab, vocab);
            bonus = true;
        } else {
            extra = argmax(check, vocab);
        }

        int keep = std::min(n_accept, room);
        if (eos >= 0) {
            for (int i = 0; i < keep; ++i) {
                if (target.tokens[static_cast<std::size_t>(prefix_len + i)] == eos) {
                    keep = i + 1;
                    break;
                }
            }
        }
        const bool saw_eos = eos >= 0 && keep > 0 &&
                             target.tokens[static_cast<std::size_t>(prefix_len + keep - 1)] == eos;
        const bool take_extra = !saw_eos && keep == n_accept && keep < room;
        target.tokens.resize(static_cast<std::size_t>(prefix_len + keep));
        target.computed = prefix_len + keep;
        stats.accepted_draft += keep;
        if (take_extra) {
            target.tokens.push_back(extra);
            target.forward(1, false, logits);
            target.pending.swap(logits);
            if (bonus) ++stats.bonus;
            if (eos >= 0 && extra == eos) {
                // stop at bonus eos
            }
        }

        draft.tokens = target.tokens;
        draft.computed = prefix_len;
        const int replay = static_cast<int>(draft.tokens.size()) - prefix_len;
        if (replay > 0) {
            draft.forward(replay, false, logits);
            draft.pending.swap(logits);
        }
        if (saw_eos || (eos >= 0 && !target.tokens.empty() && target.tokens.back() == eos)) break;
    }

    stats.wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    stats.tokens = target.tokens;
    if (target_dev) cuda::release_device(target_dev);
    if (draft_dev) cuda::release_device(draft_dev);
    return stats;
}

}  // namespace

SpeculativeStats speculative_generate(Model& target, Model& draft, const std::vector<int>& prompt,
                                      int max_new, int gamma, SpeculativeConfig cfg) {
    return run_spec(target, draft, prompt, max_new, gamma, cfg);
}

SpeculativeStats speculative_benchmark(Model& target, Model& draft, const std::vector<int>& prompt,
                                       int max_new, int gamma, SpeculativeConfig cfg) {
    // Target-only greedy via Engine for a fair wall-clock baseline.
    EngineConfig ecfg;
    ecfg.num_blocks = cfg.num_blocks;
    ecfg.block_size = cfg.block_size;
    ecfg.use_cuda = cfg.use_cuda;
    ecfg.cuda_graphs = cfg.cuda_graphs;
    ecfg.use_cublas = cfg.use_cublas;
    ecfg.fuse_kernels = cfg.fuse_kernels;
    Engine engine(target, ecfg);
    Request req;
    req.prompt = prompt;
    req.max_new_tokens = max_new;
    req.temperature = 0.f;
    // Warmup uploads weights / graphs so the timed window is decode.
    engine.generate({req});
    auto t0 = std::chrono::steady_clock::now();
    auto greedy = engine.generate({req});
    double target_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

    // Warmup speculative path too (separate device slots).
    (void)run_spec(target, draft, prompt, max_new, gamma, cfg);
    auto stats = run_spec(target, draft, prompt, max_new, gamma, cfg);
    stats.target_only_ms = target_ms;
    if (stats.tokens != greedy[0].tokens) {
        throw std::runtime_error("speculative output diverged from target-only greedy");
    }
    return stats;
}

}  // namespace inferno
