#include "inferno/speculative.hpp"

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
        model->forward(batch, *cache, logits);
        computed += n;
    }
};

}  // namespace

SpeculativeStats speculative_generate(Model& target_model, Model& draft_model,
                                      const std::vector<int>& prompt,
                                      int max_new, int gamma,
                                      int num_blocks, int block_size) {
    if (prompt.empty()) throw std::runtime_error("empty prompt");
    if (max_new < 0 || gamma < 1) throw std::runtime_error("bad speculative config");
    if (target_model.vocab() != draft_model.vocab()) {
        throw std::runtime_error("draft and target vocabularies differ");
    }
    const int vocab = target_model.vocab();
    const auto tc = target_model.config();
    const auto dc = draft_model.config();
    PagedCache target_cache(tc.n_layers, tc.n_kv_heads, tc.head_dim, num_blocks, block_size);
    PagedCache draft_cache(dc.n_layers, dc.n_kv_heads, dc.head_dim, num_blocks, block_size);

    Stream target;
    target.model = &target_model;
    target.cache = &target_cache;
    Stream draft;
    draft.model = &draft_model;
    draft.cache = &draft_cache;
    target.tokens = prompt;
    draft.tokens = prompt;

    SpeculativeStats stats;
    if (max_new == 0) {
        stats.tokens = prompt;
        return stats;
    }

    std::vector<float> logits;
    target.forward(static_cast<int>(prompt.size()), false, logits);
    target.pending.swap(logits);
    draft.forward(static_cast<int>(prompt.size()), false, logits);
    draft.pending.swap(logits);

    const int prompt_len = static_cast<int>(prompt.size());
    while (static_cast<int>(target.tokens.size()) - prompt_len < max_new) {
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
        const bool take_extra = keep == n_accept && keep < room;
        target.tokens.resize(static_cast<std::size_t>(prefix_len + keep));
        target.computed = prefix_len + keep;
        stats.accepted_draft += keep;
        if (take_extra) {
            target.tokens.push_back(extra);
            target.forward(1, false, logits);
            target.pending.swap(logits);
            if (bonus) ++stats.bonus;
        }

        draft.tokens = target.tokens;
        draft.computed = prefix_len;
        const int replay = static_cast<int>(draft.tokens.size()) - prefix_len;
        if (replay > 0) {
            draft.forward(replay, false, logits);
            draft.pending.swap(logits);
        }
    }

    stats.tokens = target.tokens;
    return stats;
}

}  // namespace inferno
