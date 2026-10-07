#pragma once

#include "inferno/paged_cache.hpp"
#include "inferno/prefix_cache.hpp"
#include "inferno/types.hpp"

#include <random>
#include <vector>

namespace inferno {

struct Step {
    std::vector<TokenIn> tokens;
    std::vector<int> computed_seq;
    std::vector<int> computed_n;
    std::vector<int> emit_seq;
};

// Continuous batching with chunked prefill and priority preemption.
// Optional automatic prefix caching: identical block-aligned prompt prefixes
// share physical KV pages (system prompts / chat templates).
class Scheduler {
public:
    Scheduler(PagedCache& cache, PrefixCache* prefixes, int max_batched_tokens, int max_seqs, int eos_id,
              bool prefix_caching, bool retain_prefixes);

    void add(Request req);
    bool has_work() const;
    Step prepare();
    void commit(const Step& step, const float* logits, int vocab);
    std::vector<GenerationResult> pop_finished();

    int preemptions() const { return preemptions_; }
    int waiting() const;
    int running() const;
    Metrics metrics() const { return metrics_; }

private:
    struct Seq {
        int slot = 0;
        int public_id = 0;
        int arrival = 0;
        std::vector<int> tokens;
        int prompt_len = 0;
        int computed = 0;
        int max_new = 0;
        int generated = 0;
        float temperature = 0.f;
        int top_k = 1;
        std::uint64_t seed = 1;
        std::mt19937 rng{1};
        std::vector<int> blocks;
        int shared_len = 0;           // block-aligned shared prefix length
        int shared_blocks = 0;        // number of leading blocks that are shared
        std::uint64_t prefix_hash = 0;
        bool has_prefix = false;
        bool prefix_owner = false;
        bool waiting_prefix = false;  // follower waiting for owner to fill shared pages
        bool running = false;
        bool waiting = false;
        bool finished = false;
        bool failed = false;
        std::string error;
    };

    bool ensure_blocks(Seq& s, int pos);
    bool preempt_lower_than(const Seq& s);
    void finish(Seq& s);
    void attach_prefix(Seq& s, int prefix_len);
    int running_count() const;

    PagedCache& cache_;
    PrefixCache* prefixes_ = nullptr;
    int max_batched_tokens_;
    int max_seqs_;
    int eos_id_;
    bool prefix_caching_ = false;
    bool retain_prefixes_ = true;
    int arrival_clock_ = 0;
    int preemptions_ = 0;
    std::vector<Seq> seqs_;
    std::vector<GenerationResult> finished_;
    Metrics metrics_{};
};

}  // namespace inferno
