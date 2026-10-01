#pragma once

#include "inferno/paged_cache.hpp"
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
// Earlier arrivals outrank later ones. A higher-priority running sequence
// may reclaim pages from a lower-priority one; the victim is recomputed
// from its already sampled tokens, so greedy outputs stay stable.
class Scheduler {
public:
    Scheduler(PagedCache& cache, int max_batched_tokens, int max_seqs, int eos_id);

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
        bool running = false;
        bool waiting = false;
        bool finished = false;
        bool failed = false;
        std::string error;
    };

    bool ensure_blocks(Seq& s, int pos);
    bool preempt_lower_than(const Seq& s);
    void finish(Seq& s);
    int running_count() const;

    PagedCache& cache_;
    int max_batched_tokens_;
    int max_seqs_;
    int eos_id_;
    int arrival_clock_ = 0;
    int preemptions_ = 0;
    std::vector<Seq> seqs_;
    std::vector<GenerationResult> finished_;
    Metrics metrics_{};
};

}  // namespace inferno
