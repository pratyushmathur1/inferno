#include "inferno/scheduler.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace inferno {
namespace {

int sample_token(const float* logits, int vocab, float temperature, int top_k, std::mt19937& rng) {
    if (vocab <= 0) throw std::runtime_error("empty vocab");
    if (temperature <= 0.f) {
        int best = 0;
        float m = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < vocab; ++i) {
            if (logits[i] > m) {
                m = logits[i];
                best = i;
            }
        }
        return best;
    }
    int k = std::max(1, std::min(top_k, vocab));
    std::vector<int> idx(vocab);
    for (int i = 0; i < vocab; ++i) idx[i] = i;
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int b) {
        if (logits[a] != logits[b]) return logits[a] > logits[b];
        return a < b;
    });
    float m = logits[idx[0]] / temperature;
    for (int i = 1; i < k; ++i) m = std::max(m, logits[idx[i]] / temperature);
    std::vector<double> probs(k);
    double sum = 0.0;
    for (int i = 0; i < k; ++i) {
        probs[i] = std::exp(static_cast<double>(logits[idx[i]] / temperature - m));
        sum += probs[i];
    }
    std::uniform_real_distribution<double> dist(0.0, sum);
    double draw = dist(rng);
    double run = 0.0;
    for (int i = 0; i < k; ++i) {
        run += probs[i];
        if (draw <= run) return idx[i];
    }
    return idx[k - 1];
}

}  // namespace

Scheduler::Scheduler(PagedCache& cache, int max_batched_tokens, int max_seqs, int eos_id)
    : cache_(cache), max_batched_tokens_(max_batched_tokens), max_seqs_(max_seqs), eos_id_(eos_id) {
    if (max_batched_tokens_ < 1 || max_seqs_ < 1) throw std::runtime_error("bad scheduler config");
}

int Scheduler::running_count() const {
    int n = 0;
    for (const auto& s : seqs_) n += s.running ? 1 : 0;
    return n;
}

int Scheduler::waiting() const {
    int n = 0;
    for (const auto& s : seqs_) n += s.waiting ? 1 : 0;
    return n;
}

int Scheduler::running() const { return running_count(); }

void Scheduler::add(Request req) {
    if (req.prompt.empty()) throw std::runtime_error("empty prompt");
    if (req.max_new_tokens < 0) throw std::runtime_error("negative max_new_tokens");
    const int slot = static_cast<int>(seqs_.size());
    const int public_id = req.id >= 0 ? req.id : slot;
    const int final_len = static_cast<int>(req.prompt.size()) + req.max_new_tokens;
    const int blocks_needed = (final_len + cache_.block_size() - 1) / cache_.block_size();

    Seq s;
    s.slot = slot;
    s.public_id = public_id;
    s.arrival = arrival_clock_++;
    s.tokens = std::move(req.prompt);
    s.prompt_len = static_cast<int>(s.tokens.size());
    s.max_new = req.max_new_tokens;
    s.temperature = req.temperature;
    s.top_k = std::max(1, req.top_k);
    s.seed = req.seed;
    s.rng.seed(static_cast<std::mt19937::result_type>(req.seed));
    metrics_.requests += 1;

    if (blocks_needed > cache_.num_blocks()) {
        s.failed = true;
        s.finished = true;
        s.error = "prompt plus generation does not fit in the KV page pool";
        GenerationResult r;
        r.id = public_id;
        r.tokens = s.tokens;
        r.ok = false;
        r.error = s.error;
        finished_.push_back(std::move(r));
        seqs_.push_back(std::move(s));
        return;
    }
    s.waiting = true;
    seqs_.push_back(std::move(s));
}

bool Scheduler::has_work() const {
    for (const auto& s : seqs_) {
        if (s.waiting || s.running) return true;
    }
    return false;
}

void Scheduler::finish(Seq& s) {
    if (s.finished) return;
    cache_.free_all(s.blocks);
    cache_.clear_seq(s.slot);
    s.blocks.clear();
    s.running = false;
    s.waiting = false;
    s.finished = true;
    GenerationResult r;
    r.id = s.public_id;
    r.tokens = s.tokens;
    r.generated = s.generated;
    r.ok = !s.failed;
    r.error = s.error;
    finished_.push_back(std::move(r));
}

bool Scheduler::preempt_lower_than(const Seq& s) {
    int victim = -1;
    int newest = -1;
    for (int i = 0; i < static_cast<int>(seqs_.size()); ++i) {
        const Seq& o = seqs_[i];
        if (!o.running || o.arrival <= s.arrival || o.blocks.empty()) continue;
        if (o.arrival > newest) {
            newest = o.arrival;
            victim = i;
        }
    }
    if (victim < 0) return false;
    Seq& v = seqs_[victim];
    cache_.free_all(v.blocks);
    cache_.clear_seq(v.slot);
    v.blocks.clear();
    v.computed = 0;
    v.running = false;
    v.waiting = true;
    ++preemptions_;
    metrics_.preemptions += 1;
    return true;
}

bool Scheduler::ensure_blocks(Seq& s, int pos) {
    const int need = pos / cache_.block_size() + 1;
    while (static_cast<int>(s.blocks.size()) < need) {
        int b = cache_.alloc_block();
        if (b < 0) {
            if (!preempt_lower_than(s)) return false;
            continue;
        }
        s.blocks.push_back(b);
    }
    cache_.set_block_table(s.slot, s.blocks);
    return true;
}

Step Scheduler::prepare() {
    for (auto& s : seqs_) {
        if (!s.waiting || s.finished) continue;
        if (running_count() >= max_seqs_) break;
        s.waiting = false;
        s.running = true;
    }

    std::vector<int> order;
    for (int i = 0; i < static_cast<int>(seqs_.size()); ++i) {
        if (seqs_[i].running) order.push_back(i);
    }
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return seqs_[a].arrival < seqs_[b].arrival;
    });

    Step step;
    int budget = max_batched_tokens_;
    for (int idx : order) {
        if (budget <= 0) break;
        Seq& s = seqs_[idx];
        // A newer sequence preempted earlier in this step is waiting and must
        // not be given pages until the next iteration.
        if (!s.running) continue;
        const int uncomputed = static_cast<int>(s.tokens.size()) - s.computed;
        if (uncomputed <= 0) {
            if (s.generated >= s.max_new) finish(s);
            continue;
        }
        const int take = std::min(uncomputed, budget);
        if (!ensure_blocks(s, s.computed + take - 1)) continue;
        for (int i = 0; i < take; ++i) {
            const int pos = s.computed + i;
            TokenIn tok;
            tok.seq_id = s.slot;
            tok.position = pos;
            tok.token_id = s.tokens[pos];
            tok.emit = (pos == static_cast<int>(s.tokens.size()) - 1) && (s.generated < s.max_new);
            step.tokens.push_back(tok);
            if (tok.emit) step.emit_seq.push_back(s.slot);
        }
        step.computed_seq.push_back(s.slot);
        step.computed_n.push_back(take);
        budget -= take;
    }
    if (step.tokens.empty() && has_work()) {
        throw std::runtime_error("scheduler made no progress; KV pool is too small for the batch");
    }
    metrics_.steps += 1;
    metrics_.tokens_processed += static_cast<std::int64_t>(step.tokens.size());
    return step;
}

void Scheduler::commit(const Step& step, const float* logits, int vocab) {
    for (std::size_t i = 0; i < step.computed_seq.size(); ++i) {
        seqs_[step.computed_seq[i]].computed += step.computed_n[i];
    }
    if (step.emit_seq.empty()) {
        for (int slot : step.computed_seq) {
            Seq& s = seqs_[slot];
            if (!s.finished && s.generated >= s.max_new && s.computed >= static_cast<int>(s.tokens.size())) {
                finish(s);
            }
        }
        return;
    }
    if (!logits) throw std::runtime_error("missing logits");
    int row = 0;
    for (int slot : step.emit_seq) {
        Seq& s = seqs_[slot];
        const int tok = sample_token(logits + static_cast<std::size_t>(row) * vocab, vocab,
                                      s.temperature, s.top_k, s.rng);
        ++row;
        s.tokens.push_back(tok);
        s.generated += 1;
        if ((eos_id_ >= 0 && tok == eos_id_) || s.generated >= s.max_new) finish(s);
    }
}

std::vector<GenerationResult> Scheduler::pop_finished() {
    std::vector<GenerationResult> out;
    out.swap(finished_);
    return out;
}

}  // namespace inferno
