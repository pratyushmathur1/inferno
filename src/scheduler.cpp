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

Scheduler::Scheduler(PagedCache& cache, PrefixCache* prefixes, int max_batched_tokens, int max_seqs,
                     int eos_id, bool prefix_caching, bool retain_prefixes)
    : cache_(cache),
      prefixes_(prefixes),
      max_batched_tokens_(max_batched_tokens),
      max_seqs_(max_seqs),
      eos_id_(eos_id),
      prefix_caching_(prefix_caching && prefixes != nullptr),
      retain_prefixes_(retain_prefixes) {
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

void Scheduler::attach_prefix(Seq& s, int requested_prefix_len) {
    if (!prefix_caching_ || !prefixes_) return;
    const int L = prefixes_->shared_len(s.tokens, requested_prefix_len);
    if (L < cache_.block_size()) return;

    const std::uint64_t h = PrefixCache::hash_tokens(s.tokens, L);
    PrefixEntry* existing = prefixes_->find(h);
    if (existing) {
        // Verify tokens match (hash collision guard).
        if (static_cast<int>(existing->tokens.size()) != L) return;
        for (int i = 0; i < L; ++i) {
            if (existing->tokens[static_cast<std::size_t>(i)] != s.tokens[static_cast<std::size_t>(i)]) {
                return;
            }
        }
        prefixes_->add_ref(h);
        s.has_prefix = true;
        s.prefix_hash = h;
        s.shared_len = L;
        s.shared_blocks = L / cache_.block_size();
        s.blocks = existing->blocks;
        cache_.set_block_table(s.slot, s.blocks);
        if (existing->ready) {
            s.computed = L;
            s.waiting_prefix = false;
            metrics_.prefix_hits += 1;
            metrics_.prefix_tokens_saved += L;
        } else {
            s.computed = 0;
            s.waiting_prefix = true;  // owner still filling pages
            s.prefix_owner = false;
            metrics_.prefix_hits += 1;  // structural hit; still waits for ready
        }
        return;
    }

    // Miss: allocate shared pages; this sequence owns the prefill.
    std::vector<int> blocks;
    blocks.reserve(static_cast<std::size_t>(L / cache_.block_size()));
    for (int i = 0; i < L / cache_.block_size(); ++i) {
        int b = cache_.alloc_block();
        if (b < 0) {
            cache_.free_all(blocks);
            return;  // fall back to uncached path
        }
        blocks.push_back(b);
    }
    std::vector<int> tok(s.tokens.begin(), s.tokens.begin() + L);
    prefixes_->insert(h, std::move(tok), blocks, s.slot);
    s.has_prefix = true;
    s.prefix_hash = h;
    s.shared_len = L;
    s.shared_blocks = static_cast<int>(blocks.size());
    s.blocks = std::move(blocks);
    s.prefix_owner = true;
    s.waiting_prefix = false;
    s.computed = 0;
    cache_.set_block_table(s.slot, s.blocks);
    metrics_.prefix_misses += 1;
}

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

    attach_prefix(s, req.prefix_len);
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
    // Free only private suffix pages; shared prefix goes through PrefixCache.
    if (s.has_prefix && prefixes_) {
        std::vector<int> private_blocks;
        if (static_cast<int>(s.blocks.size()) > s.shared_blocks) {
            private_blocks.assign(s.blocks.begin() + s.shared_blocks, s.blocks.end());
        }
        cache_.free_all(private_blocks);
        auto drop = prefixes_->release(s.prefix_hash, retain_prefixes_);
        cache_.free_all(drop);
    } else {
        cache_.free_all(s.blocks);
    }
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
        // Prefer victims that have private pages to reclaim.
        if (static_cast<int>(o.blocks.size()) <= o.shared_blocks) continue;
        if (o.arrival > newest) {
            newest = o.arrival;
            victim = i;
        }
    }
    if (victim < 0) return false;
    Seq& v = seqs_[victim];
    std::vector<int> private_blocks;
    if (static_cast<int>(v.blocks.size()) > v.shared_blocks) {
        private_blocks.assign(v.blocks.begin() + v.shared_blocks, v.blocks.end());
    }
    cache_.free_all(private_blocks);
    v.blocks.resize(static_cast<std::size_t>(v.shared_blocks));
    cache_.set_block_table(v.slot, v.blocks);
    v.computed = v.shared_len;  // keep shared prefix; recompute suffix
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
    auto promote_prefix_waiters = [&] {
        if (!prefix_caching_ || !prefixes_) return;
        for (auto& s : seqs_) {
            if (!s.waiting_prefix || !s.has_prefix) continue;
            PrefixEntry* e = prefixes_->find(s.prefix_hash);
            if (e && e->ready) {
                s.computed = std::max(s.computed, s.shared_len);
                s.waiting_prefix = false;
                metrics_.prefix_tokens_saved += s.shared_len;
            }
        }
    };

    for (int attempt = 0; attempt < 2; ++attempt) {
        promote_prefix_waiters();

        for (auto& s : seqs_) {
            if (!s.waiting || s.finished || s.waiting_prefix) continue;
            if (running_count() >= max_seqs_) break;
            s.waiting = false;
            s.running = true;
        }

        std::vector<int> order;
        for (int i = 0; i < static_cast<int>(seqs_.size()); ++i) {
            if (seqs_[i].running && !seqs_[i].waiting_prefix) order.push_back(i);
        }
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            return seqs_[a].arrival < seqs_[b].arrival;
        });

        Step step;
        int budget = max_batched_tokens_;
        for (int idx : order) {
            if (budget <= 0) break;
            Seq& s = seqs_[idx];
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
        if (!step.tokens.empty()) {
            metrics_.steps += 1;
            metrics_.tokens_processed += static_cast<std::int64_t>(step.tokens.size());
            return step;
        }
        if (!has_work()) {
            metrics_.steps += 1;
            return step;
        }
        // Promote prefix owners that are still waiting while followers block.
        bool promoted = false;
        for (auto& s : seqs_) {
            if (!s.waiting_prefix || !s.has_prefix || !prefixes_) continue;
            PrefixEntry* e = prefixes_->find(s.prefix_hash);
            if (!e || e->ready || e->owner_slot < 0 || e->owner_slot >= static_cast<int>(seqs_.size())) {
                continue;
            }
            Seq& owner = seqs_[e->owner_slot];
            if (owner.waiting) {
                owner.waiting = false;
                owner.running = true;
                owner.waiting_prefix = false;
                promoted = true;
            }
        }
        if (!promoted || attempt == 1) break;
    }
    throw std::runtime_error("scheduler made no progress; KV pool is too small for the batch");
}

void Scheduler::commit(const Step& step, const float* logits, int vocab) {
    for (std::size_t i = 0; i < step.computed_seq.size(); ++i) {
        Seq& s = seqs_[step.computed_seq[i]];
        s.computed += step.computed_n[i];
        if (s.prefix_owner && s.has_prefix && prefixes_ && s.computed >= s.shared_len) {
            prefixes_->mark_ready(s.prefix_hash);
            // Wake followers on next prepare.
        }
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
