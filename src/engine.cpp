#include "inferno/engine.hpp"

#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace inferno {

Engine::Engine(Model model, EngineConfig cfg)
    : model_(std::move(model)),
      cfg_(cfg),
      cache_(model_.config().n_layers, model_.config().n_kv_heads, model_.config().head_dim,
             cfg.num_blocks, cfg.block_size) {
    if (cfg_.use_cuda && !cfg_.cuda_graphs) {
        // Device selection is recorded for the CUDA graph path. Operator
        // dispatch stays on the CPU reference until a CUDA build is linked.
    }
}

Engine::~Engine() { stop(); }

std::vector<GenerationResult> Engine::generate(const std::vector<Request>& requests) {
    if (running_) throw std::runtime_error("stop the continuous-batching worker before generate()");
    Scheduler sched(cache_, cfg_.max_batched_tokens, cfg_.max_seqs, model_.config().eos_id);
    for (const auto& req : requests) sched.add(req);
    std::vector<GenerationResult> out;
    while (sched.has_work()) {
        Step step = sched.prepare();
        std::vector<float> logits;
        model_.forward(step.tokens, cache_, logits);
        sched.commit(step, logits.empty() ? nullptr : logits.data(), model_.vocab());
        auto done = sched.pop_finished();
        out.insert(out.end(), done.begin(), done.end());
    }
    auto rest = sched.pop_finished();
    out.insert(out.end(), rest.begin(), rest.end());
    store_metrics(sched.metrics());
    return out;
}

void Engine::step(Scheduler& sched) {
    Step step = sched.prepare();
    std::vector<float> logits;
    model_.forward(step.tokens, cache_, logits);
    sched.commit(step, logits.empty() ? nullptr : logits.data(), model_.vocab());
    store_metrics(sched.metrics());
}

void Engine::store_metrics(const Metrics& m) {
    std::lock_guard<std::mutex> lock(metrics_mu_);
    last_metrics_ = m;
}

Metrics Engine::metrics() const {
    std::lock_guard<std::mutex> lock(metrics_mu_);
    return last_metrics_;
}

void Engine::start() {
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (running_) return;
        stop_ = false;
        running_ = true;
    }
    worker_ = std::thread([this] { worker(); });
}

void Engine::stop() {
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (!running_ && !worker_.joinable()) return;
        stop_ = true;
    }
    cv_.notify_all();
    if (worker_.joinable()) worker_.join();
    stop_ = false;
}

std::future<GenerationResult> Engine::submit(Request req) {
    std::promise<GenerationResult> promise;
    auto fut = promise.get_future();
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (!running_) throw std::runtime_error("Engine::start() before submit()");
        inbox_.push_back(Pending{std::move(req), std::move(promise)});
    }
    cv_.notify_one();
    return fut;
}

void Engine::worker() {
    Scheduler sched(cache_, cfg_.max_batched_tokens, cfg_.max_seqs, model_.config().eos_id);
    std::unordered_map<int, std::promise<GenerationResult>> inflight;
    int next_id = 1;
    auto fulfill = [&](std::vector<GenerationResult> done) {
        for (auto& r : done) {
            auto it = inflight.find(r.id);
            if (it == inflight.end()) continue;
            it->second.set_value(std::move(r));
            inflight.erase(it);
        }
    };
    while (true) {
        std::vector<Pending> grabbed;
        {
            std::unique_lock<std::mutex> lock(mu_);
            cv_.wait(lock, [&] { return stop_ || !inbox_.empty() || sched.has_work(); });
            if (stop_ && inbox_.empty() && !sched.has_work()) {
                running_ = false;
                break;
            }
            grabbed.swap(inbox_);
        }
        for (auto& item : grabbed) {
            item.req.id = next_id++;
            const int id = item.req.id;
            inflight.emplace(id, std::move(item.promise));
            sched.add(std::move(item.req));
        }
        fulfill(sched.pop_finished());
        if (!sched.has_work()) continue;
        step(sched);
        fulfill(sched.pop_finished());
    }
}

}  // namespace inferno
