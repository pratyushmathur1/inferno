#pragma once

#include "inferno/cuda_api.hpp"
#include "inferno/model.hpp"
#include "inferno/prefix_cache.hpp"
#include "inferno/scheduler.hpp"
#include "inferno/types.hpp"

#include <condition_variable>
#include <future>
#include <mutex>
#include <thread>

namespace inferno {

class Engine {
public:
    Engine(Model model, EngineConfig cfg);
    ~Engine();

    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    std::vector<GenerationResult> generate(const std::vector<Request>& requests);
    Metrics metrics() const;
    const Model& model() const { return model_; }
    PagedCache& cache() { return cache_; }
    PrefixCache& prefixes() { return prefixes_; }
    const EngineConfig& config() const { return cfg_; }

    void start();
    void stop();
    std::future<GenerationResult> submit(Request req);
    int device_replays() const { return cuda::graph_replays(device_); }
    cuda::Profile device_profile() const { return cuda::last_profile(device_); }

private:
    void worker();
    void step(Scheduler& sched);
    void store_metrics(const Metrics& m);

    Model model_;
    EngineConfig cfg_;
    PagedCache cache_;
    PrefixCache prefixes_;
    Metrics last_metrics_{};
    mutable std::mutex metrics_mu_;
    void* device_ = nullptr;

    std::mutex mu_;
    std::condition_variable cv_;
    bool running_ = false;
    bool stop_ = false;
    std::thread worker_;
    struct Pending {
        Request req;
        std::promise<GenerationResult> promise;
    };
    std::vector<Pending> inbox_;
};

}  // namespace inferno
