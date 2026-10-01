#pragma once

#include "inferno/engine.hpp"

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>

namespace inferno {

std::string completions_response(const GenerationResult& result);
GenerationResult run_completion_json(Engine& engine, const std::string& body);
std::string metrics_text(const Metrics& m);

class HttpServer {
public:
    HttpServer(Engine& engine, int port);
    ~HttpServer();
    void start();
    void stop();
    int port() const { return port_; }

private:
    void loop();
    void handle(int fd);

    Engine& engine_;
    int port_;
    int listen_fd_ = -1;
    std::atomic<bool> stop_{false};
    std::atomic<int> inflight_{0};
    std::mutex idle_mu_;
    std::condition_variable idle_cv_;
    std::thread thread_;
};

}  // namespace inferno
