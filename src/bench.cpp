#include "inferno/bench.hpp"

#include "inferno/cpu_kernels.hpp"
#include "inferno/cuda_api.hpp"
#include "inferno/engine.hpp"
#include "inferno/paged_cache.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

namespace inferno {
namespace {

double ms_since(std::chrono::steady_clock::time_point a) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count();
}

ModelConfig bench_config() {
    ModelConfig cfg;
    cfg.vocab = 128;
    cfg.hidden = 64;
    cfg.n_layers = 2;
    cfg.n_heads = 4;
    cfg.n_kv_heads = 2;
    cfg.head_dim = 16;
    cfg.intermediate = 128;
    return cfg;
}

}  // namespace

void run_benchmarks() {
    std::mt19937 rng(1);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);

    const int M = 256, N = 256, K = 256;
    std::vector<float> a(M * K), b(N * K), c(M * N), c2(M * N);
    for (auto& v : a) v = dist(rng);
    for (auto& v : b) v = dist(rng);
    auto t0 = std::chrono::steady_clock::now();
    kernels::linear(a.data(), b.data(), c.data(), M, N, K);
    double naive_ms = ms_since(t0);
    t0 = std::chrono::steady_clock::now();
    kernels::linear_tiled(a.data(), b.data(), c2.data(), M, N, K, 32);
    double tiled_ms = ms_since(t0);
    float max_diff = 0.f;
    for (std::size_t i = 0; i < c.size(); ++i) max_diff = std::max(max_diff, std::fabs(c[i] - c2[i]));
    const double naive_gflops = (2.0 * M * N * K) / (naive_ms * 1e6);
    const double tiled_gflops = (2.0 * M * N * K) / (tiled_ms * 1e6);
    std::cout << "gemm " << M << "x" << N << "x" << K
              << "  naive " << naive_ms << " ms (" << naive_gflops << " GFLOP/s)"
              << "  tiled " << tiled_ms << " ms (" << tiled_gflops << " GFLOP/s)"
              << "  max_abs_diff " << max_diff << "\n";

    const int T = 128, heads = 8, dim = 32;
    std::vector<float> q(T * heads * dim), k(T * heads * dim), v(T * heads * dim);
    std::vector<float> o1(q.size()), o2(q.size());
    for (auto& x : q) x = dist(rng);
    for (auto& x : k) x = dist(rng);
    for (auto& x : v) x = dist(rng);
    t0 = std::chrono::steady_clock::now();
    kernels::causal_attention_naive(q.data(), k.data(), v.data(), o1.data(), T, heads, heads, dim);
    double naive_attn = ms_since(t0);
    t0 = std::chrono::steady_clock::now();
    kernels::causal_attention_online(q.data(), k.data(), v.data(), o2.data(), T, heads, heads, dim);
    double flash_attn = ms_since(t0);
    max_diff = 0;
    for (std::size_t i = 0; i < o1.size(); ++i) max_diff = std::max(max_diff, std::fabs(o1[i] - o2[i]));
    std::cout << "attention T=" << T << " heads=" << heads << " dim=" << dim
              << "  full-softmax " << naive_attn << " ms"
              << "  flash-online " << flash_attn << " ms"
              << "  max_abs_diff " << max_diff << "\n";

    Model model = Model::init_random(bench_config(), 7);
    EngineConfig cfg;
    cfg.block_size = 8;
    cfg.num_blocks = 256;
    cfg.max_batched_tokens = 512;
    cfg.max_seqs = 16;

    std::vector<Request> reqs;
    for (int i = 0; i < 8; ++i) {
        Request r;
        r.id = i;
        r.prompt.resize(12);
        for (int j = 0; j < 12; ++j) r.prompt[j] = (i * 3 + j) % model.vocab();
        r.max_new_tokens = 12;
        r.temperature = 0.f;
        reqs.push_back(r);
    }

    Engine batched(model, cfg);
    t0 = std::chrono::steady_clock::now();
    auto batched_out = batched.generate(reqs);
    double batched_ms = ms_since(t0);
    int gen_tokens = 0;
    for (const auto& r : batched_out) gen_tokens += r.generated;

    double serial_ms = 0;
    for (const auto& req : reqs) {
        Engine one(model, cfg);
        t0 = std::chrono::steady_clock::now();
        one.generate({req});
        serial_ms += ms_since(t0);
    }

    // No-cache baseline: every new token replays the whole prefix.
    t0 = std::chrono::steady_clock::now();
    {
        Request req = reqs[0];
        std::vector<int> tokens = req.prompt;
        Model local = model;
        for (int n = 0; n < req.max_new_tokens; ++n) {
            PagedCache cache(model.config().n_layers, model.config().n_kv_heads, model.config().head_dim,
                             cfg.num_blocks, cfg.block_size);
            std::vector<int> blocks;
            int need = (static_cast<int>(tokens.size()) + cfg.block_size - 1) / cfg.block_size;
            for (int b = 0; b < need; ++b) blocks.push_back(cache.alloc_block());
            cache.set_block_table(0, blocks);
            std::vector<TokenIn> batch(tokens.size());
            for (int i = 0; i < static_cast<int>(tokens.size()); ++i) {
                batch[i].seq_id = 0;
                batch[i].position = i;
                batch[i].token_id = tokens[i];
                batch[i].emit = i + 1 == static_cast<int>(tokens.size());
            }
            std::vector<float> logits;
            local.forward(batch, cache, logits);
            int best = 0;
            float best_v = logits[0];
            for (int i = 1; i < model.vocab(); ++i) {
                if (logits[i] > best_v) {
                    best_v = logits[i];
                    best = i;
                }
            }
            tokens.push_back(best);
        }
    }
    double recompute_ms = ms_since(t0);

    std::cout << "serve  8 sequences x (12 prompt + 12 new)\n"
              << "  continuous batch " << batched_ms << " ms, " << gen_tokens << " generated, "
              << (1000.0 * gen_tokens / batched_ms) << " tok/s\n"
              << "  serial engines   " << serial_ms << " ms\n"
              << "  prefill-every-token (1 sequence) " << recompute_ms << " ms\n";

    if (cuda::available()) {
        float graph_ms = 0.f;
        if (cuda::graph_replay_bench(128, 128, 128, 20, graph_ms)) {
            std::cout << "cuda graph replay GEMM+norm " << graph_ms << " ms\n";
        }
    } else {
        std::cout << "cuda  not linked (rebuild with -DINFERNO_CUDA=ON on a GPU node)\n";
    }
}

}  // namespace inferno
