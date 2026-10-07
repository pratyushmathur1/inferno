#include "inferno/bench.hpp"
#include "inferno/cuda_api.hpp"
#include "inferno/engine.hpp"
#include "inferno/model.hpp"
#include "inferno/server.hpp"
#include "inferno/speculative.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {

void usage() {
    std::cerr
        << "Inferno — a paged, continuously batched LLM inference engine\n\n"
        << "  inferno demo\n"
        << "  inferno bench\n"
        << "  inferno init-model -o model.bin [--seed N] [--layers N] [--hidden N] [--vocab N] [--head-dim N]\n"
        << "  inferno generate -m model.bin --tokens 1,2,3 --max-new 16 [--temp 0] [--cpu]\n"
        << "  inferno speculate -m target.inf1 --draft draft.inf1 --tokens … --max-new 64 --gamma 4\n"
        << "  inferno bench-latency -m model.inf1 --tokens … --max-new 32\n"
        << "  inferno bench-prefix -m model.inf1 --system-tokens … --user-tokens … --seqs N\n"
        << "  inferno compare-workload -m model.bin [--seqs 8] [--prompt 12] [--max-new 12] [--cpu]\n"
        << "  inferno profile-forward -m model.bin [--seqs 8] [--prompt 12] [--max-new 4] [--naive-gemm] [--no-fuse]\n"
        << "  inferno serve -m model.bin [--port 8000] [--cpu]\n"
        << "\nGPU flags: --cuda | --cpu | --no-graphs | --naive-gemm | --no-fuse | --prefix-cache | --profile-forward\n"
        << "Text demo: python3 scripts/generate_text.py -m model.inf1 --hf <hub_id> --prompt \"…\"\n";
}

std::vector<int> parse_tokens(const std::string& s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) out.push_back(std::stoi(item));
    }
    return out;
}

const char* need(int& i, int argc, char** argv, const std::string& flag) {
    if (i + 1 >= argc) throw std::runtime_error("missing value for " + flag);
    return argv[++i];
}

inferno::ModelConfig tiny() {
    inferno::ModelConfig cfg;
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

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            usage();
            return 1;
        }
        std::string cmd = argv[1];
        if (cmd == "demo") {
            inferno::Model model = inferno::Model::init_random(tiny(), 1);
            inferno::EngineConfig cfg;
            cfg.use_cuda = inferno::cuda::available();
            cfg.cuda_graphs = cfg.use_cuda;
            inferno::Engine engine(model, cfg);
            inferno::Request req;
            req.prompt = {1, 5, 9, 12, 4};
            req.max_new_tokens = 16;
            req.temperature = 0.f;
            auto out = engine.generate({req});
            std::cout << "prompt";
            for (int t : req.prompt) std::cout << ' ' << t;
            std::cout << "\ngenerated";
            for (std::size_t i = req.prompt.size(); i < out[0].tokens.size(); ++i) {
                std::cout << ' ' << out[0].tokens[i];
            }
            std::cout << "\nsteps " << engine.metrics().steps
                      << " tokens_processed " << engine.metrics().tokens_processed
                      << (cfg.use_cuda ? " device gpu" : " device cpu");
            if (cfg.use_cuda) std::cout << " graph_replays " << engine.device_replays();
            std::cout << "\n";
            return 0;
        }
        if (cmd == "bench") {
            inferno::run_benchmarks();
            return 0;
        }
        std::string model_path;
        std::string draft_path;
        std::string token_str;
        std::string system_token_str;
        std::string user_token_str;
        std::string out_path = "model.bin";
        int port = 8000;
        int max_new = 16;
        int seed = 1;
        int layers = 2;
        int hidden = 64;
        int vocab = 128;
        int seqs = 8;
        int prompt_len = 12;
        int head_dim = 16;
        int gamma = 4;
        int num_blocks = 0;  // 0 → engine default
        float temp = 0.f;
        bool want_cuda = inferno::cuda::available();
        bool graphs = true;
        bool use_cublas = true;
        bool fuse_kernels = true;
        bool profile_cuda = false;
        bool prefix_caching = false;
        for (int i = 2; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "-m" || a == "--model") model_path = need(i, argc, argv, a);
            else if (a == "--draft") draft_path = need(i, argc, argv, a);
            else if (a == "-o" || a == "--out") out_path = need(i, argc, argv, a);
            else if (a == "--tokens") token_str = need(i, argc, argv, a);
            else if (a == "--system-tokens") system_token_str = need(i, argc, argv, a);
            else if (a == "--user-tokens") user_token_str = need(i, argc, argv, a);
            else if (a == "--max-new") max_new = std::stoi(need(i, argc, argv, a));
            else if (a == "--temp") temp = std::stof(need(i, argc, argv, a));
            else if (a == "--port") port = std::stoi(need(i, argc, argv, a));
            else if (a == "--seed") seed = std::stoi(need(i, argc, argv, a));
            else if (a == "--layers") layers = std::stoi(need(i, argc, argv, a));
            else if (a == "--hidden") hidden = std::stoi(need(i, argc, argv, a));
            else if (a == "--vocab") vocab = std::stoi(need(i, argc, argv, a));
            else if (a == "--head-dim") head_dim = std::stoi(need(i, argc, argv, a));
            else if (a == "--seqs") seqs = std::stoi(need(i, argc, argv, a));
            else if (a == "--prompt") prompt_len = std::stoi(need(i, argc, argv, a));
            else if (a == "--gamma") gamma = std::stoi(need(i, argc, argv, a));
            else if (a == "--num-blocks") num_blocks = std::stoi(need(i, argc, argv, a));
            else if (a == "--cpu") want_cuda = false;
            else if (a == "--cuda") want_cuda = true;
            else if (a == "--no-graphs") graphs = false;
            else if (a == "--naive-gemm") use_cublas = false;
            else if (a == "--no-fuse") fuse_kernels = false;
            else if (a == "--prefix-cache") prefix_caching = true;
            else if (a == "--profile-forward") profile_cuda = true;
            else throw std::runtime_error("unknown flag " + a);
        }
        if (cmd == "init-model") {
            auto cfg = tiny();
            cfg.vocab = vocab;
            cfg.hidden = hidden;
            cfg.n_layers = layers;
            cfg.head_dim = head_dim;
            if (cfg.head_dim < 2 || (cfg.head_dim % 2) != 0) {
                throw std::runtime_error("head_dim must be a positive even integer");
            }
            if (hidden % cfg.head_dim != 0) throw std::runtime_error("hidden must be a multiple of head_dim");
            cfg.n_heads = std::max(1, hidden / cfg.head_dim);
            cfg.n_kv_heads = std::max(1, cfg.n_heads / 2);
            if (cfg.n_heads % cfg.n_kv_heads != 0) cfg.n_kv_heads = cfg.n_heads;
            cfg.intermediate = hidden * 2;
            inferno::Model model = inferno::Model::init_random(cfg, static_cast<std::uint32_t>(seed));
            model.save(out_path);
            std::cout << "wrote " << out_path
                      << " layers=" << cfg.n_layers << " hidden=" << cfg.hidden
                      << " heads=" << cfg.n_heads << " kv=" << cfg.n_kv_heads
                      << " head_dim=" << cfg.head_dim << " vocab=" << cfg.vocab << "\n";
            return 0;
        }
        if (model_path.empty()) throw std::runtime_error("pass -m model.bin");
        inferno::Model model = inferno::Model::load(model_path);
        inferno::EngineConfig ecfg;
        ecfg.use_cuda = want_cuda;
        ecfg.cuda_graphs = want_cuda && graphs && !profile_cuda && cmd != "profile-forward";
        ecfg.use_cublas = use_cublas;
        ecfg.fuse_kernels = fuse_kernels;
        ecfg.profile_cuda = profile_cuda || cmd == "profile-forward";
        ecfg.prefix_caching = prefix_caching || cmd == "bench-prefix";
        if (num_blocks > 0) ecfg.num_blocks = num_blocks;
        // Real models need room for prompt+new; bump if still at default and model is large.
        if (num_blocks <= 0 && model.config().n_layers >= 8) {
            ecfg.num_blocks = std::max(ecfg.num_blocks, 512);
        }
        inferno::Engine engine(model, ecfg);
        if (cmd == "generate") {
            inferno::Request req;
            req.prompt = token_str.empty() ? std::vector<int>{1, 2, 3, 4} : parse_tokens(token_str);
            req.max_new_tokens = max_new;
            req.temperature = temp;
            auto out = engine.generate({req});
            if (!out[0].ok) {
                std::cerr << out[0].error << "\n";
                return 1;
            }
            for (std::size_t i = 0; i < out[0].tokens.size(); ++i) {
                if (i) std::cout << ' ';
                std::cout << out[0].tokens[i];
            }
            std::cout << "\n";
            return 0;
        }
        if (cmd == "compare-workload") {
            if (seqs < 1 || prompt_len < 1 || max_new < 0) {
                throw std::runtime_error("bad compare-workload sizes");
            }
            std::vector<inferno::Request> reqs;
            for (int i = 0; i < seqs; ++i) {
                inferno::Request r;
                r.id = i;
                r.prompt.resize(prompt_len);
                for (int j = 0; j < prompt_len; ++j) {
                    r.prompt[j] = (i * 3 + j) % model.vocab();
                }
                r.max_new_tokens = max_new;
                r.temperature = 0.f;
                reqs.push_back(r);
            }
            // Warmup once so CUDA graphs / weight upload are not in the timed window.
            engine.generate(reqs);
            auto t0 = std::chrono::steady_clock::now();
            auto out = engine.generate(reqs);
            double ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0)
                            .count();
            int gen = 0;
            for (const auto& r : out) gen += r.generated;
            double tok_s = ms > 0.0 ? (1000.0 * gen / ms) : 0.0;
            std::cout << "COMPARE {";
            std::cout << "\"ms\":" << ms << ",\"gen\":" << gen << ",\"tok_s\":" << tok_s
                      << ",\"device\":\"" << (ecfg.use_cuda ? "gpu" : "cpu") << "\""
                      << ",\"gemm\":\"" << (ecfg.use_cublas ? "cublas" : "tiled") << "\""
                      << ",\"fuse\":" << (ecfg.fuse_kernels ? "true" : "false")
                      << ",\"graph_replays\":" << engine.device_replays()
                      << ",\"sequences\":[";
            for (std::size_t i = 0; i < out.size(); ++i) {
                if (i) std::cout << ',';
                const auto* r = &out[i];
                for (const auto& cand : out) {
                    if (cand.id == static_cast<int>(i)) {
                        r = &cand;
                        break;
                    }
                }
                std::cout << "{\"id\":" << r->id << ",\"tokens\":[";
                for (std::size_t t = 0; t < r->tokens.size(); ++t) {
                    if (t) std::cout << ',';
                    std::cout << r->tokens[t];
                }
                std::cout << "]}";
            }
            std::cout << "]}\n";
            return 0;
        }
        if (cmd == "profile-forward") {
            if (!ecfg.use_cuda) throw std::runtime_error("profile-forward needs GPU (--cuda)");
            if (seqs < 1 || prompt_len < 1) throw std::runtime_error("bad profile sizes");
            std::vector<inferno::Request> reqs;
            for (int i = 0; i < seqs; ++i) {
                inferno::Request r;
                r.id = i;
                r.prompt.resize(prompt_len);
                for (int j = 0; j < prompt_len; ++j) {
                    r.prompt[j] = (i * 3 + j) % model.vocab();
                }
                r.max_new_tokens = std::max(1, max_new);
                r.temperature = 0.f;
                reqs.push_back(r);
            }
            engine.generate(reqs);  // upload + one profiled pass
            auto p = engine.device_profile();
            auto pct = [&](float ms) {
                return p.total_ms > 0.f ? (100.f * ms / p.total_ms) : 0.f;
            };
            std::cout << "PROFILE {\n"
                      << "  \"gemm\":\"" << p.gemm_backend << "\",\n"
                      << "  \"fused\":" << (p.fused ? "true" : "false") << ",\n"
                      << "  \"tokens\":" << p.tokens << ",\"layers\":" << p.layers << ",\n"
                      << "  \"total_ms\":" << p.total_ms << ",\n"
                      << "  \"embed_ms\":" << p.embed_ms << ",\"embed_pct\":" << pct(p.embed_ms) << ",\n"
                      << "  \"gemm_ms\":" << p.gemm_ms << ",\"gemm_pct\":" << pct(p.gemm_ms) << ",\n"
                      << "  \"norm_ms\":" << p.norm_ms << ",\"norm_pct\":" << pct(p.norm_ms) << ",\n"
                      << "  \"rope_ms\":" << p.rope_ms << ",\"rope_pct\":" << pct(p.rope_ms) << ",\n"
                      << "  \"kv_write_ms\":" << p.kv_write_ms << ",\"kv_write_pct\":" << pct(p.kv_write_ms) << ",\n"
                      << "  \"attn_ms\":" << p.attn_ms << ",\"attn_pct\":" << pct(p.attn_ms) << ",\n"
                      << "  \"misc_ms\":" << p.misc_ms << ",\"misc_pct\":" << pct(p.misc_ms) << "\n"
                      << "}\n";
            std::cout << "\n  op           ms      %\n"
                      << "  ---------- ------ ------\n";
            auto row = [&](const char* name, float ms) {
                std::cout << "  " << name;
                for (int i = static_cast<int>(std::strlen(name)); i < 10; ++i) std::cout << ' ';
                std::cout << ' ' << ms << "  " << pct(ms) << "\n";
            };
            row("embed", p.embed_ms);
            row("gemm", p.gemm_ms);
            row("norm", p.norm_ms);
            row("rope", p.rope_ms);
            row("kv_write", p.kv_write_ms);
            row("attn", p.attn_ms);
            row("misc", p.misc_ms);
            row("TOTAL", p.total_ms);
            return 0;
        }
        if (cmd == "speculate") {
            if (draft_path.empty()) throw std::runtime_error("speculate needs --draft");
            if (token_str.empty()) throw std::runtime_error("speculate needs --tokens");
            inferno::Model draft = inferno::Model::load(draft_path);
            inferno::SpeculativeConfig scfg;
            scfg.num_blocks = ecfg.num_blocks;
            scfg.block_size = ecfg.block_size;
            scfg.use_cuda = ecfg.use_cuda;
            scfg.cuda_graphs = false;  // shape changes each verify step
            scfg.use_cublas = ecfg.use_cublas;
            scfg.fuse_kernels = ecfg.fuse_kernels;
            auto prompt = parse_tokens(token_str);
            auto stats = inferno::speculative_benchmark(model, draft, prompt, max_new, gamma, scfg);
            const double accept =
                stats.drafted > 0 ? (100.0 * stats.accepted_draft / stats.drafted) : 0.0;
            const double speedup =
                stats.wall_ms > 0.0 ? (stats.target_only_ms / stats.wall_ms) : 0.0;
            std::cout << "SPECULATE {\n"
                      << "  \"gamma\":" << gamma << ",\n"
                      << "  \"drafted\":" << stats.drafted << ",\n"
                      << "  \"accepted_draft\":" << stats.accepted_draft << ",\n"
                      << "  \"bonus\":" << stats.bonus << ",\n"
                      << "  \"accept_pct\":" << accept << ",\n"
                      << "  \"spec_ms\":" << stats.wall_ms << ",\n"
                      << "  \"target_only_ms\":" << stats.target_only_ms << ",\n"
                      << "  \"speedup\":" << speedup << ",\n"
                      << "  \"new_tokens\":" << (stats.tokens.size() - prompt.size()) << ",\n"
                      << "  \"device\":\"" << (scfg.use_cuda ? "gpu" : "cpu") << "\"\n"
                      << "}\n";
            for (std::size_t i = 0; i < stats.tokens.size(); ++i) {
                if (i) std::cout << ' ';
                std::cout << stats.tokens[i];
            }
            std::cout << "\n";
            return 0;
        }
        if (cmd == "bench-latency") {
            if (token_str.empty()) throw std::runtime_error("bench-latency needs --tokens");
            inferno::Request req;
            req.prompt = parse_tokens(token_str);
            req.max_new_tokens = max_new;
            req.temperature = 0.f;
            engine.generate({req});  // warmup: upload + graphs

            inferno::Request r1 = req;
            r1.max_new_tokens = 1;
            auto t0 = std::chrono::steady_clock::now();
            engine.generate({r1});
            double ttft_ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

            t0 = std::chrono::steady_clock::now();
            auto out = engine.generate({req});
            double wall_ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            int gen = out.empty() ? 0 : out[0].generated;
            double itl = gen > 1 ? (wall_ms - ttft_ms) / (gen - 1) : 0.0;
            size_t used = 0, total = 0;
            double mem_mb = 0;
            if (inferno::cuda::device_mem(used, total)) mem_mb = used / (1024.0 * 1024.0);
            std::cout << "LATENCY "
                      << "{\"wall_ms\":" << wall_ms << ",\"ttft_ms\":" << ttft_ms << ",\"itl_ms\":" << itl
                      << ",\"new_tokens\":" << gen << ",\"tok_s\":"
                      << (wall_ms > 0 ? 1000.0 * gen / wall_ms : 0.0) << ",\"gpu_mem_mb\":" << mem_mb
                      << ",\"fuse\":" << (ecfg.fuse_kernels ? "true" : "false") << ",\"device\":\""
                      << (ecfg.use_cuda ? "gpu" : "cpu") << "\"}\n";
            return 0;
        }
        if (cmd == "bench-prefix") {
            if (system_token_str.empty() || user_token_str.empty()) {
                throw std::runtime_error("bench-prefix needs --system-tokens and --user-tokens");
            }
            if (seqs < 1) throw std::runtime_error("seqs must be >= 1");
            auto system = parse_tokens(system_token_str);
            auto user = parse_tokens(user_token_str);
            // Align system length down to block size so the shared span is exact.
            const int bs = ecfg.block_size;
            const int shared = (static_cast<int>(system.size()) / bs) * bs;
            if (shared < bs) throw std::runtime_error("system prompt must cover at least one KV block");
            system.resize(static_cast<std::size_t>(shared));

            auto make_batch = [&](bool) {
                std::vector<inferno::Request> reqs;
                for (int i = 0; i < seqs; ++i) {
                    inferno::Request r;
                    r.id = i;
                    r.prompt = system;
                    // Unique user suffix per sequence (mutate last user token).
                    auto u = user;
                    if (!u.empty()) u.back() = (u.back() + i) % std::max(1, model.vocab());
                    r.prompt.insert(r.prompt.end(), u.begin(), u.end());
                    r.prefix_len = shared;
                    r.max_new_tokens = std::max(1, max_new);
                    r.temperature = 0.f;
                    r.seed = static_cast<std::uint64_t>(i + 1);
                    reqs.push_back(std::move(r));
                }
                return reqs;
            };

            // Measure TTFT (max_new=1) and full wall for the same concurrent batch.
            // Warm path: first generate fills / retains the shared system prefix pages;
            // the timed generate should hit for all N seqs when cache_on.
            auto run = [&](bool cache_on) {
                inferno::EngineConfig cfg = ecfg;
                cfg.prefix_caching = cache_on;
                cfg.retain_prefix_cache = true;
                cfg.profile_cuda = false;
                cfg.cuda_graphs = false;  // shape varies; keep timing honest
                if (cfg.num_blocks < 512) cfg.num_blocks = 512;
                if (cfg.max_seqs < seqs) cfg.max_seqs = seqs;
                inferno::Engine eng(model, cfg);
                auto batch = make_batch(cache_on);
                eng.generate(batch);  // warmup (+ fill prefix when cache_on)

                auto ttft_batch = make_batch(cache_on);
                for (auto& r : ttft_batch) r.max_new_tokens = 1;
                auto t0 = std::chrono::steady_clock::now();
                eng.generate(ttft_batch);
                double ttft_ms =
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

                batch = make_batch(cache_on);
                t0 = std::chrono::steady_clock::now();
                auto out = eng.generate(batch);
                double wall_ms =
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                int gen = 0;
                for (const auto& r : out) gen += r.generated;
                auto m = eng.metrics();
                return std::tuple<double, double, int, inferno::Metrics>(ttft_ms, wall_ms, gen, m);
            };

            auto [ttft_off, wall_off, gen_off, m_off] = run(false);
            auto [ttft_on, wall_on, gen_on, m_on] = run(true);
            const double ttft_speedup = ttft_on > 0.0 ? ttft_off / ttft_on : 0.0;
            const double wall_speedup = wall_on > 0.0 ? wall_off / wall_on : 0.0;
            std::cout << "PREFIX {\n"
                      << "  \"seqs\":" << seqs << ",\n"
                      << "  \"system_tokens\":" << shared << ",\n"
                      << "  \"user_tokens\":" << user.size() << ",\n"
                      << "  \"max_new\":" << std::max(1, max_new) << ",\n"
                      << "  \"ttft_off_ms\":" << ttft_off << ",\n"
                      << "  \"ttft_on_ms\":" << ttft_on << ",\n"
                      << "  \"ttft_speedup\":" << ttft_speedup << ",\n"
                      << "  \"wall_off_ms\":" << wall_off << ",\n"
                      << "  \"wall_on_ms\":" << wall_on << ",\n"
                      << "  \"wall_speedup\":" << wall_speedup << ",\n"
                      << "  \"off_tok_s\":" << (wall_off > 0 ? 1000.0 * gen_off / wall_off : 0.0) << ",\n"
                      << "  \"on_tok_s\":" << (wall_on > 0 ? 1000.0 * gen_on / wall_on : 0.0) << ",\n"
                      << "  \"prefix_hits\":" << m_on.prefix_hits << ",\n"
                      << "  \"prefix_misses\":" << m_on.prefix_misses << ",\n"
                      << "  \"prefix_tokens_saved\":" << m_on.prefix_tokens_saved << ",\n"
                      << "  \"device\":\"" << (ecfg.use_cuda ? "gpu" : "cpu") << "\"\n"
                      << "}\n";
            return 0;
        }
        if (cmd == "serve") {
            engine.start();
            inferno::HttpServer server(engine, port);
            server.start();
            std::cout << "Inferno listening on http://127.0.0.1:" << server.port()
                      << (ecfg.use_cuda ? " (gpu)" : " (cpu)") << "\n"
                      << "POST /v1/completions  {\"token_ids\":[1,2,3],\"max_tokens\":16}\n";
            std::cout << "press enter to stop\n";
            std::string line;
            std::getline(std::cin, line);
            server.stop();
            engine.stop();
            return 0;
        }
        usage();
        return 1;
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << "\n";
        return 1;
    }
}
