#include "inferno/bench.hpp"
#include "inferno/cuda_api.hpp"
#include "inferno/engine.hpp"
#include "inferno/model.hpp"
#include "inferno/server.hpp"

#include <chrono>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void usage() {
    std::cerr
        << "Inferno — a paged, continuously batched LLM inference engine\n\n"
        << "  inferno demo\n"
        << "  inferno bench\n"
        << "  inferno init-model -o model.bin [--seed N] [--layers N] [--hidden N] [--vocab N] [--head-dim N]\n"
        << "  inferno generate -m model.bin --tokens 1,2,3 --max-new 16 [--temp 0] [--cpu]\n"
        << "  inferno compare-workload -m model.bin [--seqs 8] [--prompt 12] [--max-new 12] [--cpu]\n"
        << "  inferno serve -m model.bin [--port 8000] [--cpu]\n";
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
        std::string token_str;
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
        float temp = 0.f;
        bool want_cuda = inferno::cuda::available();
        bool graphs = true;
        for (int i = 2; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "-m" || a == "--model") model_path = need(i, argc, argv, a);
            else if (a == "-o" || a == "--out") out_path = need(i, argc, argv, a);
            else if (a == "--tokens") token_str = need(i, argc, argv, a);
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
            else if (a == "--cpu") want_cuda = false;
            else if (a == "--cuda") want_cuda = true;
            else if (a == "--no-graphs") graphs = false;
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
        ecfg.cuda_graphs = want_cuda && graphs;
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
