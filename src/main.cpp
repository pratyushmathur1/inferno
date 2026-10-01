#include "inferno/bench.hpp"
#include "inferno/engine.hpp"
#include "inferno/model.hpp"
#include "inferno/server.hpp"

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
        << "  inferno init-model -o model.bin [--seed N] [--layers N] [--hidden N] [--vocab N]\n"
        << "  inferno generate -m model.bin --tokens 1,2,3 --max-new 16 [--temp 0]\n"
        << "  inferno serve -m model.bin [--port 8000]\n";
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
                      << " tokens_processed " << engine.metrics().tokens_processed << "\n";
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
        float temp = 0.f;
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
            else throw std::runtime_error("unknown flag " + a);
        }
        if (cmd == "init-model") {
            auto cfg = tiny();
            cfg.vocab = vocab;
            cfg.hidden = hidden;
            cfg.n_layers = layers;
            cfg.n_heads = std::max(1, hidden / cfg.head_dim);
            if (hidden % cfg.head_dim != 0) throw std::runtime_error("hidden must be a multiple of 16");
            cfg.n_kv_heads = std::max(1, cfg.n_heads / 2);
            if (cfg.n_heads % cfg.n_kv_heads != 0) cfg.n_kv_heads = cfg.n_heads;
            cfg.intermediate = hidden * 2;
            inferno::Model model = inferno::Model::init_random(cfg, static_cast<std::uint32_t>(seed));
            model.save(out_path);
            std::cout << "wrote " << out_path << "\n";
            return 0;
        }
        if (model_path.empty()) throw std::runtime_error("pass -m model.bin");
        inferno::Model model = inferno::Model::load(model_path);
        inferno::EngineConfig ecfg;
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
        if (cmd == "serve") {
            engine.start();
            inferno::HttpServer server(engine, port);
            server.start();
            std::cout << "Inferno listening on http://127.0.0.1:" << server.port() << "\n"
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
