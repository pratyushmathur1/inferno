#include "inferno/model.hpp"

#include "inferno/cpu_kernels.hpp"
#include "inferno/cuda_api.hpp"

#include <cmath>
#include <fstream>
#include <random>
#include <stdexcept>
#include <utility>

namespace inferno {
namespace {

struct Map {
    Span tok_emb, final_rms, lm_head;
    std::vector<LayerOff> layers;
};

Map build_map(const ModelConfig& cfg) {
    Map m;
    std::int64_t c = 0;
    auto take = [&](std::int64_t n) {
        Span s{c, n};
        c += n;
        return s;
    };
    m.tok_emb = take(static_cast<std::int64_t>(cfg.vocab) * cfg.hidden);
    for (int L = 0; L < cfg.n_layers; ++L) {
        LayerOff o;
        o.rms1 = take(cfg.hidden);
        o.wq = take(static_cast<std::int64_t>(cfg.q_dim()) * cfg.hidden);
        o.wk = take(static_cast<std::int64_t>(cfg.kv_dim()) * cfg.hidden);
        o.wv = take(static_cast<std::int64_t>(cfg.kv_dim()) * cfg.hidden);
        o.wo = take(static_cast<std::int64_t>(cfg.hidden) * cfg.q_dim());
        o.rms2 = take(cfg.hidden);
        o.wgate = take(static_cast<std::int64_t>(cfg.intermediate) * cfg.hidden);
        o.wup = take(static_cast<std::int64_t>(cfg.intermediate) * cfg.hidden);
        o.wdown = take(static_cast<std::int64_t>(cfg.hidden) * cfg.intermediate);
        m.layers.push_back(o);
    }
    m.final_rms = take(cfg.hidden);
    m.lm_head = take(static_cast<std::int64_t>(cfg.vocab) * cfg.hidden);
    return m;
}

void check_config(const ModelConfig& cfg) {
    if (cfg.vocab < 1 || cfg.hidden < 1 || cfg.n_layers < 1) {
        throw std::runtime_error("invalid model config");
    }
    if (cfg.n_heads < 1 || cfg.n_kv_heads < 1 || cfg.n_heads % cfg.n_kv_heads != 0) {
        throw std::runtime_error("head count must be a multiple of kv heads");
    }
    if (cfg.head_dim < 2 || cfg.head_dim % 2 != 0 || cfg.intermediate < 1) {
        throw std::runtime_error("head_dim must be even and intermediate > 0");
    }
}

float normal(std::mt19937& rng) {
    std::uniform_real_distribution<float> uni(1e-6f, 1.f);
    float u1 = uni(rng);
    float u2 = uni(rng);
    return std::sqrt(-2.f * std::log(u1)) * std::cos(6.28318530718f * u2);
}

void write_i32(std::ostream& o, int v) {
    std::int32_t x = v;
    o.write(reinterpret_cast<const char*>(&x), sizeof(x));
}
void write_f32(std::ostream& o, float v) { o.write(reinterpret_cast<const char*>(&v), sizeof(v)); }

int read_i32(std::istream& i) {
    std::int32_t x = 0;
    i.read(reinterpret_cast<char*>(&x), sizeof(x));
    return x;
}
float read_f32(std::istream& i) {
    float v = 0;
    i.read(reinterpret_cast<char*>(&v), sizeof(v));
    return v;
}

}  // namespace

std::int64_t weight_count(const ModelConfig& cfg) { return build_map(cfg).lm_head.off + build_map(cfg).lm_head.n; }

Model::Model(ModelConfig cfg) : cfg_(cfg) {
    check_config(cfg_);
    auto map = build_map(cfg_);
    tok_emb_ = map.tok_emb;
    layers_ = std::move(map.layers);
    final_rms_ = map.final_rms;
    lm_head_ = map.lm_head;
    weights_.assign(static_cast<std::size_t>(lm_head_.off + lm_head_.n), 0.f);
}

const float* Model::ptr(Span s) const { return weights_.data() + s.off; }

Model Model::init_random(ModelConfig cfg, std::uint32_t seed) {
    Model m(cfg);
    std::mt19937 rng(seed);
    for (float& v : m.weights_) v = 0.02f * normal(rng);
    // RMSNorm scales start at 1 so a random net is not tiny from the first layer.
    auto set_one = [&](Span s) {
        for (std::int64_t i = 0; i < s.n; ++i) m.weights_[static_cast<std::size_t>(s.off + i)] = 1.f;
    };
    for (const auto& layer : m.layers_) {
        set_one(layer.rms1);
        set_one(layer.rms2);
    }
    set_one(m.final_rms_);
    return m;
}

void Model::save(const std::string& path) const {
    std::ofstream o(path, std::ios::binary);
    if (!o) throw std::runtime_error("cannot write " + path);
    o.write("INF1", 4);
    write_i32(o, cfg_.vocab);
    write_i32(o, cfg_.hidden);
    write_i32(o, cfg_.n_layers);
    write_i32(o, cfg_.n_heads);
    write_i32(o, cfg_.n_kv_heads);
    write_i32(o, cfg_.head_dim);
    write_i32(o, cfg_.intermediate);
    write_f32(o, cfg_.rms_eps);
    write_f32(o, cfg_.rope_theta);
    write_i32(o, cfg_.eos_id);
    std::int64_t n = static_cast<std::int64_t>(weights_.size());
    o.write(reinterpret_cast<const char*>(&n), sizeof(n));
    o.write(reinterpret_cast<const char*>(weights_.data()), sizeof(float) * weights_.size());
    if (!o) throw std::runtime_error("short write " + path);
}

Model Model::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot read " + path);
    char magic[4] = {};
    in.read(magic, 4);
    if (std::string(magic, 4) != "INF1") throw std::runtime_error("bad inferno model magic");
    ModelConfig cfg;
    cfg.vocab = read_i32(in);
    cfg.hidden = read_i32(in);
    cfg.n_layers = read_i32(in);
    cfg.n_heads = read_i32(in);
    cfg.n_kv_heads = read_i32(in);
    cfg.head_dim = read_i32(in);
    cfg.intermediate = read_i32(in);
    cfg.rms_eps = read_f32(in);
    cfg.rope_theta = read_f32(in);
    cfg.eos_id = read_i32(in);
    Model m(cfg);
    std::int64_t n = 0;
    in.read(reinterpret_cast<char*>(&n), sizeof(n));
    if (n != static_cast<std::int64_t>(m.weights_.size())) throw std::runtime_error("weight size mismatch");
    in.read(reinterpret_cast<char*>(m.weights_.data()), sizeof(float) * m.weights_.size());
    if (!in) throw std::runtime_error("truncated model file");
    return m;
}

void Model::forward(const std::vector<TokenIn>& tokens, PagedCache& cache, std::vector<float>& logits,
                    bool use_gpu, bool cuda_graphs, void** gpu) {
    const int T = static_cast<int>(tokens.size());
    logits.clear();
    if (T == 0) return;
    if (use_gpu) {
        if (!gpu) throw std::runtime_error("GPU forward needs a device slot");
        cuda::llama_forward(cfg_, weights_.data(), static_cast<std::int64_t>(weights_.size()), layers_.data(),
                            tokens, cache, logits, cuda_graphs, *gpu);
        return;
    }
    const int H = cfg_.hidden;
    const int Q = cfg_.q_dim();
    const int Kdim = cfg_.kv_dim();
    const int I = cfg_.intermediate;

    x_.assign(static_cast<std::size_t>(T) * H, 0.f);
    xn_.resize(x_.size());
    q_.assign(static_cast<std::size_t>(T) * Q, 0.f);
    k_.assign(static_cast<std::size_t>(T) * Kdim, 0.f);
    v_.assign(static_cast<std::size_t>(T) * Kdim, 0.f);
    attn_.assign(static_cast<std::size_t>(T) * Q, 0.f);
    proj_.assign(static_cast<std::size_t>(T) * H, 0.f);
    gate_.assign(static_cast<std::size_t>(T) * I, 0.f);
    up_.resize(gate_.size());
    mid_.resize(gate_.size());

    const float* emb = ptr(tok_emb_);
    for (int t = 0; t < T; ++t) {
        int id = tokens[t].token_id;
        if (id < 0 || id >= cfg_.vocab) throw std::runtime_error("token id out of range");
        const float* row = emb + static_cast<std::size_t>(id) * H;
        float* dst = x_.data() + static_cast<std::size_t>(t) * H;
        for (int i = 0; i < H; ++i) dst[i] = row[i];
    }

    std::vector<int> positions(tokens.size());
    std::vector<int> seqs(tokens.size());
    for (int t = 0; t < T; ++t) {
        positions[t] = tokens[t].position;
        seqs[t] = tokens[t].seq_id;
    }

    for (int layer = 0; layer < cfg_.n_layers; ++layer) {
        const LayerOff& off = layers_[layer];
        std::vector<float> residual = x_;
        kernels::rmsnorm(x_.data(), ptr(off.rms1), xn_.data(), T, H, cfg_.rms_eps);
        kernels::linear(xn_.data(), ptr(off.wq), q_.data(), T, Q, H);
        kernels::linear(xn_.data(), ptr(off.wk), k_.data(), T, Kdim, H);
        kernels::linear(xn_.data(), ptr(off.wv), v_.data(), T, Kdim, H);
        kernels::apply_rope(q_.data(), k_.data(), positions.data(), T,
                            cfg_.n_heads, cfg_.n_kv_heads, cfg_.head_dim, cfg_.rope_theta);
        for (int t = 0; t < T; ++t) {
            for (int kv = 0; kv < cfg_.n_kv_heads; ++kv) {
                const float* kp = k_.data() + (static_cast<std::size_t>(t) * cfg_.n_kv_heads + kv) * cfg_.head_dim;
                const float* vp = v_.data() + (static_cast<std::size_t>(t) * cfg_.n_kv_heads + kv) * cfg_.head_dim;
                cache.write_kv(layer, tokens[t].seq_id, tokens[t].position, kv, kp, vp);
            }
        }
        cache.attention(q_.data(), seqs.data(), positions.data(), T, layer,
                        cfg_.n_heads, cfg_.n_kv_heads, cfg_.head_dim, attn_.data());
        kernels::linear(attn_.data(), ptr(off.wo), proj_.data(), T, H, Q);
        kernels::add(residual.data(), proj_.data(), x_.data(), T * H);

        residual = x_;
        kernels::rmsnorm(x_.data(), ptr(off.rms2), xn_.data(), T, H, cfg_.rms_eps);
        kernels::linear(xn_.data(), ptr(off.wgate), gate_.data(), T, I, H);
        kernels::linear(xn_.data(), ptr(off.wup), up_.data(), T, I, H);
        kernels::silu_mul(gate_.data(), up_.data(), mid_.data(), T * I);
        kernels::linear(mid_.data(), ptr(off.wdown), proj_.data(), T, H, I);
        kernels::add(residual.data(), proj_.data(), x_.data(), T * H);
    }

    kernels::rmsnorm(x_.data(), ptr(final_rms_), xn_.data(), T, H, cfg_.rms_eps);

    int emits = 0;
    for (const auto& tok : tokens) emits += tok.emit ? 1 : 0;
    gathered_.assign(static_cast<std::size_t>(emits) * H, 0.f);
    int e = 0;
    for (int t = 0; t < T; ++t) {
        if (!tokens[t].emit) continue;
        const float* src = xn_.data() + static_cast<std::size_t>(t) * H;
        float* dst = gathered_.data() + static_cast<std::size_t>(e) * H;
        for (int i = 0; i < H; ++i) dst[i] = src[i];
        ++e;
    }
    logits.assign(static_cast<std::size_t>(emits) * cfg_.vocab, 0.f);
    if (emits > 0) {
        kernels::linear(gathered_.data(), ptr(lm_head_), logits.data(), emits, cfg_.vocab, H);
    }
}

}  // namespace inferno
