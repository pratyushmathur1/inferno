#include "inferno/cpu_kernels.hpp"
#include "inferno/cuda_api.hpp"
#include "inferno/engine.hpp"
#include "inferno/model.hpp"
#include "inferno/paged_cache.hpp"
#include "inferno/server.hpp"
#include "inferno/speculative.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

int g_fails = 0;

void check(bool cond, const char* what) {
    if (!cond) {
        std::cerr << "FAIL " << what << "\n";
        ++g_fails;
    }
}

float max_abs(const std::vector<float>& a, const std::vector<float>& b) {
    float m = 0.f;
    for (std::size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

inferno::ModelConfig small_cfg() {
    inferno::ModelConfig cfg;
    cfg.vocab = 48;
    cfg.hidden = 32;
    cfg.n_layers = 2;
    cfg.n_heads = 4;
    cfg.n_kv_heads = 2;
    cfg.head_dim = 8;
    cfg.intermediate = 48;
    return cfg;
}

inferno::EngineConfig roomy(int batch_tokens = 128) {
    inferno::EngineConfig cfg;
    cfg.block_size = 4;
    cfg.num_blocks = 128;
    cfg.max_batched_tokens = batch_tokens;
    cfg.max_seqs = 8;
    return cfg;
}

std::vector<int> prompt_of(int n, int vocab, int salt) {
    std::vector<int> p(n);
    for (int i = 0; i < n; ++i) p[i] = (salt * 3 + i * 5 + 1) % vocab;
    return p;
}

const inferno::GenerationResult* by_id(const std::vector<inferno::GenerationResult>& rs, int id) {
    for (const auto& r : rs) {
        if (r.id == id) return &r;
    }
    return nullptr;
}

void test_kernels() {
    float x[4] = {1.f, 2.f, 3.f, 4.f};
    float w[4] = {1.f, 1.f, 1.f, 1.f};
    float y[4];
    inferno::kernels::rmsnorm(x, w, y, 1, 4, 1e-6f);
    float ss = (1.f + 4.f + 9.f + 16.f) / 4.f;
    float inv = 1.f / std::sqrt(ss + 1e-6f);
    check(std::fabs(y[0] - inv) < 1e-5f, "rmsnorm row 0");
    check(std::fabs(y[3] - 4.f * inv) < 1e-5f, "rmsnorm row 3");

    float a[] = {1, 2, 3, 4, 5, 6};
    float b[] = {1, 0, 0, 0, 1, 0};
    float c[4] = {};
    inferno::kernels::linear(a, b, c, 2, 2, 3);
    check(c[0] == 1.f && c[1] == 2.f && c[2] == 4.f && c[3] == 5.f, "linear 2x2");
    float c2[4] = {};
    inferno::kernels::linear_tiled(a, b, c2, 2, 2, 3, 2);
    check(max_abs({c[0], c[1], c[2], c[3]}, {c2[0], c2[1], c2[2], c2[3]}) < 1e-5f, "tiled linear");

    float q[8] = {0.1f, 0.2f, 0.3f, 0.4f, -0.2f, 0.5f, 0.1f, -0.7f};
    float k[8];
    std::memcpy(k, q, sizeof(q));
    float q0[8];
    std::memcpy(q0, q, sizeof(q));
    int pos0 = 0;
    inferno::kernels::apply_rope(q, k, &pos0, 1, 2, 2, 4, 10000.f);
    check(max_abs(std::vector<float>(q, q + 8), std::vector<float>(q0, q0 + 8)) < 1e-6f, "rope at position 0");

    float logits[] = {1.f, 2.f, 3.f};
    float probs[3];
    inferno::kernels::softmax(logits, probs, 1, 3);
    check(std::fabs((probs[0] + probs[1] + probs[2]) - 1.f) < 1e-5f, "softmax sums to 1");
    check(probs[2] > probs[1] && probs[1] > probs[0], "softmax order");

    const int T = 5, H = 4, KV = 2, D = 8;
    std::vector<float> qq(T * H * D), kk(T * KV * D), vv(T * KV * D), o1(T * H * D), o2(T * H * D);
    for (std::size_t i = 0; i < qq.size(); ++i) qq[i] = std::sin(0.17f * static_cast<float>(i));
    for (std::size_t i = 0; i < kk.size(); ++i) kk[i] = std::cos(0.13f * static_cast<float>(i));
    for (std::size_t i = 0; i < vv.size(); ++i) vv[i] = 0.01f * static_cast<float>(static_cast<int>(i % 11) - 5);
    inferno::kernels::causal_attention_naive(qq.data(), kk.data(), vv.data(), o1.data(), T, H, KV, D);
    inferno::kernels::causal_attention_online(qq.data(), kk.data(), vv.data(), o2.data(), T, H, KV, D);
    check(max_abs(o1, o2) < 1e-4f, "flash online matches full softmax");

    // A single token attends only to itself, so the output is V (GQA repeats).
    float q1[] = {1.f, 0.f, 0.f, 1.f};
    float k1[] = {0.f, 1.f, 1.f, 0.f};
    float v1[] = {0.25f, -0.5f, 0.75f, 1.5f};
    float o[4] = {};
    inferno::kernels::causal_attention_online(q1, k1, v1, o, 1, 2, 2, 2);
    check(std::fabs(o[0] - 0.25f) < 1e-5f && std::fabs(o[1] - -0.5f) < 1e-5f, "single token attention is V");
}

void test_paged_matches_dense() {
    const int T = 6, H = 4, KV = 2, D = 4, block = 4, pages = 8;
    std::vector<float> q(T * H * D), k(T * KV * D), v(T * KV * D), dense(T * H * D), paged(T * H * D);
    for (std::size_t i = 0; i < q.size(); ++i) q[i] = std::sin(0.07f * i);
    for (std::size_t i = 0; i < k.size(); ++i) k[i] = std::cos(0.05f * i);
    for (std::size_t i = 0; i < v.size(); ++i) v[i] = std::sin(0.03f * i + 1.f);
    inferno::kernels::causal_attention_online(q.data(), k.data(), v.data(), dense.data(), T, H, KV, D);

    inferno::PagedCache cache(1, KV, D, pages, block);
    std::vector<int> table;
    int logical = (T + block - 1) / block;
    for (int i = 0; i < logical; ++i) table.push_back(cache.alloc_block());
    cache.set_block_table(3, table);
    for (int t = 0; t < T; ++t) {
        for (int h = 0; h < KV; ++h) {
            cache.write_kv(0, 3, t, h, k.data() + (t * KV + h) * D, v.data() + (t * KV + h) * D);
        }
    }
    std::vector<int> seqs(T, 3), pos(T);
    for (int t = 0; t < T; ++t) pos[t] = t;
    cache.attention(q.data(), seqs.data(), pos.data(), T, 0, H, KV, D, paged.data());
    check(max_abs(dense, paged) == 0.f, "paged flash matches dense online bitwise");
}

void test_quant_and_parallel() {
    float w[] = {1.f, -2.f};
    std::int8_t wq[2];
    float scales[1];
    inferno::kernels::quantize_weight_int8(w, 1, 2, wq, scales);
    check(scales[0] == 2.f / 127.f, "int8 scale");
    check(wq[0] == 64 && wq[1] == -127, "int8 codes");
    float x[] = {1.f, 1.f};
    float y[1];
    inferno::kernels::linear_int8(x, wq, scales, y, 1, 1, 2);
    check(std::fabs(y[0] - (-126.f / 127.f)) < 1e-5f, "int8 gemm hand value");

    check(inferno::kernels::fp32_to_e4m3(1.f) == 0x38, "e4m3 1");
    check(inferno::kernels::fp32_to_e4m3(2.f) == 0x40, "e4m3 2");
    check(inferno::kernels::fp32_to_e4m3(0.5f) == 0x30, "e4m3 0.5");
    check(inferno::kernels::fp32_to_e4m3(-1.f) == static_cast<std::uint8_t>(0x80 | 0x38), "e4m3 -1");
    check(std::fabs(inferno::kernels::e4m3_to_fp32(0x38) - 1.f) < 1e-6f, "e4m3 decode 1");
    check(std::fabs(inferno::kernels::e4m3_to_fp32(0x40) - 2.f) < 1e-6f, "e4m3 decode 2");

    float wf[] = {1.f, -0.5f, 0.25f, 2.f};
    std::uint8_t w8[4];
    float sc[2];
    inferno::kernels::quantize_weight_fp8(wf, 2, 2, w8, sc);
    float xf[] = {0.5f, 1.f};
    float y8[2];
    inferno::kernels::linear_fp8(xf, w8, sc, y8, 1, 2, 2);
    float yref[2] = {0.f, 0.f};
    for (int n = 0; n < 2; ++n) {
        for (int k = 0; k < 2; ++k) {
            yref[n] += xf[k] * inferno::kernels::e4m3_to_fp32(w8[n * 2 + k]) * sc[n];
        }
    }
    check(std::fabs(y8[0] - yref[0]) < 1e-6f && std::fabs(y8[1] - yref[1]) < 1e-6f, "fp8 gemm");

    float X[] = {1.f, 2.f, 3.f, 4.f, 0.5f, -1.f};
    float W[] = {1, 0, 0, 0, 1, 0, 0, 0, 1, 2, 2, 2};
    // M=2 K=3 N=4
    float full[8];
    inferno::kernels::linear(X, W, full, 2, 4, 3);
    float col[8] = {};
    inferno::kernels::linear_column_rank(X, W, col, 2, 4, 3, 0, 2);
    inferno::kernels::linear_column_rank(X, W, col, 2, 4, 3, 1, 2);
    check(max_abs(std::vector<float>(full, full + 8), std::vector<float>(col, col + 8)) == 0.f,
          "column parallel");
    float parts[24] = {};
    inferno::kernels::linear_row_rank(X, W, parts, 2, 4, 3, 0, 3);
    inferno::kernels::linear_row_rank(X, W, parts, 2, 4, 3, 1, 3);
    inferno::kernels::linear_row_rank(X, W, parts, 2, 4, 3, 2, 3);
    float reduced[8];
    inferno::kernels::allreduce_sum(parts, reduced, 2, 4, 3);
    check(max_abs(std::vector<float>(full, full + 8), std::vector<float>(reduced, reduced + 8)) < 1e-5f,
          "row parallel allreduce");
}

std::vector<inferno::GenerationResult> gen(const inferno::Model& model, const inferno::EngineConfig& cfg,
                                           const std::vector<inferno::Request>& reqs) {
    inferno::Engine engine(model, cfg);
    return engine.generate(reqs);
}

void test_scheduler() {
    inferno::Model model = inferno::Model::init_random(small_cfg(), 11);
    const std::string path = "/tmp/inferno-test-model.bin";
    model.save(path);
    inferno::Model loaded = inferno::Model::load(path);
    auto a = gen(model, roomy(), {{0, prompt_of(7, model.vocab(), 1), 5, 0.f, 1, 1}});
    auto b = gen(loaded, roomy(), {{0, prompt_of(7, model.vocab(), 1), 5, 0.f, 1, 1}});
    check(a[0].tokens == b[0].tokens, "save/load generates the same tokens");
    check(a[0].generated == 5, "generated count");
    check(static_cast<int>(a[0].tokens.size()) == 12, "prompt plus new tokens");

    auto wide = gen(model, roomy(256), {
        {1, prompt_of(9, model.vocab(), 2), 4, 0.f, 1, 1},
        {2, prompt_of(5, model.vocab(), 3), 6, 0.f, 1, 1},
        {3, prompt_of(13, model.vocab(), 4), 3, 0.f, 1, 1},
    });
    auto chunked = gen(model, roomy(3), {
        {1, prompt_of(9, model.vocab(), 2), 4, 0.f, 1, 1},
        {2, prompt_of(5, model.vocab(), 3), 6, 0.f, 1, 1},
        {3, prompt_of(13, model.vocab(), 4), 3, 0.f, 1, 1},
    });
    for (int id : {1, 2, 3}) {
        std::vector<int> original = id == 1 ? prompt_of(9, model.vocab(), 2)
                                  : id == 2 ? prompt_of(5, model.vocab(), 3)
                                            : prompt_of(13, model.vocab(), 4);
        int max_new = id == 1 ? 4 : id == 2 ? 6 : 3;
        auto alone = gen(model, roomy(), {{id, original, max_new, 0.f, 1, 1}});
        check(alone[0].tokens == by_id(wide, id)->tokens, "batched sequences match serial decode");
    }

    inferno::EngineConfig tight = roomy();
    tight.block_size = 4;
    tight.num_blocks = 8;
    tight.max_batched_tokens = 64;
    inferno::Request first{10, prompt_of(4, model.vocab(), 8), 28, 0.f, 1, 1};
    inferno::Request second{11, prompt_of(4, model.vocab(), 9), 28, 0.f, 1, 1};
    inferno::Engine shared(model, tight);
    auto both = shared.generate({first, second});
    check(shared.metrics().preemptions > 0, "tight KV pool preempts the newer sequence");
    auto s1 = gen(model, roomy(), {first});
    auto s2 = gen(model, roomy(), {second});
    check(by_id(both, 10)->tokens == s1[0].tokens, "preempted batch matches serial A");
    check(by_id(both, 11)->tokens == s2[0].tokens, "preempted batch matches serial B");

    inferno::Request zero{1, prompt_of(4, model.vocab(), 1), 0, 0.f, 1, 1};
    auto z = gen(model, roomy(), {zero});
    check(z[0].generated == 0 && z[0].tokens == zero.prompt, "max_new 0 returns the prompt");

    inferno::Request hot{1, prompt_of(3, model.vocab(), 2), 4, 0.8f, 8, 99};
    auto sample = gen(model, roomy(), {hot});
    check(sample[0].ok && sample[0].generated == 4, "temperature sampling returns max_new tokens");
}

void test_speculative() {
    inferno::Model target = inferno::Model::init_random(small_cfg(), 3);
    inferno::Model draft = inferno::Model::init_random(small_cfg(), 9);
    std::vector<int> prompt = prompt_of(6, target.vocab(), 4);
    auto greedy = gen(target, roomy(), {{1, prompt, 10, 0.f, 1, 1}});
    auto spec = inferno::speculative_generate(target, draft, prompt, 10, 4, {64, 8});
    check(spec.tokens == greedy[0].tokens, "speculative greedy matches target-only greedy");
    check(spec.drafted > 0, "draft model proposed tokens");
}

std::string http_exchange(int port, const std::string& req) {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        ::close(fd);
        return {};
    }
    ::send(fd, req.data(), req.size(), MSG_NOSIGNAL);
    std::string out;
    char buf[1024];
    while (true) {
        ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
        if (n <= 0) break;
        out.append(buf, buf + n);
    }
    ::close(fd);
    return out;
}

void test_http_and_submit() {
    inferno::Model model = inferno::Model::init_random(small_cfg(), 21);
    inferno::Engine engine(model, roomy());
    auto body = std::string("{\"token_ids\":[1,2,3,4],\"max_tokens\":3,\"temperature\":0}");
    auto parsed = inferno::run_completion_json(engine, body);
    check(parsed.generated == 3, "json completion length");

    engine.start();
    inferno::HttpServer server(engine, 0);
    server.start();
    std::string health = http_exchange(server.port(), "GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n");
    check(health.find("200") != std::string::npos && health.find("ok") != std::string::npos, "GET /health");

    std::string payload = "{\"token_ids\":[4,5,6,7,8],\"max_tokens\":3,\"temperature\":0}";
    std::ostringstream req;
    req << "POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n"
        << "Content-Length: " << payload.size() << "\r\nConnection: close\r\n\r\n" << payload;
    std::string resp = http_exchange(server.port(), req.str());
    check(resp.find("200") != std::string::npos, "POST status");
    check(resp.find("\"generated\":3") != std::string::npos, "POST generated field");
    server.stop();
    engine.stop();

    inferno::Engine live(model, roomy());
    live.start();
    inferno::Request a{0, prompt_of(4, model.vocab(), 1), 3, 0.f, 1, 1};
    inferno::Request b{0, prompt_of(6, model.vocab(), 2), 3, 0.f, 1, 1};
    auto fa = live.submit(a);
    auto fb = live.submit(b);
    auto ra = fa.get();
    auto rb = fb.get();
    live.stop();
    auto ea = gen(model, roomy(), {a});
    b.id = 0;
    auto eb = gen(model, roomy(), {b});
    check(ra.tokens == ea[0].tokens && rb.tokens == eb[0].tokens, "live submit matches generate");

    if (inferno::cuda::available()) {
        inferno::EngineConfig gpu = roomy();
        gpu.use_cuda = true;
        gpu.cuda_graphs = true;
        inferno::Request req{7, prompt_of(4, model.vocab(), 3), 4, 0.f, 1, 1};
        auto out = gen(model, gpu, {req});
        check(out.size() == 1 && out[0].ok && out[0].generated == 4, "gpu generate returns four tokens");
    }
}

}  // namespace

int main() {
    try {
        test_kernels();
        test_paged_matches_dense();
        test_quant_and_parallel();
        test_scheduler();
        test_speculative();
        test_http_and_submit();
    } catch (const std::exception& ex) {
        std::cerr << "FAIL exception: " << ex.what() << "\n";
        return 1;
    }
    if (g_fails) {
        std::cerr << g_fails << " checks failed\n";
        return 1;
    }
    std::cout << "all inferno tests passed\n";
    return 0;
}
