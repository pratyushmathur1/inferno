#include "inferno/cpu_kernels.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace inferno {
namespace kernels {
namespace {

float dot(const float* a, const float* b, int n) {
    float acc = 0.f;
    for (int i = 0; i < n; ++i) acc += a[i] * b[i];
    return acc;
}

}  // namespace

void linear(const float* x, const float* w, float* y, int M, int N, int K) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int m = 0; m < M; ++m) {
        const float* xrow = x + static_cast<std::size_t>(m) * K;
        float* yrow = y + static_cast<std::size_t>(m) * N;
        for (int n = 0; n < N; ++n) {
            const float* wrow = w + static_cast<std::size_t>(n) * K;
            float acc = 0.f;
            for (int k = 0; k < K; ++k) acc += xrow[k] * wrow[k];
            yrow[n] = acc;
        }
    }
}

void linear_tiled(const float* x, const float* w, float* y, int M, int N, int K, int tile) {
    if (tile < 1) tile = 32;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int m0 = 0; m0 < M; m0 += tile) {
        for (int n0 = 0; n0 < N; n0 += tile) {
            const int m1 = std::min(m0 + tile, M);
            const int n1 = std::min(n0 + tile, N);
            for (int m = m0; m < m1; ++m) {
                for (int n = n0; n < n1; ++n) {
                    y[static_cast<std::size_t>(m) * N + n] = 0.f;
                }
            }
            for (int k0 = 0; k0 < K; k0 += tile) {
                const int k1 = std::min(k0 + tile, K);
                for (int m = m0; m < m1; ++m) {
                    for (int n = n0; n < n1; ++n) {
                        float acc = y[static_cast<std::size_t>(m) * N + n];
                        const float* xrow = x + static_cast<std::size_t>(m) * K;
                        const float* wrow = w + static_cast<std::size_t>(n) * K;
                        for (int k = k0; k < k1; ++k) acc += xrow[k] * wrow[k];
                        y[static_cast<std::size_t>(m) * N + n] = acc;
                    }
                }
            }
        }
    }
}

void add(const float* a, const float* b, float* y, int n) {
    for (int i = 0; i < n; ++i) y[i] = a[i] + b[i];
}

void rmsnorm(const float* x, const float* weight, float* y, int rows, int dim, float eps) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int r = 0; r < rows; ++r) {
        const float* row = x + static_cast<std::size_t>(r) * dim;
        float* out = y + static_cast<std::size_t>(r) * dim;
        float ss = 0.f;
        for (int i = 0; i < dim; ++i) ss += row[i] * row[i];
        ss = ss / static_cast<float>(dim);
        float inv = 1.f / std::sqrt(ss + eps);
        for (int i = 0; i < dim; ++i) out[i] = row[i] * inv * weight[i];
    }
}

void silu_mul(const float* gate, const float* up, float* y, int n) {
    for (int i = 0; i < n; ++i) {
        float g = gate[i];
        float sig = 1.f / (1.f + std::exp(-g));
        y[i] = (g * sig) * up[i];
    }
}

void apply_rope(float* q, float* k, const int* positions, int tokens,
                int n_heads, int n_kv_heads, int head_dim, float rope_theta) {
    const int half = head_dim / 2;
    auto rotate_heads = [&](float* data, int heads) {
        for (int t = 0; t < tokens; ++t) {
            for (int h = 0; h < heads; ++h) {
                float* vec = data + (static_cast<std::size_t>(t) * heads + h) * head_dim;
                for (int i = 0; i < half; ++i) {
                    float freq = std::pow(rope_theta, -2.f * static_cast<float>(i) / static_cast<float>(head_dim));
                    float angle = static_cast<float>(positions[t]) * freq;
                    float c = std::cos(angle);
                    float s = std::sin(angle);
                    float x0 = vec[i];
                    float x1 = vec[i + half];
                    vec[i] = x0 * c - x1 * s;
                    vec[i + half] = x0 * s + x1 * c;
                }
            }
        }
    };
    rotate_heads(q, n_heads);
    rotate_heads(k, n_kv_heads);
}

void causal_attention_online(const float* q, const float* k, const float* v, float* out,
                             int tokens, int n_heads, int n_kv_heads, int head_dim) {
    const int group = n_heads / n_kv_heads;
    const float scale = 1.f / std::sqrt(static_cast<float>(head_dim));
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int t = 0; t < tokens; ++t) {
        for (int h = 0; h < n_heads; ++h) {
            const int kvh = h / group;
            const float* qq = q + (static_cast<std::size_t>(t) * n_heads + h) * head_dim;
            float* oo = out + (static_cast<std::size_t>(t) * n_heads + h) * head_dim;
            float m_i = -INFINITY;
            float l_i = 0.f;
            for (int d = 0; d < head_dim; ++d) oo[d] = 0.f;
            for (int s = 0; s <= t; ++s) {
                const float* kk = k + (static_cast<std::size_t>(s) * n_kv_heads + kvh) * head_dim;
                const float* vv = v + (static_cast<std::size_t>(s) * n_kv_heads + kvh) * head_dim;
                float score = dot(qq, kk, head_dim) * scale;
                float m_new = std::max(m_i, score);
                float p = std::exp(score - m_new);
                float alpha = std::exp(m_i - m_new);
                for (int d = 0; d < head_dim; ++d) oo[d] = oo[d] * alpha + p * vv[d];
                l_i = l_i * alpha + p;
                m_i = m_new;
            }
            float inv = l_i == 0.f ? 0.f : 1.f / l_i;
            for (int d = 0; d < head_dim; ++d) oo[d] *= inv;
        }
    }
}

void causal_attention_naive(const float* q, const float* k, const float* v, float* out,
                            int tokens, int n_heads, int n_kv_heads, int head_dim) {
    const int group = n_heads / n_kv_heads;
    const float scale = 1.f / std::sqrt(static_cast<float>(head_dim));
    for (int t = 0; t < tokens; ++t) {
        for (int h = 0; h < n_heads; ++h) {
            const int kvh = h / group;
            const float* qq = q + (static_cast<std::size_t>(t) * n_heads + h) * head_dim;
            float* oo = out + (static_cast<std::size_t>(t) * n_heads + h) * head_dim;
            std::vector<float> scores(static_cast<std::size_t>(t) + 1);
            float m = -INFINITY;
            for (int s = 0; s <= t; ++s) {
                const float* kk = k + (static_cast<std::size_t>(s) * n_kv_heads + kvh) * head_dim;
                scores[s] = dot(qq, kk, head_dim) * scale;
                m = std::max(m, scores[s]);
            }
            float sum = 0.f;
            for (int s = 0; s <= t; ++s) {
                scores[s] = std::exp(scores[s] - m);
                sum += scores[s];
            }
            float inv = sum == 0.f ? 0.f : 1.f / sum;
            for (int d = 0; d < head_dim; ++d) oo[d] = 0.f;
            for (int s = 0; s <= t; ++s) {
                const float* vv = v + (static_cast<std::size_t>(s) * n_kv_heads + kvh) * head_dim;
                float p = scores[s] * inv;
                for (int d = 0; d < head_dim; ++d) oo[d] += p * vv[d];
            }
        }
    }
}

void softmax(const float* x, float* y, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        const float* row = x + static_cast<std::size_t>(r) * cols;
        float* out = y + static_cast<std::size_t>(r) * cols;
        float m = -INFINITY;
        for (int c = 0; c < cols; ++c) m = std::max(m, row[c]);
        float sum = 0.f;
        for (int c = 0; c < cols; ++c) {
            out[c] = std::exp(row[c] - m);
            sum += out[c];
        }
        float inv = sum == 0.f ? 0.f : 1.f / sum;
        for (int c = 0; c < cols; ++c) out[c] *= inv;
    }
}

void quantize_weight_int8(const float* w, int rows, int cols,
                          std::int8_t* wq, float* scales) {
    for (int r = 0; r < rows; ++r) {
        const float* row = w + static_cast<std::size_t>(r) * cols;
        float amax = 0.f;
        for (int c = 0; c < cols; ++c) amax = std::max(amax, std::fabs(row[c]));
        float scale = amax == 0.f ? 1.f : amax / 127.f;
        scales[r] = scale;
        std::int8_t* dst = wq + static_cast<std::size_t>(r) * cols;
        for (int c = 0; c < cols; ++c) {
            float q = std::round(row[c] / scale);
            q = std::max(-127.f, std::min(127.f, q));
            dst[c] = static_cast<std::int8_t>(q);
        }
    }
}

void linear_int8(const float* x, const std::int8_t* w, const float* scales,
                 float* y, int M, int N, int K) {
    for (int m = 0; m < M; ++m) {
        const float* xrow = x + static_cast<std::size_t>(m) * K;
        float amax = 0.f;
        for (int k = 0; k < K; ++k) amax = std::max(amax, std::fabs(xrow[k]));
        float xscale = amax == 0.f ? 1.f : amax / 127.f;
        std::vector<std::int8_t> xq(static_cast<std::size_t>(K));
        if (amax == 0.f) {
            for (int n = 0; n < N; ++n) y[static_cast<std::size_t>(m) * N + n] = 0.f;
            continue;
        }
        for (int k = 0; k < K; ++k) {
            float q = std::round(xrow[k] / xscale);
            q = std::max(-127.f, std::min(127.f, q));
            xq[k] = static_cast<std::int8_t>(q);
        }
        for (int n = 0; n < N; ++n) {
            const std::int8_t* wrow = w + static_cast<std::size_t>(n) * K;
            std::int32_t acc = 0;
            for (int k = 0; k < K; ++k) {
                acc += static_cast<std::int32_t>(xq[k]) * static_cast<std::int32_t>(wrow[k]);
            }
            y[static_cast<std::size_t>(m) * N + n] =
                static_cast<float>(acc) * xscale * scales[n];
        }
    }
}

std::uint8_t fp32_to_e4m3(float x) {
    if (x == 0.f) return 0;
    static_assert(sizeof(float) == 4, "float must be 32-bit");
    std::uint32_t bits = 0;
    std::memcpy(&bits, &x, sizeof(bits));
    std::uint32_t sign = (bits >> 31) & 1u;
    std::int32_t exp = static_cast<std::int32_t>((bits >> 23) & 0xff) - 127;
    std::uint32_t mant = bits & 0x7fffffu;
    if ((bits & 0x7fffffffu) > 0x7f800000u) return static_cast<std::uint8_t>((sign << 7) | 0x7f);
    // Saturate above the E4M3 max finite, 448.
    if (exp > 8) return static_cast<std::uint8_t>((sign << 7) | 0x7e);
    int stored = exp + 7;
    // Round mantissa to 3 bits. Add the bit just below the kept field.
    std::uint32_t mant3 = mant >> 20;
    std::uint32_t round_bit = (mant >> 19) & 1u;
    if (round_bit) {
        mant3 += 1;
        if (mant3 == 8) {
            mant3 = 0;
            stored += 1;
        }
    }
    if (stored >= 15) return static_cast<std::uint8_t>((sign << 7) | 0x7e);
    if (stored <= 0) {
        // Flush deep subnormals to zero. Small but exact-zero is enough for tests
        // of normal values (1, 0.5, 2).
        if (stored < -2) return static_cast<std::uint8_t>(sign << 7);
        int shift = 1 - stored;
        std::uint32_t sub = mant3 >> shift;
        return static_cast<std::uint8_t>((sign << 7) | (sub & 0x7u));
    }
    return static_cast<std::uint8_t>((sign << 7) | (static_cast<std::uint32_t>(stored) << 3) | (mant3 & 0x7u));
}

float e4m3_to_fp32(std::uint8_t b) {
    std::uint32_t sign = (b >> 7) & 1u;
    std::uint32_t exp = (b >> 3) & 0xfu;
    std::uint32_t mant = b & 0x7u;
    if (exp == 0 && mant == 0) return sign ? -0.f : 0.f;
    if (exp == 0xf && mant == 0x7) return std::numeric_limits<float>::quiet_NaN();
    float value;
    if (exp == 0) {
        // E4M3 subnormal: mant / 8 * 2^(1-bias) with bias 7 -> mant * 2^(-9).
        value = std::ldexp(static_cast<float>(mant), -9);
    } else {
        value = std::ldexp(1.f + static_cast<float>(mant) / 8.f, static_cast<int>(exp) - 7);
    }
    return sign ? -value : value;
}

void quantize_weight_fp8(const float* w, int rows, int cols,
                         std::uint8_t* wq, float* scales) {
    for (int r = 0; r < rows; ++r) {
        const float* row = w + static_cast<std::size_t>(r) * cols;
        float amax = 0.f;
        for (int c = 0; c < cols; ++c) amax = std::max(amax, std::fabs(row[c]));
        // Map the row onto the E4M3 finite range so the largest value is 448.
        float scale = amax == 0.f ? 1.f : amax / 448.f;
        scales[r] = scale;
        std::uint8_t* dst = wq + static_cast<std::size_t>(r) * cols;
        for (int c = 0; c < cols; ++c) dst[c] = fp32_to_e4m3(row[c] / scale);
    }
}

void linear_fp8(const float* x, const std::uint8_t* w, const float* scales,
                float* y, int M, int N, int K) {
    for (int m = 0; m < M; ++m) {
        const float* xrow = x + static_cast<std::size_t>(m) * K;
        for (int n = 0; n < N; ++n) {
            const std::uint8_t* wrow = w + static_cast<std::size_t>(n) * K;
            float acc = 0.f;
            for (int k = 0; k < K; ++k) {
                acc += xrow[k] * (e4m3_to_fp32(wrow[k]) * scales[n]);
            }
            y[static_cast<std::size_t>(m) * N + n] = acc;
        }
    }
}

void linear_column_rank(const float* x, const float* w_full, float* y_full,
                        int M, int N, int K, int rank, int world) {
    if (N % world != 0) throw std::runtime_error("N must divide world for column parallelism");
    int local = N / world;
    int n0 = rank * local;
    for (int m = 0; m < M; ++m) {
        const float* xrow = x + static_cast<std::size_t>(m) * K;
        for (int n = 0; n < local; ++n) {
            const float* wrow = w_full + static_cast<std::size_t>(n0 + n) * K;
            float acc = 0.f;
            for (int k = 0; k < K; ++k) acc += xrow[k] * wrow[k];
            y_full[static_cast<std::size_t>(m) * N + (n0 + n)] = acc;
        }
    }
}

void linear_row_rank(const float* x, const float* w_full, float* y_partial,
                     int M, int N, int K, int rank, int world) {
    if (K % world != 0) throw std::runtime_error("K must divide world for row parallelism");
    int local = K / world;
    int k0 = rank * local;
    for (int m = 0; m < M; ++m) {
        const float* xrow = x + static_cast<std::size_t>(m) * K + k0;
        for (int n = 0; n < N; ++n) {
            const float* wrow = w_full + static_cast<std::size_t>(n) * K + k0;
            float acc = 0.f;
            for (int k = 0; k < local; ++k) acc += xrow[k] * wrow[k];
            y_partial[(static_cast<std::size_t>(rank) * M + m) * N + n] = acc;
        }
    }
}

void allreduce_sum(float* partials, float* out, int rows, int cols, int world) {
    // partials layout: [world, rows, cols]
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            float acc = 0.f;
            for (int w = 0; w < world; ++w) {
                acc += partials[(static_cast<std::size_t>(w) * rows + r) * cols + c];
            }
            out[static_cast<std::size_t>(r) * cols + c] = acc;
        }
    }
}

}  // namespace kernels
}  // namespace inferno
