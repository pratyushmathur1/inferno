#pragma once

#include <cstdint>
#include <vector>

namespace inferno {
namespace kernels {

// y[m, n] = sum_k x[m, k] * w[n, k]
void linear(const float* x, const float* w, float* y, int M, int N, int K);

// Blocked GEMM, same layout as linear. Used by the benchmark path.
void linear_tiled(const float* x, const float* w, float* y, int M, int N, int K, int tile = 32);

void add(const float* a, const float* b, float* y, int n);
void rmsnorm(const float* x, const float* weight, float* y, int rows, int dim, float eps);
void silu_mul(const float* gate, const float* up, float* y, int n);

// Rotate-half RoPE applied independently to Q and K.
void apply_rope(float* q, float* k, const int* positions, int tokens,
                int n_heads, int n_kv_heads, int head_dim, float rope_theta);

// Causal online-softmax attention over a dense KV tensor.
// q: [T, n_heads, d], k/v: [T, n_kv_heads, d], out: [T, n_heads, d]
void causal_attention_online(const float* q, const float* k, const float* v, float* out,
                             int tokens, int n_heads, int n_kv_heads, int head_dim);

// Reference full-score softmax attention (same shapes).
void causal_attention_naive(const float* q, const float* k, const float* v, float* out,
                            int tokens, int n_heads, int n_kv_heads, int head_dim);

void softmax(const float* x, float* y, int rows, int cols);

// Weight-only activation quantization: per-row absmax int8 activation,
// per-output-channel int8 weights.
void quantize_weight_int8(const float* w, int rows, int cols,
                          std::int8_t* wq, float* scales);
void linear_int8(const float* x, const std::int8_t* w, const float* scales,
                 float* y, int M, int N, int K);

// OCP E4M3 (bias 7). bits are stored in uint8.
std::uint8_t fp32_to_e4m3(float x);
float e4m3_to_fp32(std::uint8_t bits);
void quantize_weight_fp8(const float* w, int rows, int cols,
                         std::uint8_t* wq, float* scales);
void linear_fp8(const float* x, const std::uint8_t* w, const float* scales,
                float* y, int M, int N, int K);

// Column-parallel: each rank owns N / world output rows of W.
// Row-parallel: each rank owns K / world input columns, partials are summed.
void linear_column_rank(const float* x, const float* w_full, float* y_full,
                        int M, int N, int K, int rank, int world);
void linear_row_rank(const float* x, const float* w_full, float* y_partial,
                     int M, int N, int K, int rank, int world);
void allreduce_sum(float* partials, float* out, int rows, int cols, int world);

}  // namespace kernels
}  // namespace inferno
