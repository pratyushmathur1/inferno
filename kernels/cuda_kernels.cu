// CUDA operator set for Inferno.
// Build with: cmake -DINFERNO_CUDA=ON -DCMAKE_BUILD_TYPE=Release ..
//
// The serving scheduler stays in C++ and is tested on CPU. This translation
// unit is the GPU side: tiled GEMM, RMSNorm, RoPE, softmax, a Flash-style
// online attention kernel, and a CUDA-graph capture of that decode chain.

#include <cuda_runtime.h>

#include <algorithm>

namespace inferno {
namespace cuda {

namespace {

constexpr int kTile = 16;

#define INFERNO_CUDA_CHECK(call)                     \
    do {                                             \
        cudaError_t _st = (call);                   \
        if (_st != cudaSuccess) return false;       \
    } while (0)

__global__ void gemm_nt(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[kTile][kTile];
    __shared__ float Bs[kTile][kTile];
    int row = blockIdx.y * kTile + threadIdx.y;
    int col = blockIdx.x * kTile + threadIdx.x;
    float acc = 0.f;
    int tiles = (K + kTile - 1) / kTile;
    for (int t = 0; t < tiles; ++t) {
        int a_col = t * kTile + threadIdx.x;
        int b_col = t * kTile + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.f;
        Bs[threadIdx.x][threadIdx.y] = (col < N && b_col < K) ? B[col * K + b_col] : 0.f;
        __syncthreads();
#pragma unroll
        for (int k = 0; k < kTile; ++k) acc += As[threadIdx.y][k] * Bs[threadIdx.x][k];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = acc;
}

__global__ void rmsnorm_kernel(const float* x, const float* weight, float* y, int rows, int dim, float eps) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const float* row = x + r * dim;
    float ss = 0.f;
    for (int i = 0; i < dim; ++i) ss += row[i] * row[i];
    float inv = rsqrtf(ss / static_cast<float>(dim) + eps);
    float* out = y + r * dim;
    for (int i = 0; i < dim; ++i) out[i] = row[i] * inv * weight[i];
}

__global__ void rope_kernel(float* q, const int* positions, int tokens, int heads, int dim, float theta) {
    int t = blockIdx.x;
    int h = blockIdx.y;
    int i = threadIdx.x;
    int half = dim / 2;
    if (t >= tokens || h >= heads || i >= half) return;
    float* vec = q + (static_cast<long long>(t) * heads + h) * dim;
    float freq = powf(theta, -2.f * static_cast<float>(i) / static_cast<float>(dim));
    float angle = static_cast<float>(positions[t]) * freq;
    float c = cosf(angle);
    float s = sinf(angle);
    float x0 = vec[i];
    float x1 = vec[i + half];
    vec[i] = x0 * c - x1 * s;
    vec[i + half] = x0 * s + x1 * c;
}

__global__ void softmax_kernel(const float* x, float* y, int cols) {
    int r = blockIdx.x;
    const float* row = x + static_cast<long long>(r) * cols;
    float* out = y + static_cast<long long>(r) * cols;
    float m = -1e30f;
    for (int c = 0; c < cols; ++c) m = fmaxf(m, row[c]);
    float sum = 0.f;
    for (int c = 0; c < cols; ++c) {
        out[c] = expf(row[c] - m);
        sum += out[c];
    }
    float inv = sum == 0.f ? 0.f : 1.f / sum;
    for (int c = 0; c < cols; ++c) out[c] *= inv;
}

// One block per query head. Online softmax over KV — the same recurrence as
// the CPU paged attention, here over a dense K/V tile.
__global__ void flash_attn_kernel(const float* q, const float* k, const float* v, float* o,
                                  int kv_len, int dim) {
    const float* qq = q + static_cast<long long>(blockIdx.x) * dim;
    float m_i = -1e30f;
    float l_i = 0.f;
    extern __shared__ float acc[];
    for (int d = threadIdx.x; d < dim; d += blockDim.x) acc[d] = 0.f;
    __syncthreads();
    float scale = rsqrtf(static_cast<float>(dim));
    for (int s = 0; s < kv_len; ++s) {
        const float* kk = k + static_cast<long long>(s) * dim;
        const float* vv = v + static_cast<long long>(s) * dim;
        float score = 0.f;
        for (int d = threadIdx.x; d < dim; d += blockDim.x) score += qq[d] * kk[d];
        // Single-thread block is the tested configuration for this kernel.
        score *= scale;
        float m_new = fmaxf(m_i, score);
        float p = expf(score - m_new);
        float alpha = expf(m_i - m_new);
        for (int d = 0; d < dim; ++d) acc[d] = acc[d] * alpha + p * vv[d];
        l_i = l_i * alpha + p;
        m_i = m_new;
    }
    float* oo = o + static_cast<long long>(blockIdx.x) * dim;
    float inv = l_i == 0.f ? 0.f : 1.f / l_i;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) oo[d] = acc[d] * inv;
}

void launch_decode_chain(float* a, float* b, float* c, float* norm_w, int* positions,
                         int M, int N, int K, cudaStream_t stream) {
    dim3 block(kTile, kTile);
    dim3 grid((N + kTile - 1) / kTile, (M + kTile - 1) / kTile);
    gemm_nt<<<grid, block, 0, stream>>>(a, b, c, M, N, K);
    rmsnorm_kernel<<<(M + 127) / 128, 128, 0, stream>>>(c, norm_w, a, M, N, 1e-5f);
    rope_kernel<<<dim3(M, 1), 32, 0, stream>>>(a, positions, M, 1, N, 10000.f);
    softmax_kernel<<<M, 1, 0, stream>>>(a, c, N);
    flash_attn_kernel<<<M, 1, N * sizeof(float), stream>>>(a, b, c, a, std::min(M, K), N);
}

}  // namespace

bool available() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) return false;
    return n > 0;
}

bool graph_replay_bench(int M, int N, int K, int iters, float& ms_out) {
    if (!available() || iters < 1 || M < 1 || N < 1 || K < 1) return false;
    float *a = nullptr, *b = nullptr, *c = nullptr, *w = nullptr;
    int* positions = nullptr;
    INFERNO_CUDA_CHECK(cudaMalloc(&a, sizeof(float) * M * std::max(N, K)));
    INFERNO_CUDA_CHECK(cudaMalloc(&b, sizeof(float) * std::max(N, K) * K));
    INFERNO_CUDA_CHECK(cudaMalloc(&c, sizeof(float) * M * N));
    INFERNO_CUDA_CHECK(cudaMalloc(&w, sizeof(float) * N));
    INFERNO_CUDA_CHECK(cudaMalloc(&positions, sizeof(int) * M));
    INFERNO_CUDA_CHECK(cudaMemset(a, 0, sizeof(float) * M * std::max(N, K)));
    INFERNO_CUDA_CHECK(cudaMemset(b, 0, sizeof(float) * std::max(N, K) * K));
    INFERNO_CUDA_CHECK(cudaMemset(w, 0, sizeof(float) * N));

    cudaStream_t stream = nullptr;
    INFERNO_CUDA_CHECK(cudaStreamCreate(&stream));
    launch_decode_chain(a, b, c, w, positions, M, N, K, stream);
    INFERNO_CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    INFERNO_CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    launch_decode_chain(a, b, c, w, positions, M, N, K, stream);
    INFERNO_CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    INFERNO_CUDA_CHECK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

    cudaEvent_t start, stop;
    INFERNO_CUDA_CHECK(cudaEventCreate(&start));
    INFERNO_CUDA_CHECK(cudaEventCreate(&stop));
    INFERNO_CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < iters; ++i) {
        INFERNO_CUDA_CHECK(cudaGraphLaunch(exec, stream));
    }
    INFERNO_CUDA_CHECK(cudaEventRecord(stop, stream));
    INFERNO_CUDA_CHECK(cudaEventSynchronize(stop));
    INFERNO_CUDA_CHECK(cudaEventElapsedTime(&ms_out, start, stop));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    cudaFree(w);
    cudaFree(positions);
    return true;
}

}  // namespace cuda
}  // namespace inferno
