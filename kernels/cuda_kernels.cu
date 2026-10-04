// GPU forward for Inferno.
//
// Weights and the paged KV cache live on the device. The host scheduler still
// owns block allocation; each step copies the block tables and token ids, then
// runs the layer stack. A decode step whose token count matches the previous
// step is captured into a CUDA graph and replayed after that.

#include "inferno/cuda_api.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace inferno {
namespace cuda {
namespace {

constexpr int kTile = 32;
constexpr int kMaxDim = 128;
constexpr int kMaxBlock = 32;

void check(cudaError_t st, const char* what) {
    if (st != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(st));
    }
}

void check_cublas(cublasStatus_t st, const char* what) {
    if (st != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(what) + ": cublas status " + std::to_string(static_cast<int>(st)));
    }
}

#define CHECK(call) ::inferno::cuda::check((call), #call)

// Row-major Y[M,N] = X[M,K] · W[N,K]^T via cuBLAS (column-major API).
void gemm_cublas(cublasHandle_t handle, const float* X, const float* W, float* Y, int M, int N, int K,
                 cudaStream_t stream) {
    check_cublas(cublasSetStream(handle, stream), "cublasSetStream");
    const float alpha = 1.f, beta = 0.f;
    check_cublas(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, W, K, X, K, &beta, Y, N),
                 "cublasSgemm");
}

__global__ void gemm_nt(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
                        int M, int N, int K) {
    __shared__ float As[kTile][kTile];
    __shared__ float Bs[kTile][kTile];
    const int row = blockIdx.y * kTile + threadIdx.y;
    const int col = blockIdx.x * kTile + threadIdx.x;
    float acc = 0.f;
    const int tiles = (K + kTile - 1) / kTile;
    for (int t = 0; t < tiles; ++t) {
        const int a_col = t * kTile + threadIdx.x;
        const int b_col = t * kTile + threadIdx.y;
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
    const int r = blockIdx.x;
    if (r >= rows) return;
    const float* row = x + static_cast<long long>(r) * dim;
    float ss = 0.f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) ss += row[i] * row[i];
    __shared__ float red;
    // One block per row. Dim is small, so a shared reduction over the block is enough.
    __shared__ float buf[128];
    buf[threadIdx.x] = ss;
    __syncthreads();
    if (threadIdx.x == 0) {
        float tot = 0.f;
        for (int i = 0; i < blockDim.x; ++i) tot += buf[i];
        red = rsqrtf(tot / static_cast<float>(dim) + eps);
    }
    __syncthreads();
    float* out = y + static_cast<long long>(r) * dim;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) out[i] = row[i] * red * weight[i];
}

// One launch: residual = x, y = rmsnorm(x). Removes a separate copy before attn/FFN norms.
__global__ void rmsnorm_save_residual_kernel(const float* x, float* residual, const float* weight, float* y,
                                             int rows, int dim, float eps) {
    const int r = blockIdx.x;
    if (r >= rows) return;
    const float* row = x + static_cast<long long>(r) * dim;
    float* res = residual + static_cast<long long>(r) * dim;
    float ss = 0.f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        const float v = row[i];
        res[i] = v;
        ss += v * v;
    }
    __shared__ float red;
    __shared__ float buf[128];
    buf[threadIdx.x] = ss;
    __syncthreads();
    if (threadIdx.x == 0) {
        float tot = 0.f;
        for (int i = 0; i < blockDim.x; ++i) tot += buf[i];
        red = rsqrtf(tot / static_cast<float>(dim) + eps);
    }
    __syncthreads();
    float* out = y + static_cast<long long>(r) * dim;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) out[i] = row[i] * red * weight[i];
}

// One launch: x = a + b, residual = x, y = rmsnorm(x).
__global__ void add_rmsnorm_kernel(const float* a, const float* b, const float* weight, float* x, float* residual,
                                   float* y, int rows, int dim, float eps) {
    const int r = blockIdx.x;
    if (r >= rows) return;
    float* xrow = x + static_cast<long long>(r) * dim;
    float* res = residual + static_cast<long long>(r) * dim;
    const float* arow = a + static_cast<long long>(r) * dim;
    const float* brow = b + static_cast<long long>(r) * dim;
    float ss = 0.f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        const float v = arow[i] + brow[i];
        xrow[i] = v;
        res[i] = v;
        ss += v * v;
    }
    __shared__ float red;
    __shared__ float buf[128];
    buf[threadIdx.x] = ss;
    __syncthreads();
    if (threadIdx.x == 0) {
        float tot = 0.f;
        for (int i = 0; i < blockDim.x; ++i) tot += buf[i];
        red = rsqrtf(tot / static_cast<float>(dim) + eps);
    }
    __syncthreads();
    float* out = y + static_cast<long long>(r) * dim;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) out[i] = xrow[i] * red * weight[i];
}

__global__ void rope_kernel(float* q, const int* positions, int tokens, int heads, int dim, float theta) {
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int i = threadIdx.x;
    const int half = dim / 2;
    if (t >= tokens || h >= heads || i >= half) return;
    float* vec = q + (static_cast<long long>(t) * heads + h) * dim;
    const float freq = powf(theta, -2.f * static_cast<float>(i) / static_cast<float>(dim));
    const float angle = static_cast<float>(positions[t]) * freq;
    const float c = cosf(angle);
    const float s = sinf(angle);
    const float x0 = vec[i];
    const float x1 = vec[i + half];
    vec[i] = x0 * c - x1 * s;
    vec[i + half] = x0 * s + x1 * c;
}

__global__ void embed_kernel(const int* ids, const float* table, float* x, int H) {
    const int t = blockIdx.x;
    for (int d = threadIdx.x; d < H; d += blockDim.x) {
        x[static_cast<long long>(t) * H + d] = table[static_cast<long long>(ids[t]) * H + d];
    }
}

__global__ void add_kernel(const float* a, const float* b, float* y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] + b[i];
}

__global__ void copy_kernel(const float* a, float* b, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] = a[i];
}

__global__ void silu_mul_kernel(const float* gate, const float* up, float* y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = gate[i];
    const float sig = 1.f / (1.f + expf(-g));
    y[i] = (g * sig) * up[i];
}

__global__ void write_kv_kernel(const float* k, const float* v, float* kcache, float* vcache,
                                const int* tables, const int* seq_compact, const int* positions, int table_stride,
                                int n_kv, int block_size, int dim, long long layer_stride) {
    const int t = blockIdx.x;
    const int kvh = blockIdx.y;
    const int d = threadIdx.x;
    if (d >= dim) return;
    const int pos = positions[t];
    const int page = pos / block_size;
    const int slot = pos % block_size;
    const int phys = tables[static_cast<long long>(seq_compact[t]) * table_stride + page];
    const long long dst =
        layer_stride + ((static_cast<long long>(phys) * n_kv + kvh) * block_size + slot) * dim + d;
    const long long src = (static_cast<long long>(t) * n_kv + kvh) * dim + d;
    kcache[dst] = k[src];
    vcache[dst] = v[src];
}

// One block per query head. Each KV page is loaded into shared memory and merged
// with the running softmax, so the page is the attention tile.
__global__ void paged_attn_kernel(const float* q, const float* kcache, const float* vcache, const int* tables,
                                  const int* seq_compact, const int* positions, float* out, int table_stride,
                                  int n_heads, int n_kv, int block_size, int dim, long long layer_stride) {
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    extern __shared__ float sm[];
    float* Ks = sm;
    float* Vs = Ks + block_size * dim;
    __shared__ float scores[kMaxBlock];
    __shared__ float acc[kMaxDim];
    __shared__ float running_m, running_l, alpha, m_new, page_sum;

    if (threadIdx.x < dim) acc[threadIdx.x] = 0.f;
    if (threadIdx.x == 0) {
        running_m = -1e30f;
        running_l = 0.f;
    }
    __syncthreads();

    const int kvh = h / (n_heads / n_kv);
    const float* qq = q + (static_cast<long long>(t) * n_heads + h) * dim;
    const int kv_len = positions[t] + 1;
    const int* table = tables + static_cast<long long>(seq_compact[t]) * table_stride;
    const float scale = rsqrtf(static_cast<float>(dim));
    const int pages = (kv_len + block_size - 1) / block_size;

    for (int page = 0; page < pages; ++page) {
        const int phys = table[page];
        const int valid = min(block_size, kv_len - page * block_size);
        const float* kb =
            kcache + layer_stride + ((static_cast<long long>(phys) * n_kv + kvh) * block_size) * dim;
        const float* vb =
            vcache + layer_stride + ((static_cast<long long>(phys) * n_kv + kvh) * block_size) * dim;
        for (int i = threadIdx.x; i < valid * dim; i += blockDim.x) {
            Ks[i] = kb[i];
            Vs[i] = vb[i];
        }
        __syncthreads();
        if (threadIdx.x < valid) {
            const float* kk = Ks + threadIdx.x * dim;
            float dot = 0.f;
            for (int d = 0; d < dim; ++d) dot += qq[d] * kk[d];
            scores[threadIdx.x] = dot * scale;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            float pm = -1e30f;
            for (int i = 0; i < valid; ++i) pm = fmaxf(pm, scores[i]);
            m_new = fmaxf(running_m, pm);
            alpha = __expf(running_m - m_new);
            float ps = 0.f;
            for (int i = 0; i < valid; ++i) {
                const float p = __expf(scores[i] - m_new);
                scores[i] = p;
                ps += p;
            }
            page_sum = ps;
        }
        __syncthreads();
        for (int d = threadIdx.x; d < dim; d += blockDim.x) acc[d] *= alpha;
        __syncthreads();
        for (int s = 0; s < valid; ++s) {
            const float p = scores[s];
            const float* vv = Vs + s * dim;
            for (int d = threadIdx.x; d < dim; d += blockDim.x) acc[d] += p * vv[d];
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            running_l = running_l * alpha + page_sum;
            running_m = m_new;
        }
        __syncthreads();
    }

    const float inv = running_l == 0.f ? 0.f : 1.f / running_l;
    float* oo = out + (static_cast<long long>(t) * n_heads + h) * dim;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) oo[d] = acc[d] * inv;
}

void gemm_tiled(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t stream) {
    dim3 block(kTile, kTile);
    dim3 grid((N + kTile - 1) / kTile, (M + kTile - 1) / kTile);
    gemm_nt<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}

void launch_rmsnorm(const float* x, const float* w, float* y, int rows, int dim, float eps, cudaStream_t stream) {
    rmsnorm_kernel<<<rows, 128, 0, stream>>>(x, w, y, rows, dim, eps);
}

void launch_rmsnorm_save(const float* x, float* residual, const float* w, float* y, int rows, int dim, float eps,
                         cudaStream_t stream) {
    rmsnorm_save_residual_kernel<<<rows, 128, 0, stream>>>(x, residual, w, y, rows, dim, eps);
}

void launch_add_rmsnorm(const float* a, const float* b, const float* w, float* x, float* residual, float* y,
                        int rows, int dim, float eps, cudaStream_t stream) {
    add_rmsnorm_kernel<<<rows, 128, 0, stream>>>(a, b, w, x, residual, y, rows, dim, eps);
}

struct EventPair {
    cudaEvent_t a = nullptr;
    cudaEvent_t b = nullptr;
    void create() {
        check(cudaEventCreate(&a), "event");
        check(cudaEventCreate(&b), "event");
    }
    void destroy() {
        if (a) cudaEventDestroy(a);
        if (b) cudaEventDestroy(b);
        a = b = nullptr;
    }
    float ms() const {
        float t = 0.f;
        check(cudaEventElapsedTime(&t, a, b), "elapsed");
        return t;
    }
};

struct Runner {
    ModelConfig cfg{};
    int max_t = 0;
    int num_blocks = -1;
    int block_size = -1;
    int table_stride = 0;
    long long layer_stride = 0;
    cudaStream_t stream = nullptr;
    cublasHandle_t blas = nullptr;
    bool use_cublas = true;
    bool fuse_kernels = true;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    int exec_t = -1;
    int last_t = -2;
    int replays = 0;
    Profile last{};

    float* weights = nullptr;
    float* kcache = nullptr;
    float* vcache = nullptr;
    float* x = nullptr;
    float* xn = nullptr;
    float* residual = nullptr;
    float* q = nullptr;
    float* k = nullptr;
    float* v = nullptr;
    float* attn = nullptr;
    float* proj = nullptr;
    float* gate = nullptr;
    float* up = nullptr;
    float* mid = nullptr;
    float* logits = nullptr;
    int* token_ids = nullptr;
    int* positions = nullptr;
    int* seq_compact = nullptr;
    int* tables = nullptr;

    ~Runner() {
        if (exec) cudaGraphExecDestroy(exec);
        if (graph) cudaGraphDestroy(graph);
        auto freef = [](void* p) { if (p) cudaFree(p); };
        freef(weights);
        freef(kcache);
        freef(vcache);
        freef(x);
        freef(xn);
        freef(residual);
        freef(q);
        freef(k);
        freef(v);
        freef(attn);
        freef(proj);
        freef(gate);
        freef(up);
        freef(mid);
        freef(logits);
        freef(token_ids);
        freef(positions);
        freef(seq_compact);
        freef(tables);
        if (blas) cublasDestroy(blas);
        if (stream) cudaStreamDestroy(stream);
    }

    void drop_graph() {
        if (exec) cudaGraphExecDestroy(exec);
        if (graph) cudaGraphDestroy(graph);
        exec = nullptr;
        graph = nullptr;
        exec_t = -1;
    }

    void gemm(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t s) {
        if (use_cublas) gemm_cublas(blas, A, B, C, M, N, K, s);
        else gemm_tiled(A, B, C, M, N, K, s);
    }
};

struct Live : Runner {
    std::vector<LayerOff> layers;
    Span tok_emb{};
    Span final_rms{};
    Span lm_head{};

    void launch(int T, cudaStream_t s) {
        const int H = cfg.hidden;
        const int Qdim = cfg.n_heads * cfg.head_dim;
        const int Kdim = cfg.n_kv_heads * cfg.head_dim;
        const int I = cfg.intermediate;
        const int dim = cfg.head_dim;
        auto blocks = [](int n) { return (n + 255) / 256; };

        embed_kernel<<<T, 128, 0, s>>>(token_ids, weights + tok_emb.off, x, H);
        for (int layer = 0; layer < cfg.n_layers; ++layer) {
            const LayerOff& off = layers[layer];
            if (fuse_kernels) {
                launch_rmsnorm_save(x, residual, weights + off.rms1.off, xn, T, H, cfg.rms_eps, s);
            } else {
                copy_kernel<<<blocks(T * H), 256, 0, s>>>(x, residual, T * H);
                launch_rmsnorm(x, weights + off.rms1.off, xn, T, H, cfg.rms_eps, s);
            }
            gemm(xn, weights + off.wq.off, q, T, Qdim, H, s);
            gemm(xn, weights + off.wk.off, k, T, Kdim, H, s);
            gemm(xn, weights + off.wv.off, v, T, Kdim, H, s);
            rope_kernel<<<dim3(T, cfg.n_heads), dim / 2, 0, s>>>(q, positions, T, cfg.n_heads, dim, cfg.rope_theta);
            rope_kernel<<<dim3(T, cfg.n_kv_heads), dim / 2, 0, s>>>(k, positions, T, cfg.n_kv_heads, dim, cfg.rope_theta);
            write_kv_kernel<<<dim3(T, cfg.n_kv_heads), dim, 0, s>>>(
                k, v, kcache, vcache, tables, seq_compact, positions, table_stride, cfg.n_kv_heads, block_size, dim,
                layer_stride * layer);
            const int smem = 2 * block_size * dim * static_cast<int>(sizeof(float));
            paged_attn_kernel<<<dim3(T, cfg.n_heads), 128, smem, s>>>(
                q, kcache, vcache, tables, seq_compact, positions, attn, table_stride, cfg.n_heads, cfg.n_kv_heads,
                block_size, dim, layer_stride * layer);
            gemm(attn, weights + off.wo.off, proj, T, H, Qdim, s);
            if (fuse_kernels) {
                // x = residual+attn, residual = x (pre-FFN), xn = rmsnorm(x)
                launch_add_rmsnorm(residual, proj, weights + off.rms2.off, x, residual, xn, T, H, cfg.rms_eps, s);
            } else {
                add_kernel<<<blocks(T * H), 256, 0, s>>>(residual, proj, x, T * H);
                copy_kernel<<<blocks(T * H), 256, 0, s>>>(x, residual, T * H);
                launch_rmsnorm(x, weights + off.rms2.off, xn, T, H, cfg.rms_eps, s);
            }
            gemm(xn, weights + off.wgate.off, gate, T, I, H, s);
            gemm(xn, weights + off.wup.off, up, T, I, H, s);
            silu_mul_kernel<<<blocks(T * I), 256, 0, s>>>(gate, up, mid, T * I);
            gemm(mid, weights + off.wdown.off, proj, T, H, I, s);
            add_kernel<<<blocks(T * H), 256, 0, s>>>(residual, proj, x, T * H);
        }
        launch_rmsnorm(x, weights + final_rms.off, xn, T, H, cfg.rms_eps, s);
        gemm(xn, weights + lm_head.off, logits, T, cfg.vocab, H, s);
    }

    // Instrument one forward with CUDA events (disables graph replay).
    Profile launch_profiled(int T, cudaStream_t s) {
        Profile p;
        p.tokens = T;
        p.layers = cfg.n_layers;
        p.gemm_backend = use_cublas ? "cublas" : "tiled";
        p.fused = fuse_kernels;
        const int H = cfg.hidden;
        const int Qdim = cfg.n_heads * cfg.head_dim;
        const int Kdim = cfg.n_kv_heads * cfg.head_dim;
        const int I = cfg.intermediate;
        const int dim = cfg.head_dim;
        auto blocks = [](int n) { return (n + 255) / 256; };

        EventPair total, emb, gemm_e, norm_e, rope_e, kv_e, attn_e, misc_e;
        total.create();
        emb.create();
        gemm_e.create();
        norm_e.create();
        rope_e.create();
        kv_e.create();
        attn_e.create();
        misc_e.create();

        auto finish = [&](EventPair& e, float& bucket) {
            check(cudaEventRecord(e.b, s), "rec");
            check(cudaEventSynchronize(e.b), "sync bucket");
            bucket += e.ms();
        };

        check(cudaEventRecord(total.a, s), "rec");
        check(cudaEventRecord(emb.a, s), "rec");
        embed_kernel<<<T, 128, 0, s>>>(token_ids, weights + tok_emb.off, x, H);
        finish(emb, p.embed_ms);

        for (int layer = 0; layer < cfg.n_layers; ++layer) {
            const LayerOff& off = layers[layer];
            if (fuse_kernels) {
                check(cudaEventRecord(norm_e.a, s), "rec");
                launch_rmsnorm_save(x, residual, weights + off.rms1.off, xn, T, H, cfg.rms_eps, s);
                finish(norm_e, p.norm_ms);
            } else {
                check(cudaEventRecord(misc_e.a, s), "rec");
                copy_kernel<<<blocks(T * H), 256, 0, s>>>(x, residual, T * H);
                finish(misc_e, p.misc_ms);
                check(cudaEventRecord(norm_e.a, s), "rec");
                launch_rmsnorm(x, weights + off.rms1.off, xn, T, H, cfg.rms_eps, s);
                finish(norm_e, p.norm_ms);
            }

            check(cudaEventRecord(gemm_e.a, s), "rec");
            gemm(xn, weights + off.wq.off, q, T, Qdim, H, s);
            gemm(xn, weights + off.wk.off, k, T, Kdim, H, s);
            gemm(xn, weights + off.wv.off, v, T, Kdim, H, s);
            finish(gemm_e, p.gemm_ms);

            check(cudaEventRecord(rope_e.a, s), "rec");
            rope_kernel<<<dim3(T, cfg.n_heads), dim / 2, 0, s>>>(q, positions, T, cfg.n_heads, dim, cfg.rope_theta);
            rope_kernel<<<dim3(T, cfg.n_kv_heads), dim / 2, 0, s>>>(k, positions, T, cfg.n_kv_heads, dim, cfg.rope_theta);
            finish(rope_e, p.rope_ms);

            check(cudaEventRecord(kv_e.a, s), "rec");
            write_kv_kernel<<<dim3(T, cfg.n_kv_heads), dim, 0, s>>>(
                k, v, kcache, vcache, tables, seq_compact, positions, table_stride, cfg.n_kv_heads, block_size, dim,
                layer_stride * layer);
            finish(kv_e, p.kv_write_ms);

            check(cudaEventRecord(attn_e.a, s), "rec");
            const int smem = 2 * block_size * dim * static_cast<int>(sizeof(float));
            paged_attn_kernel<<<dim3(T, cfg.n_heads), 128, smem, s>>>(
                q, kcache, vcache, tables, seq_compact, positions, attn, table_stride, cfg.n_heads, cfg.n_kv_heads,
                block_size, dim, layer_stride * layer);
            finish(attn_e, p.attn_ms);

            check(cudaEventRecord(gemm_e.a, s), "rec");
            gemm(attn, weights + off.wo.off, proj, T, H, Qdim, s);
            finish(gemm_e, p.gemm_ms);

            if (fuse_kernels) {
                check(cudaEventRecord(norm_e.a, s), "rec");
                launch_add_rmsnorm(residual, proj, weights + off.rms2.off, x, residual, xn, T, H, cfg.rms_eps, s);
                finish(norm_e, p.norm_ms);
            } else {
                check(cudaEventRecord(misc_e.a, s), "rec");
                add_kernel<<<blocks(T * H), 256, 0, s>>>(residual, proj, x, T * H);
                copy_kernel<<<blocks(T * H), 256, 0, s>>>(x, residual, T * H);
                finish(misc_e, p.misc_ms);
                check(cudaEventRecord(norm_e.a, s), "rec");
                launch_rmsnorm(x, weights + off.rms2.off, xn, T, H, cfg.rms_eps, s);
                finish(norm_e, p.norm_ms);
            }

            check(cudaEventRecord(gemm_e.a, s), "rec");
            gemm(xn, weights + off.wgate.off, gate, T, I, H, s);
            gemm(xn, weights + off.wup.off, up, T, I, H, s);
            finish(gemm_e, p.gemm_ms);

            check(cudaEventRecord(misc_e.a, s), "rec");
            silu_mul_kernel<<<blocks(T * I), 256, 0, s>>>(gate, up, mid, T * I);
            finish(misc_e, p.misc_ms);

            check(cudaEventRecord(gemm_e.a, s), "rec");
            gemm(mid, weights + off.wdown.off, proj, T, H, I, s);
            finish(gemm_e, p.gemm_ms);

            check(cudaEventRecord(misc_e.a, s), "rec");
            add_kernel<<<blocks(T * H), 256, 0, s>>>(residual, proj, x, T * H);
            finish(misc_e, p.misc_ms);
        }

        check(cudaEventRecord(norm_e.a, s), "rec");
        launch_rmsnorm(x, weights + final_rms.off, xn, T, H, cfg.rms_eps, s);
        finish(norm_e, p.norm_ms);

        check(cudaEventRecord(gemm_e.a, s), "rec");
        gemm(xn, weights + lm_head.off, logits, T, cfg.vocab, H, s);
        finish(gemm_e, p.gemm_ms);

        check(cudaEventRecord(total.b, s), "rec");
        check(cudaEventSynchronize(total.b), "sync total");
        p.total_ms = total.ms();

        total.destroy();
        emb.destroy();
        gemm_e.destroy();
        norm_e.destroy();
        rope_e.destroy();
        kv_e.destroy();
        attn_e.destroy();
        misc_e.destroy();
        return p;
    }
};

void* must_alloc(size_t bytes) {
    void* p = nullptr;
    check(cudaMalloc(&p, bytes), "cudaMalloc");
    return p;
}

}  // namespace

bool available() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) return false;
    return n > 0;
}

bool graph_replay_bench(int M, int N, int K, int iters, float& ms_out) {
    if (!available() || iters < 1) return false;
    float *a = nullptr, *b = nullptr, *c = nullptr, *w = nullptr;
    check(cudaMalloc(&a, sizeof(float) * static_cast<size_t>(M) * K), "malloc a");
    check(cudaMalloc(&b, sizeof(float) * static_cast<size_t>(N) * K), "malloc b");
    check(cudaMalloc(&c, sizeof(float) * static_cast<size_t>(M) * N), "malloc c");
    check(cudaMalloc(&w, sizeof(float) * static_cast<size_t>(N)), "malloc w");
    check(cudaMemset(a, 0, sizeof(float) * static_cast<size_t>(M) * K), "memset a");
    check(cudaMemset(w, 0, sizeof(float) * static_cast<size_t>(N)), "memset w");
    cudaStream_t stream = nullptr;
    check(cudaStreamCreate(&stream), "stream");
    gemm_tiled(a, b, c, M, N, K, stream);
    launch_rmsnorm(c, w, a, M, N, 1e-5f, stream);
    check(cudaStreamSynchronize(stream), "warmup");
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "begin capture");
    gemm_tiled(a, b, c, M, N, K, stream);
    launch_rmsnorm(c, w, a, M, N, 1e-5f, stream);
    check(cudaStreamEndCapture(stream, &graph), "end capture");
#if CUDART_VERSION >= 12000
    check(cudaGraphInstantiate(&exec, graph, 0), "instantiate");
#else
    check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0), "instantiate");
#endif
    cudaEvent_t start, stop;
    check(cudaEventCreate(&start), "event");
    check(cudaEventCreate(&stop), "event");
    check(cudaEventRecord(start, stream), "record");
    for (int i = 0; i < iters; ++i) check(cudaGraphLaunch(exec, stream), "replay");
    check(cudaEventRecord(stop, stream), "record stop");
    check(cudaEventSynchronize(stop), "sync event");
    check(cudaEventElapsedTime(&ms_out, start, stop), "elapsed");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    cudaFree(w);
    return true;
}

void release_device(void* slot) { delete static_cast<Live*>(slot); }

int graph_replays(void* slot) {
    if (!slot) return 0;
    return static_cast<Live*>(slot)->replays;
}

Profile last_profile(void* slot) {
    if (!slot) return {};
    return static_cast<Live*>(slot)->last;
}

bool device_mem(size_t& used_bytes, size_t& total_bytes) {
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) return false;
    total_bytes = total_b;
    used_bytes = total_b - free_b;
    return true;
}

void llama_forward(const ModelConfig& cfg, const float* weights, std::int64_t nweights, const LayerOff* layers,
                   const std::vector<TokenIn>& tokens, PagedCache& cache, std::vector<float>& logits,
                   ForwardOptions opts, void*& slot) {
    const int T = static_cast<int>(tokens.size());
    logits.clear();
    if (T == 0) return;
    if (cfg.head_dim > kMaxDim || cfg.head_dim < 2 || (cfg.head_dim % 2) != 0) {
        throw std::runtime_error("GPU forward supports even head_dim up to 128");
    }
    if (cache.block_size() > kMaxBlock) {
        throw std::runtime_error("GPU paged attention supports block_size up to 32");
    }
    for (const auto& tok : tokens) {
        if (tok.token_id < 0 || tok.token_id >= cfg.vocab) throw std::runtime_error("token id out of range");
    }
    if (!slot) slot = new Live();
    Live& dev = *static_cast<Live*>(slot);
    if (!dev.stream) check(cudaStreamCreate(&dev.stream), "stream");
    if (!dev.blas) check_cublas(cublasCreate(&dev.blas), "cublasCreate");
    if (dev.use_cublas != opts.use_cublas || dev.fuse_kernels != opts.fuse_kernels) {
        dev.use_cublas = opts.use_cublas;
        dev.fuse_kernels = opts.fuse_kernels;
        dev.drop_graph();
    }

    const bool shape_changed = dev.num_blocks != cache.num_blocks() || dev.block_size != cache.block_size() ||
                               dev.cfg.n_layers != cfg.n_layers || dev.cfg.n_kv_heads != cfg.n_kv_heads ||
                               dev.cfg.head_dim != cfg.head_dim;
    if (dev.weights == nullptr) {
        dev.cfg = cfg;
        dev.layers.assign(layers, layers + cfg.n_layers);
        dev.tok_emb.off = 0;
        dev.tok_emb.n = static_cast<std::int64_t>(cfg.vocab) * cfg.hidden;
        std::int64_t cursor = dev.tok_emb.n;
        for (int L = 0; L < cfg.n_layers; ++L) cursor = layers[L].wdown.off + layers[L].wdown.n;
        dev.final_rms.off = cursor;
        dev.final_rms.n = cfg.hidden;
        cursor += cfg.hidden;
        dev.lm_head.off = cursor;
        dev.lm_head.n = static_cast<std::int64_t>(cfg.vocab) * cfg.hidden;
        dev.weights = static_cast<float*>(must_alloc(sizeof(float) * static_cast<size_t>(nweights)));
        check(cudaMemcpy(dev.weights, weights, sizeof(float) * static_cast<size_t>(nweights), cudaMemcpyHostToDevice),
              "upload weights");
    }
    if (shape_changed || dev.kcache == nullptr) {
        if (dev.kcache) cudaFree(dev.kcache);
        if (dev.vcache) cudaFree(dev.vcache);
        dev.drop_graph();
        dev.num_blocks = cache.num_blocks();
        dev.block_size = cache.block_size();
        dev.table_stride = cache.num_blocks();
        dev.layer_stride = static_cast<long long>(cache.num_blocks()) * cfg.n_kv_heads * cache.block_size() * cfg.head_dim;
        const size_t bytes = sizeof(float) * static_cast<size_t>(dev.layer_stride) * cfg.n_layers;
        dev.kcache = static_cast<float*>(must_alloc(bytes));
        dev.vcache = static_cast<float*>(must_alloc(bytes));
        check(cudaMemset(dev.kcache, 0, bytes), "clear k");
        check(cudaMemset(dev.vcache, 0, bytes), "clear v");
        if (dev.tables) {
            cudaFree(dev.tables);
            dev.tables = nullptr;
        }
    }
    if (T > dev.max_t) {
        dev.drop_graph();
        auto grow = [&](float*& p, size_t n) {
            if (p) cudaFree(p);
            p = static_cast<float*>(must_alloc(sizeof(float) * n));
        };
        auto growi = [&](int*& p, size_t n) {
            if (p) cudaFree(p);
            p = static_cast<int*>(must_alloc(sizeof(int) * n));
        };
        dev.max_t = std::max(T, 64);
        const int H = cfg.hidden;
        const int Q = cfg.n_heads * cfg.head_dim;
        const int KV = cfg.n_kv_heads * cfg.head_dim;
        const int I = cfg.intermediate;
        grow(dev.x, static_cast<size_t>(dev.max_t) * H);
        grow(dev.xn, static_cast<size_t>(dev.max_t) * H);
        grow(dev.residual, static_cast<size_t>(dev.max_t) * H);
        grow(dev.q, static_cast<size_t>(dev.max_t) * Q);
        grow(dev.k, static_cast<size_t>(dev.max_t) * KV);
        grow(dev.v, static_cast<size_t>(dev.max_t) * KV);
        grow(dev.attn, static_cast<size_t>(dev.max_t) * Q);
        grow(dev.proj, static_cast<size_t>(dev.max_t) * H);
        grow(dev.gate, static_cast<size_t>(dev.max_t) * I);
        grow(dev.up, static_cast<size_t>(dev.max_t) * I);
        grow(dev.mid, static_cast<size_t>(dev.max_t) * I);
        grow(dev.logits, static_cast<size_t>(dev.max_t) * cfg.vocab);
        growi(dev.token_ids, dev.max_t);
        growi(dev.positions, dev.max_t);
        growi(dev.seq_compact, dev.max_t);
        if (dev.tables) cudaFree(dev.tables);
        dev.tables = nullptr;
    }
    if (dev.tables == nullptr) {
        dev.tables = static_cast<int*>(
            must_alloc(sizeof(int) * static_cast<size_t>(dev.max_t) * static_cast<size_t>(dev.table_stride)));
    }

    std::vector<int> ids(T), pos(T), compact(T);
    std::unordered_map<int, int> slot_of;
    std::vector<int> packed(static_cast<size_t>(T) * dev.table_stride, -1);
    int slots = 0;
    for (int t = 0; t < T; ++t) {
        ids[t] = tokens[t].token_id;
        pos[t] = tokens[t].position;
        const int seq = tokens[t].seq_id;
        auto it = slot_of.find(seq);
        if (it == slot_of.end()) {
            const int slot = slots++;
            slot_of.emplace(seq, slot);
            const auto& table = cache.block_table(seq);
            if (static_cast<int>(table.size()) > dev.table_stride) {
                throw std::runtime_error("block table longer than the cache");
            }
            std::copy(table.begin(), table.end(), packed.begin() + static_cast<size_t>(slot) * dev.table_stride);
            compact[t] = slot;
        } else {
            compact[t] = it->second;
        }
    }
    check(cudaMemcpyAsync(dev.token_ids, ids.data(), sizeof(int) * T, cudaMemcpyHostToDevice, dev.stream), "ids");
    check(cudaMemcpyAsync(dev.positions, pos.data(), sizeof(int) * T, cudaMemcpyHostToDevice, dev.stream), "pos");
    check(cudaMemcpyAsync(dev.seq_compact, compact.data(), sizeof(int) * T, cudaMemcpyHostToDevice, dev.stream),
          "seq");
    check(cudaMemcpyAsync(dev.tables, packed.data(), sizeof(int) * packed.size(), cudaMemcpyHostToDevice, dev.stream),
          "tables");

    const bool want_graphs = opts.graphs && !opts.profile;
    if (opts.profile) {
        dev.last = dev.launch_profiled(T, dev.stream);
        check(cudaGetLastError(), "profiled launch");
    } else {
        const bool replay = want_graphs && dev.exec && dev.exec_t == T;
        if (replay) {
            check(cudaGraphLaunch(dev.exec, dev.stream), "graph launch");
            ++dev.replays;
        } else {
            dev.launch(T, dev.stream);
            check(cudaGetLastError(), "kernel launch");
            if (want_graphs && dev.last_t == T) {
                check(cudaStreamSynchronize(dev.stream), "sync before capture");
                dev.drop_graph();
                check(cudaStreamBeginCapture(dev.stream, cudaStreamCaptureModeGlobal), "begin capture");
                dev.launch(T, dev.stream);
                check(cudaStreamEndCapture(dev.stream, &dev.graph), "end capture");
#if CUDART_VERSION >= 12000
                check(cudaGraphInstantiate(&dev.exec, dev.graph, 0), "instantiate");
#else
                check(cudaGraphInstantiate(&dev.exec, dev.graph, nullptr, nullptr, 0), "instantiate");
#endif
                dev.exec_t = T;
            }
        }
        check(cudaStreamSynchronize(dev.stream), "sync forward");
        if (!opts.profile) {
            // keep last profile only from explicit profile runs
        }
    }
    dev.last_t = T;

    std::vector<float> all(static_cast<size_t>(T) * cfg.vocab);
    check(cudaMemcpy(all.data(), dev.logits, sizeof(float) * all.size(), cudaMemcpyDeviceToHost), "logits");
    int emits = 0;
    for (const auto& tok : tokens) emits += tok.emit ? 1 : 0;
    logits.assign(static_cast<size_t>(emits) * cfg.vocab, 0.f);
    int e = 0;
    for (int t = 0; t < T; ++t) {
        if (!tokens[t].emit) continue;
        std::copy(all.begin() + static_cast<size_t>(t) * cfg.vocab,
                  all.begin() + static_cast<size_t>(t + 1) * cfg.vocab,
                  logits.begin() + static_cast<size_t>(e) * cfg.vocab);
        ++e;
    }
}

}  // namespace cuda
}  // namespace inferno
