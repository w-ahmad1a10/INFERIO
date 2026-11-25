// INFERIO v0 — correct-first GPT-2 prefill engine (CUDA + cuBLAS), Turing SM75.
// Design: sequences are processed in internal groups (sequence-streaming) so
// activations stay small; all GEMMs go through cuBLAS fp16 tensor cores with
// fp32 accumulation. No persistent KV cache (prefill-only benchmark mode).
#include "inferio.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <mma.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------- errors ---
static std::string g_err;

static int set_error(int rc, const char *what, const char *file, int line) {
    char buf[512];
    std::snprintf(buf, sizeof(buf), "%s (%s:%d)", what, file, line);
    g_err = buf;
    return rc;
}

static int cuda_check(cudaError_t e, const char *what, const char *file, int line) {
    if (e == cudaSuccess) return 0;
    int rc = (e == cudaErrorMemoryAllocation) ? INFERIO_ERR_OOM : INFERIO_ERR_CUDA;
    char buf[512];
    std::snprintf(buf, sizeof(buf), "%s: %s (%s:%d)", what, cudaGetErrorString(e), file, line);
    g_err = buf;
    return rc;
}

static int cublas_check(cublasStatus_t s, const char *what, const char *file, int line) {
    if (s == CUBLAS_STATUS_SUCCESS) return 0;
    int rc = (s == CUBLAS_STATUS_ALLOC_FAILED) ? INFERIO_ERR_OOM : INFERIO_ERR_CUDA;
    char buf[512];
    std::snprintf(buf, sizeof(buf), "%s: cublas status %d (%s:%d)", what, (int)s, file, line);
    g_err = buf;
    return rc;
}

#define CUDA_CK(expr)                                                        \
    do {                                                                     \
        int _rc = cuda_check((expr), #expr, __FILE__, __LINE__);             \
        if (_rc) return _rc;                                                 \
    } while (0)

#define CUBLAS_CK(expr)                                                      \
    do {                                                                     \
        int _rc = cublas_check((expr), #expr, __FILE__, __LINE__);           \
        if (_rc) return _rc;                                                 \
    } while (0)

// ------------------------------------------------------ allocation tracking --
static long long g_cur_bytes = 0;
static long long g_peak_bytes = 0;

static void *dev_alloc(size_t n) {
    void *p = nullptr;
    if (cudaMalloc(&p, n) != cudaSuccess) return nullptr;
    g_cur_bytes += (long long)n;
    if (g_cur_bytes > g_peak_bytes) g_peak_bytes = g_cur_bytes;
    return p;
}

static void dev_free(void *p, size_t n) {
    if (!p) return;
    cudaFree(p);
    g_cur_bytes -= (long long)n;
}

// ---------------------------------------------------------------- profiler --
static bool g_profile = false;

struct SpanRecord {
    int phase;
    int pair;
};

struct Profiler {
    std::vector<cudaEvent_t> pool;
    std::vector<SpanRecord> spans;
    int next_pair = 0;
    int capacity = 0;
    bool ready = false;
};

static Profiler g_prof;

static int prof_init(int pairs) {
    g_prof.pool.assign((size_t)pairs * 2, nullptr);
    for (auto &e : g_prof.pool) {
        if (cudaEventCreate(&e) != cudaSuccess) return 1;
    }
    g_prof.spans.reserve((size_t)pairs);
    g_prof.capacity = pairs;
    g_prof.ready = true;
    return 0;
}

static int prof_begin(int phase) {
    if (!g_profile || !g_prof.ready) return -1;
    if (g_prof.next_pair >= g_prof.capacity) return -1;
    int pair = g_prof.next_pair++;
    cudaEventRecord(g_prof.pool[(size_t)pair * 2], 0);
    g_prof.spans.push_back({phase, pair});
    return pair;
}

static void prof_end(int pair) {
    if (pair < 0) return;
    cudaEventRecord(g_prof.pool[(size_t)pair * 2 + 1], 0);
}

static void prof_collect(double *phase_ms) {
    for (auto &s : g_prof.spans) {
        float ms = 0.f;
        cudaEventElapsedTime(&ms, g_prof.pool[(size_t)s.pair * 2],
                             g_prof.pool[(size_t)s.pair * 2 + 1]);
        phase_ms[s.phase] += (double)ms;
    }
    g_prof.spans.clear();
    g_prof.next_pair = 0;
}

// ------------------------------------------------------------------ kernels --
static const int kD = 768;     // model width
static const int kHeads = 12;
static const int kHeadDim = 64;
static const int kFF = 3072;
static const int kVocab = 50257;
static const int kNLayers = 12;
static const int kQKV = 3 * kD;  // 2304

__global__ void embed_kernel(const int32_t *__restrict__ ids, const __half *__restrict__ wte,
                             const __half *__restrict__ wpe, __half *__restrict__ x,
                             long total, int seq, int D) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int t = (int)(i / D);
    int d = (int)(i % D);
    int si = t / seq;
    int p = t - si * seq;
    int id = ids[(long)si * seq + p];
    x[i] = __float2half(__half2float(wte[(long)id * D + d]) + __half2float(wpe[(long)p * D + d]));
}

template <int BLOCK>
__device__ __forceinline__ float block_reduce_sum(float val, float *shared) {
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) val += __shfl_down_sync(0xffffffffu, val, off);
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    val = (threadIdx.x < BLOCK / 32) ? shared[threadIdx.x] : 0.0f;
    if (wid == 0) {
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) val += __shfl_down_sync(0xffffffffu, val, off);
    }
    return val;
}

template <int BLOCK>
__global__ void ln_kernel(const __half *__restrict__ x, const __half *__restrict__ w,
                          const __half *__restrict__ b, __half *__restrict__ y, int D,
                          float eps) {
    const long row = blockIdx.x;
    const __half2 *x2 = reinterpret_cast<const __half2 *>(x + row * D);
    __half2 *y2 = reinterpret_cast<__half2 *>(y + row * D);
    const __half2 *w2 = reinterpret_cast<const __half2 *>(w);
    const __half2 *b2 = reinterpret_cast<const __half2 *>(b);
    const int D2 = D / 2;
    __shared__ float s1[BLOCK / 32];
    __shared__ float s2[BLOCK / 32];
    __shared__ float mean_sh, inv_sh;

    float sum = 0.f;
    for (int i = threadIdx.x; i < D2; i += BLOCK) {
        float2 f = __half22float2(x2[i]);
        sum += f.x + f.y;
    }
    sum = block_reduce_sum<BLOCK>(sum, s1);
    if (threadIdx.x == 0) mean_sh = sum / (float)D;
    __syncthreads();
    const float mean = mean_sh;

    float var = 0.f;
    for (int i = threadIdx.x; i < D2; i += BLOCK) {
        float2 f = __half22float2(x2[i]);
        const float tx = f.x - mean, ty = f.y - mean;
        var += tx * tx + ty * ty;
    }
    var = block_reduce_sum<BLOCK>(var, s2);
    if (threadIdx.x == 0) inv_sh = rsqrtf(var / (float)D + eps);
    __syncthreads();
    const float inv = inv_sh;

    for (int i = threadIdx.x; i < D2; i += BLOCK) {
        float2 f = __half22float2(x2[i]);
        float2 wf = __half22float2(w2[i]);
        float2 bf = __half22float2(b2[i]);
        y2[i] = __floats2half2_rn((f.x - mean) * inv * wf.x + bf.x,
                                  (f.y - mean) * inv * wf.y + bf.y);
    }
}

// Vectorized (half2) bias add. total2 = number of half2 elements, N2 = row width in half2.
__global__ void bias_add_kernel(__half *__restrict__ y, const __half *__restrict__ bias,
                                long total2, int N2) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total2) return;
    const int c = (int)(i % N2);
    __half2 *y2 = reinterpret_cast<__half2 *>(y);
    const __half2 *b2 = reinterpret_cast<const __half2 *>(bias);
    float2 f = __half22float2(y2[i]);
    float2 bf = __half22float2(b2[c]);
    y2[i] = __floats2half2_rn(f.x + bf.x, f.y + bf.y);
}

__device__ __forceinline__ float tanh_approx(float x) {
    float r;
    asm("tanh.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}

__device__ __forceinline__ float gelu_new(float x) {
    return 0.5f * x * (1.0f + tanh_approx(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

__global__ void bias_gelu_kernel(__half *__restrict__ y, const __half *__restrict__ bias,
                                 long total2, int N2) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total2) return;
    const int c = (int)(i % N2);
    __half2 *y2 = reinterpret_cast<__half2 *>(y);
    const __half2 *b2 = reinterpret_cast<const __half2 *>(bias);
    float2 f = __half22float2(y2[i]);
    float2 bf = __half22float2(b2[c]);
    y2[i] = __floats2half2_rn(gelu_new(f.x + bf.x), gelu_new(f.y + bf.y));
}

// One thread per query row; tiled online softmax in log2 domain (exp2f), one
// accumulator rescale per 64-key tile instead of per key. Scores for the tile
// are staged in shared memory. qkv layout: [rows, 2304], q|k|v thirds.
#define kBC 64

__global__ void attn_kernel(const __half *__restrict__ qkv, __half *__restrict__ out, int seq,
                            float scale_log2e) {
    const int qrow = blockIdx.x * blockDim.x + threadIdx.x;
    const int h = blockIdx.y;
    const int g = blockIdx.z;
    if (qrow >= seq) return;

    extern __shared__ float sc[];
    float *srow = sc + threadIdx.x * (kBC + 1);

    const long row_stride = kQKV;
    const __half *qbase = qkv + ((long)g * seq + qrow) * row_stride + h * kHeadDim;
    float q[kHeadDim];
#pragma unroll
    for (int d = 0; d < kHeadDim; ++d) q[d] = __half2float(qbase[d]);

    const __half *kbase0 = qkv + (long)g * seq * row_stride + kD + h * kHeadDim;
    const __half *vbase0 = qkv + (long)g * seq * row_stride + 2 * kD + h * kHeadDim;

    float m2 = -1e30f;  // running max in log2 domain
    float l = 0.f;
    float acc[kHeadDim];
#pragma unroll
    for (int d = 0; d < kHeadDim; ++d) acc[d] = 0.f;

    const int n_tiles = (qrow + kBC) / kBC;
    for (int t = 0; t < n_tiles; ++t) {
        const int k0 = t * kBC;
        const int kend = min(kBC, qrow - k0 + 1);

        float tmax = -1e30f;
        for (int kk = 0; kk < kend; ++kk) {
            const __half *kb = kbase0 + (long)(k0 + kk) * row_stride;
            float s = 0.f;
#pragma unroll
            for (int d = 0; d < kHeadDim; d += 2) {
                __half2 kv = *reinterpret_cast<const __half2 *>(kb + d);
                float2 f = __half22float2(kv);
                s += q[d] * f.x + q[d + 1] * f.y;
            }
            s *= scale_log2e;
            srow[kk] = s;
            tmax = fmaxf(tmax, s);
        }

        const float m2_new = fmaxf(m2, tmax);
        const float corr = exp2f(m2 - m2_new);
#pragma unroll
        for (int d = 0; d < kHeadDim; ++d) acc[d] *= corr;
        float lsum = 0.f;
        for (int kk = 0; kk < kend; ++kk) {
            const float p = exp2f(srow[kk] - m2_new);
            lsum += p;
            const __half *vb = vbase0 + (long)(k0 + kk) * row_stride;
#pragma unroll
            for (int d = 0; d < kHeadDim; d += 2) {
                __half2 vv = *reinterpret_cast<const __half2 *>(vb + d);
                float2 f = __half22float2(vv);
                acc[d] = fmaf(p, f.x, acc[d]);
                acc[d + 1] = fmaf(p, f.y, acc[d + 1]);
            }
        }
        l = l * corr + lsum;
        m2 = m2_new;
    }

    const float inv = (l > 0.f) ? 1.0f / l : 0.0f;
    __half *ob = out + ((long)g * seq + qrow) * kD + h * kHeadDim;
#pragma unroll
    for (int d = 0; d < kHeadDim; ++d) ob[d] = __float2half(acc[d] * inv);
}

// Tensor-core flash attention (two-pass: QK for row max, then QK again + PV).
// Br = 64 queries per CTA (4 warps x 16 rows), Bc = 64 keys per tile, head_dim 64.
// Scores/p rows are staged in shared memory because WMMA fragments are opaque.
#define kAttnBr 64
#define kAttnBc 64
#define kAttnStride 72
#define kAttnSmemBytes (4 * 64 * kAttnStride * 2 + 64 * kAttnStride * 4 + 2 * 64 * 4)

__global__ void attn_wmma_kernel(const __half *__restrict__ qkv, __half *__restrict__ out,
                                 int seq, float scale_log2e) {
    extern __shared__ char smem_raw[];
    __half *Qs = reinterpret_cast<__half *>(smem_raw);
    __half *Ks = Qs + 64 * kAttnStride;
    __half *Vs = Ks + 64 * kAttnStride;
    __half *Ps = Vs + 64 * kAttnStride;
    float *S = reinterpret_cast<float *>(Ps + 64 * kAttnStride);
    float *rowmax = S + 64 * kAttnStride;
    float *rowsum = rowmax + 64;

    const int qc = blockIdx.x;
    const int h = blockIdx.y;
    const int g = blockIdx.z;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row0 = warp * 16;
    const int qbase_row = qc * kAttnBr + row0;
    const long row_stride = kQKV;
    const __half *qkv_g = qkv + (long)g * seq * row_stride + h * kHeadDim;

    for (int i = threadIdx.x; i < kAttnBr * kHeadDim; i += blockDim.x) {
        const int r = i / kHeadDim, d = i % kHeadDim;
        Qs[r * kAttnStride + d] = qkv_g[(long)(qc * kAttnBr + r) * row_stride + d];
    }
    if (threadIdx.x < kAttnBr) {
        rowmax[threadIdx.x] = -1e30f;
        rowsum[threadIdx.x] = 0.f;
    }
    __syncthreads();

    const int qmax = min(qc * kAttnBr + kAttnBr - 1, seq - 1);
    const int n_tiles = (qmax + kAttnBc) / kAttnBc;

    using wmma::accumulator;
    using wmma::col_major;
    using wmma::fragment;
    using wmma::matrix_a;
    using wmma::matrix_b;
    using wmma::mem_row_major;
    using wmma::row_major;

    // Pass A: row max of scaled scores.
    for (int t = 0; t < n_tiles; ++t) {
        const int k0 = t * kAttnBc;
        for (int i = threadIdx.x; i < kAttnBc * kHeadDim; i += blockDim.x) {
            const int kk = i / kHeadDim, d = i % kHeadDim;
            Ks[kk * kAttnStride + d] = qkv_g[(long)(k0 + kk) * row_stride + kD + d];
        }
        __syncthreads();

        fragment<accumulator, 16, 16, 16, float> c[4];
        for (int nk = 0; nk < 4; ++nk) {
            fragment<matrix_a, 16, 16, 16, __half, row_major> a;
            fragment<matrix_b, 16, 16, 16, __half, col_major> b;
            wmma::fill_fragment(c[nk], 0.f);
            for (int dch = 0; dch < 4; ++dch) {
                wmma::load_matrix_sync(a, &Qs[row0 * kAttnStride + dch * 16], kAttnStride);
                wmma::load_matrix_sync(b, &Ks[(nk * 16) * kAttnStride + dch * 16], kAttnStride);
                wmma::mma_sync(c[nk], a, b, c[nk]);
            }
            wmma::store_matrix_sync(&S[row0 * kAttnStride + nk * 16], c[nk], kAttnStride,
                                    mem_row_major);
        }
        __syncthreads();

        if (lane < 16) {
            const int row = row0 + lane;
            const int qrow = qc * kAttnBr + row;
            float m = rowmax[row];
            for (int kk = 0; kk < kAttnBc; ++kk) {
                if (k0 + kk > qrow) break;
                const float s = S[row * kAttnStride + kk] * scale_log2e;
                S[row * kAttnStride + kk] = s;
                m = fmaxf(m, s);
            }
            rowmax[row] = m;
        }
        __syncthreads();
    }

    fragment<accumulator, 16, 16, 16, float> o[4];
    for (int i = 0; i < 4; ++i) wmma::fill_fragment(o[i], 0.f);

    // Pass B: p = exp2(s - m2), accumulate l and O = P @ V.
    for (int t = 0; t < n_tiles; ++t) {
        const int k0 = t * kAttnBc;
        for (int i = threadIdx.x; i < kAttnBc * kHeadDim; i += blockDim.x) {
            const int kk = i / kHeadDim, d = i % kHeadDim;
            const long krow = (long)(k0 + kk) * row_stride;
            Ks[kk * kAttnStride + d] = qkv_g[krow + kD + d];
            Vs[kk * kAttnStride + d] = qkv_g[krow + 2 * kD + d];
        }
        __syncthreads();

        fragment<accumulator, 16, 16, 16, float> c[4];
        for (int nk = 0; nk < 4; ++nk) {
            fragment<matrix_a, 16, 16, 16, __half, row_major> a;
            fragment<matrix_b, 16, 16, 16, __half, col_major> b;
            wmma::fill_fragment(c[nk], 0.f);
            for (int dch = 0; dch < 4; ++dch) {
                wmma::load_matrix_sync(a, &Qs[row0 * kAttnStride + dch * 16], kAttnStride);
                wmma::load_matrix_sync(b, &Ks[(nk * 16) * kAttnStride + dch * 16], kAttnStride);
                wmma::mma_sync(c[nk], a, b, c[nk]);
            }
            wmma::store_matrix_sync(&S[row0 * kAttnStride + nk * 16], c[nk], kAttnStride,
                                    mem_row_major);
        }
        __syncthreads();

        if (lane < 16) {
            const int row = row0 + lane;
            const int qrow = qc * kAttnBr + row;
            const float m2 = rowmax[row];
            float l = 0.f;
            for (int kk = 0; kk < kAttnBc; ++kk) {
                float p = 0.f;
                if (k0 + kk <= qrow) {
                    p = exp2f(S[row * kAttnStride + kk] * scale_log2e - m2);
                    l += p;
                }
                Ps[row * kAttnStride + kk] = __float2half(p);
            }
            rowsum[row] += l;
        }
        __syncthreads();

        for (int kc = 0; kc < 4; ++kc) {
            fragment<matrix_a, 16, 16, 16, __half, row_major> pa;
            wmma::load_matrix_sync(pa, &Ps[row0 * kAttnStride + kc * 16], kAttnStride);
            for (int n = 0; n < 4; ++n) {
                fragment<matrix_b, 16, 16, 16, __half, row_major> vb;
                wmma::load_matrix_sync(vb, &Vs[(kc * 16) * kAttnStride + n * 16], kAttnStride);
                wmma::mma_sync(o[n], pa, vb, o[n]);
            }
        }
        __syncthreads();
    }

    for (int n = 0; n < 4; ++n)
        wmma::store_matrix_sync(&S[row0 * kAttnStride + n * 16], o[n], kAttnStride,
                                mem_row_major);
    __syncthreads();

    for (int i = threadIdx.x; i < kAttnBr * kHeadDim; i += blockDim.x) {
        const int r = i / kHeadDim, d = i % kHeadDim;
        const int qrow = qc * kAttnBr + r;
        if (qrow >= seq) continue;
        const float inv = rowsum[r] > 0.f ? 1.f / rowsum[r] : 0.f;
        out[(long)(g * seq + qrow) * kD + h * kHeadDim + d] =
            __float2half(S[r * kAttnStride + d] * inv);
    }
}

__global__ void gather_last_kernel(const __half *__restrict__ xn, __half *__restrict__ hlast,
                                   int g0, int seq, int D) {
    const int si = blockIdx.x;
    const __half *src = xn + ((long)si * seq + (seq - 1)) * D;
    __half *dst = hlast + (long)(g0 + si) * D;
    for (int d = threadIdx.x; d < D; d += blockDim.x) dst[d] = src[d];
}

__global__ void cast_kernel(const __half *__restrict__ src, float *__restrict__ dst, long total) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) dst[i] = __half2float(src[i]);
}

__global__ void argmax_kernel(const float *__restrict__ logits, int32_t *__restrict__ out,
                              int vocab) {
    const int row = blockIdx.x;
    const float *r = logits + (long)row * vocab;
    float best = -1e30f;
    int bi = 0;
    for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
        float v = r[i];
        if (v > best) {
            best = v;
            bi = i;
        }
    }
    __shared__ float sv[256];
    __shared__ int si_[256];
    sv[threadIdx.x] = best;
    si_[threadIdx.x] = bi;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (sv[threadIdx.x + s] > sv[threadIdx.x]) {
                sv[threadIdx.x] = sv[threadIdx.x + s];
                si_[threadIdx.x] = si_[threadIdx.x + s];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) out[row] = si_[0];
}

// -------------------------------------------------------------- gemm helpers --
static cublasHandle_t g_cublasHandle = nullptr;
static cublasLtHandle_t g_ltHandle = nullptr;
static int g_gemm_mode = 0; /* 1 = cuBLASLt per-shape plans, 0 = cublasGemmEx */
static int g_attn_wmma = 0; /* 1 = tensor-core flash attention, 0 = tiled fp32 attention */
static void *g_ltWs = nullptr;
static size_t g_ltWsBytes = 0;

struct LtPlan {
    long M = 0, K = 0, N = 0;
    int opA = -1;
    cublasLtMatmulDesc_t op_desc = nullptr;
    cublasLtMatrixLayout_t a_desc = nullptr, b_desc = nullptr, c_desc = nullptr;
    cublasLtMatmulAlgo_t algo{};
    size_t ws_bytes = 0;
    bool valid = false;
};
static std::vector<LtPlan> g_plans;

static LtPlan *lt_get_plan(long M, long K, long N, int opA) {
    for (auto &p : g_plans)
        if (p.valid && p.M == M && p.K == K && p.N == N && p.opA == opA) return &p;
    LtPlan p;
    p.M = M;
    p.K = K;
    p.N = N;
    p.opA = opA;
    cublasOperation_t ta = (opA == 1) ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t tb = CUBLAS_OP_N;
    bool ok = cublasLtMatmulDescCreate(&p.op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F) ==
              CUBLAS_STATUS_SUCCESS;
    if (ok)
        ok &= cublasLtMatmulDescSetAttribute(p.op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &ta,
                                             sizeof(ta)) == CUBLAS_STATUS_SUCCESS;
    if (ok)
        ok &= cublasLtMatmulDescSetAttribute(p.op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &tb,
                                             sizeof(tb)) == CUBLAS_STATUS_SUCCESS;
    int a_rows = (opA == 1) ? (int)K : (int)N;
    int a_cols = (opA == 1) ? (int)N : (int)K;
    int lda = (opA == 1) ? (int)K : (int)N;
    if (ok)
        ok &= cublasLtMatrixLayoutCreate(&p.a_desc, CUDA_R_16F, a_rows, a_cols, lda) ==
              CUBLAS_STATUS_SUCCESS;
    if (ok)
        ok &= cublasLtMatrixLayoutCreate(&p.b_desc, CUDA_R_16F, (int)K, (int)M, (int)K) ==
              CUBLAS_STATUS_SUCCESS;
    if (ok)
        ok &= cublasLtMatrixLayoutCreate(&p.c_desc, CUDA_R_16F, (int)N, (int)M, (int)N) ==
              CUBLAS_STATUS_SUCCESS;
    cublasLtMatmulPreference_t pref = nullptr;
    int nres = 0;
    cublasLtMatmulHeuristicResult_t heur[8];
    if (ok) {
        ok &= cublasLtMatmulPreferenceCreate(&pref) == CUBLAS_STATUS_SUCCESS;
        if (ok) {
            size_t max_ws = g_ltWsBytes;
            ok &= cublasLtMatmulPreferenceSetAttribute(
                      pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_ws,
                      sizeof(max_ws)) == CUBLAS_STATUS_SUCCESS;
        }
        if (ok)
            ok &= cublasLtMatmulAlgoGetHeuristic(g_ltHandle, p.op_desc, p.a_desc, p.b_desc,
                                                 p.c_desc, p.c_desc, pref, 8, heur, &nres) ==
                      CUBLAS_STATUS_SUCCESS &&
                  nres > 0;
    }
    if (pref) cublasLtMatmulPreferenceDestroy(pref);
    p.valid = ok;
    if (ok) {
        p.algo = heur[0].algo;
        p.ws_bytes = heur[0].workspaceSize;
    }
    g_plans.push_back(p);
    return p.valid ? &g_plans.back() : nullptr;
}

// Row-major C[M,N] = alpha * X[M,K] @ W[K,N] + beta * C, fp16 in/out, fp32 accum.
// Implemented by viewing the row-major buffers as their column-major transposes:
// C_cm[N,M] = W_cm[N,K] @ X_cm[K,M].
static cublasStatus_t gemm_xw(const __half *X, const __half *W, __half *C, long M, long K, long N,
                              float beta) {
    if (g_gemm_mode == 1 && g_ltHandle) {
        LtPlan *p = lt_get_plan(M, K, N, 0);
        if (p) {
            float alpha = 1.0f;
            return cublasLtMatmul(g_ltHandle, p->op_desc, &alpha, W, p->a_desc, X, p->b_desc,
                                  &beta, C, p->c_desc, C, p->c_desc, &p->algo, g_ltWs,
                                  p->ws_bytes, 0);
        }
    }
    float alpha = 1.0f;
    return cublasGemmEx(g_cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, (int)N, (int)M, (int)K, &alpha,
                        W, CUDA_R_16F, (int)N, X, CUDA_R_16F, (int)K, &beta, C, CUDA_R_16F,
                        (int)N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

// Row-major C[M,N] = X[M,K] @ Wt[N,K]^T (tied-embedding logits), fp16 in/out.
static cublasStatus_t gemm_xwt(const __half *X, const __half *Wt, __half *C, long M, long K,
                               long N) {
    float beta = 0.0f;
    if (g_gemm_mode == 1 && g_ltHandle) {
        LtPlan *p = lt_get_plan(M, K, N, 1);
        if (p) {
            float alpha = 1.0f;
            return cublasLtMatmul(g_ltHandle, p->op_desc, &alpha, Wt, p->a_desc, X, p->b_desc,
                                  &beta, C, p->c_desc, C, p->c_desc, &p->algo, g_ltWs,
                                  p->ws_bytes, 0);
        }
    }
    float alpha = 1.0f;
    return cublasGemmEx(g_cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, (int)N, (int)M, (int)K, &alpha,
                        Wt, CUDA_R_16F, (int)K, X, CUDA_R_16F, (int)K, &beta, C, CUDA_R_16F,
                        (int)N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

// -------------------------------------------------------------- model state --
struct WTensor {
    __half *dev = nullptr;
    long long rows = 0, cols = 0, bytes = 0;
};

struct EngineState {
    bool loaded = false;
    std::unordered_map<std::string, WTensor> tensors;
    long long weight_bytes = 0;

    int buf_batch = 0;
    long long buf_rows = 0;
    int group_seqs = 0;

    __half *x = nullptr;
    size_t x_bytes = 0;
    __half *xn = nullptr;
    size_t xn_bytes = 0;
    __half *qkv = nullptr;
    size_t qkv_bytes = 0;
    __half *att = nullptr;
    size_t att_bytes = 0;
    __half *ffn = nullptr;
    size_t ffn_bytes = 0;
    __half *hlast = nullptr;
    size_t hlast_bytes = 0;
    __half *logits16 = nullptr;
    size_t logits16_bytes = 0;
    float *logits32 = nullptr;
    size_t logits32_bytes = 0;
    int32_t *argmax_dev = nullptr;
    size_t argmax_bytes = 0;
    int32_t *ids_dev = nullptr;
    size_t ids_bytes = 0;

    int64_t last_batch = 0;
    int64_t last_seqlen = 0;
    int64_t last_group = 0;
    int64_t last_groups = 0;
};

static EngineState g_m;
static cudaEvent_t g_ev0 = nullptr, g_ev1 = nullptr;

static WTensor *get_t(const char *name) {
    auto it = g_m.tensors.find(std::string(name));
    return it == g_m.tensors.end() ? nullptr : &it->second;
}

static int load_model(const char *dir) {
    std::string manifest = std::string(dir) + "/manifest.txt";
    std::ifstream mf(manifest);
    if (!mf) return set_error(INFERIO_ERR_IO, "cannot open manifest.txt", __FILE__, __LINE__);
    std::string name;
    long long rows = 0, cols = 0, bytes = 0;
    std::vector<char> hostbuf;
    while (mf >> name >> rows >> cols >> bytes) {
        std::string path = std::string(dir) + "/" + name + ".f16";
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (!f) return set_error(INFERIO_ERR_IO, "missing weight file", __FILE__, __LINE__);
        std::streamsize sz = f.tellg();
        if ((long long)sz != bytes)
            return set_error(INFERIO_ERR_IO, "weight file size mismatch", __FILE__, __LINE__);
        hostbuf.resize((size_t)sz);
        f.seekg(0);
        f.read(hostbuf.data(), sz);
        __half *dev = (__half *)dev_alloc((size_t)bytes);
        if (!dev) return set_error(INFERIO_ERR_OOM, "weight alloc failed", __FILE__, __LINE__);
        cudaError_t e = cudaMemcpy(dev, hostbuf.data(), (size_t)bytes, cudaMemcpyHostToDevice);
        if (e != cudaSuccess) return cuda_check(e, "weight upload", __FILE__, __LINE__);
        WTensor t;
        t.dev = dev;
        t.rows = rows;
        t.cols = cols;
        t.bytes = bytes;
        g_m.tensors[name] = t;
        g_m.weight_bytes += bytes;
    }
    return 0;
}

static int ensure_buffers(int batch, int seqlen) {
    const char *genv = std::getenv("INFERIO_GROUP");
    int group_cap = genv ? std::atoi(genv) : 0;
    if (group_cap <= 0) group_cap = std::max(1, 131072 / seqlen);
    int group = std::min(batch, group_cap);
    long long rows = (long long)group * seqlen;

    bool need = g_m.buf_rows < rows || g_m.buf_batch < batch || g_m.x == nullptr;
    if (!need) {
        g_m.group_seqs = group;
        return 0;
    }

    dev_free(g_m.x, g_m.x_bytes);
    dev_free(g_m.xn, g_m.xn_bytes);
    dev_free(g_m.qkv, g_m.qkv_bytes);
    dev_free(g_m.att, g_m.att_bytes);
    dev_free(g_m.ffn, g_m.ffn_bytes);
    dev_free(g_m.hlast, g_m.hlast_bytes);
    dev_free(g_m.logits16, g_m.logits16_bytes);
    dev_free(g_m.logits32, g_m.logits32_bytes);
    dev_free(g_m.argmax_dev, g_m.argmax_bytes);
    dev_free(g_m.ids_dev, g_m.ids_bytes);
    g_m.x = nullptr;
    g_m.xn = nullptr;
    g_m.qkv = nullptr;
    g_m.att = nullptr;
    g_m.ffn = nullptr;
    g_m.hlast = nullptr;
    g_m.logits16 = nullptr;
    g_m.logits32 = nullptr;
    g_m.argmax_dev = nullptr;
    g_m.ids_dev = nullptr;

    g_m.buf_rows = rows;
    g_m.buf_batch = batch;
    g_m.group_seqs = group;

    const size_t r_d = (size_t)rows * kD;
    const size_t r_qkv = (size_t)rows * kQKV;
    const size_t r_ff = (size_t)rows * kFF;
    const size_t lg = (size_t)batch * kVocab;

    g_m.x_bytes = r_d * sizeof(__half);
    g_m.xn_bytes = r_d * sizeof(__half);
    g_m.qkv_bytes = r_qkv * sizeof(__half);
    g_m.att_bytes = r_d * sizeof(__half);
    g_m.ffn_bytes = r_ff * sizeof(__half);
    g_m.hlast_bytes = (size_t)batch * kD * sizeof(__half);
    g_m.logits16_bytes = lg * sizeof(__half);
    g_m.logits32_bytes = lg * sizeof(float);
    g_m.argmax_bytes = (size_t)batch * sizeof(int32_t);
    g_m.ids_bytes = (size_t)batch * seqlen * sizeof(int32_t);

    g_m.x = (__half *)dev_alloc(g_m.x_bytes);
    g_m.xn = (__half *)dev_alloc(g_m.xn_bytes);
    g_m.qkv = (__half *)dev_alloc(g_m.qkv_bytes);
    g_m.att = (__half *)dev_alloc(g_m.att_bytes);
    g_m.ffn = (__half *)dev_alloc(g_m.ffn_bytes);
    g_m.hlast = (__half *)dev_alloc(g_m.hlast_bytes);
    g_m.logits16 = (__half *)dev_alloc(g_m.logits16_bytes);
    g_m.logits32 = (float *)dev_alloc(g_m.logits32_bytes);
    g_m.argmax_dev = (int32_t *)dev_alloc(g_m.argmax_bytes);
    g_m.ids_dev = (int32_t *)dev_alloc(g_m.ids_bytes);

    if (!g_m.x || !g_m.xn || !g_m.qkv || !g_m.att || !g_m.ffn || !g_m.hlast || !g_m.logits16 ||
        !g_m.logits32 || !g_m.argmax_dev || !g_m.ids_dev)
        return set_error(INFERIO_ERR_OOM, "workspace alloc failed", __FILE__, __LINE__);
    return 0;
}

// ------------------------------------------------------------ debug dumps --
// INFERIO_DUMP_DIR=<dir> dumps group-0 intermediates for bit-level debugging.
static void dump_tensor(const char *name, const void *dev, size_t bytes) {
    const char *dir = std::getenv("INFERIO_DUMP_DIR");
    if (!dir) return;
    std::vector<char> h(bytes);
    if (cudaMemcpy(h.data(), dev, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) return;
    char path[1024];
    std::snprintf(path, sizeof(path), "%s/%s.bin", dir, name);
    FILE *f = std::fopen(path, "wb");
    if (!f) return;
    std::fwrite(h.data(), 1, bytes, f);
    std::fclose(f);
}

// ------------------------------------------------------------- public API ---
extern "C" int inferio_init(const char *model_dir, int gpu_index) {
    g_err.clear();
    if (g_m.loaded) return 0;
    if (!model_dir) return set_error(INFERIO_ERR_ARG, "null model_dir", __FILE__, __LINE__);

    CUDA_CK(cudaSetDevice(gpu_index));
    CUDA_CK(cudaFree(nullptr));
    CUBLAS_CK(cublasCreate(&g_cublasHandle));

    const char *genv = std::getenv("INFERIO_GEMM");
    g_gemm_mode = (genv && std::strcmp(genv, "ex") == 0) ? 0 : 1;
    if (g_gemm_mode == 1) {
        if (cublasLtCreate(&g_ltHandle) != CUBLAS_STATUS_SUCCESS) {
            g_ltHandle = nullptr;
            g_gemm_mode = 0;
        } else {
            g_ltWsBytes = 96ull << 20;
            g_ltWs = dev_alloc(g_ltWsBytes);
            if (!g_ltWs) g_gemm_mode = 0;
        }
    }

    cudaDeviceProp prop{};
    CUDA_CK(cudaGetDeviceProperties(&prop, gpu_index));
    std::printf("INFERIO_INIT version=%s device=%s cc=%d.%d sm_count=%d vram_mib=%d\n",
                INFERIO_VERSION, prop.name, prop.major, prop.minor, prop.multiProcessorCount,
                (int)(prop.totalGlobalMem >> 20));
    std::fflush(stdout);

    CUDA_CK(cudaEventCreate(&g_ev0));
    CUDA_CK(cudaEventCreate(&g_ev1));

    const char *penv = std::getenv("INFERIO_PROFILE");
    g_profile = penv && std::atoi(penv) == 1;
    if (g_profile && prof_init(2048) != 0) g_profile = false;

    const char *aenv = std::getenv("INFERIO_ATTN");
    g_attn_wmma = (aenv && std::strcmp(aenv, "wmma") == 0) ? 1 : 0;
    if (g_attn_wmma) {
        cudaError_t e = cudaFuncSetAttribute((const void *)attn_wmma_kernel,
                                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                                             kAttnSmemBytes);
        if (e != cudaSuccess) {
            g_attn_wmma = 0;
            cudaGetLastError();
        }
    }
    std::printf("INFERIO_INIT attn=%s\n", g_attn_wmma ? "wmma" : "tiled");
    std::fflush(stdout);

    int rc = load_model(model_dir);
    if (rc) return rc;
    g_m.loaded = true;
    return 0;
}

extern "C" int inferio_prefill(const int32_t *host_ids, int batch, int seqlen,
                               inferio_stats *st) {
    g_err.clear();
    if (!st) return set_error(INFERIO_ERR_ARG, "null stats", __FILE__, __LINE__);
    std::memset(st, 0, sizeof(*st));
    st->rc = INFERIO_ERR_CUDA;
    if (!g_m.loaded) return set_error(INFERIO_ERR_ARG, "not initialized", __FILE__, __LINE__);
    if (!host_ids) return set_error(INFERIO_ERR_ARG, "null ids", __FILE__, __LINE__);
    if (seqlen != 1024)
        return set_error(INFERIO_ERR_ARG, "v0 supports seqlen==1024 only", __FILE__, __LINE__);
    if (batch <= 0 || batch > 4096)
        return set_error(INFERIO_ERR_ARG, "batch out of range", __FILE__, __LINE__);

    int rc = ensure_buffers(batch, seqlen);
    if (rc) {
        st->rc = rc;
        return rc;
    }

    WTensor *wte = get_t("wte"), *wpe = get_t("wpe");
    WTensor *lnf_w = get_t("lnf_w"), *lnf_b = get_t("lnf_b");
    if (!wte || !wpe || !lnf_w || !lnf_b)
        return set_error(INFERIO_ERR_IO, "missing top-level tensors", __FILE__, __LINE__);

    const int group = g_m.group_seqs;
    const float attn_scale_log2e = (1.0f / std::sqrt((float)kHeadDim)) * 1.4426950408889634f;

    CUDA_CK(cudaEventRecord(g_ev0, 0));
    CUDA_CK(cudaMemcpyAsync(g_m.ids_dev, host_ids, (size_t)batch * seqlen * sizeof(int32_t),
                            cudaMemcpyHostToDevice, 0));

    for (int g0 = 0; g0 < batch; g0 += group) {
        const int gn = std::min(group, batch - g0);
        const long tr = (long)gn * seqlen;
        {
            int ph = prof_begin(INFERIO_PHASE_EMBED);
            long total = tr * kD;
            int blocks = (int)((total + 255) / 256);
            embed_kernel<<<blocks, 256>>>(g_m.ids_dev + (size_t)g0 * seqlen, wte->dev, wpe->dev,
                                          g_m.x, total, seqlen, kD);
            prof_end(ph);
            if (g0 == 0) dump_tensor("01_embed_x", g_m.x, (size_t)tr * kD * 2);
        }

        for (int l = 0; l < kNLayers; ++l) {
            char nm[64];
            auto layer_t = [&](const char *suffix) -> WTensor * {
                std::snprintf(nm, sizeof(nm), "h%d.%s", l, suffix);
                return get_t(nm);
            };
            WTensor *ln1w = layer_t("ln1_w"), *ln1b = layer_t("ln1_b");
            WTensor *qkvw = layer_t("qkv_w"), *qkvb = layer_t("qkv_b");
            WTensor *pw = layer_t("proj_w"), *pb = layer_t("proj_b");
            WTensor *ln2w = layer_t("ln2_w"), *ln2b = layer_t("ln2_b");
            WTensor *f1w = layer_t("fc1_w"), *f1b = layer_t("fc1_b");
            WTensor *f2w = layer_t("fc2_w"), *f2b = layer_t("fc2_b");
            if (!ln1w || !ln1b || !qkvw || !qkvb || !pw || !pb || !ln2w || !ln2b || !f1w ||
                !f1b || !f2w || !f2b)
                return set_error(INFERIO_ERR_IO, "missing layer tensors", __FILE__, __LINE__);

            int ph = prof_begin(INFERIO_PHASE_LN);
            ln_kernel<256><<<(int)tr, 256>>>(g_m.x, ln1w->dev, ln1b->dev, g_m.xn, kD, 1e-5f);
            prof_end(ph);
            if (g0 == 0 && l == 0) dump_tensor("02_ln1_xn", g_m.xn, (size_t)tr * kD * 2);

            ph = prof_begin(INFERIO_PHASE_QKV);
            CUBLAS_CK(gemm_xw(g_m.xn, qkvw->dev, g_m.qkv, tr, kD, kQKV, 0.0f));
            {
                long total2 = tr * kQKV / 2;
                bias_add_kernel<<<(int)((total2 + 255) / 256), 256>>>(g_m.qkv, qkvb->dev, total2,
                                                                     kQKV / 2);
            }
            prof_end(ph);
            if (g0 == 0 && l == 0) dump_tensor("03_qkv", g_m.qkv, (size_t)tr * kQKV * 2);

            ph = prof_begin(INFERIO_PHASE_ATTN);
            if (g_attn_wmma) {
                dim3 wgrid((seqlen + kAttnBr - 1) / kAttnBr, kHeads, gn);
                attn_wmma_kernel<<<wgrid, 128, kAttnSmemBytes>>>(g_m.qkv, g_m.att, seqlen,
                                                                 attn_scale_log2e);
            } else {
                dim3 agrid((seqlen + 127) / 128, kHeads, gn);
                attn_kernel<<<agrid, 128, 128 * (kBC + 1) * 4>>>(g_m.qkv, g_m.att, seqlen,
                                                                 attn_scale_log2e);
            }
            prof_end(ph);
            if (g0 == 0 && l == 0) dump_tensor("04_att", g_m.att, (size_t)tr * kD * 2);

            ph = prof_begin(INFERIO_PHASE_PROJ);
            CUBLAS_CK(gemm_xw(g_m.att, pw->dev, g_m.x, tr, kD, kD, 1.0f));
            {
                long total2 = tr * kD / 2;
                bias_add_kernel<<<(int)((total2 + 255) / 256), 256>>>(g_m.x, pb->dev, total2,
                                                                     kD / 2);
            }
            prof_end(ph);
            if (g0 == 0 && l == 0) dump_tensor("05_proj_x", g_m.x, (size_t)tr * kD * 2);

            ph = prof_begin(INFERIO_PHASE_LN);
            ln_kernel<256><<<(int)tr, 256>>>(g_m.x, ln2w->dev, ln2b->dev, g_m.xn, kD, 1e-5f);
            prof_end(ph);
            if (g0 == 0 && l == 0) dump_tensor("06_ln2_xn", g_m.xn, (size_t)tr * kD * 2);

            ph = prof_begin(INFERIO_PHASE_FC1);
            CUBLAS_CK(gemm_xw(g_m.xn, f1w->dev, g_m.ffn, tr, kD, kFF, 0.0f));
            {
                long total2 = tr * kFF / 2;
                bias_gelu_kernel<<<(int)((total2 + 255) / 256), 256>>>(g_m.ffn, f1b->dev, total2,
                                                                      kFF / 2);
            }
            prof_end(ph);
            if (g0 == 0 && l == 0) dump_tensor("07_fc1_gelu", g_m.ffn, (size_t)tr * kFF * 2);

            ph = prof_begin(INFERIO_PHASE_FC2);
            CUBLAS_CK(gemm_xw(g_m.ffn, f2w->dev, g_m.x, tr, kFF, kD, 1.0f));
            {
                long total2 = tr * kD / 2;
                bias_add_kernel<<<(int)((total2 + 255) / 256), 256>>>(g_m.x, f2b->dev, total2,
                                                                     kD / 2);
            }
            prof_end(ph);
            if (g0 == 0 && l == 0) dump_tensor("08_fc2_x", g_m.x, (size_t)tr * kD * 2);
        }

        int ph = prof_begin(INFERIO_PHASE_LN);
        ln_kernel<256><<<(int)tr, 256>>>(g_m.x, lnf_w->dev, lnf_b->dev, g_m.xn, kD, 1e-5f);
        prof_end(ph);
        if (g0 == 0) dump_tensor("09_final_xn", g_m.xn, (size_t)tr * kD * 2);

        ph = prof_begin(INFERIO_PHASE_HEAD);
        gather_last_kernel<<<gn, 128>>>(g_m.xn, g_m.hlast, g0, seqlen, kD);
        prof_end(ph);
    }

    int ph = prof_begin(INFERIO_PHASE_HEAD);
    CUBLAS_CK(gemm_xwt(g_m.hlast, wte->dev, g_m.logits16, batch, kD, kVocab));
    {
        long total = (long)batch * kVocab;
        cast_kernel<<<(int)((total + 255) / 256), 256>>>(g_m.logits16, g_m.logits32, total);
    }
    prof_end(ph);
    dump_tensor("10_hlast", g_m.hlast, (size_t)batch * kD * 2);
    dump_tensor("11_logits32", g_m.logits32, (size_t)batch * kVocab * 4);

    CUDA_CK(cudaEventRecord(g_ev1, 0));
    {
        int aph = prof_begin(INFERIO_PHASE_ARGMAX);
        argmax_kernel<<<batch, 256>>>(g_m.logits32, g_m.argmax_dev, kVocab);
        prof_end(aph);
    }
    CUDA_CK(cudaStreamSynchronize(0));

    float ms = 0.f;
    cudaEventElapsedTime(&ms, g_ev0, g_ev1);

    st->prefill_ms = (double)ms;
    st->peak_bytes = g_peak_bytes;
    st->weight_bytes = g_m.weight_bytes;
    st->batch = batch;
    st->seqlen = seqlen;
    st->group_seqs = group;
    st->n_groups = (batch + group - 1) / group;
    st->rc = 0;
    if (g_profile) prof_collect(st->phase_ms);

    g_m.last_batch = batch;
    g_m.last_seqlen = seqlen;
    g_m.last_group = group;
    g_m.last_groups = st->n_groups;
    return 0;
}

extern "C" int inferio_last_argmax(int32_t *host_out, int batch) {
    if (!g_m.loaded) return set_error(INFERIO_ERR_ARG, "not initialized", __FILE__, __LINE__);
    if (!host_out) return set_error(INFERIO_ERR_ARG, "null out", __FILE__, __LINE__);
    if (batch != g_m.last_batch)
        return set_error(INFERIO_ERR_ARG, "batch mismatch", __FILE__, __LINE__);
    CUDA_CK(cudaMemcpy(host_out, g_m.argmax_dev, (size_t)batch * sizeof(int32_t),
                       cudaMemcpyDeviceToHost));
    return 0;
}

extern "C" int inferio_copy_last_logits(float *host_out, int batch, int vocab) {
    if (!g_m.loaded) return set_error(INFERIO_ERR_ARG, "not initialized", __FILE__, __LINE__);
    if (!host_out) return set_error(INFERIO_ERR_ARG, "null out", __FILE__, __LINE__);
    if (batch != g_m.last_batch || vocab != kVocab)
        return set_error(INFERIO_ERR_ARG, "shape mismatch", __FILE__, __LINE__);
    CUDA_CK(cudaMemcpy(host_out, g_m.logits32, (size_t)batch * vocab * sizeof(float),
                       cudaMemcpyDeviceToHost));
    return 0;
}

extern "C" const char *inferio_last_error(void) { return g_err.c_str(); }

extern "C" const char *inferio_version(void) { return INFERIO_VERSION; }

extern "C" void inferio_shutdown(void) {
    for (auto &p : g_plans) {
        if (p.op_desc) cublasLtMatmulDescDestroy(p.op_desc);
        if (p.a_desc) cublasLtMatrixLayoutDestroy(p.a_desc);
        if (p.b_desc) cublasLtMatrixLayoutDestroy(p.b_desc);
        if (p.c_desc) cublasLtMatrixLayoutDestroy(p.c_desc);
    }
    g_plans.clear();
    dev_free(g_ltWs, g_ltWsBytes);
    g_ltWs = nullptr;
    g_ltWsBytes = 0;
    if (g_ltHandle) {
        cublasLtDestroy(g_ltHandle);
        g_ltHandle = nullptr;
    }
    if (g_cublasHandle) {
        cublasDestroy(g_cublasHandle);
        g_cublasHandle = nullptr;
    }
    for (auto &kv : g_m.tensors) dev_free(kv.second.dev, (size_t)kv.second.bytes);
    g_m.tensors.clear();
    g_m.weight_bytes = 0;
    dev_free(g_m.x, g_m.x_bytes);
    dev_free(g_m.xn, g_m.xn_bytes);
    dev_free(g_m.qkv, g_m.qkv_bytes);
    dev_free(g_m.att, g_m.att_bytes);
    dev_free(g_m.ffn, g_m.ffn_bytes);
    dev_free(g_m.hlast, g_m.hlast_bytes);
    dev_free(g_m.logits16, g_m.logits16_bytes);
    dev_free(g_m.logits32, g_m.logits32_bytes);
    dev_free(g_m.argmax_dev, g_m.argmax_bytes);
    dev_free(g_m.ids_dev, g_m.ids_bytes);
    g_m = EngineState{};
    if (g_ev0) cudaEventDestroy(g_ev0);
    if (g_ev1) cudaEventDestroy(g_ev1);
    g_ev0 = g_ev1 = nullptr;
    if (g_prof.ready) {
        for (auto &e : g_prof.pool)
            if (e) cudaEventDestroy(e);
        g_prof.pool.clear();
        g_prof.spans.clear();
        g_prof.ready = false;
        g_prof.next_pair = 0;
        g_prof.capacity = 0;
    }
}
