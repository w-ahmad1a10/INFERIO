// INFERIO v0 — correct-first GPT-2 prefill engine (CUDA + cuBLAS), Turing SM75.
// Design: sequences are processed in internal groups (sequence-streaming) so
// activations stay small; all GEMMs go through cuBLAS fp16 tensor cores with
// fp32 accumulation. No persistent KV cache (prefill-only benchmark mode).
#include "inferio.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

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
    const __half *xr = x + row * D;
    __half *yr = y + row * D;
    __shared__ float s1[BLOCK / 32];
    __shared__ float s2[BLOCK / 32];
    __shared__ float mean_sh, inv_sh;

    float sum = 0.f;
    for (int i = threadIdx.x; i < D; i += BLOCK) sum += __half2float(xr[i]);
    sum = block_reduce_sum<BLOCK>(sum, s1);
    if (threadIdx.x == 0) mean_sh = sum / (float)D;
    __syncthreads();
    const float mean = mean_sh;

    float var = 0.f;
    for (int i = threadIdx.x; i < D; i += BLOCK) {
        float t = __half2float(xr[i]) - mean;
        var += t * t;
    }
    var = block_reduce_sum<BLOCK>(var, s2);
    if (threadIdx.x == 0) inv_sh = rsqrtf(var / (float)D + eps);
    __syncthreads();
    const float inv = inv_sh;

    for (int i = threadIdx.x; i < D; i += BLOCK) {
        yr[i] = __float2half((__half2float(xr[i]) - mean) * inv * __half2float(w[i]) +
                             __half2float(b[i]));
    }
}

__global__ void bias_add_kernel(__half *__restrict__ y, const __half *__restrict__ bias,
                                long total, int N) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int c = (int)(i % N);
    y[i] = __float2half(__half2float(y[i]) + __half2float(bias[c]));
}

__device__ __forceinline__ float gelu_new(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

__global__ void bias_gelu_kernel(__half *__restrict__ y, const __half *__restrict__ bias,
                                 long total, int N) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int c = (int)(i % N);
    y[i] = __float2half(gelu_new(__half2float(y[i]) + __half2float(bias[c])));
}

// One thread per query row; online softmax (flash-style, no S x S materialization).
// qkv layout: [rows, 2304] with q = [0:768], k = [768:1536], v = [1536:2304].
__global__ void attn_kernel(const __half *__restrict__ qkv, __half *__restrict__ out, int seq,
                            float scale) {
    const int qrow = blockIdx.x * blockDim.x + threadIdx.x;
    const int h = blockIdx.y;
    const int g = blockIdx.z;
    if (qrow >= seq) return;

    const long row_stride = kQKV;
    const __half *qbase = qkv + ((long)g * seq + qrow) * row_stride + h * kHeadDim;
    float q[kHeadDim];
#pragma unroll
    for (int d = 0; d < kHeadDim; ++d) q[d] = __half2float(qbase[d]);

    const __half *kbase0 = qkv + (long)g * seq * row_stride + kD + h * kHeadDim;
    const __half *vbase0 = qkv + (long)g * seq * row_stride + 2 * kD + h * kHeadDim;

    float m = -1e30f;
    float l = 0.f;
    float acc[kHeadDim];
#pragma unroll
    for (int d = 0; d < kHeadDim; ++d) acc[d] = 0.f;

    for (int k = 0; k <= qrow; ++k) {
        const __half *kb = kbase0 + (long)k * row_stride;
        float s = 0.f;
#pragma unroll
        for (int d = 0; d < kHeadDim; d += 2) {
            __half2 kv = *reinterpret_cast<const __half2 *>(kb + d);
            float2 f = __half22float2(kv);
            s += q[d] * f.x + q[d + 1] * f.y;
        }
        s *= scale;
        const float m_new = fmaxf(m, s);
        const float p = __expf(s - m_new);
        const float corr = __expf(m - m_new);
        l = l * corr + p;
        const __half *vb = vbase0 + (long)k * row_stride;
#pragma unroll
        for (int d = 0; d < kHeadDim; d += 2) {
            __half2 vv = *reinterpret_cast<const __half2 *>(vb + d);
            float2 f = __half22float2(vv);
            acc[d] = acc[d] * corr + p * f.x;
            acc[d + 1] = acc[d + 1] * corr + p * f.y;
        }
        m = m_new;
    }
    const float inv = (l > 0.f) ? 1.0f / l : 0.0f;
    __half *ob = out + ((long)g * seq + qrow) * kD + h * kHeadDim;
#pragma unroll
    for (int d = 0; d < kHeadDim; ++d) ob[d] = __float2half(acc[d] * inv);
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

// Row-major C[M,N] = alpha * X[M,K] @ W[K,N] + beta * C, fp16 in/out, fp32 accum.
// Implemented by viewing the row-major buffers as their column-major transposes:
// C_cm[N,M] = W_cm[N,K] @ X_cm[K,M].
static cublasStatus_t gemm_xw(const __half *X, const __half *W, __half *C, long M, long K, long N,
                              float beta) {
    float alpha = 1.0f;
    return cublasGemmEx(g_cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, (int)N, (int)M, (int)K, &alpha,
                        W, CUDA_R_16F, (int)N, X, CUDA_R_16F, (int)K, &beta, C, CUDA_R_16F,
                        (int)N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

// Row-major C[M,N] = X[M,K] @ Wt[N,K]^T (tied-embedding logits), fp16 in/out.
static cublasStatus_t gemm_xwt(const __half *X, const __half *Wt, __half *C, long M, long K,
                               long N) {
    float alpha = 1.0f, beta = 0.0f;
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
    const float attn_scale = 1.0f / std::sqrt((float)kHeadDim);

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
                long total = tr * kQKV;
                bias_add_kernel<<<(int)((total + 255) / 256), 256>>>(g_m.qkv, qkvb->dev, total,
                                                                     kQKV);
            }
            prof_end(ph);
            if (g0 == 0 && l == 0) dump_tensor("03_qkv", g_m.qkv, (size_t)tr * kQKV * 2);

            ph = prof_begin(INFERIO_PHASE_ATTN);
            dim3 agrid((seqlen + 127) / 128, kHeads, gn);
            attn_kernel<<<agrid, 128>>>(g_m.qkv, g_m.att, seqlen, attn_scale);
            prof_end(ph);
            if (g0 == 0 && l == 0) dump_tensor("04_att", g_m.att, (size_t)tr * kD * 2);

            ph = prof_begin(INFERIO_PHASE_PROJ);
            CUBLAS_CK(gemm_xw(g_m.att, pw->dev, g_m.x, tr, kD, kD, 1.0f));
            {
                long total = tr * kD;
                bias_add_kernel<<<(int)((total + 255) / 256), 256>>>(g_m.x, pb->dev, total, kD);
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
                long total = tr * kFF;
                bias_gelu_kernel<<<(int)((total + 255) / 256), 256>>>(g_m.ffn, f1b->dev, total,
                                                                      kFF);
            }
            prof_end(ph);
            if (g0 == 0 && l == 0) dump_tensor("07_fc1_gelu", g_m.ffn, (size_t)tr * kFF * 2);

            ph = prof_begin(INFERIO_PHASE_FC2);
            CUBLAS_CK(gemm_xw(g_m.ffn, f2w->dev, g_m.x, tr, kFF, kD, 1.0f));
            {
                long total = tr * kD;
                bias_add_kernel<<<(int)((total + 255) / 256), 256>>>(g_m.x, f2b->dev, total, kD);
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
