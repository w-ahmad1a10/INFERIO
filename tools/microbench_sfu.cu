// Microbenchmark: EX2 (exp2f) and FP32 FMA throughput on the target GPU.
// Purpose: decide whether attention is bound by softmax exp (SFU) or by FMA.
// Build: nvcc -O3 -arch=sm_75 tools/microbench_sfu.cu -o sfu
#include <cstdio>

#include <cuda_runtime.h>

__global__ void bench_exp2(float *out, long n, float seed) {
    long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
    long stride = (long)gridDim.x * blockDim.x;
    float acc = seed;
    for (long i = idx; i < n; i += stride) {
        float x = (float)(i & 1023) * -0.0009765625f;
        acc += exp2f(x);
    }
    out[idx] = acc;
}

__global__ void bench_fma(float *out, long n, float seed) {
    long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
    long stride = (long)gridDim.x * blockDim.x;
    float a0 = seed, a1 = seed + 1.f, a2 = seed + 2.f, a3 = seed + 3.f;
    const float b = 1.0000001f, c = 0.0000001f;
    for (long i = idx; i < n; i += stride) {
        a0 = fmaf(a0, b, c);
        a1 = fmaf(a1, b, c);
        a2 = fmaf(a2, b, c);
        a3 = fmaf(a3, b, c);
    }
    out[idx] = a0 + a1 + a2 + a3;
}

int main() {
    cudaSetDevice(0);
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    printf("device=%s sms=%d clock=%d kHz\n", p.name, p.multiProcessorCount, p.clockRate);

    int blocks = p.multiProcessorCount * 8, threads = 256;
    long n = 1L << 30;
    float *d = nullptr;
    cudaMalloc(&d, sizeof(float) * blocks * threads);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);

    bench_exp2<<<blocks, threads>>>(d, n, 1.f);
    cudaDeviceSynchronize();
    cudaEventRecord(e0);
    bench_exp2<<<blocks, threads>>>(d, n, 1.f);
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, e0, e1);
    printf("exp2f: %.2f Gexp/s (%.2f ms for %ld ops)\n", n / (ms * 1e6), ms, n);

    bench_fma<<<blocks, threads>>>(d, n, 1.f);
    cudaDeviceSynchronize();
    cudaEventRecord(e0);
    bench_fma<<<blocks, threads>>>(d, n, 1.f);
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    cudaEventElapsedTime(&ms, e0, e1);
    printf("fma  : %.2f GFMA/s (%.2f ms, %ld iterations)\n", 4.0 * n / (ms * 1e6), ms, n);
    return 0;
}
