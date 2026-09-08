// Elementwise CUDA kernels shared across stages.
#include "tensor.h"

__global__ void k_cast_f32_bf16(const float* src, bf16* dst, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2bfloat16_rn(src[i]);
}
__global__ void k_cast_bf16_f32(const bf16* src, float* dst, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __bfloat162float(src[i]);
}
__global__ void k_fill_bf16(bf16* dst, float v, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2bfloat16_rn(v);
}
__global__ void k_fill_fp32(float* dst, float v, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = v;
}

static int blocks_for(int64_t n) { return (int)((n + 255) / 256); }

void cast_fp32_to_bf16(const float* src, bf16* dst, int64_t n) {
    k_cast_f32_bf16<<<blocks_for(n), 256>>>(src, dst, n);
    CUDA_CHECK(cudaGetLastError());
}
void cast_bf16_to_fp32(const bf16* src, float* dst, int64_t n) {
    k_cast_bf16_f32<<<blocks_for(n), 256>>>(src, dst, n);
    CUDA_CHECK(cudaGetLastError());
}
void fill_bf16(bf16* dst, float v, int64_t n) {
    k_fill_bf16<<<blocks_for(n), 256>>>(dst, v, n);
    CUDA_CHECK(cudaGetLastError());
}
void fill_fp32(float* dst, float v, int64_t n) {
    k_fill_fp32<<<blocks_for(n), 256>>>(dst, v, n);
    CUDA_CHECK(cudaGetLastError());
}
