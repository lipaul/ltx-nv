#include "philox.h"
#include <curand_kernel.h>

// Mirrors calc_execution_policy (DistributionTemplates.h): block=256,
// grid = min(SM * (maxThreadsPerMultiProcessor/256), ceil(numel/256)).
static void calc_policy(int64_t numel, int64_t unroll, int64_t& grid, int64_t& block, uint64_t& counter_offset) {
    block = 256;
    grid = (numel + block - 1) / block;
    int dev;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    int64_t blocks_per_sm = prop.maxThreadsPerMultiProcessor / block;
    int64_t cap = (int64_t)prop.multiProcessorCount * blocks_per_sm;
    if (grid > cap) grid = cap;
    counter_offset = ((numel - 1) / (block * grid * unroll) + 1) * unroll;
}

template <typename T>
__global__ void __launch_bounds__(256) randn_kernel(int64_t numel, uint64_t seed, uint64_t offset, T* out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandStatePhilox4_32_10_t state;
    curand_init(seed, idx, offset, &state);
    int64_t total_threads = (int64_t)blockDim.x * gridDim.x;
    int64_t rounded_size = ((numel - 1) / (total_threads * 4) + 1) * total_threads * 4;
    for (int64_t linear_index = idx; linear_index < rounded_size; linear_index += total_threads * 4) {
        float4 rand = curand_normal4(&state);
        const float vals[4] = {rand.x, rand.y, rand.z, rand.w};
#pragma unroll
        for (int ii = 0; ii < 4; ii++) {
            int64_t li = linear_index + total_threads * ii;
            if (li < numel) out[li] = vals[ii];  // normal(0,1): mean + rand*std with mean=0,std=1
        }
    }
}

template <typename T>
static void randn_dispatch(T* out, int64_t numel, uint64_t seed, uint64_t offset) {
    int64_t grid, block;
    uint64_t counter_offset;
    calc_policy(numel, 4, grid, block, counter_offset);
    randn_kernel<T><<<(int)grid, (int)block>>>(numel, seed, offset, out);
    CUDA_CHECK(cudaGetLastError());
    (void)counter_offset;  // caller advances offset by the same formula
}

static int64_t policy_counter_offset(int64_t numel) {
    int64_t grid, block;
    uint64_t co;
    calc_policy(numel, 4, grid, block, co);
    return (int64_t)co;
}

void TorchRNG::randn_bf16(bf16* out, int64_t numel) {
    randn_dispatch(out, numel, seed, offset);
    offset += policy_counter_offset(numel);
}
void TorchRNG::randn_fp32(float* out, int64_t numel) {
    randn_dispatch(out, numel, seed, offset);
    offset += policy_counter_offset(numel);
}
