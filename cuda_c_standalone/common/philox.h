// torch-compatible CUDA randn: replicates at::native::distribution_elementwise_grid_stride_kernel
// (float path, curand4_engine_calls=4) so that randn(seed=10) is bit-identical to PyTorch's.
#pragma once
#include "tensor.h"

struct TorchRNG {
    uint64_t seed = 0;
    uint64_t offset = 0;  // philox_offset_per_thread_

    void manual_seed(uint64_t s) {
        seed = s;
        offset = 0;
    }
    // normal(mean=0, std=1), bf16 output — matches torch.randn(..., dtype=bf16)
    void randn_bf16(bf16* out, int64_t numel);
    // fp32 output variant (same philox stream semantics)
    void randn_fp32(float* out, int64_t numel);
};
