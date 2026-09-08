// GPU tensor + CUDA/cuBLAS error checking helpers.
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CUDA_CHECK(x)                                                                  \
    do {                                                                               \
        cudaError_t e_ = (x);                                                          \
        if (e_ != cudaSuccess) {                                                       \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_),        \
                    __FILE__, __LINE__);                                               \
            exit(1);                                                                   \
        }                                                                              \
    } while (0)

#define CUBLAS_CHECK(x)                                                                \
    do {                                                                               \
        cublasStatus_t s_ = (x);                                                       \
        if (s_ != CUBLAS_STATUS_SUCCESS) {                                             \
            fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)s_, __FILE__,           \
                    __LINE__);                                                         \
            exit(1);                                                                   \
        }                                                                              \
    } while (0)

#define CUDNN_CHECK(x)                                                                 \
    do {                                                                               \
        cudnnStatus_t s_ = (x);                                                        \
        if (s_ != CUDNN_STATUS_SUCCESS) {                                              \
            fprintf(stderr, "cuDNN error %s at %s:%d\n", cudnnGetErrorString(s_),      \
                    __FILE__, __LINE__);                                               \
            exit(1);                                                                   \
        }                                                                              \
    } while (0)

using bf16 = __nv_bfloat16;

enum DType { F32 = 0, BF16 = 1, F64 = 2, I64 = 3, U8 = 4 };

inline const char* dtype_name(DType t) {
    switch (t) {
        case F32: return "float32";
        case BF16: return "bfloat16";
        case F64: return "float64";
        case I64: return "int64";
        case U8: return "uint8";
    }
    return "?";
}
inline int64_t itemsize(DType t) {
    switch (t) {
        case F32: return 4;
        case BF16: return 2;
        case F64: return 8;
        case I64: return 8;
        case U8: return 1;
    }
    return 0;
}

struct Tensor {
    void* data = nullptr;  // device pointer
    std::vector<int64_t> shape;
    DType dtype = BF16;

    int64_t numel() const {
        int64_t n = 1;
        for (auto s : shape) n *= s;
        return n;
    }
    int64_t nbytes() const { return numel() * itemsize(dtype); }
    Tensor reshaped(std::vector<int64_t> s) const {
        Tensor t = *this;
        t.shape = std::move(s);
        return t;
    }
};

// Simple bump allocator over one big cudaMalloc.
struct Arena {
    char* base = nullptr;
    size_t cap = 0, used = 0;

    void reserve(size_t bytes) {
        if (cap >= bytes) return;
        if (base) CUDA_CHECK(cudaFree(base));
        CUDA_CHECK(cudaMalloc(&base, bytes));
        cap = bytes;
        used = 0;
    }
    void* alloc(size_t bytes) {
        size_t a = 512;
        used = (used + a - 1) / a * a;
        if (used + bytes > cap) {
            fprintf(stderr, "arena OOM: used %zu + %zu > cap %zu\n", used, bytes, cap);
            exit(1);
        }
        void* p = base + used;
        used += bytes;
        return p;
    }
    void reset() { used = 0; }
};

// -- elementwise helpers (implemented in common.cu) --
void* cuda_malloc_copy(const void* host, int64_t bytes);
void cast_fp32_to_bf16(const float* src, bf16* dst, int64_t n);
void cast_bf16_to_fp32(const bf16* src, float* dst, int64_t n);
void fill_bf16(bf16* dst, float v, int64_t n);
void fill_fp32(float* dst, float v, int64_t n);

// Download a tensor to host (returns malloc'd buffer).
void* to_host(const Tensor& t);
