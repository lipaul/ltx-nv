// cuBLAS / cuBLASLt wrappers: bf16 weights & activations, fp32 compute.
//   linear:  y[M,N] = x[M,K] @ W[N,K]^T + b[N]        (row-major, fused bias epilogue)
//   gemm_tt: batched  C[M,N] = A[M,K] @ B[N,K]^T *alpha (row-major, fp32 C out)
#pragma once
#include <cublasLt.h>
#include <cublas_v2.h>
#include "tensor.h"

struct GpuCtx {
    cublasHandle_t cublas;
    cublasLtHandle_t lt;
    void* ws = nullptr;
    size_t ws_size = 256ull << 20;  // 256 MB workspace (needed for the 188160-wide K GEMMs)

    void init();
};

// y = x@W^T + b; x bf16 [M,K], w bf16 [N,K], b bf16 [N] (may be null), y bf16 [M,N].
void linear(GpuCtx& ctx, const void* x, const void* w, const void* b, void* y, int64_t M, int64_t K, int64_t N);

// Row-major batched: C_b[i,j] = alpha * sum_k A_b[i,k] * B_b[j,k]; C fp32, A/B bf16.
// A: [B, M, K] ld K; B: [B, N, K] ld K (bstride 0 shares B); C: [B, M, N] ld M.
void gemm_tt_fp32out(GpuCtx& ctx, int64_t batch, int64_t M, int64_t N, int64_t K, float alpha,
                     const bf16* A, const bf16* B, float* C);

// Same but bf16 output (for O·V with P bf16).
void gemm_nt_bf16out(GpuCtx& ctx, int64_t batch, int64_t M, int64_t N, int64_t K, float alpha,
                     const bf16* A, const bf16* B, bf16* C);
