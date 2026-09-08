#include "gemm.h"

void GpuCtx::init() {
    CUBLAS_CHECK(cublasCreate(&cublas));
    CUBLAS_CHECK(cublasLtCreate(&lt));
    CUDA_CHECK(cudaMalloc(&ws, ws_size));
    // torch defaults: TF32 disabled for matmul — keep full fp32 compute
    CUBLAS_CHECK(cublasSetMathMode(cublas, CUBLAS_DEFAULT_MATH));
}

static const float kBetaZero = 0.0f;

// Row-major batched: C_b = A_b @ B_b^T * alpha.  col-major recipe: opA=T, opB=T.
// A row-major [M,K] == col-major [K,M] ld K; op_T gives (m,k) element A[m,k].
// B row-major [N,K] == col-major [K,N] ld K; op_T gives (k,n) element B[n,k].
// C col-major [M,N] ld M == row-major [M,N].
static void gemm_batched_common(GpuCtx& ctx, int64_t batch, int64_t M, int64_t N, int64_t K, float alpha,
                                const void* A, int64_t as, const void* B, int64_t bs, void* C, int64_t cs,
                                cudaDataType_t ctype) {
    CUBLAS_CHECK(cublasGemmStridedBatchedEx(
        ctx.cublas, CUBLAS_OP_T, CUBLAS_OP_T, (int)M, (int)N, (int)K, &alpha, A, CUDA_R_16BF, (int)K, as,
        B, CUDA_R_16BF, (int)K, bs, &kBetaZero, C, ctype, (int)M, cs, (int)batch, CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT));
}

void gemm_tt_fp32out(GpuCtx& ctx, int64_t batch, int64_t M, int64_t N, int64_t K, float alpha,
                     const bf16* A, const bf16* B, float* C) {
    gemm_batched_common(ctx, batch, M, N, K, alpha, A, M * K, B, N * K, C, M * N, CUDA_R_32F);
}

void gemm_nt_bf16out(GpuCtx& ctx, int64_t batch, int64_t M, int64_t N, int64_t K, float alpha,
                     const bf16* A, const bf16* B, bf16* C) {
    gemm_batched_common(ctx, batch, M, N, K, alpha, A, M * K, B, N * K, C, M * N, CUDA_R_16BF);
}

void linear(GpuCtx& ctx, const void* x, const void* w, const void* b, void* y, int64_t M, int64_t K, int64_t N) {
    cublasLtMatmulDesc_t op;
    CUBLAS_CHECK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t tA = CUBLAS_OP_T, tB = CUBLAS_OP_N;
    // A = W col-major [K,N] (row-major [N,K]) op_T -> (n,k) = W[n,k]; B = x col-major [K,M] op_N
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &tA, sizeof(tA)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tB, sizeof(tB)));
    if (b) {
        cublasLtEpilogue_t epi = CUBLASLT_EPILOGUE_BIAS;
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi)));
        const void* bp = b;
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bp, sizeof(bp)));
    }
    cublasLtMatrixLayout_t la, lb, lc;
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&la, CUDA_R_16BF, K, N, K));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&lb, CUDA_R_16BF, K, M, K));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&lc, CUDA_R_16BF, N, M, N));
    float alpha = 1.0f, beta = 0.0f;

    cublasLtMatmulPreference_t pref;
    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                      &ctx.ws_size, sizeof(ctx.ws_size)));
    cublasLtMatmulHeuristicResult_t heur;
    int nres = 0;
    CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(ctx.lt, op, la, lb, lc, lc, pref, 1, &heur, &nres));
    if (nres == 0) {
        fprintf(stderr, "no cublasLt algo for M=%ld K=%ld N=%ld\n", (long)M, (long)K, (long)N);
        exit(1);
    }
    CUBLAS_CHECK(cublasLtMatmul(ctx.lt, op, &alpha, w, la, x, lb, &beta, y, lc, y, lc, &heur.algo,
                                ctx.ws, ctx.ws_size, 0));
    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(la);
    cublasLtMatrixLayoutDestroy(lb);
    cublasLtMatrixLayoutDestroy(lc);
    cublasLtMatmulDescDestroy(op);
}
