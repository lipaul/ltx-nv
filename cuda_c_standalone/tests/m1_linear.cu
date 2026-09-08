// M1 gate: linear (cuBLASLt bias epilogue) vs torch F.linear on the DiT patchify_proj.
#include "gemm.h"
#include "philox.h"
#include "st.h"
#include "dump.h"
#include <cstdio>

int main(int argc, char** argv) {
    const char* ckpt = argc > 1 ? argv[1]
        : "/home/acm/work/models/ltx-2.5/diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors";
    GpuCtx ctx;
    ctx.init();
    Arena arena;
    arena.reserve(1ull << 30);
    SafeTensorsFile f;
    f.open(ckpt);
    Tensor w = f.load("model.diffusion_model.patchify_proj.weight", BF16, arena);
    Tensor b = f.load("model.diffusion_model.patchify_proj.bias", BF16, arena);
    // x = video noise [3456,128] from m1 dump (regenerated here via philox for simplicity)
    TorchRNG rng;
    rng.manual_seed(10);
    Tensor x{nullptr, {3456, 128}, BF16};
    x.data = arena.alloc(x.nbytes());
    rng.randn_bf16((bf16*)x.data, x.numel());

    Tensor y{nullptr, {3456, 4096}, BF16};
    y.data = arena.alloc(y.nbytes());
    linear(ctx, x.data, w.data, b.data, y.data, 3456, 128, 4096);
    CUDA_CHECK(cudaDeviceSynchronize());

    DumpWriter dw;
    dw.begin("artifacts/m1_linear");
    dw.dump("patchify_out.bin", y);
    dw.end();
    printf("m1_linear done -> artifacts/m1/patchify_out.bin\n");
    return 0;
}
