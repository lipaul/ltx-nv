// M1 gate: randn(seed=10) bit-identical to torch for the video (442368) and audio (8704)
// noise tensors, in the exact call order of the baseline pipeline.
#include "philox.h"
#include "dump.h"
#include <cstdio>

int main() {
    TorchRNG rng;
    rng.manual_seed(10);
    Arena arena;
    arena.reserve(64 << 20);
    // baseline order: video noise [1,3456,128] then audio noise [1,68,128]
    Tensor v{nullptr, {1, 3456, 128}, BF16};
    Tensor a{nullptr, {1, 68, 128}, BF16};
    v.data = arena.alloc(v.nbytes());
    a.data = arena.alloc(a.nbytes());
    rng.randn_bf16((bf16*)v.data, v.numel());
    rng.randn_bf16((bf16*)a.data, a.numel());

    DumpWriter dw;
    dw.begin("artifacts/m1_philox");
    dw.dump("video_noise.bin", v);
    dw.dump("audio_noise.bin", a);
    dw.end();
    printf("m1_philox done -> artifacts/m1_philox\n");
    return 0;
}
