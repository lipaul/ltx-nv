"""Compare standalone artifacts against the official-pipeline baseline.

Reports per-stage cosine similarity / max-abs error between
baseline/artifacts and standalone/artifacts.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from common import ARTIFACTS, BASELINE, cosine  # noqa: E402


def cmp(name: str, a: torch.Tensor, b: torch.Tensor) -> None:
    if a.shape != b.shape:
        print(f"{name:38s} SHAPE MISMATCH {tuple(a.shape)} vs {tuple(b.shape)}")
        return
    a, b = a.float(), b.float()
    diff = (a - b).abs()
    rel = diff.max().item() / (a.abs().max().item() + 1e-9)
    print(f"{name:38s} cos={cosine(a, b):.6f} maxabs={diff.max().item():.5f} meanabs={diff.mean().item():.6f} rel={rel:.4f}")


def main() -> None:
    print("== text encoder ==")
    for tag in ("te_positive", "te_negative"):
        base = torch.load(BASELINE / f"{tag}.pt")
        mine = torch.load(ARTIFACTS / f"{tag}.pt")
        for k in ("video_encoding", "audio_encoding", "attention_mask"):
            cmp(f"{tag}.{k}", base[k], mine[k])

    print("== scheduler ==")
    cmp("sigmas", torch.load(BASELINE / "dit_sigmas.pt"), torch.load(ARTIFACTS / "dit_sigmas.pt"))

    print("== DiT steps ==")
    for idx in (0, 1, 15, 29):
        bp, mp = BASELINE / "dit_steps" / f"step_{idx:02d}.pt", ARTIFACTS / "dit_steps" / f"step_{idx:02d}.pt"
        if not mp.exists():
            continue
        base, mine = torch.load(bp), torch.load(mp)
        for k in ("video_latent_in", "audio_latent_in", "video_denoised", "audio_denoised",
                  "video_cond", "video_uncond", "video_ptb", "video_mod",
                  "audio_cond", "audio_uncond", "audio_ptb", "audio_mod"):
            if k in base and k in mine and base[k] is not None:
                cmp(f"step{idx:02d}.{k}", base[k], mine[k])
        if idx == 0:
            for k in ("video_positions", "audio_positions"):
                cmp(f"step0.{k}", base[k], mine[k])

    print("== final latents ==")
    cmp("final_video_latent", torch.load(BASELINE / "final_video_latent.pt"),
        torch.load(ARTIFACTS / "final_video_latent.pt"))

    print("== VAE outputs ==")
    bchunks = torch.load(BASELINE / "video_chunks.pt")
    mchunks = torch.load(ARTIFACTS / "video_chunks.pt")
    cmp("video_chunks", torch.cat([c.reshape(-1) for c in bchunks]), torch.cat([c.reshape(-1) for c in mchunks]))
    bwave = torch.load(BASELINE / "audio_waveform.pt")
    mwave = torch.load(ARTIFACTS / "audio_waveform.pt")
    n = min(bwave["waveform"].shape[-1], mwave["waveform"].shape[-1])
    cmp("audio_waveform", bwave["waveform"][..., :n], mwave["waveform"][..., :n])
    print(f"baseline sr={bwave['sampling_rate']} standalone sr={mwave['sampling_rate']}")


if __name__ == "__main__":
    main()
