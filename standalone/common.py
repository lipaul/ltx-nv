"""Shared helpers for the standalone LTX-2.5 inference implementation.

Everything here is plain PyTorch + safetensors: no ltx_core / ltx_pipelines imports.
"""

import json
import math
import time
from pathlib import Path

import torch
from safetensors import safe_open

MODELS = Path("/home/acm/work/models/ltx-2.5")
TRANSFORMER = str(MODELS / "diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors")
TEXT_ENCODER = str(MODELS / "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors")
VIDEO_VAE = str(MODELS / "vae/ltx-2.5-video-vae-conv-bf16.safetensors")
AUDIO_VAE = str(MODELS / "vae/ltx-2.5-audio-vae-bf16.safetensors")
DURATION_HEAD = str(MODELS / "model_patches/ltx-2.5-duration-head-bf16.safetensors")

ARTIFACTS = Path(__file__).parent / "artifacts"
BASELINE = Path(__file__).parent.parent / "baseline" / "artifacts"

DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16


def read_metadata(path: str) -> dict:
    with safe_open(path, framework="pt") as f:
        meta = f.metadata() or {}
    out = {}
    for k, v in meta.items():
        try:
            out[k] = json.loads(v)
        except (json.JSONDecodeError, TypeError):
            out[k] = v
    return out


def load_state_dict(
    path: str,
    *,
    device: torch.device = DEVICE,
    dtype: torch.dtype | None = DTYPE,
    rename=None,
    keep=None,
) -> dict[str, torch.Tensor]:
    """Load tensors from a safetensors file, optionally renaming/filtering keys.

    rename: callable old_key -> new_key | None (None drops the key).
    keep:   callable old_key -> bool, applied before rename.
    """
    sd: dict[str, torch.Tensor] = {}
    with safe_open(path, framework="pt", device=str(device)) as f:
        for name in f.keys():  # noqa: SIM118
            if keep is not None and not keep(name):
                continue
            new = name if rename is None else rename(name)
            if new is None:
                continue
            t = f.get_tensor(name)
            if dtype is not None and t.is_floating_point():
                t = t.to(dtype)
            sd[new] = t
    return sd


def load_into(module: torch.nn.Module, sd: dict[str, torch.Tensor]) -> None:
    """Assign-style load (params replaced by the given tensors)."""
    missing = module.load_state_dict(sd, strict=False, assign=True)
    unexpected = [k for k in sd if k not in dict(module.named_parameters()) and k not in dict(module.named_buffers())]
    if unexpected:
        print(f"[load_into] {len(unexpected)} unexpected keys ignored, e.g. {unexpected[:4]}")


def free_cuda() -> None:
    import gc

    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


class StageTimer:
    """Context manager recording wall time + VRAM peak for one pipeline stage."""

    def __init__(self, name: str, log: dict | None = None):
        self.name = name
        self.log = log if log is not None else {}

    def __enter__(self):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        torch.cuda.synchronize()
        entry = {
            "sec": round(time.perf_counter() - self.t0, 2),
            "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
        }
        self.log[self.name] = entry
        print(f"[stage] {self.name}: {entry['sec']}s peak {entry['peak_gib']} GiB")


def tensor_stats(t: torch.Tensor) -> dict:
    tf = t.float()
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype).replace("torch.", ""),
        "mean": round(tf.mean().item(), 5),
        "std": round(tf.std().item(), 5),
        "absmax": round(tf.abs().max().item(), 5),
    }


def save_pt(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, path)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.gelu(x, approximate="tanh")


def rms_norm(x: torch.Tensor, eps: float = 1e-6, weight: torch.Tensor | None = None) -> torch.Tensor:
    return torch.nn.functional.rms_norm(x, (x.shape[-1],), weight=weight, eps=eps)


def get_timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Diffusers Timesteps(flip_sin_to_cos=True, downscale_freq_shift=0)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps[:, None].float() * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.nn.functional.pad(emb, (0, 1))
    return emb
