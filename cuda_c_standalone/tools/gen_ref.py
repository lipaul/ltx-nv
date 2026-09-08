"""Generate torch reference tensors for cuda_c_standalone tests.

Usage: gen_ref.py <out_dir> <which> ...
which:
  philox     -> video_noise.pt / audio_noise.pt (torch.randn, seed 10, baseline order)
  patchify   -> x (noise [3456,128]), y = F.linear(x, W, b) from dev transformer
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "standalone"))
from common import TRANSFORMER, DTYPE, DEVICE  # noqa: E402
from safetensors import safe_open  # noqa: E402


def philox(out_dir: str) -> None:
    g = torch.Generator(device=DEVICE).manual_seed(10)
    v = torch.randn(1, 3456, 128, device=DEVICE, dtype=torch.bfloat16, generator=g)
    a = torch.randn(1, 68, 128, device=DEVICE, dtype=torch.bfloat16, generator=g)
    torch.save(v.cpu(), out_dir + "/video_noise.pt")
    torch.save(a.cpu(), out_dir + "/audio_noise.pt")


def patchify(out_dir: str) -> None:
    g = torch.Generator(device=DEVICE).manual_seed(10)
    v = torch.randn(1, 3456, 128, device=DEVICE, dtype=torch.bfloat16, generator=g)  # offset 0
    a = torch.randn(1, 68, 128, device=DEVICE, dtype=torch.bfloat16, generator=g)  # offset 4
    with safe_open(TRANSFORMER, framework="pt", device=str(DEVICE)) as f:
        w = f.get_tensor("model.diffusion_model.patchify_proj.weight").to(DTYPE)
        b = f.get_tensor("model.diffusion_model.patchify_proj.bias").to(DTYPE)
    x = v.reshape(3456, 128)
    with torch.inference_mode():
        y = torch.nn.functional.linear(x, w, b)
    torch.save(x.cpu(), out_dir + "/x.pt")
    torch.save(y.cpu(), out_dir + "/patchify_out.pt")


if __name__ == "__main__":
    out_dir = sys.argv[1]
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        globals()[sys.argv[2]](out_dir)
    print("gen_ref done")
