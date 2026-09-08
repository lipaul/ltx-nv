"""Compare cuda_c_standalone .bin dumps against baseline/standalone .pt artifacts.

Usage: tools/compare_c.py <manifest.json> <baseline_dir> [spec...]
spec = c_name:pt_name   (defaults: same names as manifest keys, mapped per-stage)
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "standalone"))
from common import cosine  # noqa: E402


def read_bin(path: str, dtype: str, shape: list) -> torch.Tensor:
    dts = {"bfloat16": torch.bfloat16, "float32": torch.float32, "int64": torch.int64, "uint8": torch.uint8}
    t = torch.frombuffer(Path(path).read_bytes(), dtype=dts[dtype]).clone()
    return t.reshape(shape)


def main() -> None:
    manifest = json.loads(Path(sys.argv[1]).read_text())
    art_dir = str(Path(sys.argv[1]).parent)
    baseline_dir = sys.argv[2]
    specs = {}
    for s in sys.argv[3:]:
        # spec: c_name[:pt_file[:pt_key]]   (pt_file relative to baseline_dir)
        parts = s.split(":")
        specs[parts[0]] = parts[1:] if len(parts) > 1 else []
    n_bad = 0
    for name, meta in manifest.items():
        if name == "_end":
            continue
        mine = read_bin(f"{art_dir}/{name}", meta["dtype"], meta["shape"]).float()
        spec = specs.get(name, [])
        if spec:
            pt_file = baseline_dir + "/" + spec[0]
            key = spec[1] if len(spec) > 1 else None
        else:
            pt_file = baseline_dir + "/" + name.replace(".bin", ".pt")
            key = None
        base = torch.load(pt_file)
        if isinstance(base, dict):
            key = key or next(iter(base))
            base = base[key]
        if isinstance(base, list):
            base = base[0]
        base = base.cpu().float().reshape(-1)
        mine = mine.float().reshape(-1)
        if base.shape != mine.shape:
            print(f"{name:34s} SHAPE {tuple(base.shape)} vs {tuple(mine.shape)}")
            n_bad += 1
            continue
        d = (base - mine).abs()
        bitwise = d.max().item() == 0.0
        if not bitwise:
            n_bad += 1
            print(f"{name:34s} cos={cosine(base, mine):.6f} maxabs={d.max().item():.6f} meanabs={d.mean().item():.7f}")
        else:
            print(f"{name:34s} BITWISE")
    print("all bitwise" if n_bad == 0 else f"{n_bad} tensor(s) differ")


if __name__ == "__main__":
    main()
