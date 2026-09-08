#!/usr/bin/env bash
# Set up the LTX-2.5 analysis environment from scratch (uv).
#
# Creates reference/.venv — the single environment used by baseline/ and standalone/
# scripts: torch 2.13+cu132, transformers 5.14.1 (ltx-core pins <5.15; Gemma-4 TE
# needs >=5.8), ltx-core + ltx-pipelines installed editable from the workspace.
#
# Usage: ./setup.sh          (idempotent; safe to re-run)
set -euo pipefail
cd "$(dirname "$0")"

REPO_URL="https://github.com/Lightricks/LTX-2.git"
REF_DIR="reference"
PYTHON_VERSION="3.12"
MODELS_DIR="${LTX_MODELS_DIR:-/home/acm/work/models/ltx-2.5}"

# --- 1. uv ------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
    echo "[setup] uv not found; installing"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "[setup] uv $(uv --version | awk '{print $2}')"

# --- 2. reference clone ------------------------------------------------------
if [ ! -d "$REF_DIR/.git" ]; then
    echo "[setup] cloning Lightricks/LTX-2 -> $REF_DIR/"
    git clone --depth 1 "$REPO_URL" "$REF_DIR"
else
    echo "[setup] $REF_DIR/ already cloned"
fi

# --- 3. sync the ltx-pipelines package (pulls ltx-core + torch cu132) --------
# --package keeps ltx-trainer and the opt-in CUDA `kernels` group out of the env
# (no nvcc needed; natten/DiffVAE intentionally unused — we decode with the conv VAE).
echo "[setup] uv sync --package ltx-pipelines (downloads ~5 GB of wheels on first run)"
(cd "$REF_DIR" && uv sync --package ltx-pipelines --python "$PYTHON_VERSION")

PY="$REF_DIR/.venv/bin/python"

# --- 4. verify the GPU stack --------------------------------------------------
"$PY" - <<'EOF'
import torch, transformers
import ltx_core, ltx_pipelines  # noqa: F401

assert torch.cuda.is_available(), "CUDA unavailable — check driver/NVIDIA visibility"
print(f"[setup] torch {torch.__version__} (cuda {torch.version.cuda}) on "
      f"{torch.cuda.get_device_name(0)} sm_{torch.cuda.get_device_capability(0)[0]}"
      f"{torch.cuda.get_device_capability(0)[1]}")
print(f"[setup] transformers {transformers.__version__} (must be >=5.8,<5.15)")
print("[setup] ltx_core / ltx_pipelines import OK")
EOF

# --- 5. weights (gated HF repo; this script does NOT download them) ----------
required=(
    "diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors"
    "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
    "vae/ltx-2.5-video-vae-conv-bf16.safetensors"
    "vae/ltx-2.5-audio-vae-bf16.safetensors"
    "model_patches/ltx-2.5-duration-head-bf16.safetensors"
)
missing=0
for f in "${required[@]}"; do
    if [ ! -f "$MODELS_DIR/$f" ]; then
        echo "[setup] MISSING $MODELS_DIR/$f" >&2
        missing=1
    fi
done
if [ "$missing" = 1 ]; then
    echo "[setup] download Lightricks/LTX-2.5 (gated — accept the license first), e.g.:" >&2
    echo "  hf download Lightricks/LTX-2.5 --local-dir $MODELS_DIR" >&2
    exit 1
fi
echo "[setup] weights present under $MODELS_DIR/"

echo "[setup] done. Run:"
echo "  $PY baseline/run_baseline.py                                  # official pipeline + instrumentation (~3 min)"
echo "  $PY standalone/run_all.py --stage all --block-timing          # standalone re-implementation (~5 min)"
echo "  $PY standalone/compare.py                                     # standalone vs baseline fidelity report"
