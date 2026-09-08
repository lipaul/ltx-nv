"""Run the official LTX-2.5 dev one-stage pipeline with instrumentation.

Saves per-stage intermediates (text contexts, per-step latents/denoised passes,
final latents, decoded video/audio) + timing/VRAM stats so the standalone
implementation can be validated against them.

Run: reference/.venv/bin/python baseline/run_baseline.py
"""

import json
import logging
import time
from pathlib import Path

import torch
from safetensors import safe_open

import ltx_pipelines.utils.blocks as blocks_mod
import ltx_pipelines.utils.denoisers as denoisers_mod
from ltx_core.components.guiders import MultiModalGuiderParams
from ltx_pipelines.ti2vid_one_stage import TI2VidOneStagePipeline
from ltx_pipelines.utils.constants import DEFAULT_NEGATIVE_PROMPT
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.model_paths import ModelPaths

logging.basicConfig(level=logging.INFO)

MODELS = Path("/home/acm/work/models/ltx-2.5")
TRANSFORMER = str(MODELS / "diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors")
TEXT_ENCODER = str(MODELS / "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors")
VIDEO_VAE = str(MODELS / "vae/ltx-2.5-video-vae-conv-bf16.safetensors")
AUDIO_VAE = str(MODELS / "vae/ltx-2.5-audio-vae-bf16.safetensors")
DURATION_HEAD = str(MODELS / "model_patches/ltx-2.5-duration-head-bf16.safetensors")

PROMPT = "A red fox trots through fresh snow in a pine forest at dawn, soft golden light through the trees, cinematic."
HEIGHT, WIDTH, FRAMES, FPS = 512, 768, 65, 24.0
STEPS = 30
SEED = 10
MAX_BATCH_SIZE = 4

ART = Path(__file__).parent / "artifacts"
STEPS_DIR = ART / "dit_steps"
for d in [ART, STEPS_DIR, ART / "checkpoint_configs"]:
    d.mkdir(parents=True, exist_ok=True)

STATS: dict = {"args": {"prompt": PROMPT, "height": HEIGHT, "width": WIDTH, "frames": FRAMES, "fps": FPS,
                        "steps": STEPS, "seed": SEED, "max_batch_size": MAX_BATCH_SIZE,
                        "transformer": TRANSFORMER, "text_encoder": TEXT_ENCODER,
                        "video_vae": VIDEO_VAE, "audio_vae": AUDIO_VAE}}
_step_counter = {"i": 0}


def _vram() -> dict:
    return {
        "allocated_gib": round(torch.cuda.memory_allocated() / 2**30, 3),
        "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
    }


def _save(name: str, obj: object) -> None:
    torch.save(obj, ART / name)


def timed_stage(name: str):
    def deco(fn):
        def wrapper(*args, **kwargs):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            out = fn(*args, **kwargs)
            if name == "video_decode":  # lazy iterator: materialize to time + save
                out = list(out)
                _save("video_chunks.pt", [c.cpu() for c in out])
                out = iter(out)
            torch.cuda.synchronize()
            STATS.setdefault("stages", {})[name] = {"sec": round(time.perf_counter() - t0, 2), **_vram()}
            logging.info("[baseline] %s done in %.1fs %s", name, STATS["stages"][name]["sec"], _vram())
            return out
        return wrapper
    return deco


# ---------------------------------------------------------------------------
# Instrument the official blocks without touching the reference checkout.
# ---------------------------------------------------------------------------
_orig_prompt_encoder = blocks_mod.PromptEncoder.__call__


@timed_stage("text_encode")
def _pe(self, *args, **kwargs):
    out = _orig_prompt_encoder(self, *args, **kwargs)
    ctx_p, ctx_n = out
    _save("te_positive.pt", {"video_encoding": ctx_p.video_encoding.cpu(),
                             "audio_encoding": ctx_p.audio_encoding.cpu(),
                             "attention_mask": ctx_p.attention_mask.cpu()})
    _save("te_negative.pt", {"video_encoding": ctx_n.video_encoding.cpu(),
                             "audio_encoding": ctx_n.audio_encoding.cpu(),
                             "attention_mask": ctx_n.attention_mask.cpu()})
    return out


blocks_mod.PromptEncoder.__call__ = _pe

_orig_stage = blocks_mod.DiffusionStage.__call__


@timed_stage("dit")
def _ds(self, *args, **kwargs):
    sigmas = kwargs["sigmas"]
    _save("dit_sigmas.pt", sigmas.detach().cpu())
    STATS["sigmas"] = [round(float(s), 6) for s in sigmas]
    return _orig_stage(self, *args, **kwargs)


blocks_mod.DiffusionStage.__call__ = _ds

_orig_denoiser_call = denoisers_mod.FactoryGuidedDenoiser.__call__


def _fd(self, transformer, video_state, audio_state, sigmas, step_idx):
    result = _orig_denoiser_call(self, transformer, video_state, audio_state, sigmas, step_idx)
    v_res, a_res = result
    rec = {
        "step_idx": step_idx,
        "sigma": float(sigmas[step_idx]),
        "video_latent_in": video_state.latent.detach().cpu(),
        "video_denoise_mask": video_state.denoise_mask.detach().cpu(),
        "audio_latent_in": audio_state.latent.detach().cpu(),
        "audio_denoise_mask": audio_state.denoise_mask.detach().cpu(),
        "video_denoised": v_res.denoised.detach().cpu(),
        "audio_denoised": a_res.denoised.detach().cpu(),
    }
    for pass_name in ("cond", "uncond", "ptb", "mod"):
        v = getattr(v_res, pass_name, None)
        a = getattr(a_res, pass_name, None)
        if v is not None:
            rec[f"video_{pass_name}"] = v.detach().cpu()
        if a is not None:
            rec[f"audio_{pass_name}"] = a.detach().cpu()
    if step_idx == 0:
        rec["video_positions"] = video_state.positions.detach().cpu()
        rec["audio_positions"] = audio_state.positions.detach().cpu()
        rec["video_clean_latent"] = video_state.clean_latent.detach().cpu()
        rec["audio_clean_latent"] = audio_state.clean_latent.detach().cpu()
    torch.save(rec, STEPS_DIR / f"step_{step_idx:02d}.pt")
    _step_counter["i"] += 1
    return result


denoisers_mod.FactoryGuidedDenoiser.__call__ = _fd

_orig_video_decoder = blocks_mod.VideoDecoder.__call__
blocks_mod.VideoDecoder.__call__ = timed_stage("video_decode")(_orig_video_decoder)

_orig_audio_decoder = blocks_mod.AudioDecoder.__call__


@timed_stage("audio_decode")
def _ad(self, latent):
    out = _orig_audio_decoder(self, latent)
    _save("audio_latent_in.pt", latent.detach().cpu())
    _save("audio_waveform.pt", {"waveform": out.waveform.detach().cpu(), "sampling_rate": out.sampling_rate})
    return out


blocks_mod.AudioDecoder.__call__ = _ad

# ---------------------------------------------------------------------------
# Checkpoint configs (safetensors metadata) for the standalone build.
# ---------------------------------------------------------------------------
for tag, path in [("transformer", TRANSFORMER), ("text_encoder", TEXT_ENCODER),
                  ("video_vae", VIDEO_VAE), ("audio_vae", AUDIO_VAE)]:
    with safe_open(path, framework="pt") as f:
        meta = f.metadata() or {}
    (ART / "checkpoint_configs" / f"{tag}.json").write_text(json.dumps(meta, indent=1))

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
model_paths = ModelPaths.from_split(
    transformer_path=TRANSFORMER,
    text_encoder_path=TEXT_ENCODER,
    video_vae_path=VIDEO_VAE,
    audio_vae_path=AUDIO_VAE,
    duration_head_path=DURATION_HEAD,
)

pipeline = TI2VidOneStagePipeline(
    model_paths=model_paths,
    loras=(),
)

t0 = time.perf_counter()
with torch.inference_mode():  # matches the official CLI's main(); without it autograd keeps activations
    out = pipeline(
        prompt=PROMPT,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        seed=SEED,
        height=HEIGHT,
        width=WIDTH,
        num_frames=FRAMES,
        frame_rate=FPS,
        num_inference_steps=STEPS,
        video_guider_params=MultiModalGuiderParams(
            cfg_scale=3.0, stg_scale=1.0, rescale_scale=0.7, modality_scale=3.0, skip_step=0, stg_blocks=[28]
        ),
        audio_guider_params=MultiModalGuiderParams(
            cfg_scale=7.0, stg_scale=1.0, rescale_scale=0.7, modality_scale=3.0, skip_step=0, stg_blocks=[28]
        ),
        images=[],
        max_batch_size=MAX_BATCH_SIZE,
    )
    STATS["total_sec"] = round(time.perf_counter() - t0, 1)
    STATS["steps_recorded"] = _step_counter["i"]

    _save("final_video_latent.pt", out.video_latent.detach().cpu())

    encode_video(
        video=out.video,
        fps=FPS,
        audio=out.audio,
        output_path=str(ART / "baseline.mp4"),
        video_chunks_number=1,
    )

(ART / "meta.json").write_text(json.dumps(STATS, indent=1))
logging.info("[baseline] complete in %.1fs -> %s", STATS["total_sec"], ART)
