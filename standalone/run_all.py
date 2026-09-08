"""Standalone LTX-2.5 inference: TE -> DiT -> VAE, each stage explicit and instrumented.

No ltx_core / ltx_pipelines imports: weights are loaded straight from safetensors and every
op is plain PyTorch (see ltx_te.py / ltx_dit.py / ltx_vae.py).

Usage:
    reference/.venv/bin/python standalone/run_all.py [--steps 30] [--no-save-steps]
    reference/.venv/bin/python standalone/run_all.py --stage te|dit|vae   (single stage)
"""

import argparse
import json
import subprocess
import sys
import time
import wave
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from common import (ARTIFACTS, AUDIO_VAE, BASELINE, DEVICE, DTYPE, TEXT_ENCODER, TRANSFORMER,
                    VIDEO_VAE, free_cuda, save_pt, tensor_stats)

PROMPT = "A red fox trots through fresh snow in a pine forest at dawn, soft golden light through the trees, cinematic."
NEGATIVE = (
    "has_subtitles, has_blurbox, transition from black, transition to black, speech_ending_short, "
    "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, "
    "grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, "
    "deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, "
    "wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of "
    "field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent "
    "lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny "
    "valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, "
    "mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, "
    "off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward "
    "pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, "
    "inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts."
)
HEIGHT, WIDTH, FRAMES, FPS = 512, 768, 65, 24.0
SEED = 10
V_PARAMS = dict(cfg_scale=3.0, stg_scale=1.0, rescale_scale=0.7, modality_scale=3.0, stg_blocks=(28,))
A_PARAMS = dict(cfg_scale=7.0, stg_scale=1.0, rescale_scale=0.7, modality_scale=3.0, stg_blocks=(28,))

STATS: dict = {"args": {"prompt": PROMPT, "height": HEIGHT, "width": WIDTH, "frames": FRAMES, "fps": FPS,
                        "seed": SEED}}


def stage_te() -> list[dict]:
    from ltx_te import encode_prompts
    with torch.inference_mode():
        out = encode_prompts([PROMPT, NEGATIVE], TEXT_ENCODER, TRANSFORMER, stats=STATS)
    save_pt(ARTIFACTS / "te_positive.pt", out[0])
    save_pt(ARTIFACTS / "te_negative.pt", out[1])
    STATS["te_stats"] = {"video": tensor_stats(out[0]["video_encoding"]),
                         "audio": tensor_stats(out[0]["audio_encoding"])}
    free_cuda()
    return out


def stage_dit(contexts_cpu: list[dict] | None = None) -> None:
    from ltx_dit import (DiTConfig, GuiderParams, create_noised_states, ltx2_scheduler_sigmas,
                         load_transformer, run_dit, unpatchify_audio, unpatchify_video)

    if contexts_cpu is None:
        contexts_cpu = [torch.load(ARTIFACTS / "te_positive.pt"), torch.load(ARTIFACTS / "te_negative.pt")]
    pos, neg = contexts_cpu
    contexts = {
        "v_pos": pos["video_encoding"].to(DEVICE, DTYPE), "a_pos": pos["audio_encoding"].to(DEVICE, DTYPE),
        "v_neg": neg["video_encoding"].to(DEVICE, DTYPE), "a_neg": neg["audio_encoding"].to(DEVICE, DTYPE),
    }
    cfg = DiTConfig.from_checkpoint(TRANSFORMER)
    sigmas = ltx2_scheduler_sigmas(STATS["steps"]).to(DEVICE)
    save_pt(ARTIFACTS / "dit_sigmas.pt", sigmas.cpu())
    v_state, a_state = create_noised_states(1, FRAMES, HEIGHT, WIDTH, FPS, SEED, DEVICE, DTYPE)
    v_params = GuiderParams(**V_PARAMS)
    a_params = GuiderParams(**A_PARAMS)

    steps_dir = ARTIFACTS / "dit_steps"
    steps_dir.mkdir(exist_ok=True)
    save_steps = not STATS.get("no_save_steps")
    block_timing = bool(STATS.get("block_timing"))
    prev = {"v": v_state.latent.cpu().clone(), "a": a_state.latent.cpu().clone()}
    step_times: list[float] = []

    def on_step(idx, sec, dv, da, passes, vs, as_):
        step_times.append(round(sec, 3))
        if not save_steps:
            return
        rec = {"step_idx": idx, "sigma": float(sigmas[idx]),
               "video_latent_in": prev["v"], "audio_latent_in": prev["a"],
               "video_denoised": dv.cpu(), "audio_denoised": da.cpu()}
        if idx == 0:
            rec["video_positions"] = v_state.positions.cpu()
            rec["audio_positions"] = a_state.positions.cpu()
            rec["video_denoise_mask"] = v_state.denoise_mask.cpu()
            rec["audio_denoise_mask"] = a_state.denoise_mask.cpu()
        for name, (v, a) in passes.items():
            rec[f"video_{name}"] = v.cpu()
            rec[f"audio_{name}"] = a.cpu()
        torch.save(rec, steps_dir / f"step_{idx:02d}.pt")
        prev["v"] = vs.latent.cpu()
        prev["a"] = as_.latent.cpu()

    t0 = time.perf_counter()
    with torch.inference_mode():
        model = load_transformer(TRANSFORMER, cfg)
        STATS["dit_load_sec"] = round(time.perf_counter() - t0, 1)
        if block_timing:
            block_secs = [0.0] * cfg.num_layers
            stamp = {"t": None}

            def pre(_m, _i, k):
                torch.cuda.synchronize()
                now = time.perf_counter()
                if stamp["t"] is not None:
                    block_secs[k - 1] += now - stamp["t"]
                stamp["t"] = now

            for k, blk in enumerate(model.blocks):
                blk.register_forward_pre_hook(lambda m, i, k=k: pre(m, i, k))
            stamp["t"] = None
        torch.cuda.reset_peak_memory_stats()
        t1 = time.perf_counter()
        v_tokens, a_tokens = run_dit(model, cfg, contexts, v_state, a_state, sigmas, v_params, a_params, on_step)
        torch.cuda.synchronize()
        STATS.setdefault("stages", {})["dit"] = {"sec": round(time.perf_counter() - t1, 2),
                                                 "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3)}
        STATS["step_times_sec"] = step_times
        if block_timing:
            n_steps = len(step_times)
            STATS["block_times_sec_per_step"] = [round(x / n_steps, 4) for x in block_secs]
            STATS["block_total_sec_per_step"] = round(sum(block_secs) / n_steps, 2)
        print(f"[stage] dit: {STATS['stages']['dit']}")
        f = (FRAMES - 1) // 8 + 1
        h, w = HEIGHT // 32, WIDTH // 32
        final_video = unpatchify_video(v_tokens, f, h, w)
        final_audio = unpatchify_audio(a_tokens, 8, 16)
    save_pt(ARTIFACTS / "final_video_latent.pt", final_video.cpu())
    save_pt(ARTIFACTS / "final_audio_latent.pt", final_audio.cpu())
    del model, contexts
    free_cuda()


def stage_vae() -> None:
    from ltx_vae import (decode_audio, load_audio_decoder, load_video_decoder, load_vocoder)

    video_latent = torch.load(ARTIFACTS / "final_video_latent.pt").to(DEVICE)
    audio_latent = torch.load(ARTIFACTS / "final_audio_latent.pt").to(DEVICE)
    with torch.inference_mode():
        t0 = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        vdec = load_video_decoder(VIDEO_VAE)
        chunks = vdec.decode_video(video_latent)
        torch.cuda.synchronize()
        STATS.setdefault("stages", {})["video_decode"] = {
            "sec": round(time.perf_counter() - t0, 2),
            "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3)}
        save_pt(ARTIFACTS / "video_chunks.pt", [chunks.cpu()])
        del vdec
        free_cuda()

        t1 = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        adec = load_audio_decoder(AUDIO_VAE)
        voc = load_vocoder(AUDIO_VAE)
        waveform = decode_audio(audio_latent, adec, voc)
        torch.cuda.synchronize()
        STATS["stages"]["audio_decode"] = {
            "sec": round(time.perf_counter() - t1, 2),
            "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3)}
        save_pt(ARTIFACTS / "audio_waveform.pt", {"waveform": waveform.cpu(), "sampling_rate": voc.output_sampling_rate})
    write_mp4(chunks.cpu(), waveform.cpu(), voc.output_sampling_rate, ARTIFACTS / "standalone.mp4")


def write_mp4(video: torch.Tensor, waveform: torch.Tensor, sr: int, path: Path) -> None:
    """video: (F,H,W,C) float [0,1]; waveform: (2,N) float [-1,1]. Uses ffmpeg."""
    f, h, w, _ = video.shape
    raw = (video.clamp(0, 1) * 255).to(torch.uint8).contiguous().numpy().tobytes()
    tmp_raw = path.with_suffix(".raw")
    tmp_raw.write_bytes(raw)
    tmp_wav = str(path.with_suffix(".wav"))
    with wave.open(tmp_wav, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        pcm = (waveform.clamp(-1, 1).T * 32767).to(torch.int16).contiguous().numpy().tobytes()
        wf.writeframes(pcm)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
         "-r", str(FPS), "-i", str(tmp_raw), "-i", tmp_wav,
         "-vf", "scale=in_color_matrix=bt709:out_color_matrix=bt709",
         "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
         "-c:a", "aac", "-shortest", str(path)],
        check=True,
    )
    tmp_raw.unlink()
    Path(tmp_wav).unlink()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["te", "dit", "vae", "all"], default="all")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--no-save-steps", action="store_true")
    ap.add_argument("--block-timing", action="store_true")
    args = ap.parse_args()
    STATS["steps"] = args.steps
    STATS["no_save_steps"] = args.no_save_steps
    STATS["block_timing"] = args.block_timing
    ARTIFACTS.mkdir(exist_ok=True)
    t0 = time.perf_counter()
    if args.stage in ("te", "all"):
        contexts = stage_te()
        if args.stage == "all":
            stage_dit(contexts)
    elif args.stage == "dit":
        stage_dit()
    if args.stage in ("vae", "all"):
        stage_vae()
    STATS["total_sec"] = round(time.perf_counter() - t0, 1)
    (ARTIFACTS / f"meta_{args.stage}.json").write_text(json.dumps(STATS, indent=1, default=str))
    (ARTIFACTS / "meta.json").write_text(json.dumps(STATS, indent=1, default=str))
    print(f"[standalone] done in {STATS['total_sec']}s -> {ARTIFACTS}")


if __name__ == "__main__":
    main()
