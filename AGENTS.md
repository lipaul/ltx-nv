# AGENTS.md — ltx3_nv

Standalone re-implementation of LTX-2.5 (Lightricks audio+video diffusion) inference on an
RTX 6000 Ada (48 GB, SM 8.9), built to analyze TE / DiT / VAE step by step.

## Layout
- `reference/` — clone of https://github.com/Lightricks/LTX-2 (uv workspace). Its `.venv` is THE
  environment for everything here (`reference/.venv/bin/python`). Do not use `qwen_nv/.venv`
  (transformers 5.16 violates ltx-core's `<5.15` pin; Gemma-4 TE cannot load there).
- `baseline/` — `run_baseline.py` runs the OFFICIAL `TI2VidOneStagePipeline` (dev bf16) with
  monkey-patched instrumentation; artifacts in `baseline/artifacts/` (per-step latents, TE
  contexts, decoded video/audio, `meta.json`).
- `standalone/` — the deliverable: pure-PyTorch re-implementation, no `ltx_core`/`ltx_pipelines`
  imports. `ltx_nn.py` (RoPE/attention/AdaLN primitives) → `ltx_te.py` (Gemma-4 body via HF class
  + re-implemented feature extractor/connectors) → `ltx_dit.py` (LTXModel + CFG/STG batching +
  LTX2Scheduler + Euler loop) → `ltx_vae.py` (conv video decoder + audio decoder + BigVGAN/BWE).
  `run_all.py` orchestrates (`--stage te|dit|vae|all`, `--block-timing`), `compare.py` diffs
  against baseline, `debug_dit.py` compares intermediates vs the reference model.

## Weights (already on disk, do not re-download)
`/home/acm/work/models/ltx-2.5/` — split Comfy layout. Analysis config uses:
dev transformer bf16 (42 GB), `gemma4-12b-with-proj-ltx-2.5-bf16.safetensors` (25 GB, config +
tokenizer embedded in safetensors metadata), `ltx-2.5-video-vae-conv-bf16.safetensors` (conv
decoder — the non-`-conv` file is the DiffVAE and needs `natten`, which is NOT installed),
audio VAE + vocoder, duration head.

## Critical gotchas (all cost real debugging time)
- Run inference under `torch.inference_mode()`. Without it, autograd retains activations and the
  25 GB Gemma TE forward OOMs at ~45 GB (the official CLI's `main()` is decorated; scripts must too).
- TE (25 GB) and DiT (42 GB) cannot be GPU-resident together in 48 GB. Load → run → `del` →
  `free_cuda()` per stage; `free_cuda()` MUST `gc.collect()` before `empty_cache()` (nn.Module
  reference cycles keep 42 GB alive otherwise).
- SM 8.9: nvfp4 and FlashAttention-4 are unavailable (Blackwell-only). Attention = SDPA with
  priority `[CUDNN, FLASH, EFFICIENT, MATH]` (matches reference kernel choice).
- Background jobs from this shell: use `setsid ... & disown` or the tool's timeout kills the
  process group mid-run.
- Checkpoint configs live in safetensors metadata (`config.transformer`, `config.vae`,
  `gemma_config`), not in YAML. `baseline/artifacts/checkpoint_configs/*.json` are extracted copies.
- Weight keys need renaming on load: strip `model.diffusion_model.` (DiT), `decoder.`/
  `audio_vae.decoder.`/`vocoder.` (VAEs), `model.layers.` → `model.language_model.layers.` (Gemma,
  comfy-flat layout), `to_out.0.`→`to_out.`, `ff.net.0.proj.`→`ff.proj_in.`, `ff.net.2.`→`ff.proj_out.`,
  `transformer_1d_blocks`→`blocks` (connectors).
- LTX-2.5 dev pipeline defaults (from checkpoint `model_version=2.5.0`): 30 steps, CFG video 3.0 /
  audio 7.0, STG 1.0 on block 28, modality 3.0, rescale 0.7, seed 10, 512×768×121@24 (analysis run
  uses 65 frames). Guidance = 4 passes (cond/uncond/ptb/mod) batched into ONE B=4 transformer call.
- Gemma-4 needs non-persistent buffers (rotary inv_freq, embed_scale) materialized after meta
  build — see `ltx_te._populate_gemma_buffers`; lm_head is tied to embed_tokens.
- Video VAE `unpatchify` channel order is `(c, p_t, r→width, q→height)` with q fastest — easy to
  transpose wrongly.

## Commands
```bash
# fresh machine: bootstrap env (uv sync reference/.venv + verify + weights check)
./setup.sh
# baseline (official pipeline + instrumentation), ~3 min
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True reference/.venv/bin/python baseline/run_baseline.py
# standalone full run, ~5 min; add --block-timing for per-DiT-block GPU timing
reference/.venv/bin/python standalone/run_all.py --stage all --block-timing
# single stages / comparison
reference/.venv/bin/python standalone/run_all.py --stage dit
reference/.venv/bin/python standalone/compare.py
reference/.venv/bin/python standalone/debug_dit.py both   # ref-vs-mine intermediates
```
Frame counts must satisfy `(F-1) % 8 == 0`; H, W divisible by 32.

## Git workflow
Remote: `git@github.com:lipaul/ltx-nv.git` (branch `main`, SSH key `~/.ssh/id_ed25519_github`).
Per user instruction: **commit and push after every change** — don't wait to be asked.
`.gitignore` excludes `reference/` (nested clone + venv) and all run artifacts; never force-add them.

## Fidelity status
Standalone matches baseline: TE contexts cos≈1.0 (bf16 noise), scheduler + initial noise +
positions bit-exact, DiT per-step cos≥0.996 (bf16 kernel-order noise accumulates over 30 steps),
VAE decoders bit-exact given the same latent. Outputs are equivalent, not bit-identical.
