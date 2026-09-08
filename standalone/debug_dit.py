"""Debug: compare standalone DiT intermediates against ltx_core reference, stage by stage."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from common import TRANSFORMER, DEVICE, DTYPE, BASELINE, ARTIFACTS

STEP = torch.load(BASELINE / "dit_steps" / "step_00.pt")
TE_POS = torch.load(BASELINE / "te_positive.pt")


def get_modality_inputs():
    from ltx_core.model.transformer import Modality
    v_latent = STEP["video_latent_in"].to(DEVICE, DTYPE)
    a_latent = STEP["audio_latent_in"].to(DEVICE, DTYPE)
    v_mask = STEP["video_denoise_mask"].to(DEVICE)
    a_mask = STEP["audio_denoise_mask"].to(DEVICE)
    sigma = torch.tensor(1.0, device=DEVICE)
    v = Modality(
        latent=v_latent, sigma=sigma.expand(1), timesteps=v_mask * sigma,
        positions=STEP["video_positions"].to(DEVICE), context=TE_POS["video_encoding"].to(DEVICE, DTYPE),
        context_mask=None, attention_mask=None, keyframes_mask=torch.zeros_like(v_mask),
    )
    a = Modality(
        latent=a_latent, sigma=sigma.expand(1), timesteps=a_mask * sigma,
        positions=STEP["audio_positions"].to(DEVICE), context=TE_POS["audio_encoding"].to(DEVICE, DTYPE),
        context_mask=None, attention_mask=None, keyframes_mask=None,
    )
    return v, a


def run_reference():
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from ltx_core.loader.registry import ModelRegistry
    from ltx_core.model.transformer import LTXModelConfigurator, LTXV_MODEL_COMFY_RENAMING_MAP, X0Model
    from ltx_core.model.transformer.model import LTXModel  # noqa: F401

    builder = SingleGPUModelBuilder(
        model_path=TRANSFORMER, model_class_configurator=LTXModelConfigurator,
        model_sd_ops=LTXV_MODEL_COMFY_RENAMING_MAP,
        registry=ModelRegistry(cache_models=False, cache_weights=False),
    )
    model = builder.build(device=DEVICE, dtype=DTYPE).eval()
    v, a = get_modality_inputs()
    out = {}

    def hook(name):
        def fn(_m, _i, o):
            if isinstance(o, torch.Tensor):
                out[name] = o.detach().float().cpu()
        return fn

    def block_hook(tag):
        def fn(_m, _i, o):
            out[f"v_{tag}"] = o[0].x.detach().float().cpu()
            if o[1] is not None:
                out[f"a_{tag}"] = o[1].x.detach().float().cpu()
        return fn

    handles = [
        model.transformer_blocks[0].register_forward_hook(block_hook("block0")),
        model.transformer_blocks[1].register_forward_hook(block_hook("block1")),
    ]
    with torch.inference_mode():
        vx, ax = model(video=v, audio=a, perturbations=None)
    out["v_x0"] = (v.latent.float() - vx.float() * v.timesteps.float()).cpu()
    out["a_x0"] = (a.latent.float() - ax.float() * a.timesteps.float()).cpu()
    # also capture prepared args
    vp = model.video_args_preprocessor.prepare(v, a)
    out["v_x"] = vp.x.detach().float().cpu()
    out["v_timestep"] = vp.timesteps.detach().float().cpu()
    out["v_embedded"] = vp.embedded_timestep.detach().float().cpu()
    out["v_pe_cos"] = vp.positional_embeddings[0].detach().float().cpu()
    out["v_cross_pe_cos"] = vp.cross_positional_embeddings[0].detach().float().cpu()
    for h in handles:
        h.remove()
    torch.save(out, ARTIFACTS / "ref_intermediates.pt")
    del model, vp, v, a
    import gc

    gc.collect()
    torch.cuda.empty_cache()


def run_standalone():
    from ltx_dit import DiTConfig, Modality as SModality, load_transformer
    cfg = DiTConfig.from_checkpoint(TRANSFORMER)
    model = load_transformer(TRANSFORMER, cfg)
    v, a = get_modality_inputs()
    sv = SModality(latent=v.latent, sigma=v.sigma, timesteps=v.timesteps, positions=v.positions,
                   context=v.context, keyframes_mask=v.keyframes_mask)
    sa = SModality(latent=a.latent, sigma=a.sigma, timesteps=a.timesteps, positions=a.positions, context=a.context)
    out = {}

    def hook(name):
        def fn(_m, _i, o):
            if isinstance(o, torch.Tensor):
                out[name] = o.detach().float().cpu()
        return fn

    def block_hook(tag):
        def fn(_m, _i, o):
            out[f"v_{tag}"] = o[0].x.detach().float().cpu()
            out[f"a_{tag}"] = o[1].x.detach().float().cpu()
        return fn

    model.blocks[0].register_forward_hook(block_hook("block0"))
    model.blocks[1].register_forward_hook(block_hook("block1"))
    with torch.inference_mode():
        vx, ax = model(sv, sa, None)
    out["v_x0"] = (sv.latent.float() - vx.float() * sv.timesteps.float()).cpu()
    out["a_x0"] = (sa.latent.float() - ax.float() * sa.timesteps.float()).cpu()
    vp = model.video_pre.prepare(sv, sa)
    out["v_x"] = vp.x.detach().float().cpu()
    out["v_timestep"] = vp.timesteps.detach().float().cpu()
    out["v_embedded"] = vp.embedded_timestep.detach().float().cpu()
    out["v_pe_cos"] = vp.pe[0].detach().float().cpu()
    out["v_cross_pe_cos"] = vp.cross_pe[0].detach().float().cpu()
    torch.save(out, ARTIFACTS / "mine_intermediates.pt")


def compare():
    r = torch.load(ARTIFACTS / "ref_intermediates.pt")
    m = torch.load(ARTIFACTS / "mine_intermediates.pt")
    for k in r:
        if k not in m:
            continue
        a, b = r[k], m[k]
        if a.shape != b.shape:
            print(f"{k:16s} SHAPE {tuple(a.shape)} vs {tuple(b.shape)}")
            continue
        d = (a - b).abs()
        print(f"{k:16s} maxabs={d.max().item():.6f} meanabs={d.mean().item():.7f} ref_absmax={a.abs().max().item():.4f}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("ref", "both"):
        run_reference()
    if which in ("mine", "both"):
        run_standalone()
    compare()
