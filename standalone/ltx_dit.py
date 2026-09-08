"""Standalone LTX-2.5 DiT (LTXModel): 48-block audio-video transformer + guidance + sampling.

Re-implements ltx_core.model.transformer with plain PyTorch, loading the dev
transformer safetensors directly (key rename: strip ``model.diffusion_model.``).

Data flow per denoising step (dev model, CFG+STG+modality guidance):
    4 guidance passes (cond | uncond | STG-perturbed | modality-isolated) are batched
    into ONE transformer call with B=4; per-sample perturbation keep-masks disable the
    skipped attentions; the blended x0 = cond + (cfg-1)(cond-uncond) + stg(cond-ptb)
    + (mod-1)(cond-mod), rescaled, then an Euler step advances the latent.
"""

import math
import time
from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F

from common import DTYPE, DEVICE, read_metadata, load_state_dict
from ltx_nn import AdaLayerNormSingle, Attention, FeedForward, precompute_freqs_cis, rms_norm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiTConfig:
    num_layers: int = 48
    video_heads: int = 32
    video_head_dim: int = 128
    video_in: int = 128
    video_out: int = 128
    video_cross_dim: int = 4096
    audio_heads: int = 32
    audio_head_dim: int = 64
    audio_in: int = 128
    audio_out: int = 128
    audio_cross_dim: int = 2048
    norm_eps: float = 1e-6
    theta: float = 10000.0
    video_max_pos: tuple = (20, 2048, 2048)
    audio_max_pos: tuple = (20,)
    timestep_scale_multiplier: int = 1000
    av_ca_timestep_scale_multiplier: float = 1000.0
    double_rope: bool = True
    gated_attention: bool = True
    cross_attention_adaln: bool = True
    ff_bias: bool = False
    audio_ff_bias: bool = True
    use_keyframes: bool = True

    @staticmethod
    def from_checkpoint(transformer_path: str) -> "DiTConfig":
        c = read_metadata(transformer_path)["config"]["transformer"]
        return DiTConfig(
            num_layers=c["num_layers"], video_heads=c["num_attention_heads"],
            video_head_dim=c["attention_head_dim"], video_in=c["in_channels"], video_out=c["out_channels"],
            video_cross_dim=c["cross_attention_dim"], audio_heads=c["audio_num_attention_heads"],
            audio_head_dim=c["audio_attention_head_dim"], audio_in=c.get("audio_in_channels", 128),
            audio_out=c["audio_out_channels"], audio_cross_dim=c["audio_cross_attention_dim"],
            norm_eps=c["norm_eps"], theta=c["positional_embedding_theta"],
            video_max_pos=tuple(c["positional_embedding_max_pos"]),
            audio_max_pos=tuple(c["audio_positional_embedding_max_pos"]),
            timestep_scale_multiplier=c["timestep_scale_multiplier"],
            av_ca_timestep_scale_multiplier=c["av_ca_timestep_scale_multiplier"],
            double_rope=c.get("frequencies_precision", False) == "float64",
            gated_attention=c.get("apply_gated_attention", False),
            cross_attention_adaln=c.get("cross_attention_adaln", False),
            ff_bias=c.get("ff_bias", True), audio_ff_bias=c.get("audio_ff_bias", True),
            use_keyframes=c.get("use_keyframes_abs_pos_embedding", False),
        )

    @property
    def video_dim(self) -> int:
        return self.video_heads * self.video_head_dim

    @property
    def audio_dim(self) -> int:
        return self.audio_heads * self.audio_head_dim


# ---------------------------------------------------------------------------
# Transformer args (per-modality prepared inputs)
# ---------------------------------------------------------------------------


@dataclass
class Modality:
    latent: torch.Tensor  # (B, T, 128) patchified
    sigma: torch.Tensor  # (B,)
    timesteps: torch.Tensor  # (B, T, 1) = denoise_mask * sigma
    positions: torch.Tensor  # video (B,3,T,2) pixel-space bounds; audio (B,1,T,2) seconds
    context: torch.Tensor  # (B, S, cross_dim) text encoding
    keyframes_mask: torch.Tensor | None = None  # (B, T, 1)
    enabled: bool = True


@dataclass
class TransformerArgs:
    x: torch.Tensor
    context: torch.Tensor
    context_mask: torch.Tensor | None
    timesteps: torch.Tensor  # (B, T, 9*D) embedded
    embedded_timestep: torch.Tensor  # (B, T, D)
    pe: tuple | None
    cross_pe: tuple | None
    cross_scale_shift_timestep: torch.Tensor | None
    cross_gate_timestep: torch.Tensor | None
    enabled: bool
    prompt_timestep: torch.Tensor | None
    self_attention_mask: torch.Tensor | None = None
    self_attn_perturbation_mask: torch.Tensor | None = None
    self_attn_all_perturbed: bool = False
    cross_attn_perturbation_mask: torch.Tensor | None = None
    cross_attn_skip_all: bool = False


class ArgsPreprocessor:
    """Mirrors TransformerArgsPreprocessor / MultiModalTransformerArgsPreprocessor."""

    def __init__(self, model: "LTXModel", stream: str):
        self.m = model
        self.stream = stream  # "video" | "audio"

    @property
    def cfg(self) -> DiTConfig:
        return self.m.cfg

    def prepare(self, modality: Modality, cross: Modality | None) -> TransformerArgs:
        m = self.m
        cfg = self.cfg
        stream = self.stream
        patchify = m.audio_patchify_proj if stream == "audio" else m.patchify_proj
        adaln = m.audio_adaln_single if stream == "audio" else m.adaln_single
        prompt_adaln = m.audio_prompt_adaln_single if stream == "audio" else m.prompt_adaln_single
        dim = cfg.audio_dim if stream == "audio" else cfg.video_dim
        inner = cfg.audio_in if stream == "audio" else cfg.video_in

        x = patchify(modality.latent)
        if stream == "video" and modality.keyframes_mask is not None and m.keyframes_abs_pos_embedding is not None:
            mask = (modality.keyframes_mask > 0).to(x.dtype)
            x = x + mask * m.keyframes_abs_pos_embedding.to(x.dtype)

        ts = modality.timesteps * cfg.timestep_scale_multiplier
        timestep, embedded = adaln(ts.flatten(), hidden_dtype=modality.latent.dtype)
        b = x.shape[0]
        timestep = timestep.view(b, -1, timestep.shape[-1])
        embedded = embedded.view(b, -1, embedded.shape[-1])
        prompt_timestep = None
        if prompt_adaln is not None:
            pt, _ = prompt_adaln((modality.sigma * cfg.timestep_scale_multiplier).flatten(),
                                 hidden_dtype=modality.latent.dtype)
            prompt_timestep = pt.view(b, -1, pt.shape[-1])

        context = modality.context.view(b, -1, x.shape[-1])
        max_pos = list(cfg.audio_max_pos) if stream == "audio" else list(cfg.video_max_pos)
        heads = cfg.audio_heads if stream == "audio" else cfg.video_heads
        pe = precompute_freqs_cis(
            modality.positions, dim=dim, out_dtype=x.dtype, theta=cfg.theta, max_pos=max_pos,
            use_middle_indices_grid=True, num_attention_heads=heads, double_precision=cfg.double_rope,
        )
        cross_pe = cross_ss = cross_gate = None
        if cross is not None:
            cross_pe = precompute_freqs_cis(
                modality.positions[:, 0:1, :], dim=cfg.audio_cross_dim, out_dtype=x.dtype, theta=cfg.theta,
                max_pos=[max(cfg.video_max_pos[0], cfg.audio_max_pos[0])], use_middle_indices_grid=True,
                num_attention_heads=heads, double_precision=cfg.double_rope,
            )
            cross_adaln = m.av_ca_audio_scale_shift_adaln_single if stream == "audio" \
                else m.av_ca_video_scale_shift_adaln_single
            gate_adaln = m.av_ca_v2a_gate_adaln_single if stream == "audio" else m.av_ca_a2v_gate_adaln_single
            cst, _ = cross_adaln(ts.flatten(), hidden_dtype=x.dtype)
            cross_ss = cst.view(b, -1, cst.shape[-1])
            factor = cfg.av_ca_timestep_scale_multiplier / cfg.timestep_scale_multiplier
            cgt, _ = gate_adaln((cross.sigma * cfg.timestep_scale_multiplier * factor).flatten(),
                                hidden_dtype=x.dtype)
            cross_gate = cgt.view(b, -1, cgt.shape[-1])
        return TransformerArgs(
            x=x, context=context, context_mask=None, timesteps=timestep, embedded_timestep=embedded,
            pe=pe, cross_pe=cross_pe, cross_scale_shift_timestep=cross_ss, cross_gate_timestep=cross_gate,
            enabled=modality.enabled, prompt_timestep=prompt_timestep,
        )


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------


def get_ada_values(table: torch.Tensor, batch: int, timestep: torch.Tensor, indices: slice) -> tuple:
    num = table.shape[0]
    values = (
        table[indices].unsqueeze(0).unsqueeze(0).to(device=timestep.device, dtype=timestep.dtype)
        + timestep.reshape(batch, timestep.shape[1], num, -1)[:, :, indices, :]
    ).unbind(dim=2)
    return values


def get_av_ca_ada_values(table, batch, scale_shift_ts, gate_ts, ss_indices):
    ss = get_ada_values(table[:4, :], batch, scale_shift_ts, ss_indices)
    g = get_ada_values(table[4:, :], batch, gate_ts, slice(None, None))
    scale, shift = (t.squeeze(2) for t in ss)
    (gate,) = (t.squeeze(2) for t in g)
    return scale, shift, gate


class Block(torch.nn.Module):
    """BasicAVTransformerBlock: video + audio streams with A<->V cross attention."""

    def __init__(self, cfg: DiTConfig):
        super().__init__()
        v, a = cfg.video_dim, cfg.audio_dim
        n_ca = 9 if cfg.cross_attention_adaln else 6
        self.attn1 = Attention(v, None, cfg.video_heads, cfg.video_head_dim, cfg.norm_eps, cfg.gated_attention)
        self.attn2 = Attention(v, cfg.video_cross_dim, cfg.video_heads, cfg.video_head_dim, cfg.norm_eps,
                               cfg.gated_attention)
        self.ff = FeedForward(v, v, bias=cfg.ff_bias)
        self.scale_shift_table = torch.nn.Parameter(torch.empty(n_ca, v))
        self.audio_attn1 = Attention(a, None, cfg.audio_heads, cfg.audio_head_dim, cfg.norm_eps, cfg.gated_attention)
        self.audio_attn2 = Attention(a, cfg.audio_cross_dim, cfg.audio_heads, cfg.audio_head_dim, cfg.norm_eps,
                                     cfg.gated_attention)
        self.audio_ff = FeedForward(a, a, bias=cfg.audio_ff_bias)
        self.audio_scale_shift_table = torch.nn.Parameter(torch.empty(n_ca, a))
        # A2V: Q video (heads=audio_heads, d_head=audio_head_dim per reference), KV audio
        self.audio_to_video_attn = Attention(v, a, cfg.audio_heads, cfg.audio_head_dim, cfg.norm_eps,
                                             cfg.gated_attention)
        self.video_to_audio_attn = Attention(a, v, cfg.audio_heads, cfg.audio_head_dim, cfg.norm_eps,
                                             cfg.gated_attention)
        self.scale_shift_table_a2v_ca_audio = torch.nn.Parameter(torch.empty(5, a))
        self.scale_shift_table_a2v_ca_video = torch.nn.Parameter(torch.empty(5, v))
        if cfg.cross_attention_adaln:
            self.prompt_scale_shift_table = torch.nn.Parameter(torch.empty(2, v))
            self.audio_prompt_scale_shift_table = torch.nn.Parameter(torch.empty(2, a))

    def _text_cross(self, x_normed, context, attn, sst, prompt_sst, timestep, prompt_timestep,
                    context_mask, cross_adaln):
        if cross_adaln:
            shift_q, scale_q, gate = get_ada_values(sst, x_normed.shape[0], timestep, slice(6, 9))
            kv = prompt_sst[None, None].to(x_normed.device, x_normed.dtype)
            if prompt_timestep is not None:
                kv = kv + prompt_timestep.reshape(x_normed.shape[0], prompt_timestep.shape[1], 2, -1)
            shift_kv, scale_kv = kv.unbind(dim=2)
            attn_input = x_normed * (1 + scale_q) + shift_q
            enc = context * (1 + scale_kv) + shift_kv
            return attn(attn_input, context=enc, mask=context_mask) * gate
        return attn(x_normed, context=context, mask=context_mask)

    def forward(self, video: TransformerArgs | None, audio: TransformerArgs | None):
        cfg_ca = self.scale_shift_table.shape[0] == 9
        vx = video.x if video is not None else None
        ax = audio.x if audio is not None else None
        run_vx = video is not None and video.enabled and vx.numel() > 0
        run_ax = audio is not None and audio.enabled and ax.numel() > 0
        run_a2v = run_vx and ax.numel() > 0
        run_v2a = run_ax and vx.numel() > 0

        if run_vx:
            vshift, vscale, vgate = get_ada_values(self.scale_shift_table, vx.shape[0], video.timesteps, slice(0, 3))
            norm_vx = rms_norm(vx, eps=self.norm_eps) * (1 + vscale) + vshift
            out = self.attn1(norm_vx, pe=video.pe, mask=video.self_attention_mask,
                             perturbation_mask=video.self_attn_perturbation_mask,
                             all_perturbed=video.self_attn_all_perturbed)
            vx_fma = vx + out * vgate
            vx_normed = rms_norm(vx_fma, eps=self.norm_eps)
            vx = vx_fma + self._text_cross(vx_normed, video.context, self.attn2, self.scale_shift_table,
                                           getattr(self, "prompt_scale_shift_table", None),
                                           video.timesteps, video.prompt_timestep, video.context_mask, cfg_ca)
        if run_ax:
            ashift, ascale, agate = get_ada_values(self.audio_scale_shift_table, ax.shape[0], audio.timesteps,
                                                   slice(0, 3))
            norm_ax = rms_norm(ax, eps=self.norm_eps) * (1 + ascale) + ashift
            out = self.audio_attn1(norm_ax, pe=audio.pe, mask=audio.self_attention_mask,
                                   perturbation_mask=audio.self_attn_perturbation_mask,
                                   all_perturbed=audio.self_attn_all_perturbed)
            ax_fma = ax + out * agate
            ax_normed = rms_norm(ax_fma, eps=self.norm_eps)
            ax = ax_fma + self._text_cross(ax_normed, audio.context, self.audio_attn2,
                                           self.audio_scale_shift_table,
                                           getattr(self, "audio_prompt_scale_shift_table", None),
                                           audio.timesteps, audio.prompt_timestep, audio.context_mask, cfg_ca)

        if run_a2v or run_v2a:
            vx_pre, ax_pre = vx, ax
            if run_a2v and not video.cross_attn_skip_all:
                s, sh, gate = get_av_ca_ada_values(self.scale_shift_table_a2v_ca_video, vx.shape[0],
                                                   video.cross_scale_shift_timestep, video.cross_gate_timestep,
                                                   slice(0, 2))
                a2v_v = rms_norm(vx_pre, eps=self.norm_eps) * (1 + s) + sh
                s, sh, _ = get_av_ca_ada_values(self.scale_shift_table_a2v_ca_audio, ax.shape[0],
                                                audio.cross_scale_shift_timestep, audio.cross_gate_timestep,
                                                slice(0, 2))
                a2v_a = rms_norm(ax_pre, eps=self.norm_eps) * (1 + s) + sh
                vx = vx + self.audio_to_video_attn(a2v_v, context=a2v_a, pe=video.cross_pe,
                                                   k_pe=audio.cross_pe) * gate * video.cross_attn_perturbation_mask
            if run_v2a and not audio.cross_attn_skip_all:
                s, sh, gate = get_av_ca_ada_values(self.scale_shift_table_a2v_ca_audio, ax.shape[0],
                                                   audio.cross_scale_shift_timestep, audio.cross_gate_timestep,
                                                   slice(2, 4))
                v2a_a = rms_norm(ax_pre, eps=self.norm_eps) * (1 + s) + sh
                s, sh, _ = get_av_ca_ada_values(self.scale_shift_table_a2v_ca_video, vx.shape[0],
                                                video.cross_scale_shift_timestep, video.cross_gate_timestep,
                                                slice(2, 4))
                v2a_v = rms_norm(vx_pre, eps=self.norm_eps) * (1 + s) + sh
                ax = ax + self.video_to_audio_attn(v2a_a, context=v2a_v, pe=audio.cross_pe,
                                                   k_pe=video.cross_pe) * gate * audio.cross_attn_perturbation_mask

        if run_vx:
            sh, sc, gate = get_ada_values(self.scale_shift_table, vx.shape[0], video.timesteps, slice(3, 6))
            vx = vx + self.ff(rms_norm(vx, eps=self.norm_eps) * (1 + sc) + sh) * gate
        if run_ax:
            sh, sc, gate = get_ada_values(self.audio_scale_shift_table, ax.shape[0], audio.timesteps, slice(3, 6))
            ax = ax + self.audio_ff(rms_norm(ax, eps=self.norm_eps) * (1 + sc) + sh) * gate

        return (replace(video, x=vx) if video is not None else None,
                replace(audio, x=ax) if audio is not None else None)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class LTXModel(torch.nn.Module):
    def __init__(self, cfg: DiTConfig):
        super().__init__()
        self.cfg = cfg
        v, a = cfg.video_dim, cfg.audio_dim
        n_adaln = 9 if cfg.cross_attention_adaln else 6
        self.patchify_proj = torch.nn.Linear(cfg.video_in, v, bias=True)
        self.keyframes_abs_pos_embedding = torch.nn.Parameter(torch.zeros(1, v)) if cfg.use_keyframes else None
        self.adaln_single = AdaLayerNormSingle(v, n_adaln)
        self.prompt_adaln_single = AdaLayerNormSingle(v, 2) if cfg.cross_attention_adaln else None
        self.scale_shift_table = torch.nn.Parameter(torch.empty(2, v))
        self.norm_out = torch.nn.LayerNorm(v, elementwise_affine=False, eps=cfg.norm_eps)
        self.proj_out = torch.nn.Linear(v, cfg.video_out)
        self.audio_patchify_proj = torch.nn.Linear(cfg.audio_in, a, bias=True)
        self.audio_adaln_single = AdaLayerNormSingle(a, n_adaln)
        self.audio_prompt_adaln_single = AdaLayerNormSingle(a, 2) if cfg.cross_attention_adaln else None
        self.audio_scale_shift_table = torch.nn.Parameter(torch.empty(2, a))
        self.audio_norm_out = torch.nn.LayerNorm(a, elementwise_affine=False, eps=cfg.norm_eps)
        self.audio_proj_out = torch.nn.Linear(a, cfg.audio_out)
        self.av_ca_video_scale_shift_adaln_single = AdaLayerNormSingle(v, 4)
        self.av_ca_audio_scale_shift_adaln_single = AdaLayerNormSingle(a, 4)
        self.av_ca_a2v_gate_adaln_single = AdaLayerNormSingle(v, 1)
        self.av_ca_v2a_gate_adaln_single = AdaLayerNormSingle(a, 1)
        self.blocks = torch.nn.ModuleList([Block(cfg) for _ in range(cfg.num_layers)])
        for blk in self.blocks:
            blk.norm_eps = cfg.norm_eps
        self.video_pre = ArgsPreprocessor(self, "video")
        self.audio_pre = ArgsPreprocessor(self, "audio")

    def forward(self, video: Modality | None, audio: Modality | None, perturbations) -> tuple:
        va = self.video_pre.prepare(video, audio) if video is not None else None
        aa = self.audio_pre.prepare(audio, video) if audio is not None else None
        if perturbations is None:
            ref = (va or aa).x
            perturbations = torch.ones(4, self.cfg.num_layers, ref.shape[0], device=ref.device, dtype=ref.dtype)
        for idx, block in enumerate(self.blocks):
            if va is not None:
                va = apply_block_perturbations(va, perturbations, idx, 0, 2)
            if aa is not None:
                aa = apply_block_perturbations(aa, perturbations, idx, 1, 3)
            va, aa = block(va, aa)
        vx = self._output(self.scale_shift_table, self.norm_out, self.proj_out, va) if va is not None else None
        ax = self._output(self.audio_scale_shift_table, self.audio_norm_out, self.audio_proj_out, aa) \
            if aa is not None else None
        return vx, ax

    @staticmethod
    def _output(sst, norm_out, proj_out, args) -> torch.Tensor:
        x = args.x
        ss = sst[None, None].to(x.device, x.dtype) + args.embedded_timestep[:, :, None]
        shift, scale = ss[:, :, 0], ss[:, :, 1]
        x = norm_out(x)
        x = x * (1 + scale) + shift
        return proj_out(x)


def apply_block_perturbations(args: TransformerArgs, pcfg, block_idx: int, self_type: int, cross_type: int):
    """BlockPerturbationsProcessor equivalent: attach this block's keep-masks."""
    if pcfg is None:
        return args
    all_self = bool((pcfg[self_type, block_idx] == 0).all())
    any_self = bool((pcfg[self_type, block_idx] == 0).any())
    all_cross = bool((pcfg[cross_type, block_idx] == 0).all())
    return replace(
        args,
        self_attn_perturbation_mask=pcfg[self_type, block_idx].reshape(-1, 1, 1) if any_self and not all_self else None,
        self_attn_all_perturbed=all_self,
        cross_attn_perturbation_mask=None if all_cross else pcfg[cross_type, block_idx].reshape(-1, 1, 1),
        cross_attn_skip_all=all_cross,
    )


def build_perturbation_config(num_blocks: int, passes: list[str], v_stg_blocks: list[int],
                              a_stg_blocks: list[int], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """(4, num_blocks, B) keep-mask. Types: 0 skip-video-self, 1 skip-audio-self, 2 skip-A2V, 3 skip-V2A."""
    b = len(passes)
    keep = torch.ones(4, num_blocks, b, device=device, dtype=dtype)
    for i, name in enumerate(passes):
        if name == "ptb":
            for blk in v_stg_blocks:
                keep[0, blk, i] = 0
            for blk in a_stg_blocks:
                keep[1, blk, i] = 0
        elif name == "mod":
            keep[2, :, i] = 0
            keep[3, :, i] = 0
    return keep


def load_transformer(path: str, cfg: DiTConfig) -> LTXModel:
    with torch.device("meta"):
        model = LTXModel(cfg)

    def rename(key: str) -> str | None:
        if not key.startswith("model.diffusion_model."):
            return None
        k = key.removeprefix("model.diffusion_model.")
        k = k.replace("transformer_blocks.", "blocks.")
        k = k.replace("to_out.0.", "to_out.").replace("ff.net.0.proj.", "ff.proj_in.").replace("ff.net.2.", "ff.proj_out.")
        k = k.replace("audio_ff.net.0.proj.", "audio_ff.proj_in.").replace("audio_ff.net.2.", "audio_ff.proj_out.")
        return k

    sd = load_state_dict(path, rename=rename)
    res = model.load_state_dict(sd, strict=False, assign=True)
    missing = [k for k in res.missing_keys if k != "keyframes_abs_pos_embedding"]
    if missing:
        raise RuntimeError(f"transformer missing keys: {missing[:10]} ({len(missing)} total)")
    return model.to(DEVICE).eval()


# ---------------------------------------------------------------------------
# Latent state construction (video/audio tools + noiser)
# ---------------------------------------------------------------------------

SCALE_T, SCALE_H, SCALE_W = 8, 32, 32


def video_positions(batch: int, f: int, h: int, w: int, fps: float, device) -> torch.Tensor:
    """(B, 3, T, 2) pixel-space [start, end) bounds per token, time axis divided by fps, causal fix."""
    coords = torch.meshgrid(
        torch.arange(f, device=device), torch.arange(h, device=device), torch.arange(w, device=device),
        indexing="ij",
    )
    starts = torch.stack(coords, dim=0)  # (3, f, h, w)
    ends = starts + 1
    bounds = torch.stack([starts, ends], dim=-1)  # (3, f, h, w, 2)
    bounds = bounds.reshape(3, -1, 2).unsqueeze(0).expand(batch, -1, -1, -1).clone().float()  # (B,3,T,2)
    scale = torch.tensor([SCALE_T, SCALE_H, SCALE_W], device=device).view(1, 3, 1, 1)
    bounds = bounds * scale
    bounds[:, 0] = (bounds[:, 0] + 1 - SCALE_T).clamp(min=0)
    bounds[:, 0] = bounds[:, 0] / fps
    return bounds


def audio_positions(batch: int, frames: int, device) -> torch.Tensor:
    """(B, 1, T, 2) latent-frame [start, end) bounds in seconds (causal mel timing)."""
    idx = torch.arange(frames, dtype=torch.float32, device=device)
    mel_start = ((idx * 4 + 1 - 4).clamp(min=0)) * 160 / 16000
    mel_end = (((idx + 1) * 4 + 1 - 4).clamp(min=0)) * 160 / 16000
    bounds = torch.stack([mel_start, mel_end], dim=-1)  # (T, 2)
    return bounds[None, None].expand(batch, 1, -1, -1).contiguous()


def patchify_video(latent: torch.Tensor) -> torch.Tensor:
    return latent.permute(0, 2, 3, 4, 1).reshape(latent.shape[0], -1, latent.shape[1])


def unpatchify_video(tokens: torch.Tensor, f: int, h: int, w: int) -> torch.Tensor:
    b, _, c = tokens.shape
    return tokens.view(b, f, h, w, c).permute(0, 4, 1, 2, 3)


def patchify_audio(latent: torch.Tensor) -> torch.Tensor:
    return latent.permute(0, 2, 1, 3).reshape(latent.shape[0], latent.shape[2], -1)


def unpatchify_audio(tokens: torch.Tensor, channels: int, mel_bins: int) -> torch.Tensor:
    b, t, cf = tokens.shape
    return tokens.view(b, t, channels, mel_bins).permute(0, 2, 1, 3)


@dataclass
class LatentState:
    latent: torch.Tensor
    denoise_mask: torch.Tensor
    positions: torch.Tensor
    clean_latent: torch.Tensor
    keyframes_mask: torch.Tensor | None = None


def create_noised_states(batch: int, num_video_frames: int, height: int, width: int, fps: float,
                         seed: int, device: torch.device, dtype: torch.dtype) -> tuple[LatentState, LatentState]:
    """Replicates DiffusionStage._build_state for the unconditioned t2v case (noise_scale=1)."""
    generator = torch.Generator(device=device).manual_seed(seed)
    f = (num_video_frames - 1) // SCALE_T + 1
    h, w = height // SCALE_H, width // SCALE_W
    v_tokens = f * h * w
    video_noise = torch.randn(batch, v_tokens, 128, device=device, dtype=dtype, generator=generator)
    keyframes_mask = torch.zeros(batch, v_tokens, 1, device=device, dtype=torch.float32)
    keyframes_mask[:, : h * w] = 1.0  # first latent frame encodes a single pixel frame (causal VAE)
    video_state = LatentState(
        latent=video_noise,
        denoise_mask=torch.ones(batch, v_tokens, 1, device=device, dtype=torch.float32),
        positions=video_positions(batch, f, h, w, fps, device),
        clean_latent=torch.zeros_like(video_noise),
        keyframes_mask=keyframes_mask,
    )
    audio_frames = round(num_video_frames / fps * 25.0)
    audio_noise = torch.randn(batch, audio_frames, 128, device=device, dtype=dtype, generator=generator)
    audio_state = LatentState(
        latent=audio_noise,
        denoise_mask=torch.ones(batch, audio_frames, 1, device=device, dtype=torch.float32),
        positions=audio_positions(batch, audio_frames, device),
        clean_latent=torch.zeros_like(audio_noise),
    )
    return video_state, audio_state


# ---------------------------------------------------------------------------
# Scheduler + guidance + sampling loop
# ---------------------------------------------------------------------------


def ltx2_scheduler_sigmas(steps: int, tokens: int = 4096, max_shift: float = 2.05, base_shift: float = 0.95,
                          terminal: float = 0.1) -> torch.Tensor:
    sigmas = torch.linspace(1.0, 0.0, steps + 1)
    mm = (max_shift - base_shift) / (4096 - 1024)
    shift = tokens * mm + (base_shift - mm * 1024)
    e = math.exp(shift)
    sigmas = torch.where(sigmas != 0, e / (e + (1 / sigmas - 1)), sigmas)
    nz = sigmas != 0
    one_minus = 1.0 - sigmas[nz]
    scale = one_minus[-1] / (1.0 - terminal)
    sigmas[nz] = 1.0 - one_minus / scale
    return sigmas.to(torch.float32)


@dataclass(frozen=True)
class GuiderParams:
    cfg_scale: float
    stg_scale: float
    rescale_scale: float
    modality_scale: float
    stg_blocks: tuple[int, ...]


def guide(cond, uncond, ptb, mod, p: GuiderParams) -> torch.Tensor:
    pred = (
        cond.float()
        + (p.cfg_scale - 1) * (cond.float() - uncond.float())
        + p.stg_scale * (cond.float() - ptb.float())
        + (p.modality_scale - 1) * (cond.float() - mod.float())
    )
    if p.rescale_scale != 0:
        factor = cond.float().std() / pred.std()
        factor = p.rescale_scale * factor + (1 - p.rescale_scale)
        pred = pred * factor
    return pred.to(cond.dtype)


def guided_denoise_step(model: LTXModel, video_state: LatentState, audio_state: LatentState,
                        contexts: dict, sigma: torch.Tensor, step_idx: int,
                        v_params: GuiderParams, a_params: GuiderParams, cfg: DiTConfig) -> tuple:
    """One transformer call with B=4 guidance passes -> blended x0 per modality."""
    passes = ["cond", "uncond", "ptb", "mod"]
    n = len(passes)
    v_ctx = torch.cat([contexts["v_pos"], contexts["v_neg"], contexts["v_pos"], contexts["v_pos"]], dim=0)
    a_ctx = torch.cat([contexts["a_pos"], contexts["a_neg"], contexts["a_pos"], contexts["a_pos"]], dim=0)
    sigma_b = sigma.expand(n)  # (4,) float32
    v_latent = video_state.latent.repeat(n, 1, 1)
    a_latent = audio_state.latent.repeat(n, 1, 1)
    video = Modality(
        latent=v_latent, sigma=sigma_b,
        timesteps=video_state.denoise_mask.repeat(n, 1, 1) * sigma_b.view(n, 1, 1),
        positions=video_state.positions.repeat(n, 1, 1, 1), context=v_ctx,
        keyframes_mask=video_state.keyframes_mask.repeat(n, 1, 1),
    )
    audio = Modality(
        latent=a_latent, sigma=sigma_b,
        timesteps=audio_state.denoise_mask.repeat(n, 1, 1) * sigma_b.view(n, 1, 1),
        positions=audio_state.positions.repeat(n, 1, 1, 1), context=a_ctx,
    )
    perturbations = build_perturbation_config(cfg.num_layers, passes, list(v_params.stg_blocks),
                                              list(a_params.stg_blocks), video_state.latent.device,
                                              video_state.latent.dtype)
    vx, ax = model(video, audio, perturbations)
    # X0Model: per-token velocity -> x0 = latent - v * timesteps (fp32 math, cast back)
    x0_v = (v_latent.float() - vx.float() * video.timesteps.float()).to(vx.dtype)
    x0_a = (a_latent.float() - ax.float() * audio.timesteps.float()).to(ax.dtype)
    vs = x0_v.chunk(n)
    as_ = x0_a.chunk(n)
    r = dict(zip(passes, zip(vs, as_, strict=True), strict=True))
    denoised_video = guide(*[r[p][0] for p in passes], v_params)
    denoised_audio = guide(*[r[p][1] for p in passes], a_params)
    return denoised_video, denoised_audio, r


def euler_step(sample: torch.Tensor, denoised: torch.Tensor, sigmas: torch.Tensor, idx: int) -> torch.Tensor:
    sigma = sigmas[idx].float()
    dt = (sigmas[idx + 1] - sigmas[idx]).float()
    velocity = ((sample.float() - denoised.float()) / sigma).to(sample.dtype)
    return (sample.float() + velocity.float() * dt).to(sample.dtype)


@torch.inference_mode()
def run_dit(model: LTXModel, cfg: DiTConfig, contexts: dict, video_state: LatentState, audio_state: LatentState,
            sigmas: torch.Tensor, v_params: GuiderParams, a_params: GuiderParams,
            on_step=None) -> tuple[torch.Tensor, torch.Tensor]:
    for idx in range(len(sigmas) - 1):
        t0 = time.perf_counter()
        dv, da, passes = guided_denoise_step(model, video_state, audio_state, contexts, sigmas[idx], idx,
                                             v_params, a_params, cfg)
        video_state = replace(video_state, latent=euler_step(video_state.latent, dv, sigmas, idx))
        audio_state = replace(audio_state, latent=euler_step(audio_state.latent, da, sigmas, idx))
        if on_step:
            on_step(idx, time.perf_counter() - t0, dv, da, passes, video_state, audio_state)
    return video_state.latent, audio_state.latent
