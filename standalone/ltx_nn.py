"""Plain-PyTorch neural building blocks shared by the standalone TE and DiT.

Mirrors ltx_core's transformer primitives (RoPE, Attention, AdaLN, FF) with
explicit shapes in the docstrings. No ltx_core imports.
"""

import functools
import math

import numpy
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

SDPA_PRIORITY = [SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]


def rms_norm(x: torch.Tensor, eps: float = 1e-6, weight: torch.Tensor | None = None) -> torch.Tensor:
    return F.rms_norm(x, (x.shape[-1],), weight=weight, eps=eps)


def sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """q,k,v: (B, H, T, D). Same backend-priority walk as the reference PytorchAttention."""
    with sdpa_kernel(SDPA_PRIORITY, set_priority=True):
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
    return out


# ---------------------------------------------------------------------------
# RoPE (LTXRopeType.SPLIT)
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=8)
def generate_freq_grid_np(theta: float, n_axes: int, inner_dim: int) -> torch.Tensor:
    n_elem = 2 * n_axes
    pow_indices = numpy.power(
        theta,
        numpy.linspace(
            numpy.log(1.0) / numpy.log(theta),
            numpy.log(theta) / numpy.log(theta),
            inner_dim // n_elem,
            dtype=numpy.float64,
        ),
    )
    return torch.as_tensor(pow_indices * math.pi / 2, dtype=torch.float32)


@functools.lru_cache(maxsize=8)
def generate_freq_grid_pytorch(theta: float, n_axes: int, inner_dim: int) -> torch.Tensor:
    n_elem = 2 * n_axes
    indices = theta ** (
        torch.linspace(
            math.log(1.0, theta),
            math.log(theta, theta),
            inner_dim // n_elem,
            dtype=torch.float32,
        )
    )
    return indices.to(torch.float32) * math.pi / 2


def precompute_freqs_cis(
    indices_grid: torch.Tensor,  # (B, n_axes, T) or (B, n_axes, T, 2) with use_middle
    dim: int,
    out_dtype: torch.dtype,
    theta: float = 10000.0,
    max_pos: list[int] | None = None,
    use_middle_indices_grid: bool = True,
    num_attention_heads: int = 32,
    double_precision: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (cos, sin) of shape (B, H, T, D_head/2)."""
    if max_pos is None:
        max_pos = [20, 2048, 2048]
    n_axes = indices_grid.shape[1]
    generator = generate_freq_grid_np if double_precision else generate_freq_grid_pytorch
    indices = generator(theta, n_axes, dim)

    grid = indices_grid
    if use_middle_indices_grid and grid.ndim == 4:
        grid = (grid[..., 0] + grid[..., 1]) / 2.0
    elif grid.ndim == 4:
        grid = grid[..., 0]
    fractional = torch.stack([grid[:, i] / max_pos[i] for i in range(n_axes)], dim=-1)  # (B, T, n_axes)
    freqs = (indices.to(grid.device) * (fractional.unsqueeze(-1) * 2 - 1)).transpose(-1, -2).flatten(2)  # (B,T,F)

    pad_size = dim // 2 - freqs.shape[-1]
    cos_freq, sin_freq = freqs.cos(), freqs.sin()
    if pad_size:
        cos_freq = torch.cat([torch.ones_like(cos_freq[:, :, :pad_size]), cos_freq], dim=-1)
        sin_freq = torch.cat([torch.zeros_like(sin_freq[:, :, :pad_size]), sin_freq], dim=-1)
    b, t = cos_freq.shape[0], cos_freq.shape[1]
    cos_freq = cos_freq.reshape(b, t, num_attention_heads, -1).swapaxes(1, 2)
    sin_freq = sin_freq.reshape(b, t, num_attention_heads, -1).swapaxes(1, 2)
    return cos_freq.to(out_dtype), sin_freq.to(out_dtype)


def apply_split_rotary_emb(x: torch.Tensor, freqs_cis: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """x: (B, T, H*D) or (B, H, T, D); freqs: (B, H, T, D/2)."""
    cos_freqs, sin_freqs = freqs_cis
    needs_reshape = x.ndim != 4 and cos_freqs.ndim == 4
    h = cos_freqs.shape[1]
    if needs_reshape:
        x = x.unflatten(-1, (h, -1)).transpose(1, 2)
    split = x.reshape(*x.shape[:-1], 2, x.shape[-1] // 2)  # (...,2,D/2)
    first, second = split[..., :1, :], split[..., 1:, :]
    c, s = cos_freqs.unsqueeze(-2), sin_freqs.unsqueeze(-2)
    out_first = first * c - second * s
    out_second = second * c + first * s
    out = torch.cat([out_first, out_second], dim=-2)
    out = out.reshape(*x.shape)
    if needs_reshape:
        out = out.transpose(1, 2).flatten(-2)
    return out


# ---------------------------------------------------------------------------
# Timestep embedding (diffusers PixArt-alpha style)
# ---------------------------------------------------------------------------


class TimestepEmbedder(torch.nn.Module):
    def __init__(self, in_channels: int, time_embed_dim: int):
        super().__init__()
        self.linear_1 = torch.nn.Linear(in_channels, time_embed_dim)
        self.linear_2 = torch.nn.Linear(time_embed_dim, time_embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_2(F.silu(self.linear_1(x)))


class CombinedTimestepEmbeddings(torch.nn.Module):
    """time_proj(256 sinusoidal) -> TimestepEmbedding(embedding_dim)."""

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.timestep_embedder = TimestepEmbedder(256, embedding_dim)

    def forward(self, timestep: torch.Tensor, hidden_dtype: torch.dtype) -> torch.Tensor:
        from common import get_timestep_embedding

        # common.get_timestep_embedding already emits [cos, sin] (flip_sin_to_cos=True)
        proj = get_timestep_embedding(timestep, 256, max_period=10000)
        return self.timestep_embedder(proj.to(hidden_dtype))


class AdaLayerNormSingle(torch.nn.Module):
    """SiLU -> Linear(embedding_dim -> coefficient*embedding_dim)."""

    def __init__(self, embedding_dim: int, embedding_coefficient: int = 6):
        super().__init__()
        self.emb = CombinedTimestepEmbeddings(embedding_dim)
        self.linear = torch.nn.Linear(embedding_dim, embedding_coefficient * embedding_dim, bias=True)

    def forward(self, timestep: torch.Tensor, hidden_dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.emb(timestep, hidden_dtype=hidden_dtype)
        return self.linear(F.silu(embedded)), embedded


# ---------------------------------------------------------------------------
# Attention + FFN
# ---------------------------------------------------------------------------


class Attention(torch.nn.Module):
    """Multi-head attention with RMS q/k-norm, split RoPE and per-head gating.

    q/k/v projections operate on (B, T, D) tokens; heads are virtual (view).
    """

    def __init__(
        self,
        query_dim: int,
        context_dim: int | None = None,
        heads: int = 8,
        dim_head: int = 64,
        norm_eps: float = 1e-6,
        apply_gated_attention: bool = False,
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = heads * dim_head
        context_dim = query_dim if context_dim is None else context_dim
        self.q_norm = torch.nn.RMSNorm(inner_dim, eps=norm_eps)
        self.k_norm = torch.nn.RMSNorm(inner_dim, eps=norm_eps)
        self.to_q = torch.nn.Linear(query_dim, inner_dim, bias=True)
        self.to_k = torch.nn.Linear(context_dim, inner_dim, bias=True)
        self.to_v = torch.nn.Linear(context_dim, inner_dim, bias=True)
        self.to_gate_logits = torch.nn.Linear(query_dim, heads, bias=True) if apply_gated_attention else None
        self.to_out = torch.nn.Linear(inner_dim, query_dim, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        pe: tuple[torch.Tensor, torch.Tensor] | None = None,
        k_pe: tuple[torch.Tensor, torch.Tensor] | None = None,
        perturbation_mask: torch.Tensor | None = None,
        all_perturbed: bool = False,
    ) -> torch.Tensor:
        context = x if context is None else context
        b = x.shape[0]
        dim_head = self.dim_head
        h = self.heads
        v = self.to_v(context)
        if all_perturbed:
            out = v
        else:
            q = self.to_q(x)
            k = self.to_k(context)
            q = self.q_norm(q)
            k = self.k_norm(k)
            if pe is not None:
                q = apply_split_rotary_emb(q, pe)
                k = apply_split_rotary_emb(k, pe if k_pe is None else k_pe)
            q = q.view(b, -1, h, dim_head).transpose(1, 2)
            k = k.view(b, -1, h, dim_head).transpose(1, 2)
            v4 = v.view(b, -1, h, dim_head).transpose(1, 2)
            if mask is not None:
                if mask.ndim == 2:
                    mask = mask.unsqueeze(0)
                if mask.ndim == 3:
                    mask = mask.unsqueeze(1)
            out = sdpa(q, k, v4, mask)
            out = out.transpose(1, 2).reshape(b, -1, h * dim_head)
            if perturbation_mask is not None:
                out = out * perturbation_mask + v * (1 - perturbation_mask)
        if self.to_gate_logits is not None:
            gate_logits = self.to_gate_logits(x)  # (B, T, H)
            t = out.shape[1]
            out = out.view(b, t, h, dim_head) * (2.0 * torch.sigmoid(gate_logits)).unsqueeze(-1)
            out = out.view(b, t, h * dim_head)
        return self.to_out(out)


class FeedForward(torch.nn.Module):
    def __init__(self, dim: int, dim_out: int | None = None, mult: int = 4, bias: bool = True):
        super().__init__()
        dim_out = dim if dim_out is None else dim_out
        inner = int(dim * mult)
        self.proj_in = torch.nn.Linear(dim, inner, bias=bias)
        self.proj_out = torch.nn.Linear(inner, dim_out, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj_out(F.gelu(self.proj_in(x), approximate="tanh"))
