"""Standalone LTX-2.5 VAEs: conv video decoder + audio decoder + BigVGAN vocoder with BWE.

Plain PyTorch re-implementations; weights loaded directly from the split safetensors files.
Video decode is a single full-latent forward pass (the baseline's AUTO_TILING resolves to a
single tile at 512x768x65, so this is the same computation).
"""

import math

import torch
import torch.nn.functional as F

from common import DTYPE, DEVICE, read_metadata, load_state_dict

# ---------------------------------------------------------------------------
# Shared primitives
# ---------------------------------------------------------------------------


class PixelNorm(torch.nn.Module):
    """Per-location RMS norm over the channel dim (dim=1 for 5D/4D tensors)."""

    def __init__(self, dim: int = 1, eps: float = 1e-8):
        super().__init__()
        self.dim, self.eps = dim, eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / torch.sqrt(torch.mean(x**2, dim=self.dim, keepdim=True) + self.eps)


class CausalConv3d(torch.nn.Module):
    """Conv3d with symmetric spatial padding and manual temporal padding.

    causal=True pads the front with the first frame; causal=False pads both ends
    with the edge frames (the reference passes causal=self.causal=False for this decoder).
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, spatial_padding_mode="zeros", **kw):
        super().__init__()
        k = (kernel_size, kernel_size, kernel_size)
        self.time_kernel = k[0]
        self.conv = torch.nn.Conv3d(
            in_channels, out_channels, k, padding=(0, k[1] // 2, k[2] // 2),
            padding_mode=spatial_padding_mode, **kw,
        )

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        if causal:
            pad = x[:, :, :1].repeat(1, 1, self.time_kernel - 1, 1, 1)
            x = torch.cat([pad, x], dim=2)
        else:
            n = (self.time_kernel - 1) // 2
            first = x[:, :, :1].repeat(1, 1, n, 1, 1)
            last = x[:, :, -1:].repeat(1, 1, n, 1, 1)
            x = torch.cat([first, x, last], dim=2)
        return self.conv(x)


class CausalConv2d(torch.nn.Module):
    """Conv2d with manual asymmetric padding along the causality axis (height=time)."""

    def __init__(self, in_channels, out_channels, kernel_size=3, causality_axis="height", **kw):
        super().__init__()
        k = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        pad_h, pad_w = k[0] - 1, k[1] - 1
        if causality_axis == "height":
            self.padding = (pad_w // 2, pad_w - pad_w // 2, pad_h, 0)
        elif causality_axis == "width":
            self.padding = (pad_w, 0, pad_h // 2, pad_h - pad_h // 2)
        else:
            self.padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
        self.conv = torch.nn.Conv2d(in_channels, out_channels, k, padding=0, **kw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, self.padding))


class PerChannelStatistics(torch.nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.register_buffer("std-of-means", torch.ones(channels))
        self.register_buffer("mean-of-means", torch.zeros(channels))

    def un_normalize(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 5:
            return x * self.get_buffer("std-of-means").view(1, -1, 1, 1, 1).to(x) + \
                self.get_buffer("mean-of-means").view(1, -1, 1, 1, 1).to(x)
        return x * self.get_buffer("std-of-means").to(x) + self.get_buffer("mean-of-means").to(x)


# ---------------------------------------------------------------------------
# Video conv decoder
# ---------------------------------------------------------------------------


class ResnetBlock3D(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, spatial_padding_mode: str):
        super().__init__()
        self.norm1 = PixelNorm()
        self.conv1 = CausalConv3d(in_channels, out_channels, 3, spatial_padding_mode)
        self.norm2 = PixelNorm()
        self.conv2 = CausalConv3d(out_channels, out_channels, 3, spatial_padding_mode)
        if in_channels != out_channels:
            self.norm3 = torch.nn.GroupNorm(1, in_channels, eps=1e-6)
            self.conv_shortcut = torch.nn.Conv3d(in_channels, out_channels, 1)
        else:
            self.norm3 = torch.nn.Identity()
            self.conv_shortcut = torch.nn.Identity()

    def forward(self, x: torch.Tensor, causal: bool = False) -> torch.Tensor:
        h = self.nonlin(self.norm1(x))
        h = self.conv1(h, causal=causal)
        h = self.nonlin(self.norm2(h))
        h = self.conv2(h, causal=causal)
        return self.conv_shortcut(self.norm3(x)) + h

    @staticmethod
    def nonlin(x):
        return F.silu(x)


class UNetMidBlock3D(torch.nn.Module):
    def __init__(self, in_channels: int, num_layers: int, spatial_padding_mode: str):
        super().__init__()
        self.res_blocks = torch.nn.ModuleList(
            [ResnetBlock3D(in_channels, in_channels, spatial_padding_mode) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor, causal: bool = False) -> torch.Tensor:
        for blk in self.res_blocks:
            x = blk(x, causal=causal)
        return x


class DepthToSpaceUpsample(torch.nn.Module):
    """conv -> rearrange (c p1 p2 p3) -> (d p1)(h p2)(w p3); drop first frame when stride[0]==2."""

    def __init__(self, in_channels: int, stride: tuple, out_channels_reduction: int, spatial_padding_mode: str):
        super().__init__()
        self.stride = stride
        self.out_channels = math.prod(stride) * in_channels // out_channels_reduction
        self.final_channels = in_channels // out_channels_reduction  # after depth-to-space
        self.conv = CausalConv3d(in_channels, self.out_channels, 3, spatial_padding_mode)

    def forward(self, x: torch.Tensor, causal: bool = False) -> torch.Tensor:
        p1, p2, p3 = self.stride
        x = self.conv(x, causal=causal)
        b, c, d, h, w = x.shape
        x = x.view(b, c // (p1 * p2 * p3), p1, p2, p3, d, h, w)
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).reshape(b, c // (p1 * p2 * p3), d * p1, h * p2, w * p3)
        if p1 == 2:
            x = x[:, :, 1:]
        return x


def _bottleneck_channels(base: int, decoder_blocks: list) -> int:
    mult = 1
    for name, params in decoder_blocks:
        cfg = {"multiplier": params} if isinstance(params, int) else params
        if name in ("compress_time", "compress_space", "compress_all"):
            mult *= cfg.get("multiplier", 1)
        elif name == "res_x_y":
            mult *= cfg.get("multiplier", 2)
    return base * mult


class ConvVideoDecoder(torch.nn.Module):
    """LTX-2.5 conv video VAE decoder (timestep_conditioning=False variant)."""

    def __init__(self, vae_cfg: dict):
        super().__init__()
        blocks = vae_cfg["decoder_blocks"]
        base = vae_cfg.get("decoder_base_channels", 128)
        patch = vae_cfg.get("patch_size", 4)
        pad_mode = vae_cfg.get("spatial_padding_mode", "zeros")
        in_ch = vae_cfg.get("latent_channels", 128)
        out_ch = vae_cfg.get("out_channels", 3) * patch**2
        ch = _bottleneck_channels(base, blocks)
        self.patch_size = patch
        self.per_channel_statistics = PerChannelStatistics(in_ch)
        self.conv_in = CausalConv3d(in_ch, ch, 3, pad_mode)
        self.up_blocks = torch.nn.ModuleList()
        for name, params in reversed(blocks):
            cfg = {"num_layers": params} if isinstance(params, int) else params
            if name == "res_x":
                blk: torch.nn.Module = UNetMidBlock3D(ch, cfg["num_layers"], pad_mode)
            elif name == "res_x_y":
                ch //= cfg.get("multiplier", 2)
                blk = ResnetBlock3D(ch * cfg.get("multiplier", 2), ch, pad_mode)
            elif name == "compress_time":
                blk = DepthToSpaceUpsample(ch, (2, 1, 1), cfg.get("multiplier", 1), pad_mode); ch = blk.final_channels
            elif name == "compress_space":
                blk = DepthToSpaceUpsample(ch, (1, 2, 2), cfg.get("multiplier", 1), pad_mode); ch = blk.final_channels
            elif name == "compress_all":
                blk = DepthToSpaceUpsample(ch, (2, 2, 2), cfg.get("multiplier", 1), pad_mode); ch = blk.final_channels
            else:
                raise ValueError(name)
            self.up_blocks.append(blk)
        self.conv_norm_out = PixelNorm()
        self.conv_out = CausalConv3d(ch, out_ch, 3, pad_mode)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        sample = self.per_channel_statistics.un_normalize(sample)
        sample = self.conv_in(sample, causal=False)
        for blk in self.up_blocks:
            sample = blk(sample, causal=False)
        sample = self.conv_norm_out(sample)
        sample = F.silu(sample)
        sample = self.conv_out(sample, causal=False)
        p = self.patch_size
        b, c, f, h, w = sample.shape
        # einops "b (c p r q) f h w -> b c (f p) (h q) (w r)": channel = (c, p_t, r->w, q->h), q fastest
        sample = sample.view(b, c // (p * p), 1, p, p, f, h, w)  # (b,c,pt,r,q,f,h,w)
        sample = sample.permute(0, 1, 5, 2, 6, 4, 7, 3)  # (b,c,f,pt,h,q,w,r)
        return sample.reshape(b, c // (p * p), f, h * p, w * p)

    def decode_video(self, latent: torch.Tensor) -> torch.Tensor:
        """latent (B,128,F',H',W') -> RGB chunks (f,h,w,c) in [0,1] (single chunk)."""
        frames = self(latent.to(DTYPE))
        video = frames[0].permute(1, 2, 3, 0)  # (f,h,w,c)
        return (video.float().add_(1.0).mul_(0.5)).clamp_(0.0, 1.0)


def load_video_decoder(path: str) -> ConvVideoDecoder:
    cfg = read_metadata(path)["config"]["vae"]
    with torch.device("meta"):
        model = ConvVideoDecoder(cfg)
    sd = load_state_dict(path, rename=lambda k: k.removeprefix("decoder.") if k.startswith("decoder.") else (
        k if k.startswith("per_channel_statistics.") else None))
    res = model.load_state_dict(sd, strict=False, assign=True)
    if res.missing_keys:
        raise RuntimeError(f"video decoder missing: {res.missing_keys[:8]}")
    return model.to(DEVICE).eval()


# ---------------------------------------------------------------------------
# Audio decoder (VQGAN-style 2D, causal along height=time axis)
# ---------------------------------------------------------------------------


class AudioResnetBlock(torch.nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.norm1 = PixelNorm(eps=1e-6)
        self.conv1 = CausalConv2d(in_ch, out_ch, 3)
        self.norm2 = PixelNorm(eps=1e-6)
        self.conv2 = CausalConv2d(out_ch, out_ch, 3)
        self.nin_shortcut = CausalConv2d(in_ch, out_ch, 1) if in_ch != out_ch else torch.nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return self.nin_shortcut(x) + h


class AudioUpsample(torch.nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = CausalConv2d(channels, channels, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)[:, :, 1:, :]  # drop first time row (causal undo)


class AudioDecoder(torch.nn.Module):
    def __init__(self, dd: dict):
        super().__init__()
        ch, ch_mult = dd["ch"], tuple(dd["ch_mult"])
        num_res, z = dd["num_res_blocks"], dd["z_channels"]
        out_ch, mel_bins = dd["out_ch"], dd["mel_bins"]
        self.out_ch, self.mel_bins = out_ch, mel_bins
        self.per_channel_statistics = PerChannelStatistics(ch)  # stats over (c*mel_bins)=128
        base_block = ch * ch_mult[-1]
        self.conv_in = CausalConv2d(z, base_block, 3)
        mid = torch.nn.Module()
        mid.block_1 = AudioResnetBlock(base_block, base_block)
        mid.block_2 = AudioResnetBlock(base_block, base_block)
        self.mid = mid
        self.up = torch.nn.ModuleList()
        block_in = base_block
        for level in reversed(range(len(ch_mult))):
            stage = torch.nn.Module()
            stage.block = torch.nn.ModuleList()
            block_out = ch * ch_mult[level]
            for _ in range(num_res + 1):
                stage.block.append(AudioResnetBlock(block_in, block_out))
                block_in = block_out
            if level != 0:
                stage.upsample = AudioUpsample(block_in)
            self.up.insert(0, stage)
        self.norm_out = PixelNorm(eps=1e-6)
        self.conv_out = CausalConv2d(block_in, out_ch, 3)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        b, c, t, f = sample.shape
        patched = sample.permute(0, 2, 1, 3).reshape(b, t, c * f)
        patched = self.per_channel_statistics.un_normalize(patched)
        sample = patched.view(b, t, c, f).permute(0, 2, 1, 3)
        h = self.conv_in(sample)
        h = self.mid.block_2(self.mid.block_1(h))
        for level in reversed(range(len(self.up))):
            stage = self.up[level]
            for blk in stage.block:
                h = blk(h)
            if hasattr(stage, "upsample"):
                h = stage.upsample(h)
        h = self.conv_out(F.silu(self.norm_out(h)))
        target_t = t * 4 - 3
        return h[:, : self.out_ch, :min(h.shape[2], target_t), : self.mel_bins]


def load_audio_decoder(path: str) -> AudioDecoder:
    dd = read_metadata(path)["config"]["audio_vae"]["model"]["params"]["ddconfig"]
    with torch.device("meta"):
        model = AudioDecoder(dd)
    sd = load_state_dict(path, rename=lambda k: k.removeprefix("audio_vae.decoder.") if k.startswith("audio_vae.decoder.") else (
        k.replace("audio_vae.per_channel_statistics.", "per_channel_statistics.")
        if k.startswith("audio_vae.per_channel_statistics.") else None))
    res = model.load_state_dict(sd, strict=False, assign=True)
    if res.missing_keys:
        raise RuntimeError(f"audio decoder missing: {res.missing_keys[:8]}")
    return model.to(DEVICE).eval()


# ---------------------------------------------------------------------------
# Vocoder (BigVGAN AMP1) + BWE
# ---------------------------------------------------------------------------


def _sinc(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x == 0, torch.tensor(1.0, device=x.device, dtype=x.dtype), torch.sin(math.pi * x) / math.pi / x)


def kaiser_sinc_filter1d(cutoff: float, half_width: float, kernel_size: int) -> torch.Tensor:
    even = kernel_size % 2 == 0
    half = kernel_size // 2
    delta_f = 4 * half_width
    amp = 2.285 * (half - 1) * math.pi * delta_f + 7.95
    beta = 0.1102 * (amp - 8.7) if amp > 50 else (0.5842 * (amp - 21) ** 0.4 + 0.07886 * (amp - 21) if amp >= 21 else 0.0)
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)
    time = torch.arange(-half, half) + 0.5 if even else torch.arange(kernel_size) - half
    if cutoff == 0:
        return torch.zeros_like(time).view(1, 1, kernel_size)
    filt = 2 * cutoff * window * _sinc(2 * cutoff * time)
    return (filt / filt.sum()).view(1, 1, kernel_size)


class UpSample1d(torch.nn.Module):
    def __init__(self, ratio: int = 2, kernel_size: int | None = None, window_type: str = "kaiser"):
        super().__init__()
        self.ratio = ratio
        if window_type == "hann":
            rolloff, width_lf = 0.99, 6
            width = math.ceil(width_lf / rolloff)
            self.kernel_size = 2 * width * ratio + 1
            self.pad = width
            self.pad_left, self.pad_right = 2 * width * ratio, self.kernel_size - ratio
            t = (torch.arange(self.kernel_size) / ratio - width) * rolloff
            tc = t.clamp(-width_lf, width_lf)
            window = torch.cos(tc * math.pi / width_lf / 2) ** 2
            filt = (torch.sinc(t) * window * rolloff / ratio).view(1, 1, -1)
        else:
            self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
            self.pad = self.kernel_size // ratio - 1
            self.pad_left = self.pad * ratio + (self.kernel_size - ratio) // 2
            self.pad_right = self.pad * ratio + (self.kernel_size - ratio + 1) // 2
            filt = kaiser_sinc_filter1d(0.5 / ratio, 0.6 / ratio, self.kernel_size)
        self.register_buffer("filter", filt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[1]
        x = F.pad(x, (self.pad, self.pad), mode="replicate")
        x = self.ratio * F.conv_transpose1d(x, self.filter.expand(n, -1, -1).to(x.dtype), stride=self.ratio, groups=n)
        return x[..., self.pad_left : -self.pad_right]


class LowPassFilter1d(torch.nn.Module):
    def __init__(self, cutoff=0.5, half_width=0.6, stride=1, kernel_size=12):
        super().__init__()
        self.kernel_size, self.stride = kernel_size, stride
        even = kernel_size % 2 == 0
        self.pad_left = kernel_size // 2 - int(even)
        self.pad_right = kernel_size // 2
        self.register_buffer("filter", kaiser_sinc_filter1d(cutoff, half_width, kernel_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[1]
        x = F.pad(x, (self.pad_left, self.pad_right), mode="replicate")
        return F.conv1d(x, self.filter.expand(n, -1, -1).to(x.dtype), stride=self.stride, groups=n)


class DownSample1d(torch.nn.Module):
    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        ks = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.lowpass = LowPassFilter1d(0.5 / ratio, 0.6 / ratio, ratio, ks)

    def forward(self, x):
        return self.lowpass(x)


class SnakeBeta(torch.nn.Module):
    def __init__(self, features: int):
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.zeros(features))
        self.beta = torch.nn.Parameter(torch.zeros(features))

    def forward(self, x):
        a, b = torch.exp(self.alpha)[None, :, None], torch.exp(self.beta)[None, :, None]
        return x + (1.0 / (b + 1e-9)) * torch.sin(x * a).pow(2)


class Activation1d(torch.nn.Module):
    def __init__(self, act: torch.nn.Module):
        super().__init__()
        self.act = act
        self.upsample = UpSample1d(2, 12)
        self.downsample = DownSample1d(2, 12)

    def forward(self, x):
        return self.downsample(self.act(self.upsample(x)))


class AMPBlock1(torch.nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = torch.nn.ModuleList(
            [torch.nn.Conv1d(channels, channels, kernel_size, 1, dilation=d, padding=(kernel_size * d - d) // 2)
             for d in dilation]
        )
        self.convs2 = torch.nn.ModuleList(
            [torch.nn.Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=(kernel_size - 1) // 2)
             for _ in dilation]
        )
        self.acts1 = torch.nn.ModuleList([Activation1d(SnakeBeta(channels)) for _ in dilation])
        self.acts2 = torch.nn.ModuleList([Activation1d(SnakeBeta(channels)) for _ in dilation])

    def forward(self, x):
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, self.acts1, self.acts2, strict=True):
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = x + xt
        return x


class Vocoder(torch.nn.Module):
    def __init__(self, cfg: dict, apply_final_activation: bool = True):
        super().__init__()
        rates = cfg["upsample_rates"]
        k_up = cfg["upsample_kernel_sizes"]
        k_res = cfg["resblock_kernel_sizes"]
        dil = cfg["resblock_dilation_sizes"]
        init_ch = cfg["upsample_initial_channel"]
        self.num_upsamples = len(rates)
        self.num_kernels = len(k_res)
        self.apply_final_activation = apply_final_activation
        self.use_tanh_at_final = cfg.get("use_tanh_at_final", False)
        self.conv_pre = torch.nn.Conv1d(128, init_ch, 7, 1, 3)
        self.ups = torch.nn.ModuleList(
            torch.nn.ConvTranspose1d(init_ch // 2**i, init_ch // 2 ** (i + 1), ks, r, (ks - r) // 2)
            for i, (r, ks) in enumerate(zip(rates, k_up, strict=True))
        )
        self.resblocks = torch.nn.ModuleList(
            AMPBlock1(init_ch // 2 ** (i + 1), k, tuple(d))
            for i in range(len(rates)) for k, d in zip(k_res, dil, strict=True)
        )
        final_ch = init_ch // 2 ** len(rates)
        self.act_post = Activation1d(SnakeBeta(final_ch))
        self.conv_post = torch.nn.Conv1d(final_ch, 2, 7, 1, 3, bias=cfg.get("use_bias_at_final", True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(2, 3)  # (B,C,T,M) -> (B,C,M,T)
        x = x.reshape(x.shape[0], -1, x.shape[-1])  # (B, C*M, T)
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = self.ups[i](x)
            xs = torch.stack([self.resblocks[i * self.num_kernels + j](x) for j in range(self.num_kernels)], 0)
            x = xs.mean(0)
        x = self.act_post(x)
        x = self.conv_post(x)
        if self.apply_final_activation:
            x = torch.tanh(x) if self.use_tanh_at_final else torch.clamp(x, -1, 1)
        return x


class STFTFn(torch.nn.Module):
    def __init__(self, filter_length: int, hop_length: int, win_length: int):
        super().__init__()
        self.hop_length, self.win_length = hop_length, win_length
        n_freqs = filter_length // 2 + 1
        self.register_buffer("forward_basis", torch.zeros(n_freqs * 2, 1, filter_length))
        self.register_buffer("inverse_basis", torch.zeros(n_freqs * 2, 1, filter_length))

    def forward(self, y: torch.Tensor):
        if y.dim() == 2:
            y = y.unsqueeze(1)
        y = F.pad(y, (max(0, self.win_length - self.hop_length), 0))
        spec = F.conv1d(y, self.forward_basis, stride=self.hop_length)
        nf = spec.shape[1] // 2
        real, imag = spec[:, :nf], spec[:, nf:]
        return torch.sqrt(real**2 + imag**2), torch.atan2(imag.float(), real.float()).to(real.dtype)


class MelSTFT(torch.nn.Module):
    def __init__(self, filter_length: int, hop_length: int, win_length: int, n_mel: int):
        super().__init__()
        self.stft_fn = STFTFn(filter_length, hop_length, win_length)
        self.register_buffer("mel_basis", torch.zeros(n_mel, filter_length // 2 + 1))

    def mel_spectrogram(self, y: torch.Tensor):
        magnitude, phase = self.stft_fn(y)
        mel = torch.matmul(self.mel_basis.to(magnitude.dtype), magnitude)
        return torch.log(torch.clamp(mel, min=1e-5)), magnitude, phase, torch.norm(magnitude, dim=1)


class VocoderWithBWE(torch.nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        v_cfg, b_cfg = cfg["vocoder"], cfg["bwe"]
        self.vocoder = Vocoder(v_cfg)
        self.bwe_generator = Vocoder(b_cfg, apply_final_activation=False)
        self.mel_stft = MelSTFT(b_cfg["n_fft"], b_cfg["hop_length"], b_cfg["win_size"], b_cfg["num_mels"])
        self.input_sampling_rate = b_cfg["input_sampling_rate"]
        self.output_sampling_rate = b_cfg["output_sampling_rate"]
        self.hop_length = b_cfg["hop_length"]
        with torch.device("cpu"):
            self.resampler = UpSample1d(self.output_sampling_rate // self.input_sampling_rate, window_type="hann")

    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        x = self.vocoder(mel_spec.float())
        _, _, length = x.shape
        out_len = length * self.output_sampling_rate // self.input_sampling_rate
        rem = length % self.hop_length
        if rem:
            x = F.pad(x, (0, self.hop_length - rem))
        mel = self.mel_stft.mel_spectrogram(x.reshape(x.shape[0] * x.shape[1], -1))[0]
        mel = mel.reshape(x.shape[0], x.shape[1], mel.shape[1], mel.shape[2]).transpose(2, 3)
        residual = self.bwe_generator(mel)
        skip = self.resampler(x)
        return torch.clamp(residual + skip, -1, 1)[..., :out_len]


def load_vocoder(path: str) -> VocoderWithBWE:
    cfg = read_metadata(path)["config"]["vocoder"]
    with torch.device("meta"):
        model = VocoderWithBWE(cfg)
    sd = load_state_dict(path, rename=lambda k: k.removeprefix("vocoder.") if k.startswith("vocoder.") else None)
    res = model.load_state_dict(sd, strict=False, assign=True)
    missing = [k for k in res.missing_keys if not k.startswith("resampler")]
    if missing:
        raise RuntimeError(f"vocoder missing: {missing[:8]}")
    return model.to(DEVICE).eval()


@torch.inference_mode()
def decode_audio(latent: torch.Tensor, decoder: AudioDecoder, vocoder: VocoderWithBWE) -> torch.Tensor:
    mel = decoder(latent.to(DTYPE))
    with torch.autocast(device_type="cuda", dtype=torch.float32):
        waveform = vocoder(mel).squeeze(0).float()
    return waveform
