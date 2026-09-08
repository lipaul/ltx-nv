"""Standalone LTX-2.5 text encoder (TE): Gemma-4-12B body + LTX feature extractor + connectors.

The Gemma LM body itself is instantiated from the ``transformers`` Gemma4Unified model class
(plain PyTorch) with weights loaded directly from the TE safetensors file; everything
LTX-specific (tokenizer BOS/padding handling, multi-layer feature extraction, projection,
the 1D connector transformers) is re-implemented here with explicit torch ops.

Run: standalone/.venv python te_run.py --prompt "..." (see te_run.py)
"""

import json
import math
import time
from pathlib import Path

import torch
from safetensors import safe_open
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast, AutoModelForImageTextToText
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from common import DTYPE, DEVICE, read_metadata, load_state_dict
from ltx_nn import Attention, FeedForward, precompute_freqs_cis, rms_norm

TOKENIZER_MAX_LENGTH = 1024
_TOKENIZER_CONFIG_SKIP = {
    "tokenizer_class", "auto_map", "model_max_length", "backend", "is_local",
    "local_files_only", "processor_class", "added_tokens_decoder",
}


# ---------------------------------------------------------------------------
# Assets / tokenizer
# ---------------------------------------------------------------------------


def load_assets(te_path: str) -> dict:
    with safe_open(te_path, framework="pt") as f:
        meta = f.metadata() or {}
        cfg = json.loads(meta["gemma_config"])
        tok_json = f.get_tensor("tokenizer_json").numpy().tobytes()
        sidecars = {}
        for k in f.keys():  # noqa: SIM118
            if k.startswith("hf_asset__"):
                sidecars[k.removeprefix("hf_asset__")] = f.get_tensor(k).numpy().tobytes()
    return {"config": cfg, "tokenizer_json": tok_json, "sidecars": sidecars}


def build_tokenizer(assets: dict) -> PreTrainedTokenizerFast:
    tok_cfg = json.loads(assets["sidecars"]["tokenizer_config.json"])
    kwargs = {k: v for k, v in tok_cfg.items() if k not in _TOKENIZER_CONFIG_SKIP}
    return PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer.from_buffer(assets["tokenizer_json"]), model_max_length=TOKENIZER_MAX_LENGTH, **kwargs
    )


def tokenize(tokenizer: PreTrainedTokenizerFast, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """LTX encode-path tokenization: strip, ensure leading BOS, left-pad to 1024."""
    bos_id = tokenizer.bos_token_id
    ids, masks = [], []
    for text in texts:
        enc = tokenizer(text.strip(), padding=False, truncation=True, max_length=TOKENIZER_MAX_LENGTH)
        seq = enc["input_ids"]
        if not seq or seq[0] != bos_id:
            seq = [bos_id, *seq][:TOKENIZER_MAX_LENGTH]
        padded = tokenizer.pad(
            {"input_ids": [seq]},
            padding="max_length",
            max_length=TOKENIZER_MAX_LENGTH,
            return_tensors="pt",
            return_attention_mask=True,
        )
        ids.append(padded.input_ids[0])
        masks.append(padded.attention_mask[0])
    return torch.stack(ids), torch.stack(masks)


# ---------------------------------------------------------------------------
# Gemma body (HF class, direct safetensors load)
# ---------------------------------------------------------------------------

_GEMMA_RENAME = [
    ("model.layers.", "model.language_model.layers."),
    ("model.embed_tokens.", "model.language_model.embed_tokens."),
    ("model.norm.", "model.language_model.norm."),
    ("vision_model.", "model.embed_vision."),
    ("multi_modal_projector.embedding_projection.", "model.embed_vision.multimodal_embedder.embedding_projection."),
    ("audio_projector.", "model.embed_audio."),
]


def _rename_gemma_key(key: str) -> str | None:
    for old, new in _GEMMA_RENAME:
        if key.startswith(old):
            return new + key[len(old):]
    return None  # drop: text_embedding_projection / tokenizer_json / hf_asset__*


def _populate_gemma_buffers(model: torch.nn.Module, config) -> None:
    """Materialize the non-persistent rotary / embed-scale buffers (meta init leaves them empty)."""
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    l_model = model.model.language_model
    text_cfg = config.text_config if hasattr(config, "text_config") else config
    rope_emb = l_model.rotary_emb
    for layer_type in dict.fromkeys(text_cfg.layer_types):
        rope_params = text_cfg.rope_parameters[layer_type]
        if rope_params is None:
            continue
        rope_type = rope_params["rope_type"]
        if rope_type == "default":
            inv_freq, scaling = rope_emb.compute_default_rope_parameters(text_cfg, layer_type=layer_type)
        else:
            kw: dict = {"layer_type": layer_type}
            if layer_type == "full_attention" and rope_type == "proportional":
                kw["head_dim_key"] = "global_head_dim"
            inv_freq, scaling = ROPE_INIT_FUNCTIONS[rope_type](text_cfg, **kw)
        rope_emb.register_buffer(f"{layer_type}_inv_freq", inv_freq, persistent=False)
        rope_emb.register_buffer(f"{layer_type}_original_inv_freq", inv_freq.clone(), persistent=False)
        setattr(rope_emb, f"{layer_type}_attention_scaling", scaling)
    l_model.embed_tokens.register_buffer("embed_scale", torch.tensor(text_cfg.hidden_size**0.5), persistent=False)


def build_gemma_body(te_path: str, layer_hooks=None) -> torch.nn.Module:
    """Instantiate Gemma4UnifiedModel from the embedded config and load LM weights onto GPU."""
    cfg_dict = read_metadata(te_path)["gemma_config"]
    config = CONFIG_MAPPING[cfg_dict["model_type"]].from_dict(cfg_dict)
    with torch.device("meta"):
        model = AutoModelForImageTextToText.from_config(config)
    _populate_gemma_buffers(model, config)
    sd = load_state_dict(te_path, rename=_rename_gemma_key)
    # lm_head shares embed_tokens (tied weights; not used by the encode path but must not stay on meta)
    emb = sd.get("model.language_model.embed_tokens.weight")
    if emb is not None:
        sd["lm_head.weight"] = emb
    missing = model.load_state_dict(sd, strict=False, assign=True)
    real_missing = [k for k in missing.missing_keys if not k.startswith(("lm_head", "model.lm_head"))]
    if real_missing:
        raise RuntimeError(f"Gemma body missing weights: {real_missing[:8]}")
    model = model.to(DEVICE).eval()
    if layer_hooks:
        for i, layer in enumerate(model.model.language_model.layers):
            layer.register_forward_hook(_make_hook(f"gemma.layer{i}", layer_hooks))
    return model


def _make_hook(name: str, store: list):
    def hook(_mod, _inp, out):
        x = out[0] if isinstance(out, tuple) else out
        store.append({"name": name, "shape": list(x.shape), "std": round(x.float().std().item(), 4)})
    return hook


# ---------------------------------------------------------------------------
# Feature extractor (V2: per-token RMS over hidden dim, dual aggregate projections)
# ---------------------------------------------------------------------------


class FeatureExtractorV2(torch.nn.Module):
    def __init__(self, gemma_hidden: int, num_layers: int, video_dim: int, audio_dim: int):
        super().__init__()
        flat = gemma_hidden * (num_layers + 1)
        self.video_aggregate_embed = torch.nn.Linear(flat, video_dim, bias=True)
        self.audio_aggregate_embed = torch.nn.Linear(flat, audio_dim, bias=True)
        self.embedding_dim = gemma_hidden

    def forward(self, hidden_states: tuple[torch.Tensor, ...], attention_mask: torch.Tensor):
        encoded = torch.stack(hidden_states, dim=-1)  # [B, T, D, L]
        variance = torch.mean(encoded**2, dim=2, keepdim=True)
        normed = encoded * torch.rsqrt(variance + 1e-6)
        b, t, d, l = normed.shape
        normed = normed.reshape(b, t, d * l).to(encoded.dtype)
        mask3 = attention_mask.bool().unsqueeze(-1)
        normed = torch.where(mask3, normed, torch.zeros_like(normed))
        vd = self.video_aggregate_embed.out_features
        ad = self.audio_aggregate_embed.out_features
        video = self.video_aggregate_embed(normed * math.sqrt(vd / self.embedding_dim))
        audio = self.audio_aggregate_embed(normed * math.sqrt(ad / self.embedding_dim))
        return video, audio


# ---------------------------------------------------------------------------
# Embeddings1DConnector (bidirectional 1D transformer with learnable registers)
# ---------------------------------------------------------------------------


class ConnectorBlock(torch.nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, gated: bool, ff_bias: bool, norm_eps: float = 1e-6):
        super().__init__()
        self.attn1 = Attention(dim, None, heads, dim_head, norm_eps, apply_gated_attention=gated)
        self.ff = FeedForward(dim, dim, bias=ff_bias)

    def forward(self, x, additive_mask=None, pe=None):
        x = x + self.attn1(rms_norm(x), mask=additive_mask, pe=pe)
        x = x + self.ff(rms_norm(x))
        return x


class Embeddings1DConnector(torch.nn.Module):
    def __init__(self, heads: int, dim_head: int, num_layers: int, max_pos: list[int],
                 registers: int, gated: bool, ff_bias: bool):
        super().__init__()
        self.inner_dim = heads * dim_head
        self.positional_max_pos = list(max_pos)
        self.blocks = torch.nn.ModuleList(
            [ConnectorBlock(self.inner_dim, heads, dim_head, gated, ff_bias) for _ in range(num_layers)]
        )
        self.num_registers = registers
        if registers:
            self.learnable_registers = torch.nn.Parameter(torch.empty(registers, self.inner_dim))

    def forward(self, hidden_states: torch.Tensor, additive_mask: torch.Tensor):
        b, s, _ = hidden_states.shape
        if self.num_registers:
            regs = self.learnable_registers.repeat(s // self.num_registers, 1).to(hidden_states.dtype)
            regs = regs.unsqueeze(0).expand(b, -1, -1)
            binary = (additive_mask[:, 0, 0, :] >= 0).to(hidden_states.dtype).unsqueeze(-1)
            hidden_states = binary * hidden_states + (1 - binary) * regs
            additive_mask = torch.zeros_like(additive_mask)
        grid = torch.arange(s, dtype=torch.float32, device=hidden_states.device)[None, None, :].expand(b, -1, -1)
        pe = precompute_freqs_cis(
            grid, dim=self.inner_dim, out_dtype=hidden_states.dtype, theta=10000.0,
            max_pos=list(self.positional_max_pos), use_middle_indices_grid=False,
            num_attention_heads=self.blocks[0].attn1.heads, double_precision=True,
        )
        for block in self.blocks:
            hidden_states = block(hidden_states, additive_mask, pe)
        return rms_norm(hidden_states), additive_mask


# ---------------------------------------------------------------------------
# EmbeddingsProcessor (feature extractor + video/audio connectors)
# ---------------------------------------------------------------------------


class EmbeddingsProcessor(torch.nn.Module):
    def __init__(self, feature_extractor: FeatureExtractorV2, video_connector: Embeddings1DConnector,
                 audio_connector: Embeddings1DConnector):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.video_connector = video_connector
        self.audio_connector = audio_connector

    def process_hidden_states(self, hidden_states, attention_mask):
        video_feats, audio_feats = self.feature_extractor(hidden_states, attention_mask)
        additive = (attention_mask.to(torch.int64) - 1).to(video_feats.dtype).reshape(
            attention_mask.shape[0], 1, -1, attention_mask.shape[-1]
        ) * torch.finfo(video_feats.dtype).max
        binary = (additive[:, 0, 0, :] >= 0).to(torch.int32)
        sort_idx = torch.argsort(binary, dim=-1, descending=True, stable=True)
        new_binary = torch.gather(binary, 1, sort_idx)
        reordered = (new_binary.to(additive.dtype) - 1) * torch.finfo(additive.dtype).max
        reordered = reordered[:, None, None, :]

        def gather_feats(f):
            return torch.gather(f, 1, sort_idx.unsqueeze(-1).expand_as(f))

        video_enc, video_mask = self.video_connector(gather_feats(video_feats), reordered)
        video_bin = (video_mask < 1e-6).to(torch.int64).reshape(*video_enc.shape[:2], 1)
        video_enc = video_enc * video_bin
        audio_enc, _ = self.audio_connector(gather_feats(audio_feats), reordered)
        return video_enc, audio_enc, video_bin.squeeze(-1)


# ---------------------------------------------------------------------------
# Assembly + weight loading
# ---------------------------------------------------------------------------


def build_embeddings_processor(te_path: str, transformer_path: str) -> EmbeddingsProcessor:
    te_cfg = read_metadata(te_path)["gemma_config"]
    text_cfg = te_cfg["text_config"] if "text_config" in te_cfg else te_cfg
    hidden, n_layers = text_cfg["hidden_size"], text_cfg["num_hidden_layers"]
    tr_cfg = read_metadata(transformer_path)["config"]["transformer"]

    with torch.device("meta"):
        fe = FeatureExtractorV2(hidden, n_layers,
                                tr_cfg["num_attention_heads"] * tr_cfg["attention_head_dim"],
                                tr_cfg["audio_num_attention_heads"] * tr_cfg["audio_attention_head_dim"])
        video_conn = Embeddings1DConnector(
            tr_cfg.get("connector_num_attention_heads", 32), tr_cfg.get("connector_attention_head_dim", 128),
            tr_cfg.get("connector_num_layers", 8), tr_cfg.get("connector_positional_embedding_max_pos", [4096]),
            tr_cfg.get("connector_num_learnable_registers", 128),
            tr_cfg.get("connector_apply_gated_attention", False), tr_cfg.get("connector_ff_bias", True))
        audio_conn = Embeddings1DConnector(
            tr_cfg.get("audio_connector_num_attention_heads", tr_cfg.get("connector_num_attention_heads", 32)),
            tr_cfg.get("audio_connector_attention_head_dim", tr_cfg.get("connector_attention_head_dim", 128)),
            tr_cfg.get("audio_connector_num_layers", tr_cfg.get("connector_num_layers", 8)),
            tr_cfg.get("connector_positional_embedding_max_pos", [4096]),
            tr_cfg.get("connector_num_learnable_registers", 128),
            tr_cfg.get("connector_apply_gated_attention", False), tr_cfg.get("connector_ff_bias", True))
        proc = EmbeddingsProcessor(fe, video_conn, audio_conn)
    sd = {}
    sd.update(load_state_dict(te_path, rename=lambda k: (
        "feature_extractor." + k.removeprefix("text_embedding_projection.")
        if k.startswith("text_embedding_projection.") else None)))
    sd.update(load_state_dict(transformer_path, rename=lambda k: (
        ("video_connector." if "video_embeddings" in k else "audio_connector.")
        + _conn_leaf(k) if k.startswith("model.diffusion_model.") and "_embeddings_connector." in k else None)))
    proc.load_state_dict(sd, strict=False, assign=True)
    return proc.to(DEVICE).eval()


def _conn_leaf(key: str) -> str:
    leaf = key.split("_embeddings_connector.", 1)[1]
    return (leaf.replace("transformer_1d_blocks", "blocks")
                .replace("to_out.0.", "to_out.")
                .replace("ff.net.0.proj.", "ff.proj_in.")
                .replace("ff.net.2.", "ff.proj_out."))


@torch.inference_mode()
def encode_prompts(prompts: list[str], te_path: str, transformer_path: str,
                   stats: dict | None = None) -> list[dict]:
    """Full TE: tokenize -> Gemma hidden states -> feature extract -> connectors.

    Returns one dict per prompt: {video_encoding, audio_encoding, attention_mask} (CPU tensors).
    """
    t0 = time.perf_counter()
    assets = load_assets(te_path)
    tokenizer = build_tokenizer(assets)
    hooks: list = []
    gemma = build_gemma_body(te_path, layer_hooks=hooks)
    ids, mask = tokenize(tokenizer, prompts)
    ids, mask = ids.to(DEVICE), mask.to(DEVICE)
    t_tok = time.perf_counter()

    outputs = gemma.model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    del outputs
    torch.cuda.synchronize()
    t_gemma = time.perf_counter()

    proc = build_embeddings_processor(te_path, transformer_path)
    results = []
    for i in range(len(prompts)):
        per = tuple(h[i : i + 1] for h in hidden_states)
        v, a, m = proc.process_hidden_states(per, mask[i : i + 1])
        results.append({"video_encoding": v.cpu(), "audio_encoding": a.cpu(), "attention_mask": m.cpu()})
    torch.cuda.synchronize()
    t_proc = time.perf_counter()

    if stats is not None:
        stats["te_tokenize_sec"] = round(t_tok - t0, 2)
        stats["te_gemma_sec"] = round(t_gemma - t_tok, 2)
        stats["te_processor_sec"] = round(t_proc - t_gemma, 2)
        stats["te_gemma_layers"] = hooks[:2] + hooks[-2:]
        stats["te_gemm_hidden"] = [list(h.shape) for h in hidden_states[:2]]
    del gemma, proc
    return results
