# LTX-2.5 推理逐步分析 — TE / DiT / VAE

硬件:RTX 6000 Ada 48 GB (SM 8.9) · torch 2.13.0+cu132 · bf16 · SDPA[CUDNN>FLASH>EFFICIENT>MATH]
模型:`ltx-2.5-22b-dev-transformer-bf16` (42 GB) + `gemma4-12b-with-proj` TE (25 GB) + conv video VAE + audio VAE/vocoder
配置:512×768×65 帧 @24fps,30 步,seed 10,CFG v3.0/a7.0,STG 1.0@block28,modality 3.0,rescale 0.7

## 总览(standalone 单进程,阶段间释放显存)

| 阶段 | 耗时 | 显存峰值 | 说明 |
|---|---|---|---|
| TE(tokenize+Gemma+processor) | ~6.3 s | ~24 GB | Gemma 48 层 forward ~0.7 s;processor ~1.1 s |
| DiT 权重加载 | ~5 s | 42 GB | NVMe→GPU 42 GB |
| DiT 30 步采样 | ~153 s | 39.8 GB | 每步 ~5.0 s(4.45→5.25 s 递增) |
| Video VAE decode | ~0.8 s | 3.5 GB | 单次全量 decode(无 tiling) |
| Audio VAE + vocoder + BWE | ~0.5 s | 0.7 GB | 16 kHz mel → 48 kHz 波形 |
| **合计** | **~168 s** | | 官方管线 baseline:165.7 s |

## Stage 1 — Text Encoder(Gemma-4-12B + LTX 投影)

数据流(每条 prompt 独立处理,B=1,序列固定 1024、左 padding、强制前导 BOS):

1. **Tokenize**:`tokenizer(text, truncation@1024)` → 补 BOS → 左 pad 到 1024。
   本例正/负 prompt 实际 token 数 ~20 / ~350。
2. **Gemma body**:`Gemma4UnifiedModel(input_ids, attention_mask, output_hidden_states=True)`
   → 49 个 `[1,1024,3840]` hidden states(embedding + 48 层;16 q-heads × 256,GQA 8 kv-heads,
   sliding(1024)/full 交替层)。跳过 lm_head。
3. **FeatureExtractorV2**:stack → `[B,T,D,L]`;逐 token 对 D 维 RMS 归一化(eps 1e-6)→
   reshape `[B,T,3840×49=188160]`;pad 位置置零;rescale `sqrt(out/3840)`;
   双线性投影:`video_aggregate_embed` 188160→4096、`audio_aggregate_embed` 188160→2048
   (权重在 TE 文件 `text_embedding_projection.*`,共 2.2 GB,是 TE 的主要参数)。
4. **Embeddings1DConnector**(video/audio 各一个,8 层,1D 双向 transformer):
   - pad 位置替换为 128 个 learnable registers 平铺;mask 归零(全连接)。
   - 1D RoPE(arange 位置,max_pos=4096,fp64 频率网格);每层:RMSNorm→gated self-attn→残差→
     RMSNorm→FFN(gelu-tanh ×4)→残差;最后 RMSNorm。
   - video 输出 `[1,1024,4096]`(乘 valid mask),audio 输出 `[1,1024,2048]`(不乘 mask)。

要点:video/audio 是**两套独立投影+connector**,DiT 的两个流各自 cross-attend 自己的 context。

## Stage 2 — DiT(LTXModel,48 块 AV 双流)

**输入构造**(每步):
- 初始噪声:`randn(1,3456,128)`(video 9×16×24 token)+ `randn(1,68,128)`(audio,25 token/s),
  CUDA generator seed=10,与 baseline 逐位一致。
- positions:video `[B,3,T,2]` 像素空间 [start,end) 框(×8×32×32,causal 首帧修正,/fps 转秒);
  audio `[B,1,T,2]` 秒(causal mel 时序)。RoPE 取区间中点。
- keyframes_mask:首 latent 帧 384 token 标记(加 `keyframes_abs_pos_embedding`)。
- timesteps = denoise_mask × sigma(逐 token);sigma 同时喂 prompt-AdaLN 与 A↔V 门控。

**每步一次 B=4 forward**(guidance 四路合批):
`[cond | uncond | ptb | mod]`,context 按路拼接;
- ptb 路:block 28 的 video/audio self-attn 输出被替换为纯 V 投影(perturbation keep-mask=0);
- mod 路:所有块的 A2V/V2A cross-attn 贡献被 mask 掉(模态隔离);
- 混合:`x0 = cond + 2·(cond−uncond) + 1·(cond−ptb) + 2·(cond−mod)`,再按 std 比例 rescale(0.7);
- 速度场:x0_pred = latent − v·timesteps;Euler:`x += ((x−x0)/σ).to(bf16)·(σ_next−σ)`(fp32 累加)。

**单块结构**(video 4096/32头×128,audio 2048/32头×64):
AdaLN-single(9 组 scale/shift/gate,来自 timestep 嵌入)→ self-attn(qk RMSNorm+split-RoPE+
per-head gating 2σ)→ text cross-attn(query 侧 AdaLN + prompt 侧 K/V AdaLN)→
A2V/V2A cross-attn(1D 时间 RoPE,交叉模态 AdaLN scale/shift + 对方 sigma 门控)→ FFN(×4 gelu-tanh,
video 无 bias / audio 有 bias)。输出头:scale_shift_table + LayerNorm(affine=False) + proj。

**计时**:48 块 ≈ 104 ms/块/步(合计 ~5.0 s/步;block47 的 182 ms 含输出头+guidance+落盘)。
30 步 sigma 由 LTX2Scheduler 生成(tokens=4096 锚点,shift=2.05,terminal 拉伸到 0.1)——与 baseline 逐位一致。

## Stage 3 — VAE

**Video conv decoder**(1.45 GB,无 timestep 条件):latent `[1,128,9,16,24]` →
per-channel un_normalize → conv_in 128→1024 → 9 个上采样块(res_x×4 组 + compress_all/space/time,
DepthToSpace + CausalConv3d 对称时间 pad)→ PixelNorm → conv_out 128→48 → unpatchify(4×4)→
`[65,512,768,3]` → `[-1,1]→[0,1]`。全量单次 decode 峰值 3.5 GB(baseline 的 AUTO_TILING 在
512×768×65 下也恰好是单 tile,计算等价)。
注意 unpatchify 通道序 `(c, p_t, r→w, q→h)`,q 最快——写反会花屏。

**Audio**:latent `[1,8,68,16]` → un_normalize(patchified 128 维统计)→ 2D VQGAN decoder
(causal-conv2d 沿时间轴,8→512→…→128→2)→ mel `[1,2,269,64]` → BigVGAN-AMP1 vocoder
(snakebeta + 抗混叠 Activation1d,×160 → 43040@16k)→ BWE 生成器(mel 残差 + hann-sinc 上采样 skip,
×3 → 129120@48k)。vocoder 全程 fp32 autocast(bf16 会显著劣化频谱)。

## 与官方管线的一致性(standalone/compare.py)

| 对象 | 结果 |
|---|---|
| sigmas / 初始噪声 / positions | 逐位一致 |
| TE video/audio encoding | cos≈1.0,valid 区 maxabs=0.125(=bf16 ulp) |
| DiT 每步 x0(cond/uncond/ptb/mod) | step0 cos≥0.9996 → step29 cos≈0.996(bf16 内核序噪声逐步累积) |
| 最终 video latent | cos 0.9963 |
| 解码视频(同 latent) | 逐位一致;端到端 meanabs 0.010/像素 |
| 解码音频(同 latent) | cos 0.99999;端到端 cos 0.952(vocoder 对 latent 漂移敏感) |

结论:standalone 与官方管线**数学等价**,差异全部来自 bf16 下不同 kernel 顺序的舍入,
在 30 步采样中按混沌方式放大;同 seed 下轨迹高度一致,输出视觉/听觉等价。

## 复现

```bash
reference/.venv/bin/python baseline/run_baseline.py            # 官方管线 + 插桩 (~3 min)
reference/.venv/bin/python standalone/run_all.py --stage all --block-timing   # 独立实现 (~5 min)
reference/.venv/bin/python standalone/compare.py               # 逐阶段对比
reference/.venv/bin/python standalone/debug_dit.py both        # DiT 中间张量定位
```
