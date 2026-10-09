# qwen-image21-swift — Porting spec: Qwen-Image-2.1 → Swift/MLX

Status: **WIP port, research/evaluation licence** (see §1). Started 2026-09-20 (release day).
Reference = diffusers **main** `QwenImage21Pipeline` (huggingface/diffusers#14804, Day-0),
transformers 5.14 `Qwen3VLForConditionalGeneration`. Oracle + goldens:
`mlxengine-image/WIP/qwen-image21-oracle` (`make_goldens.py`, fp32 CPU torch).
Upstream model card: https://huggingface.co/Qwen/Qwen-Image-2.1 · code: github.com/QwenLM/Qwen-Image-2.1.

## 0. What "update to 2.1" actually is

Qwen-Image-2.1 is **not a checkpoint bump** of the 2511 family our `qwen-image-edit-swift`
serves. Every component changed:

| | Qwen-Image / Edit-2511 (`PROD/qwen-image-edit-swift`) | **Qwen-Image-2.1** |
|---|---|---|
| DiT | 20B, 60 **dual**-stream MMDiT blocks, hidden 3072, GELU FFN 4×, per-block img/txt modulation (6 params), biases everywhere | **7B, 32 single-stream blocks**, hidden 4096 (32 heads × 128), **SwiGLU** FFN 3× (12288), **one shared modulation** linear (4096→16384 = scale/gate ×2) for all blocks, **tanh-squashed gates**, **no biases anywhere** |
| Attention | joint [text, image] full attention (+ text padding mask) | **block-causal**: `(q ≥ k) ∨ same_image_block` — text strictly causal, every image block bidirectional, target sees everything; **prefix KV cache** across denoise steps |
| Conditioning tokens | text stream separate; cond-image latents concatenated after target on the image stream | **one interleaved sequence**: cond-image latent tokens are *substituted* at the VL `<|image_pad|>` slots (1 slot → 2×2 latents), target tokens appended; text/cond modulate from **t = 0** (`causal_condition`) |
| Text encoder | Qwen2.5-VL-7B, `hidden_states[-1]` **post**-norm, 3584-d | **Qwen3-VL-8B** (byte-identical to `Qwen/Qwen3-VL-8B-Instruct`, §3), last layer **PRE-final-norm**, 4096-d, real 3-D M-RoPE positions |
| VAE | Wan2.1 3-D causal, 16-ch, 8× spatial, RGB | **Wan2.2-style residual** (AvgDown3D/DupUp3D shortcuts), **64-ch, 16× spatial**, **RGBA** in/out, plain 2-D convs (image specialisation), fp32 checkpoint |
| Latent packing | 2×2 patchify → 64-ch tokens (patch_size 2) | **patch_size 1**: plain spatial flatten, 64-ch tokens; size multiple of **32** |
| Scheduler | FlowMatchEuler dyn-shift (256/4096, 0.5/1.15); Flash static shift 3 | FlowMatchEuler dyn-shift **(256/8192, 0.5/0.9)** + **shift_terminal 0.02** |
| Guidance | true CFG 4.0 (norm-rescaled) | **none by default** (true_cfg_scale 1.0); plain CFG if enabled |
| Default output | 1024² (edit follows the first image) | **native 2048²** T2I; edit follows the **last** image at `output_resolution`² (default 1024) |
| Steps | 20–50 | **40** |
| Extras | LoRA (Lightning/TeleStyle), int4/int8, FireRed swap | native transparency (RGBA prompt format), ≤10 reference images, mask/circle-guided edits; PE prompt-rewriter models (Qwen3.5-VL-9B, separate repos) |
| Licence | **Apache-2.0** | **Qwen RESEARCH License — non-commercial only** |

So the work is a **new port** (new package, new core), not an update of the existing one.
What carries over: the fleet's Qwen3-VL Swift backbone, the Wan-VAE block idioms, the residency /
cancellation / loader conventions, and the NAX workaround.

## 1. Licence — the gate that decides where this can go

`Qwen/Qwen-Image-2.1`, `-PE-T2I`, `-PE-I2I`: **Qwen RESEARCH LICENSE AGREEMENT** (release date
2026-09-20). §1(i) *"Non-Commercial" shall mean for research or evaluation purposes only*; §2(a)
grant is *FOR NON-COMMERCIAL PURPOSES ONLY*; §2(b) commercial use needs a separate licence
(model-business@notice.qwencloud.com). Redistribution is permitted with notice/attribution (§3),
"Built with Qwen" on derived models (§4b), Chinese law / Hangzhou courts (§8).

Every previous Qwen-Image release (Qwen-Image, Edit, Edit-2509, Edit-2511, 2512, Layered) was
Apache-2.0. This is a policy change by Qwen for the 2.x line.

Fleet consequence (C7, `MLXServeCore` two-layer licence gate):
- `weightLicense` must be declared as a package-local `SPDXLicense("LicenseRef-Qwen-Research")`,
  **never added to `permissiveAllowlist`** (same call as AB-D-0055 for Audio8 — but stricter: this
  one has no commercial tier at all).
- Forge ships `MLXServeEngine(policy: .permissiveOnly, licenseEnforcement: .blocking)` → the
  package is **refused** in shipping consumers by construction; advisory consumers surface it.
- Position: same as klein-9B (`license:other`, CLAUDE.md watchlist): **a research/eval tier and a
  recipe existence-proof, not a product asset** — unless a commercial licence is obtained. The
  port is still worth having: it is the lightest Qwen image model by far (§7) and the
  architecture (single-stream + prefix cache + RGBA VAE) is what the next open-licensed one
  will look like.

## 2. Reference implementation facts (all verified in the oracle, goldens on disk)

### 2.1 Transformer (`QwenImage21Transformer2DModel`, 297 tensors, 14.23 GB bf16)
- `img_in` 64→4096; `txt_in` = ZeroCenterRMSNorm(4096, eps 1e-6, scale = w+1, fp32) → Linear
  4096→4096 → GELU(tanh) → Linear 4096→4096.
- Timestep: cos-first sinusoid (256, freqs `exp(-ln 1e4 · i/128)`, ×1000) → Linear 256→4096 →
  SiLU → Linear 4096→4096 (`time_text_embed.timestep_embedder`, no bias). The model casts the
  timestep to the activation dtype BEFORE the sinusoid.
- `modulation` = SiLU → Linear 4096→16384 (`modulation.1.weight`), chunks
  `[mod1.scale, mod1.gate, mod2.scale, mod2.gate]`, shared by all 32 blocks. With
  `causal_condition` the timestep gets an extra t=0 row: prefix tokens read row 1, target row 0.
- Block: `h += tanh(g1)·Attn(LN(h)·(1+s1))`; `h += tanh(g2)·SwiGLU(LN(h)·(1+s2))`;
  LN affine-less eps 1e-6; SwiGLU `out(silu(gate_layer(x))·proj(x))`.
- Attention: q/k/v/out no bias; per-head RMSNorm(128, eps 1e-6) (standard diffusers RMSNorm);
  complex RoPE (interleaved pairs), theta 10000, axes (16, 56, 56).
- RoPE positions (`QwenImage21Rope.forward`): cursor walk — text tokens advance one shared
  position on all 3 axes; an image block freezes frame = position, h ∈ [−⌈H/2⌉, ⌊H/2⌋),
  w likewise, then position += max(H, W); trailing text continues.
- `norm_out`: LN(affine-less) · (1 + Linear(silu(temb))) — **scale only**; `proj_out` 4096→64.
- Prefill = exact multi-pass: per prefix segment attend to keys `[0, end)` (text segments with a
  causal triangle), then target rows attend to everything. Decode (`kv_cache_mode="cached"`):
  ONLY target rows run; per-layer K/V of the prefix (post-RoPE) were stored at step 0.
- Joint sequence: each VL image slot → 4 latent tokens (`_IMG_TOKENS_PER_SLOT`); target slots
  = `n_target / 4` appended `True`s to `img_mask`.

### 2.2 Pipeline (`QwenImage21Pipeline.__call__`)
- Templates are RAW strings (not apply_chat_template). System prompt
  `Comprehend and analyze the provided prompt.` → **drop_idx = 14** (tokens
  `[151644, 8948, 198, 1092, 30782, 408, 323, 23643, 279, 3897, 9934, 13, 151645, 198]`).
  T2I: `<|im_start|>system\n{sys}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n`.
  Edit: user content prefixed `<image1><|vision_start|><|image_pad|><|vision_end|>`
  (+ ` <image{i}>…` for more images, space-separated; `<imageN>` is plain BPE text).
- Encoder output = `hidden_states[-1]` with the final RMSNorm **neutralised** (forward hook) —
  the pre-norm last-layer output (|x| mean ≈ 9 vs ≈ 1.5 post-norm). transformers ≥5.0 would
  otherwise return the normed tensor. Left padding; batch 1 → no mask.
- Condition images: `img.convert("RGBA")`; ONE LANCZOS resize (`VaeImageProcessor.resize`, PIL
  premultiplied RGBa round trip) to `calculate_dimensions(output_resolution², aspect)` (/32);
  the VL copy is composited over white → RGB → `Qwen3VLProcessor` (smart_resize factor 32, bicubic,
  min 256², max 4096², mean/std 0.5 → identity resize for our sizes); the VAE copy is the RGBA
  in [−1, 1] (alpha included, `(1,4,1,H,W)`).
- ⚠ Grid coupling: the DiT needs `4 × VL merged tokens == VAE latent tokens` per image. The
  processor's `min_pixels` (65536) can inflate the VL grid for small `output_resolution` (a 3:4
  image at 256² → 224×288 = 64512 px → VL 256×320 vs VAE 224×288 → the REFERENCE raises in
  `build_token_metadata`). Irrelevant at the default 1024; our generator throws a clear error.
- Latents `(B, 1, 64, H/16, W/16)` → pack = `view(B, 64, HW).transpose(1, 2)`; unpack reverse;
  `height/width // 32 * 32`.
- Schedule: `sigmas = linspace(1, 1/N, N)`; `mu = calculate_shift(n_target, 256, 8192, 0.5,
  0.9)`; exponential shift `e^mu / (e^mu + 1/s − 1)`; `stretch_shift_to_terminal(0.02)`;
  trailing 0; `t = sigma` (timestep/1000). Euler `x += (σ_{i+1} − σ_i)·v`.
- Output: `vae.decode(z·std + mean)[:, :, 0]` → `postprocess` → **RGBA** PIL.

### 2.3 VAE (`AutoencoderKLQwenImage21`, 238 tensors, 1.35 GB fp32)
- `is_residual=True`, base 96 / decoder base 144, dim_mult [1,2,4,8,8], num_res_blocks 2,
  temperal_downsample [F,T,T,T], z 64, in/out 4, patch_size None, 16× spatial.
- `QwenImage21CausalConv3d` is a **Conv2d** (frame folded away; weights 4-D). Norm =
  `F.normalize(dim=C)·√C·gamma` (eps 1e-12) = our `WanRMSNorm`.
- Encoder stages (in→out, down, temporal): 96→96 (↓, F), 96→192 (↓, T), 192→384 (↓, T),
  384→768 (↓, T), 768→768 (–). Each: 2 resnets → `downsampler` (ZeroPad2d(0,1,0,1) + stride-2
  conv; 3d modes carry an unused-for-T=1 `time_conv`) then **+ AvgDown3D(x)**. AvgDown3D
  front-zero-pads T to a multiple of factor_t — for T = 1 with factor_t 2 the even output
  channels are the ZERO frame's mean (exactly what the reference computes).
- Decoder stages: 1152→1152 (↑, T), 1152→1152 (↑, T), 1152→576 (↑, T), 576→288 (↑2d), 288→144
  (–); 3 resnets each → `upsampler` (nearest 2× + conv dim→dim) then **+ DupUp3D(x,
  first_chunk=True)** (drops the duplicated leading frame; a channel→space shuffle when
  in ≠ out). Mid blocks: resnet → single-head attention over HW → resnet.
- `quant_conv` 128→128, mode = first 64 channels; `post_quant_conv` 64→64; decode clamps [−1, 1].
- latents_mean/std: 64 each (vae/config.json).

### 2.4 Text encoder
`text_encoder/` = 750 tensors, total 17,534,247,392 bytes, key set identical to
`Qwen/Qwen3-VL-8B-Instruct`, sampled tensors byte-identical (all 750 checked per-tensor, first
64 KB each: `qwen-image21-oracle/text_encoder_identity.json`). → **reuse the local
`weights/Qwen3-VL-8B-Instruct` snapshot and `qwen3vl-mlx-swift`**; no download, no new port.
The `processor/` tokenizer is the same vocab/merges re-serialised by transformers 5.

## 3. Reuse map

| Need | Source | Delta |
|---|---|---|
| Qwen3-VL-8B backbone (LM + ViT + deepstack + M-RoPE, parity-locked) | `mlxengine-think/PROD/qwen3vl-mlx-swift` | +`lastHiddenState(applyFinalNorm:)` (branch `pre-norm-hidden-state`, additive, default unchanged) |
| Qwen3-VL image preprocessing (smart_resize, PIL bicubic, patchify) | `Qwen3VLImageProcessor` (same package) | none (defaults == 2.1 processor config) |
| Wan-VAE blocks (RMS norm, resnet, attention, resample) | `qwen-image-edit-swift/QwenVAE.swift` | 2-D convs instead of causal 3-D |
| Residual shortcuts AvgDown3D / DupUp3D / up-down stages | `wan-core-mlx-swift/WanVAE22.swift` | copied reshape order; explicit T axis for the T=1 semantics |
| Timestep sinusoid, complex RoPE apply, strict loader, residency, CAN seams, NAX chunk | `qwen-image-edit-swift` | cos-first / no-bias variants; NAX window recomputed (§6) |
| PIL LANCZOS | `qwen-image-edit-swift/PromptEncoder.swift` | 4-band + Pillow's RGBa premultiply/unpremultiply |

## 4. Package layout (`WIP/qwen-image21-swift`)

```
Sources/QwenImage21/
  QwenImage21Transformer.swift      model, blocks, attention (segment prefill + KV cache), RoPE, layout
  AutoencoderKLQwenImage21.swift    RGBA residual VAE (channels-last 2-D)
  QwenImage21PromptEncoder.swift    templates, drop_idx, pad expansion, pre-norm features
  QwenImage21Pipeline.swift         scheduler, packing, calculate_dimensions, generator
  QwenImage21Weights.swift          loaders (two-way strict) + key renames
  QwenImage21ImageIO.swift          RGBA LANCZOS, white composite, PNG read/write
Sources/QwenImage21Gate/main.swift  --sched --attn-probe --resize --vae --encoder --dit --generate
Tests/QwenImage21Tests/             scheduler / dimension unit tests
```
Later: `Sources/MLXQwenImage21` — the MLXEngine `ModelPackage` wrapper (textToImage + imageEdit
on one core, PackageID `qwen-image-2.1`, C7 `LicenseRef-Qwen-Research` package-local).

## 5. Parity plan (fp32, CPU stream) — goldens in `qwen-image21-oracle/goldens/`

| Gate | Golden | What it locks |
|---|---|---|
| `--sched` | `scheduler.json` (10 schedules) | mu, exponential shift, terminal stretch |
| `--attn-probe` | self-contained | MLX SDPA `.causal` end-alignment for Lq < Lk (the segment prefill premise) |
| `--resize` | `img_*.png` → `*_resized_rgba.png` | PIL LANCZOS RGBA premultiplied round trip |
| `--vae` | `vae_img_a/b`, `vae_random_decode` | encode mode, normalisation, decode, PNG→VAE input |
| `--encoder` | `encoder_{t2i_short,t2i_card,edit_1img,edit_2img}` | token ids, image_pad_mask, pixel_values, PRE-norm embeds |
| `--dit` | `dit_{t2i_short,edit_1img,edit_2img}` | layout/segments, rotary, temb, modulation, blocks 0/1/7/15/31, prefill output, KV cache, cached step, uncached step |
| e2e | `e2e_*.png` (bf16 MPS reference render) | decoded-output eyeball at 1024² |

Layouts: T2I 16×16 target (256 tokens); edit 320² → 20×20 cond + 20×20 target; two images
(320² + 288×384) → target 24×18 (follows the last image). Per-block goldens localise any break
without a Python twin (mlx-porting: direct PyTorch → Swift with granular goldens).

## 6. Known hazards carried in

- **mlx#3797 NAX split-K GEMM** (mlx-swift ≤ 0.31.6; fixed by mlx#3810, shipped in mlx-swift
  0.32.3): `img_mlp.out` is K = 12288, N = 4096 → the bad window was **1024 ≤ rows ≤ 4096** at
  half precision — the CACHED-decode pass of every 512²…1024² render. It was row-chunked at ≤896
  rows until 2026-10-01; the chunk is gone and the manifest floor is mlx-swift 0.32.3.
- Long-graph fused-dispatch corruption family: `chainBlockGraphs` lever kept (off).
- RNG: MLX normal ≠ torch randn; parity injects the torch noise.
- bf16 vs fp32 timestep rounding: the reference rounds sigma to bf16 before the sinusoid.

## 7. Cost / footprint (estimates — measure before declaring; AB-L-0063)

Resident: DiT 14.2 GB bf16 + VAE 1.35 GB fp32 (0.68 bf16) ≈ **15.6 GB** — vs the 2511 core's
41.4 GB floor. Encoder 17.5 GB bf16 per request (evicted before denoise, or resident on ≥64 GB).
KV cache: prefix_len × 32 layers × 2 × 32×128 × 2 B = **0.5 MB per prefix token** (1024² cond
image = 2.1 GB; 10 refs ≈ 21 GB). Attention at 2048² = 16384 target tokens: fused SDPA (no N²
materialisation). This is the first Qwen image model that plausibly fits the 32 GB tier.

## 8. Decisions log

- 2026-09-20 New package, not a fourth wrapper in `qwen-image-edit-swift`: different core,
  different encoder backbone, different licence posture (must not contaminate the Apache-2.0
  package's story).
- 2026-09-20 Text encoder: reuse `Qwen3-VL-8B-Instruct` weights (identity verified) rather than
  downloading 17.5 GB of identical bytes; the pipeline's `text_encoder/` is only referenced.
- 2026-09-20 `qwen3vl-mlx-swift` gets an additive `applyFinalNorm` flag instead of a fork; local
  path dependency until the tag ships (fleet sweep flags path deps — temporary by design).
- 2026-09-20 Oracle goldens at `output_resolution=320` (not 256) because of the min_pixels
  grid-coupling edge (§2.2).
- 2026-09-20 First DiT gate: T2I layout green, edit layouts cos 0.53 at block 0 → the joint gather
  addressed image tokens at `text-positions + i`; the encoder tensor still holds the VL pad rows,
  so images live at `T + i` (reference: cat → repeat_interleave → overwrite). Fixed; all three
  layouts green on target rows (AB-R receipts). Per-block goldens made it a one-run diagnosis.
- 2026-10-09 Qwen-Image-2.1-Turbo (8-step distill, same research licence) is a SIBLING PACKAGE on
  this core (`QwenImage21TurboPackage`, same module), not a mode: the DiT weights differ (14.2 GB),
  a mode would swap them per request — the `MLXQwenImageFlash` precedent. Its VAE is the base VAE
  cast to bf16 and its text encoder is byte-identical to Qwen3-VL-8B-Instruct (AB-R-0437), so the
  package has three roots (Turbo DiT + pipeline config, base `vae/`, Instruct) and the mirror
  `xocialize/Qwen-Image-2.1-Turbo` carries only what Turbo changed. The schedule is the
  checkpoint's `sample_sigmas` (diffusers PR #14950) used verbatim — `generate(sigmas:)` is the
  generic seam (a LoRA-form distill such as Viggle's could be a mode through it); a request's
  `steps` is ignored, as in the reference. Memo: `TURBO-EVAL.md`; task AB-T-0212.

## 9. Gate results (running log)

- 2026-10-09 TURBO (weights `Qwen/Qwen-Image-2.1-Turbo` @ d65dbc9, shards sha256-verified; goldens
  `goldens/turbo/` from the Turbo DiT on diffusers main 1d5d056, fp32 CPU): `--sched` exact on the
  fixed grid at 4,096 and 16,384 tokens (maxAbs 0; base entries unchanged at ≤1.2e-7); `--dit` on
  Turbo weights green on all three layouts — step-0 target rows cos 1.0000000 (relMax ≤ 1.8e-5),
  cached step cos 1.0000000, KV cache L0/L31 within 8e-4 relMax, blocks 0/1/7/15/31 cos ≥
  0.9999998. bf16 GPU e2e at the 8 fixed steps: fox 1024² seed 42 coherent beside the oracle's
  MPS render (`e2e_turbo_fox_1024.png`; different RNG, so an eyeball gate), 1.34 s/step, 13.2 s
  for 8 steps, peak 31.2 GB (Release, lightly contended). Offline wrapper gates green: manifest,
  MAT (three repos), CAN-1..3 on both surfaces.
- 2026-10-09 TURBO timing + membench, Release, uncontended (AB-R-0438): T2I 1024² 1.33 s/step,
  13.1 s per generate, peak 31.2 GB; 1-ref edit 1024² prefill 4.7 s + 1.50 s/step, 16.9 s; T2I
  2048² 8.0 s/step, 71 s, peak 33.8 GB. Membench floor 15.58 GB, activation 16.3 / 23.5 / 17.5 /
  23.3 GB (T2I 1024², 2048², edit 1, 4 refs) — inside the base declaration, so the Turbo manifest
  carries the base split. Edit e2e: Swift's 8-step "red scarf" on `e2e_photo_dog.png` is correct;
  the oracle's seed-42 MPS edit of the same file is the haloed no-op — that PNG is the base model's
  own 1024²/seed-42 render, so seed 42 replays its noise (§12, diffusers#14824); Swift's edit seed
  offset is what makes it immune. Oracle re-run at seed 4242 for the eyeball comparison.
- 2026-09-20 sched/attn-probe/resize/vae/encoder: green (AB-R-0253). DiT GPU fp32: green on
  target rows, prefill, KV cache, cached + uncached steps for all layouts; 2-image prefix rows
  relMax 3.9e-2 at cos 0.99999 (GPU noise, CPU run pending).
- 2026-09-20 CPU-stream DiT gate: T2I and 1-image layouts target rows cos 1.0000001, relMax ≤7e-6
  (prefill, KV cache, cached and uncached steps all within 1e-5 relative) — bit-level parity.
- 2026-09-20 E2E bf16, 1024², 40 steps, the reference's own seed-42 noise injected: Swift (MLX
  GPU) vs torch (MPS) final latents cos 0.99933, PSNR 36.5 dB RGB, visually identical (the neon
  "QWEN IMAGE 2.1" sign renders legibly in both). Debug build 3.8–10 s/step under contention
  (torch render + CPU gate concurrently); Release sustained timing owed (AB-L-0063).
- 2026-09-20 Engine wrapper `MLXQwenImage21` (PackageID qwen-image-2.1, textToImage + imageEdit):
  builds against mlx-engine-swift 0.56.0 (contract 1.42.0); offline gates green — manifest
  (licence outside the allowlist), MAT-1..5 across both repos, CAN-1..3 on both surfaces.
  Registry row added (mlx-engine-swift 784c9c3). Footprint still ESTIMATED.
- 2026-09-20 E2E bf16 1024² EDIT (the T2I render as input, "Change the background to a sunset
  beach", seed-42 noise injected on both sides): Swift vs torch final latents cos 0.99935, PSNR
  36.3 dB — the port reproduces the reference at production scale for editing too. ⚠ Both outputs
  are an over-sharpened, haloed copy of the input with the background unchanged: that is the
  reference pipeline's own behaviour here (identical on torch), NOT a port defect. Whether it is
  input/prompt-specific or a Day-0 `QwenImage21Pipeline` edit-path issue is being probed with a
  natural-photo T2I→edit chain on the torch side (`run_photo_edit_probe.sh`).
- 2026-09-20 E2E bf16 320² EDIT (synthetic img_a, same instruction, matched noise): Swift vs torch
  cos 0.99973 / PSNR 35.1 dB and the edit is CORRECT on both sides (background → sunset beach with
  palms, the circles and the "Qwen 2.1" text preserved; outputs/edit_320_side_by_side.png). So
  the haloed 1024² result above is input/prompt-specific reference behaviour, not the edit path.
- 2026-09-20 CPU-stream gate note: the first CPU run crashed on the 2-image layout with a Metal
  command-buffer timeout (`kIOGPUCommandBufferCallbackErrorTimeout`) while three GPU renders were
  running concurrently — the Metal-watchdog family; the T2I and 1-image layouts had already
  passed at relMax ≤ 1e-5. Re-run of that case queued uncontended.
- 2026-09-20 PRODUCTION-SCALE DiT gate (`--dit-large`, 1024² edit: 8,213 joint / 4,117 prefix /
  4,096 target tokens, golden fp32 CPU torch): fp32 GPU step-0 target rows median per-token cos
  0.9999964, 17/4,096 tokens above relMax 2e-2 (worst cos 0.976 at grid spots (8–9, 39–40) and
  (60, 13)); step-1 cached 0/4,096 tokens above 2e-2, min row cos 0.9998; KV cache L0/L31 cos
  0.9999999 / 0.9999987. bf16 GPU: median row cos 0.99990, 497 tokens above 2e-2. **Self-control:**
  our bf16 vs our fp32 disagrees on 522 tokens with the SAME worst rows (552, 3853, 591, 623, 392)
  — the outliers are precision-sensitive tokens, not a scale-dependent port defect (mlx-porting
  "dtype self-control" rule). Whole-tensor relMax is the wrong statistic at this scale; the gate
  should use per-row percentiles (todo). Forward times (Debug, GPU, contended): prefill 17.1 s
  bf16 / 22.5 s fp32, cached 10.2 s / 12.6 s.
- 2026-09-20 Oracle-version check: the model card requires transformers ≥ 5.17 and the oracle
  had 5.14.1; a second env (`.venv517`: transformers 5.17.0, same diffusers commit 80c7ed26)
  reproduces the encoder goldens BIT-IDENTICALLY (maxAbs 0 on ids, position_ids, pixel_values
  and pre-norm embeds for all four cases, incl. two images) — the Qwen3-VL rotary refactors
  between the versions are numerically neutral here. Encoder goldens stand.
- 2026-09-20 Reference-side edit probe (torch/MPS, 1024², natural photo from the model's own
  T2I): "Change the background to a sunset beach" and "Make the dog wear a red scarf" BOTH return
  the over-sharpened haloed near-copy with the instruction ignored
  (`qwen-image21-oracle/goldens/probe_side_by_side.png`), while the 320² edit is correct. This is
  systematic reference behaviour at 1024² under diffusers main @ 80c7ed26 (defaults: KV cache on,
  no CFG, 40 steps). Discriminators queued: cache off, output_resolution 512/768, true CFG 4.
  No upstream fix or report yet (issues #14804/#14817/#14820/#14821 are features, not this).
- 2026-09-20 Release timing (uncontended GPU, Debug→Release binary): T2I 1024²/40 steps 128.4 s
  wall, sustained **3.0–3.5 s/step**, peak 31.2 GB (torch MPS bf16: 80 s / ~2 s/step on the same
  box → ~1.5× headroom: fp32 RoPE cast path, modulation slice/concat, 5-way NAX chunk at 4096
  rows are the suspects). Edit 1024²: prefill 11.5 s (8.2k tokens), cached steps ~3.6 s.
- 2026-09-20 Release timing, native 2048² T2I (20 steps): sustained **19.3 s/step** (16,384
  tokens; attention-bound: 6.2× the 1024² step for 4× tokens), denoise peak 31.2 GB, but the
  **fp32 VAE decode at 2048² spikes the process peak to 62.4 GB** — AB-T-0021 (tiled VAE decode
  becomes a prerequisite the day an output cap rises) applies to this package on day one; a bf16
  decoder or the reference's `enable_tiling` (256-px tiles, 192 stride, blend) is the fix. Render
  coherent (`outputs/rel_t2i_2048.png`). Edit 1024² Release: 164.1 s wall / 40 steps.
- 2026-09-20 CPU-stream DiT gate COMPLETE (GPU idle): 2-image layout block_31 relMax 7.6e-6,
  target rows cos 1.0000000 relMax 6.5e-5, cached step 4.9e-5, cached-vs-uncached 7.0e-6 — all
  three layouts CPU-exact; the earlier crashes on this case were the cross-process GPU watchdog.
- 2026-09-20 Reference-side discriminators (torch/MPS bf16, natural photo, "Make the dog wear a
  red scarf"; `qwen-image21-oracle/goldens/probe_scale_grid.png`): **output_resolution 512 and 768
  → perfect edits** (scarf added, photo intact); 1024 → haloed near-copy, instruction ignored;
  1024 with `use_kv_cache=False` → identical halo (not the cache); 1024 with true CFG 4 →
  scarf appears but still haloed. So the reference degrades specifically at the 1024² edit size
  (4,096 condition + 4,096 target tokens). Remaining suspect: bf16 precision at 8k tokens —
  fp32 runs on both sides launched (`QI21_DTYPE=fp32`, `--fp32-dit`). Until resolved, the
  package's edit surface should default `output_resolution` to 768.
- 2026-09-20 transformers 5.17.0 full-pipeline check: the 1024² scarf edit and the 320² beach
  edit re-rendered in the 5.17 env are BIT-IDENTICAL to the 5.14.1 renders (latent cos 1.000000,
  maxAbs 0) — the transformers floor on the model card changes nothing here; the 1024² halo is
  not a version artefact. fp32 probes (torch MPS, Swift `--fp32-dit`) are the last discriminator.
- 2026-09-20 fp32 discriminator, Swift side (`--fp32-dit`, fp32 DiT + fp32 VAE, bf16 encoder,
  same noise): the 1024² scarf edit is the SAME haloed near-copy (outputs/probe_scarf_fp32_swift.png)
  — precision is ruled out. With cache, CFG, resolution, transformers version and dtype all
  probed, the 1024²-edit degradation is a property of the reference pipeline's math/config for
  that size (candidates for upstream: RoPE frame-axis offset between condition and target blocks
  scaling with max(h, w); `calculate_shift` fed target-only tokens; target/condition resolution
  coupling). Upstream report drafted (`qwen-image21-oracle/UPSTREAM-REPORT-1024-edit.md`) and later filed as huggingface/diffusers#14824.
  fp32 DiT cost for the record: 18–23 s/step at 1024², peak 53.6 GB.
- 2026-09-20 fp32 discriminator, both sides, CLOSED: torch fp32 (MPS, 1,079 s / 40 steps) is the
  same haloed near-copy; torch fp32 vs torch bf16 43.4 dB (cos 0.99979); **Swift fp32 vs torch
  fp32 53.6 dB (cos 0.99999)** at the 8.2k-token production layout — the tightest end-to-end
  agreement so far, and final proof that the 1024²-edit behaviour is the reference pipeline's,
  not the port's (upstream draft: qwen-image21-oracle/UPSTREAM-REPORT-1024-edit.md).

## 10. Status summary (2026-09-20 EOD)

Done: oracle + goldens (incl. 8.2k-token layout); Swift core (DiT, RGBA VAE, Qwen3-VL prompt
encoder, pipeline, image I/O); every parity gate green (CPU-stream 1e-6 at 320², production
scale fp32 median row cos 0.9999964); e2e == torch at 36 dB (bf16) / 53.6 dB (fp32); engine
wrapper with manifest/MAT/CAN tests 9/9; registry row; Release timing; licence decision filed.
Open (AB-T-0154): footprint split measured (QI21_MEMBENCH + in-app phys); tiled/bf16 VAE decode
for 2048² (62 GB peak, AB-T-0021); qwen3vl-mlx-swift v0.3.0 tag + path-dep flip (AB-A-0081);
in-app validation through a consumer; the reference's 1024²-edit degradation (upstream report
draft; wrapper defaults edits to 768² meanwhile); Swift step-time headroom (~1.5× vs torch MPS).
- 2026-09-20 (operator go) qwen3vl-mlx-swift **v0.3.0 tagged and pushed** (main fast-forwarded to
  5cbd4f1); this package now pins `from: "0.3.0"` (no local path dependency). Upstream report
  **filed: https://github.com/huggingface/diffusers/issues/14824** (QwenImage21Pipeline editing
  degrades at output_resolution=1024; 512/768 correct; cache/CFG/version/precision excluded).
- 2026-09-20 **Independent-implementation oracle (AB-T-0155, ComfyUI native @ c194dd00, MPS, no diffusers
  code; note: `qwen-image21-oracle/COMFY-ORACLE-1024-edit.md`, outputs `goldens/comfy/`).** The 1024² scarf
  and beach edits collapse identically in ComfyUI — **40.1 / 39.7 dB PSNR to the diffusers outputs with
  ComfyUI's own noise and schedule**, 40.8 dB with the diffusers noise + exact sigmas injected, and **49.1 dB
  between two ComfyUI noise draws** (noise-independent, reference-dominated). 768² agrees with diffusers at
  33.2 dB and is correct on both. New boundary probes: **896² and 960² edits are clean and correct**; a
  **768² reference paired with a 1024² target still degrades** (HF ratio 2.33, no scarf) → the trigger is
  the *target* grid at 64 latents per side, not the joint token count and not the reference. Verdict: the
  reference pipeline's math is the model's published recipe and the model degrades in that regime; the
  wrapper's 768² edit default can move to **960²** after an in-app eyeball; 1024² stays capped until the
  authors answer diffusers#14824 (follow-up comment drafted, not posted).
- 2026-09-21 Upstream #14824 activity: **@peterc independently confirmed on CUDA** (RTX 3090 Ti,
  diffusers @80c7ed26, torch 2.14+cu130, transformers 5.17, bf16, no ComfyUI) — same class of
  failure with a harsher symptom (the subject is replaced, e.g. the dog becomes a cat, rather
  than preserved-and-haloed), seed-independent, and his boundary is LOWER: 896² broken / 864²
  fine (56 vs 54 latents per side) against our 960² clean / 1024² broken (60 vs 64). So the
  threshold is fixture- and platform-dependent, not a hard cutoff — which argues against moving
  the wrapper's edit default to 960² on our fixture alone; **768² stays the default**.
  @sayakpaul (maintainer) asked whether other platforms are affected and whether prompts/CFG
  help, and is routing to the Qwen team (@naykun). Follow-up comment revised
  (`qwen-image21-oracle/UPSTREAM-COMMENT-14824-draft.md`), awaiting the operator's go.
- 2026-09-21 Package **published: https://github.com/xocialize/qwen-image21-swift** (public,
  17 tracked files, no weights/goldens/outputs; qwen3vl-mlx-swift pinned by tag).
- 2026-09-21 Follow-up POSTED to #14824
  (https://github.com/huggingface/diffusers/issues/14824#issuecomment-5762252236): ComfyUI
  cross-check, noise-independence, prompt/CFG results, the two boundaries side by side, and the
  mismatched-grid probe. Ball is with the Qwen team; watch for their answer before revisiting
  the 768² edit cap.

## 11. State at the 2026-09-21 reboot (resume from here)

**Pushed to GitHub (survives anything on this box):**
- `xocialize/qwen-image21-swift` @ main — this package (17 files, no weights/goldens/outputs).
- `xocialize/qwen3vl-mlx-swift` @ **v0.3.0** — the conditioner backbone this package pins by tag.
- `xocialize/mlx-engine-swift` @ main — the registry row for this package.
- `xocialize/AgentBridge-Store` @ main — AB-T-0154/0155, AB-D-0085, AB-L-0133, AB-R-0253…0264, AB-A-0081.
- Upstream: https://github.com/huggingface/diffusers/issues/14824 (+ our follow-up comment).

**On the volume only (no git remote; regenerable, but single-copy):**
- `WIP/qwen-image21-oracle` (3.9 GB): `make_goldens.py` (every golden phase incl. `dit_large`),
  `check_text_encoder_identity.py`, the probe scripts, the two upstream docs, `goldens/` (1.9 GB),
  `.venv` (transformers 5.14.1) and `.venv517` (5.17.0).
- `WIP/comfyui-oracle` (32 GB): the ComfyUI clone + venv + merged single-file weights, `submit.py`,
  `poll.py`, `compare.py`, `merge_shards.py`. **Server is stopped**; restart per COMFY-ORACLE-1024-edit.md.
- `weights/Qwen-Image-2.1` (15 GB) and `weights/Qwen3-VL-8B-Instruct` (16 GB).

**Nothing was left in `/private/tmp`** (AB-L-0086): no harness, no goldens, no fixtures — only
disposable build logs and re-downloadable upstream reference copies.

**Cold resume:** `swift build` (resolves the qwen3vl tag from the network), then
`.build/debug/QwenImage21Gate --sched ../qwen-image21-oracle/goldens` as the 2-second smoke test.
Next tasks, in order: (1) `QI21_MEMBENCH`-style measured footprint split + in-app phys re-baseline;
(2) tiled or bf16 VAE decode for 2048² (62.4 GB decode peak, AB-T-0021); (3) in-app validation
through a consumer; (4) step-time headroom (~1.5× vs torch MPS). The 768² edit cap stays until
the Qwen team answers #14824.

## 12. Noise replay — the real cause of the "1024² edit" failure (2026-09-21, CLOSED)

The Qwen team (@naykun) called it on #14824 and it is confirmed. The fixture photo was generated
T2I at 1024²/seed 42; every failing edit then ran at 1024²/seed 42, so **the edit's initial noise
was the draw that generated the image**, and the trajectory re-ran the generation instead of
following the instruction. There is no size threshold and no diffusers bug.

| run (same photo + prompt, 1024², 40 steps) | PSNR vs input | grad-energy ratio | result |
|---|---:|---:|---|
| torch, seed 42 (= the T2I seed) | 13.00 | 2.88 | replay: no scarf, haloed |
| torch, seed 12345 | 22.22 | 1.02 | correct |
| torch, seed 777 | 21.28 | 0.97 | correct |
| Swift, MLX-native noise (different RNG) | 20.86 | 0.94 | correct |
| Swift, injected torch seed-42 noise | 13.05 | 2.88 | replay (matches torch exactly) |
| **Swift, our own T2I → edit, same size + seed** | **13.73** | **2.97** | **replay — the user-facing trap** |
| Swift, same but seed 4242 | 21.09 | 1.05 | correct |
| **Swift, same seed 42, AFTER the fix below** | **17.27** | **1.07** | **correct** |

**Mechanism is RNG-independent**: it needs only that the two draws coincide (same stream, seed and
shape). Reproduced with torch's RNG and with MLX's.

**Fix (`QwenImage21Latents.noiseSeed`)**: the edit path offsets the seed by a fixed constant, so an
edit can never draw the text-to-image noise for the same seed and size. Deterministic; injected
`latents` bypass it, so parity fixtures and a deliberate replay repro still work. Unit-tested.
**The 768² edit cap is lifted — `defaultEditOutputResolution` is back to 1024.**

**Corrections to earlier entries in this file.** The §9/§10 "reference degrades at 1024² edits"
framing was wrong, and so were the discriminator conclusions built on it: 512/768/896/960 worked
because a different target shape draws different noise, not because of a size threshold; the
"768² reference + 1024² target still degrades" probe had a 1024² target, so it was replay too;
the ComfyUI cross-check was NOT noise-independent (`prepare_noise` uses `torch.manual_seed` +
CPU `torch.randn`, the same stream `randn_tensor` uses — measured cos 0.999999 vs the diffusers
draw, max diff 0.015 = bf16 rounding), so it rules out a diffusers coding bug but says nothing
about noise. The clean 1024² edit of a synthetic (never-generated) image in §9 was the control
that should have been run first.
- 2026-09-21 Correction POSTED to #14824
  (https://github.com/huggingface/diffusers/issues/14824#issuecomment-5770745487): confirms the
  noise-replay diagnosis with the seed table, retracts the "noise-independent" claim with the
  measurement, reports the RNG-independence + our generate-then-edit repro and the seed
  domain-separation fix, and flags that @peterc's case is probably NOT covered by ours.

## 13. Measured footprint + enforced envelope (2026-09-22, AB-R-0290)

`QwenImage21Gate --membench` (Release, GPU idle): floor = active memory after load with the cache
cleared; per envelope, cache cleared and peak reset, one full request (the per-request Qwen3-VL
load lands in the activation term).

| envelope (fp32 VAE) | peak GB | activation GB |
|---|---:|---:|
| T2I 1024² | 32.87 | 17.29 |
| edit 1024², 1 ref | 35.03 | 19.45 |
| edit 1024², 4 refs | 41.52 | 25.94 |
| edit 1024², 10 refs (model max) | 51.57 | 35.98 |
| T2I 2048² | 72.91 | 57.33 |

Floor 15.58 GB. **Declared: resident 15.6 GB + activation 43.2 GB (10-ref × 1.2) = 58.8 GB**,
replacing the 15.6 + 19 GB estimate, and below the 2511 package's 63 GB. **Enforced** in `run()`
by `QwenImage21Envelope`: output area ≤ 1024², ≤ 10 references, edit output_resolution ≤ 1024 —
outside requests are refused with a clear error (not clamped), so the declaration is an upper
bound. The target-size rule now lives in one place (`QwenImage21Latents.targetSize`) shared by the
generator and the guard.

bf16 VAE, measured and NOT adopted: floor 14.91 GB, 4-ref edit 35.44 GB (−6 GB), but 2048² still
63.59 GB (−13%), so it does not fix the native size; decode quality 42.8–44.9 dB PSNR vs fp32 with
0.15–0.35% of 8-bit values off by > 8 levels and outliers to 75. **2048² needs tiled decode
(AB-T-0021) — that is the next task**, and it lifts the area cap.

## 14. Halo-tiled VAE decode — native 2048² unlocked (2026-09-22, AB-R-0310, closes AB-T-0021 here)

`AutoencoderKLQwenImage21.decodeTiled`: the wan-core vae22 recipe. PREFIX (conv_in, mid-block
attention, up_blocks 0–1) runs whole on the latent grid; the local SUFFIX (up_blocks 2–4, norm_out,
conv_out) is tiled with a halo of real neighbour pixels and cropped — no blending. Suffix receptive
field computed first as ≈ 11.5 suffix-input px → halo 12.

- **Exact:** CPU stream, halo sweep — 12 is the first bit-identical halo (max|Δ| 0.0); 11 → 1.2e-7,
  8 → 108 dB, 0 → 27 dB with visible seams.
- **GPU, real latents:** 64–65 dB at 1024², 66.7–67.2 dB at 2048² vs untiled.
- **GPU, random latents — the trap:** max|Δ| 1.84 looks like a bug, but untiled GPU is equally far
  from torch-CPU ground truth (1.906 vs 1.893 max, 2.53e-3 vs 2.52e-3 mean). Judge exactness on
  CPU and fidelity on real latents, never on off-manifold noise on the GPU.
- **Memory:** 2048² decode 53.6 → 17.8 GB. Whole-request T2I peaks: 2048² 42.03 GB (was 72.91),
  2400×1792 41.60, 2752×1536 41.15 — all inside the declared 15.6 + 43.2 GB, which is unchanged.
- **Wiring:** decodes above 1024² tile automatically (2×2 at 2048²); ≤ 1024² stays untiled and
  parity-locked. Text-to-image now admits every model-card size (cap = largest by area,
  2400×1792); edits stay ≤ 1024².

Remaining for AB-T-0154: in-app validation + phys re-baseline; step-time headroom vs torch.
