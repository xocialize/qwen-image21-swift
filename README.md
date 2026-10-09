# qwen-image21-swift

Swift/MLX port of [Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) — a unified
text-to-image + multi-reference editing model: 7B single-stream block-causal DiT with a prefix KV
cache, 64-ch 16× **RGBA** VAE (native transparency), Qwen3-VL-8B conditioner.

> **Licence: the Qwen-Image-2.1 weights are under the Qwen RESEARCH License Agreement — research /
> evaluation only, no commercial use** (§1(i), §2(a); commercial use needs a separate licence from
> Qwen, §2(b)). The Swift port code in this repository is MIT (`LICENSE`); that does not change the
> weights' terms, and anything you generate with them is bound by the research licence. "Built with
> Qwen".
>
> **Status: research / evaluation tier — not a product asset.** Both `MLXQwenImage21` packages
> (base and Turbo) declare `weightLicense = LicenseRef-Qwen-Research` package-locally, and that id
> is never added to any permissive allowlist. A `.permissiveOnly` + `.blocking` engine refuses
> them; an `.advisory` engine registers them with a licence advisory. Consumers must not route to
> them by default. Use them only when explicitly named. See `PORTING-SPEC.md` §1 and fleet
> decision AB-D-0085.

Reference: diffusers main `QwenImage21Pipeline` (PR #14804). Spec, architecture delta vs the
2511 port, reuse map and parity plan: `PORTING-SPEC.md`. Goldens + oracle:
`../qwen-image21-oracle`.

> Edit + seed: an edit that reuses the text-to-image seed and size of the image being edited
> re-draws the generation noise and replays the generation instead of following the instruction
> (diagnosed on https://github.com/huggingface/diffusers/issues/14824). The edit path offsets its
> seed (`QwenImage21Latents.noiseSeed`), so edits default to the reference's 1024² again
> (`PORTING-SPEC.md` §12).

```
swift build
.build/debug/QwenImage21Gate --sched  ../qwen-image21-oracle/goldens
.build/debug/QwenImage21Gate --vae    ../../weights/Qwen-Image-2.1 ../qwen-image21-oracle/goldens
.build/debug/QwenImage21Gate --encoder ../../weights/Qwen3-VL-8B-Instruct ../qwen-image21-oracle/goldens
.build/debug/QwenImage21Gate --dit    ../../weights/Qwen-Image-2.1 ../qwen-image21-oracle/goldens
.build/debug/QwenImage21Gate --generate ../../weights/Qwen-Image-2.1 ../../weights/Qwen3-VL-8B-Instruct \
    --prompt "a red fox in fresh snow" --size 1024 --steps 40 --out fox.png
```

Weights: `transformer/` + `vae/` + `processor/` + `scheduler/` from the 2.1 repo
(`weights/Qwen-Image-2.1`); the text encoder is byte-identical to `Qwen/Qwen3-VL-8B-Instruct`
and is loaded from that snapshot. On a fresh machine the engine materialises the 2.1 part from
[`xocialize/Qwen-Image-2.1`](https://huggingface.co/xocialize/Qwen-Image-2.1). That repo is an
unmodified, hash-verified mirror of `Qwen/Qwen-Image-2.1` at `b3179ad`, shipped with the Qwen
Research LICENSE and NOTICE and without `text_encoder/`.

## Turbo tier: Qwen-Image-2.1-Turbo (2026-10-09)

[Qwen/Qwen-Image-2.1-Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo) is the same model
distilled to **8 fixed steps** for text-to-image and editing, under the same research licence.
What it adds is small (AB-R-0437): the DiT re-weighted (identical config and tensor set, 14.2 GB)
and the schedule it was distilled on, shipped as `sample_sigmas` in `model_index.json` (diffusers
PR #14950) with the scheduler's dynamic shift and terminal stretch switched off, so the grid
`[1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568]` is used verbatim
plus a trailing 0, whatever the token count. Its `vae/` is the base VAE cast to bf16 and its
`text_encoder/` is byte-identical to Qwen3-VL-8B-Instruct, so neither is downloaded.

- Core: `QwenImage21Generator.generate(sigmas:)` takes a fixed grid (`steps` is then ignored);
  `QwenImage21Scheduler.loadSampleSigmas(snapshot:)` reads it from a snapshot and refuses a
  scheduler config that would still shift it.
- Package: `QwenImage21TurboPackage` (`MLXQwenImage21`, PackageID `qwen-image-2.1-turbo`,
  textToImage + imageEdit) — three roots: the Turbo snapshot (`transformer/`, `model_index.json`,
  `scheduler/`; mirror [`xocialize/Qwen-Image-2.1-Turbo`](https://huggingface.co/xocialize/Qwen-Image-2.1-Turbo)
  at upstream `d65dbc9`), the base snapshot for the fp32 `vae/`, and Qwen3-VL-8B-Instruct. Same
  footprint envelope as the base package (same shapes). A request's `steps` is ignored; guidance
  stays at 1.
- Gate: `--turbo` samples on the snapshot's grid; `--vae-root` points at the base VAE.

```
.build/release/QwenImage21Gate --sched ../qwen-image21-oracle/goldens            # incl. the fixed-grid entries
.build/release/QwenImage21Gate --dit   ../../weights/Qwen-Image-2.1-Turbo ../qwen-image21-oracle/goldens/turbo
.build/release/QwenImage21Gate --generate ../../weights/Qwen-Image-2.1-Turbo ../../weights/Qwen3-VL-8B-Instruct \
    --vae-root ../../weights/Qwen-Image-2.1 --turbo --prompt "a red fox in fresh snow" --size 1024 --out fox8.png
```

Parity on the Turbo weights (fp32 CPU, 2026-10-09): all three DiT layouts green, block/step
cos ≥ 0.9999998, schedule exact. Evaluation memo and the plan: `TURBO-EVAL.md`; task AB-T-0212.
Oracle goldens: `../qwen-image21-oracle/goldens/turbo/` (`make_goldens.py dit --turbo`,
`e2e --turbo`, on the `.venv314` env).

## GPU numerics: the VAE's 3×3 convs (2026-09-24)

mlx's Metal `conv2d` takes a Winograd F(6×6,3×3) path when the conv is 3×3, stride 1, dilation 1,
groups 1, C % 32 == 0, O % 32 == 0, C + O ≥ 256 and N·H·W ≥ 4096. On M5 that path loses about
6.4e-3 relL2 per conv in fp32, because its inner GEMM runs TF32.

This VAE hits it in 33 decoder convs at 1024² (conv_in 64→1152, the 1152/576/288-channel resnets
and upsamplers) and 21 convs per edit-image encode. Coverage from the existing gates was thin:
`--vae` runs on the CPU lane, and `--vae-tile` only compares GPU against GPU.

Every stride-1 3×3 conv is now a `WinogradFreeConv2d`. **Default `.conv3d` for both encoder and
decoder** (`encoderConvRoute` / `decoderConvRoute`, type `QwenImage21VAEConvRoute`).

Measurements, against the torch fp32 golden (320² / 288×384 real images) and the CPU lane (1024²
DIV2K photo, alpha 1):

| | Raw conv2d (Winograd) | conv3d route |
|---|---|---|
| Golden `vae_img_b` decode | 1.2e-2 · **max 2.0** (47 dB) | 4.5e-5 · max 2.8e-3 (96 dB) |
| Golden `vae_img_a` decode | 2.7e-4 · 79.9 dB | 8.7e-6 · 109.7 dB |
| Golden encode (a / b) | 7.1e-4 / 1.7e-3 | 3.0e-4 / 3.0e-4 |
| 1024² encode vs CPU lane | 5.6e-3 · max 1.13 | 4.9e-4 |
| 1024² decode vs CPU lane | 1.1e-3 · 69 dB | 2.1e-5 · 103 dB |
| GPU halo-tiled (2×2, halo 12) vs untiled, 1024² | 5.0e-4 · max 3.5e-2 | **exactly 0** |
| 1024² time, decode / encode | 1586 ms / 413 ms | +1253 ms / +142 ms |

**Why the decoder routes here, when the FLUX-class RGB decoders don't:**

- The raw path's worst errors land in the alpha channel at transparent image edges, and they depend
  on context. On `vae_img_b` the top-edge alpha is off by 0.32 in a fresh process, but by 1.996 (a
  full −1 → +1 flip) once any CPU-lane forward has run in the same process.
- Neither isolated conv probes nor a stale-buffer poison test reproduce this; `testRawDecodeBisect`
  is the repro.
- The route is stable in every context.
- It also makes the GPU halo-tiled decode bit-identical to the untiled one. On the CPU stream this
  was already exact.

This corrects the attribution in AB-R-0310. The 64–67 dB GPU tiled-vs-untiled gap was this
Winograd window: its 6×6 tiles realign at tile edges and its GEMM runs TF32. It was not generic
"GPU accumulation order".

Remaining encode residual: the ~3e-4 left on the encoder is TF32 in the mid-block attention, the
same effect measured in the other fleet VAEs.

Controls and tests:

- Environment override: `QWEN21_VAE_CONV_ROUTE=winograd|conv3d|fp32Winograd`.
- `swift test --filter WinogradProbeTests` is weight-free.
- `QWEN21_PARITY=1 QWEN21_ROOT=<weights/Qwen-Image-2.1> swift test -c release -Xswiftc
  -enable-testing --filter VAEGPULaneTests` covers the goldens, the 1024² photo, tiling and timing.
