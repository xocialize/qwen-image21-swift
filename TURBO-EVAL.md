# Qwen-Image-2.1-Turbo — desk evaluation for a fast tier on qwen-image21-swift

Date: 2026-10-09 (Turbo published 2026-10-09, https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo).
Nothing downloaded or run; every fact below is from the Hub (file headers + range-fetched
tensors), diffusers PR #14950, and our own receipts. Recommendation at the end.

## 1. What it is

An **8-step accelerated checkpoint of Qwen-Image-2.1** for both text-to-image and editing,
`QwenImage21Pipeline`-loadable, CFG 1 by default, prefix KV cache on. Qwen gives no benchmark
numbers — the card is a showcase only (portrait, poses, RGBA, typography, UI, single- and
multi-reference edits). Licence: **Qwen Research** (same 7,831-byte text as the base, `license_name:
qwen-research`) — AB-D-0085 applies verbatim: research/eval tier, `LicenseRef-Qwen-Research`
package-local, never allowlisted, named-only.

## 2. What actually changed vs the base we already serve (verified)

| Component | Turbo repo | vs `Qwen/Qwen-Image-2.1` | Consequence for us |
|---|---|---|---|
| `transformer/config.json` | identical (32 layers, 4096, 32×128 heads, mlp 3×, `causal_condition: true`, patch 1) | identical | drop-in weight swap on the existing `QwenImage21Transformer2DModel` |
| DiT weights | same two shards, same byte sizes (9.97 GB + 4.26 GB), 297 tensors | same names + shapes; probed RMSNorm q/k weights bit-identical, the linears are the distilled ones | **14.2 GB download** — the only new weights |
| `vae/` | 238 tensors, **bf16**, 675 MB | base VAE cast to bf16 (sampled tensors differ from base fp32 by relL2 1.6–1.9e-3 = bf16 rounding) | keep loading **our fp32 base VAE** (parity-locked, conv3d route); do not fetch Turbo's |
| `text_encoder/` | single 17.5 GB file, 750 tensors | names identical to `Qwen/Qwen3-VL-8B-Instruct`; 3 probed tensors (final norm, layer-35 norm, deepstack merger bias) bit-identical | reuse the existing Qwen3-VL-8B-Instruct snapshot — **no 17.5 GB download** |
| `processor/` | transformers-5 layout (`processor_config.json` + `tokenizer.json`) | `chat_template.jinja` byte-identical; same image processor (patch 16, merge 2, 65,536–16,777,216 px) | no preprocessing change; our tokenizer already comes from the Instruct snapshot |
| `scheduler/scheduler_config.json` | `use_dynamic_shifting: false`, `shift: 1.0`, `shift_terminal: null` | base: dynamic 0.5/0.9 + terminal 0.02 | the shift pipeline is **switched off** |
| `model_index.json` | `sample_sigmas: [1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568]` | new key (diffusers PR #14950, merged 2026-10-05) | the whole "Turbo" is **these 8 sigmas + the DiT weights** |

### 2.1 The schedule, precisely

diffusers `__call__` (PR #14950): `sigmas = self.config.sample_sigmas` when the caller passes none;
`mu` is still computed but the scheduler ignores it because `use_dynamic_shifting` is false; the
static shift is `1.0·σ/(1+0·σ)` = identity; no terminal stretch; the scheduler appends 0. The
PR's own test pins exactly this grid: effective sigmas `[…, 0.414568, 0.0]`, timesteps `= σ·1000`.

So the Turbo grid is **resolution-independent and front-loaded**: six steps between σ=1.0 and
0.845, then 0.705 → 0.415 → 0. It is not a shortened version of our `QwenImage21Scheduler.sigmas(steps:mu:)`
(linspace → exponential shift by `mu(tokens)` → stretch to 0.02); that function must be bypassed,
not re-parameterised. `num_inference_steps` is ignored by the reference when the grid is present,
and the card says other grids "have not been evaluated" — a Turbo wrapper should run exactly 8.

## 3. What it buys

Same shapes, same resident set, one DiT forward per step (CFG 1) — so the footprint is the base
package's (AB-R-0290: 15.6 GB resident, envelope peaks 33–52 GB) and the win is purely steps:

| Path (Release, M5 Max, AB-R-0259 step times) | Base | Turbo (8 steps) |
|---|---|---|
| T2I 1024² denoise | 40 × 3.2 s ≈ 128 s | 8 × 3.2 s ≈ **26 s** (+ ~10 s encoder load/encode + decode) |
| Edit 1024², 1 ref | 11.5 s prefill + 39 × 3.75 s ≈ 164 s | 11.5 s + 7 × 3.75 s ≈ **38 s** |
| T2I 2048² (native) | 20 × 19.3 s ≈ 395 s | 8 × 19.3 s ≈ **154 s** |

In ML[X] Image Server, 1024² T2I was 73 s cold / 60 s warm at 40 steps (AB-R-0425), so Turbo lands
around **20–25 s warm**, and the card's native sizes (2048² edits, 1680×2512 T2I) become usable
instead of a five-minute wait. Memory does not move.

What it costs, generically for few-step distills (and what Viggle's own card admits for its
distill): diversity, small dense text, complicated multi-reference edits. Qwen published no
numbers, so **the quality A/B is the evaluation**, not an afterthought.

## 4. Where it goes: a sibling package on the same core (the Flash precedent)

`MLXQwenImageFlash` is the fleet's answer for "distilled full checkpoint of a model whose core we
already have": a second wrapper target + PackageID over the shared core, its own `WeightSourcing`,
`defaultSteps` fixed, CFG 1.0 with the note that re-applying guidance double-counts what the
student absorbed. Turbo is the same shape — different 14 GB DiT, so it is a **package, not a
mode** (a mode would have to swap the resident DiT per request). The server already has one queue
owning the resident model, so base and Turbo simply alternate like any two entries.

Plan (effort S–M; ~1 day plus download/golden time):

1. **Core** (`QwenImage21`): `generate(sigmas: [Float]?)` override — when given, skip
   `calculateShift`/`sigmas(steps:mu:)`, use the grid + trailing 0, `steps = grid.count`. Read
   `sample_sigmas` from `model_index.json` in `QwenImage21Weights` (the Flash `readSchedulerShift`
   pattern). Gate CLI: `--sigmas a,b,c` / `--turbo`. ~30 lines.
2. **Wrapper** `MLXQwenImage21Turbo` (PackageID `qwen-image-2.1-turbo`, surfaces textToImage +
   imageEdit, `modes: []`): same manifest shape as `QwenImage21Package`, `weightSources` →
   `transformer/*` + `model_index.json` + `scheduler/*` + LICENSE/NOTICE from a new mirror
   `xocialize/Qwen-Image-2.1-Turbo` (unmodified, hash-verified, no `text_encoder/`, no `vae/` —
   the §3 redistribution terms we already follow for the base mirror); `vae/*` from
   `xocialize/Qwen-Image-2.1`; encoder from `Qwen/Qwen3-VL-8B-Instruct`. `steps` in the request
   is ignored (or refused when ≠ 8) and the summary says so. Factor the shared bits out of
   `QwenImage21Package` rather than copy them.
3. **Oracle**: `.venv`'s interpreter symlink is dangling (Homebrew python 3.12.14 moved) — rebuild
   via `setup_env.sh`. The pinned diffusers (80c7ed2) predates `sample_sigmas` but takes
   call-time `sigmas=`, and the Turbo `scheduler/` loads as-is, so Turbo goldens need no diffusers
   bump: `phase_dit` with the Turbo transformer at 1024² (step 0 + cached step), `phase_e2e` at
   8 explicit sigmas, T2I 1024² seed 42 + the 1-ref `photo_dog` scarf edit.
4. **Gates**: `--dit` on Turbo weights (same code — this verifies the download and the
   `sample_sigmas` read, cos ≥ 0.9999), `--sched` on the fixed grid, bf16 e2e side-by-side with the
   MPS reference, `--membench` once for the record (expected to reproduce AB-R-0290).
5. **Quality A/B, base-40 vs Turbo-8**, fixed seeds, sizes 1024² and 2048²: fox / neon-sign
   (small text) / RGBA-transparency prompts, the dog-scarf 1-ref edit, a 2-ref composition,
   plus a 4-seed diversity spread per prompt. Score with `mlx-siglip2-iqa-swift` + eyes; file the
   receipt. This is what decides whether Turbo becomes the server's default *when 2.1 is named*,
   or a separate `qwen-image-2.1-turbo` entry.
6. **Server**: tag `qwen-image21-swift` 0.2.0, bump the pin from 0.1.1, add a `qwenImage21Turbo`
   catalog entry (`ModelCatalog.swift:368` pattern, `GenerationDefaults(steps: 8, guidance: 1.0)`,
   restricted, `.wip`, named-only) + the SelectionTests case.

Disk: 509 GiB free on the volume; the download is 14.2 GB.

## 5. The alternative worth knowing about: Viggle's turbo LoRA

`Viggle/Qwen-Image-2.1-viggle-turbo` (2026-09-23 → v0.3 2026-09-29; 358 K downloads, 690 likes
vs the official Turbo's 201 likes on day one) is a **rank-256 LoRA (1.3 GB)** on the base DiT, 6
steps on `[1.0, 0.9375, 0.875, 0.75, 0.5, 0.25]`, `shift_terminal: null`, no CFG, T2I + 1–3-ref
edits, plus a 9-step hybrid (7 LoRA steps, base finishes). Same Qwen Research licence (adapter
of the base). Being a LoRA, it would be a **mode on the existing package** (the 2511
`MLXQwenImageEditTurbo` pattern: runtime apply on the resident DiT, no second 14 GB), and the
`generate(sigmas:)` seam in step 1 serves it unchanged. Honest trade: Qwen's own distill vs a
community one with a quality write-up and a hybrid tail. Not proposed now; listed so the core
seam is designed generically.

## 6. Recommendation

**Yes — port it as `MLXQwenImage21Turbo`**, a sibling package on the existing core, research
tier like the base. It is the cheapest fast tier in the fleet (one weight swap + eight fixed
sigmas on a parity-locked core, no new encoder/VAE/processor work), it makes the model's native
2048² sizes practical, and it keeps the base package untouched for parity. Gate it on the §4.5
A/B before the server treats it as more than a named option. Licence posture unchanged — it is
still not a product asset (AB-D-0085); revisit if Qwen grants a commercial licence for 2.x.

## 7. Outcome (2026-10-09, same day)

Ported as `QwenImage21TurboPackage` in the `MLXQwenImage21` module (v0.2.0), the way §4 laid out:
`generate(sigmas:)` seam, three-root configuration, mirror
[`xocialize/Qwen-Image-2.1-Turbo`](https://huggingface.co/xocialize/Qwen-Image-2.1-Turbo) (DiT +
configs only, hash-verified). In ML[X] Image Server, naming **`qwen-image-2.1` now serves the Turbo
checkpoint** (operator decision after the A/B below); the 40-step original is `qwen-image-2.1-base`.

- Parity on the Turbo weights: `--sched` exact, `--dit` green on all three layouts, cos ≥
  0.9999998 (goldens from the Turbo DiT on diffusers main 1d5d056).
- Timing, Release, uncontended (AB-R-0441): 1024² T2I **1.33 s/step, 13.1 s per generate**;
  1-ref edit 16.9 s; 2048² 71 s. Membench inside the base envelope, so the manifest carries the
  base's measured split. (The §3 estimate assumed the 2026-09-20 step rate; the port had already
  sped up to ~1.4 s/step, so base-40 is ~60 s today and the ratio, not the absolute, held.)
- A/B base-40 vs Turbo-8, 10 cases (AB-R-0443): 4.3–4.9× at 1024², 4.5× at 2048²; SigLIP2
  NR-IQA within ±0.02 on every pair, split 5/5; text letter-perfect on both arms (neon sign,
  poster title + subtitle); 1-ref and 2-ref edits identity-preserving; four seeds give four
  distinct foxes. One adherence miss: the poster subtitle rendered at the top, not the prompted
  bottom. Verdict: serve Turbo as the fast tier of the named model; keep the base for
  layout-critical typography and as the parity reference.
- Oracle note for the next person: `e2e_photo_dog.png` is the base's own seed-42 render, so an
  edit of it at seed 42 replays its noise in the reference (diffusers#14824) — use another seed.
