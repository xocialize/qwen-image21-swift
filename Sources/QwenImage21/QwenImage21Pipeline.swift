// Qwen-Image-2.1 generation pipeline — Swift mirror of diffusers main `pipeline_qwenimage21.py`
// (`QwenImage21Pipeline.__call__`): one path for text-to-image and image-conditioned editing.
//
//   1. condition images: one LANCZOS resize to calculate_dimensions(output_resolution², aspect)
//      (/32) feeds BOTH the VL encoder (composited over white) and the VAE (RGBA, [-1, 1]);
//      the output size defaults to the LAST image's aspect at output_resolution² area;
//   2. encode_prompt -> pre-norm embeds + image_pad_mask (+ negative branch under true CFG);
//   3. cond latents = VAE mode, normalised, packed (plain spatial flatten — patch_size 1),
//      concatenated on the sequence axis BEFORE the target noise;
//   4. FlowMatchEuler: sigmas linspace(1, 1/N, N) -> exponential dynamic shift with
//      mu = calculate_shift(target tokens; 256/8192, 0.5/0.9) -> stretch to shift_terminal 0.02
//      — or a FIXED grid (`sigmas:`; Qwen-Image-2.1-Turbo's `sample_sigmas`, diffusers PR #14950)
//      used verbatim + trailing 0, the shift pipeline off;
//   5. denoise with the prefix KV cache (step 0 "extract", then "cached"); Euler update;
//      guidance is OFF by default (true_cfg_scale 1.0 — 2.1 is meant to run without it);
//   6. unpack -> de-normalise -> VAE decode -> RGBA8.

import Foundation
import MLX
import MLXRandom

public enum QwenImage21Scheduler {
    public static let baseImageSeqLen = 256
    public static let maxImageSeqLen = 8192
    public static let baseShift: Float = 0.5
    public static let maxShift: Float = 0.9
    public static let shiftTerminal: Float = 0.02

    /// diffusers `calculate_shift` with the 2.1 scheduler config.
    public static func calculateShift(imageSeqLen: Int) -> Float {
        let m = (maxShift - baseShift) / Float(maxImageSeqLen - baseImageSeqLen)
        let b = baseShift - m * Float(baseImageSeqLen)
        return Float(imageSeqLen) * m + b
    }

    /// `FlowMatchEulerDiscreteScheduler.set_timesteps(sigmas=linspace(1, 1/N, N), mu=mu)`:
    /// exponential time shift, `stretch_shift_to_terminal`, trailing 0. Computed in Double and
    /// rounded to Float like the reference's numpy(float64) -> torch.float32 path.
    public static func sigmas(steps: Int, mu: Float) -> [Float] {
        precondition(steps > 0)
        let n = steps
        var s: [Double] = (0..<n).map { i in
            n == 1 ? 1.0 : 1.0 - Double(i) * (1.0 - 1.0 / Double(n)) / Double(n - 1)
        }
        let eMu = exp(Double(mu))
        s = s.map { eMu / (eMu + (1.0 / $0 - 1.0)) }
        // stretch_shift_to_terminal
        let oneMinusLast = 1.0 - s[n - 1]
        let scale = oneMinusLast / (1.0 - Double(shiftTerminal))
        s = s.map { 1.0 - (1.0 - $0) / scale }
        var out = s.map { Float($0) }
        out.append(0)
        return out
    }

    /// A checkpoint-supplied grid used verbatim (diffusers PR #14950 `sample_sigmas` — how
    /// Qwen-Image-2.1-Turbo ships its 8 steps): no `mu`, no shift, no terminal stretch. The
    /// reference scheduler still "processes" an explicit grid, but Turbo's scheduler config
    /// (`use_dynamic_shifting: false`, `shift: 1.0`, `shift_terminal: null`) makes that the
    /// identity, so only the trailing 0 is appended and timesteps = σ·1000. The grid excludes
    /// the terminal sigma; its length is the step count, whatever the token count.
    public static func fixedGrid(_ grid: [Float]) throws -> [Float] {
        guard !grid.isEmpty else { throw QwenImage21Error.invalidInput("sampling grid is empty") }
        guard grid[0] <= 1, grid.last! > 0, zip(grid, grid.dropFirst()).allSatisfy({ $0 > $1 }) else {
            throw QwenImage21Error.invalidInput("sampling grid must be strictly decreasing within (0, 1]: \(grid)")
        }
        return grid + [0]
    }

    /// The snapshot's `model_index.json` `sample_sigmas`, or nil when the checkpoint has none
    /// (the base 2.1). When present, the snapshot's scheduler config must be the identity
    /// (shift 1, no dynamic shifting, no terminal) — the only processing `fixedGrid` reproduces;
    /// anything else is refused rather than silently mis-sampled.
    public static func loadSampleSigmas(snapshot: URL) throws -> [Float]? {
        let indexURL = snapshot.appendingPathComponent("model_index.json")
        guard FileManager.default.fileExists(atPath: indexURL.path) else { return nil }
        let index = try JSONSerialization.jsonObject(with: Data(contentsOf: indexURL)) as? [String: Any] ?? [:]
        guard let raw = index["sample_sigmas"] as? [Double] else { return nil }
        let schedURL = snapshot.appendingPathComponent("scheduler/scheduler_config.json")
        if let data = try? Data(contentsOf: schedURL),
           let sched = try JSONSerialization.jsonObject(with: data) as? [String: Any] {
            let dynamic = sched["use_dynamic_shifting"] as? Bool ?? false
            let shift = sched["shift"] as? Double ?? 1
            let terminal = sched["shift_terminal"] as? Double
            guard !dynamic, shift == 1, terminal == nil else {
                throw QwenImage21Error.loading(
                    "model_index.json has sample_sigmas but scheduler_config.json would still shift them "
                        + "(use_dynamic_shifting \(dynamic), shift \(shift), shift_terminal \(terminal.map { "\($0)" } ?? "null")) "
                        + "— a fixed grid is only supported with the identity scheduler")
            }
        }
        return raw.map { Float($0) }
    }
}

public enum QwenImage21Latents {
    /// (B, 64, 1, h, w) -> (B, h·w, 64) — 2.1 consumes latents unpatched.
    public static func pack(_ x5: MLXArray) -> MLXArray {
        let x = x5.squeezed(axis: 2)
        let (b, c, h, w) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        return x.reshaped(b, c, h * w).transposed(0, 2, 1)
    }

    /// (B, h·w, 64) -> (B, 64, 1, h, w) with h = pixelHeight / 16.
    public static func unpack(_ x: MLXArray, pixelHeight: Int, pixelWidth: Int) -> MLXArray {
        let b = x.dim(0)
        let h = 2 * (pixelHeight / 32)
        let w = 2 * (pixelWidth / 32)
        return x.transposed(0, 2, 1).reshaped(b, x.dim(2), 1, h, w)
    }

    /// Seed actually handed to the RNG, domain-separated by path.
    ///
    /// **Noise replay (diffusers#14824, confirmed with the Qwen team 2026-09-21).** An edit whose
    /// initial noise happens to be the draw that GENERATED the reference image re-runs that
    /// generation instead of following the instruction: the result is a sharpened, saturated
    /// near-copy with the instruction ignored (gradient energy ~2.9x the input, PSNR ~13 dB vs
    /// ~21 dB for a real edit). It is not resolution-dependent and not RNG-dependent — it needs
    /// only the same noise, which a generate-then-edit flow produces whenever both steps share a
    /// seed and a shape. Measured in this package: T2I 1024²/seed 42 then editing that output at
    /// 1024²/seed 42 replays (13.73 dB, 2.97x); the same edit at seed 4242 is correct (21.09 dB,
    /// 1.05x).
    ///
    /// So the edit path offsets the seed by a fixed constant. Deterministic — a given
    /// (seed, size, images) still reproduces exactly — but it can no longer collide with the
    /// text-to-image draw at the same seed and size. Injected `latents` bypass this entirely, so
    /// parity fixtures and a deliberate replay repro are unaffected.
    public static func noiseSeed(_ seed: UInt64, isEdit: Bool) -> UInt64 {
        isEdit ? seed &+ 0x9E37_79B9_7F4A_7C15 : seed
    }

    /// The output size the pipeline will produce — the reference `__call__`'s rule, in one place so
    /// the generator and any caller-side envelope guard cannot drift: explicit width/height win;
    /// otherwise the LAST condition image's aspect at `outputResolution²` area; otherwise
    /// `outputResolution` square; then floor to a multiple of 32.
    public static func targetSize(imageSizes: [(width: Int, height: Int)], width: Int?, height: Int?,
                                  outputResolution: Int) -> (width: Int, height: Int) {
        var w = width ?? outputResolution
        var h = height ?? outputResolution
        if let last = imageSizes.last, width == nil || height == nil {
            let (cw, ch) = calculateDimensions(
                targetArea: outputResolution * outputResolution, ratio: Double(last.width) / Double(last.height))
            w = width ?? cw
            h = height ?? ch
        }
        return (w / 32 * 32, h / 32 * 32)
    }

    /// diffusers `calculate_dimensions`: sqrt-area, ratio preserved, /32 with Python round().
    public static func calculateDimensions(targetArea: Int, ratio: Double) -> (width: Int, height: Int) {
        let width = (Double(targetArea) * ratio).squareRoot()
        let height = width / ratio
        return (Int((width / 32).rounded(.toNearestOrEven)) * 32, Int((height / 32).rounded(.toNearestOrEven)) * 32)
    }
}

/// End-to-end generator (batch 1). Holds the DiT + VAE; the ~17 GB Qwen3-VL encoder is loaded
/// per request through `encoderProvider` and dropped before the denoise peak unless
/// `keepEncoderResident` (the qwen-image-edit-swift residency contract).
public final class QwenImage21Generator {
    public let encoderProvider: () async throws -> QwenImage21PromptEncoder
    public let transformer: QwenImage21Transformer2DModel
    public let vae: AutoencoderKLQwenImage21
    public let keepEncoderResident: Bool
    private var residentEncoder: QwenImage21PromptEncoder?

    public init(encoderProvider: @escaping () async throws -> QwenImage21PromptEncoder,
                transformer: QwenImage21Transformer2DModel, vae: AutoencoderKLQwenImage21,
                keepEncoderResident: Bool = false) {
        self.encoderProvider = encoderProvider
        self.transformer = transformer
        self.vae = vae
        self.keepEncoderResident = keepEncoderResident
    }

    private func loadEncoder(isolation: isolated (any Actor)? = #isolation) async throws -> QwenImage21PromptEncoder {
        if keepEncoderResident, let residentEncoder { return residentEncoder }
        let e = try await encoderProvider()
        if keepEncoderResident { residentEncoder = e }
        return e
    }

    private func evictEncoder(_ e: inout QwenImage21PromptEncoder?) {
        guard !keepEncoderResident else { return }
        e = nil
        Memory.clearCache()
    }

    /// Build the first-forward graphs at load time (a 256² T2I DiT step + VAE decode) so the
    /// first request does not pay kernel/graph build. The encoder is deliberately not warmed —
    /// it loads per request.
    public func warmup() {
        let dtype: DType = .bfloat16
        let side = 256
        let lh = side / 16
        let txt = MLXArray.zeros([1, 16, 4096]).asType(dtype)
        let slots = [Bool](repeating: false, count: 16) + [Bool](repeating: true, count: lh * lh / 4)
        guard let layout = try? transformer.buildLayout(imgMask: slots, imgShapes: [(1, lh, lh)]) else { return }
        let latents = MLXArray.zeros([1, lh * lh, 64]).asType(dtype)
        let v = transformer(hiddenStates: latents, encoderHiddenStates: txt, timestep: MLXArray([Float(1)]), layout: layout, mode: .none)
        let unpacked = QwenImage21Latents.unpack(v[0..., (v.dim(1) - lh * lh)...].asType(vae.weightDtype), pixelHeight: side, pixelWidth: side)
        let decoded = vae.decode(AutoencoderKLQwenImage21.deNormalize(unpacked))
        eval(decoded)
        Memory.clearCache()
    }

    public struct Result {
        public let image: QwenImage21RGBAImage
        public let latentsPacked: MLXArray
        public let sigmas: [Float]
    }

    /// - images: condition images (any size); resized here like the reference.
    /// - width/height: explicit output size (rounded down to /32); nil -> derived.
    /// - sigmas: a fixed sampling grid (terminal 0 excluded) that replaces the shifted linspace
    ///   schedule — Turbo's `sample_sigmas`. Its length is the step count; `steps` is then ignored.
    /// - latents: injected packed noise [1, h·w, 64] for parity gates (torch RNG ≠ MLX RNG).
    public func generate(
        prompt: String, images: [QwenImage21RGBAImage] = [], negativePrompt: String? = nil,
        trueCFGScale: Float = 1.0, width: Int? = nil, height: Int? = nil, outputResolution: Int = 1024,
        steps: Int = 40, sigmas grid: [Float]? = nil, seed: UInt64 = 0, useKVCache: Bool = true,
        latents injected: MLXArray? = nil,
        progress: ((Int, Int) -> Void)? = nil, isolation: isolated (any Actor)? = #isolation
    ) async throws -> Result {
        // Validate the grid before the encoder is paid for.
        let fixedSigmas = try grid.map { try QwenImage21Scheduler.fixedGrid($0) }
        guard fixedSigmas != nil || steps > 0 else { throw QwenImage21Error.invalidInput("steps must be > 0") }
        let area = outputResolution * outputResolution

        // 1. Condition images: one resize for both consumers; output size from the last image.
        var resized: [QwenImage21RGBAImage] = []
        for img in images {
            let (iw, ih) = QwenImage21Latents.calculateDimensions(targetArea: area, ratio: Double(img.width) / Double(img.height))
            resized.append(QwenImage21PILResize.resizeRGBA(img, outWidth: iw, outHeight: ih))
        }
        let (w, h) = QwenImage21Latents.targetSize(
            imageSizes: images.map { ($0.width, $0.height) }, width: width, height: height,
            outputResolution: outputResolution)
        guard w >= 32, h >= 32 else { throw QwenImage21Error.invalidInput("output size below 32x32") }

        // 2. Prompt encoding (encoder evicted before the DiT peak).
        let doCFG = trueCFGScale > 1 && negativePrompt != nil
        var encoderRef: QwenImage21PromptEncoder? = try await loadEncoder()
        let pos = try encoderRef!.encode(prompt: prompt, images: resized)
        let neg = doCFG ? try encoderRef!.encode(prompt: negativePrompt!, images: resized) : nil
        if let neg { eval(pos.embeds, neg.embeds) } else { eval(pos.embeds) }
        evictEncoder(&encoderRef)
        // Denoise in the DiT's own dtype (bf16 in production; fp32 for precision probes) — the
        // encoder may run at a different precision.
        let dtype = transformer.computeDType
        let posEmbeds = pos.embeds.asType(dtype)
        let negEmbeds = neg?.embeds.asType(dtype)
        try Task.checkCancellation()

        // 3. Condition latents (VAE mode, normalised, packed) + shapes; grid consistency check.
        var condParts: [MLXArray] = []
        var imgShapes: [(Int, Int, Int)] = []
        for (i, img) in resized.enumerated() {
            let pixels = MLXArray(img.vaePixelsCHW(), [1, 4, 1, img.height, img.width]).asType(dtype)
            let lat = vae.encode(pixels)  // (1, 64, 1, h, w)
            let (lh, lw) = (lat.dim(3), lat.dim(4))
            let g = pos.mergedGrids[i]
            guard g.0 * g.1 * g.2 * 4 == lh * lw else {
                throw QwenImage21Error.invalidInput(
                    "image \(i + 1): VL grid \(g.1)x\(g.2) (x4 = \(g.0 * g.1 * g.2 * 4) tokens) does not match the VAE latent grid \(lh)x\(lw); the processor's min/max_pixels re-sized it — raise output_resolution")
            }
            condParts.append(QwenImage21Latents.pack(lat).asType(dtype))
            imgShapes.append((1, lh, lw))
        }
        let condLatents: MLXArray? = condParts.isEmpty ? nil : (condParts.count == 1 ? condParts[0] : concatenated(condParts, axis: 1))

        // 4. Target noise (torch-compatible only via injection).
        let (lh, lw) = (h / 16, w / 16)
        var latents: MLXArray
        if let injected {
            latents = injected.asType(dtype)
        } else {
            let key = MLXRandom.key(QwenImage21Latents.noiseSeed(seed, isEdit: !images.isEmpty))
            latents = QwenImage21Latents.pack(MLXRandom.normal([1, 64, 1, lh, lw], key: key)).asType(dtype)
        }
        let nTarget = lh * lw
        imgShapes.append((1, lh, lw))

        // 5. Layout + schedule.
        let slots = pos.imagePadMask + Array(repeating: true, count: nTarget / 4)
        let layout = try transformer.buildLayout(imgMask: slots, imgShapes: imgShapes)
        let negLayout = try neg.map { try transformer.buildLayout(imgMask: $0.imagePadMask + Array(repeating: true, count: nTarget / 4), imgShapes: imgShapes) }
        let sigmas = try fixedSigmas
            ?? QwenImage21Scheduler.sigmas(steps: steps, mu: QwenImage21Scheduler.calculateShift(imageSeqLen: nTarget))
        let nSteps = sigmas.count - 1
        let cacheOn = useKVCache && transformer.causalCondition
        let posCache = cacheOn ? QwenImage21KVCache(numLayers: transformer.numLayers) : nil
        let negCache = cacheOn && doCFG ? QwenImage21KVCache(numLayers: transformer.numLayers) : nil

        // 6. Denoise.
        for i in 0..<nSteps {
            try Task.checkCancellation()
            let mode: QwenImage21KVCacheMode = cacheOn ? (i == 0 ? .extract : .cached) : .none
            let t = MLXArray([sigmas[i]])
            let hidden = condLatents.map { concatenated([$0, latents], axis: 1) } ?? latents
            var v = transformer(hiddenStates: hidden, encoderHiddenStates: posEmbeds, timestep: t, layout: layout,
                                kvCache: posCache, mode: mode)
            v = v[0..., (v.dim(1) - nTarget)...]
            if let negEmbeds, let negLayout {
                var nv = transformer(hiddenStates: hidden, encoderHiddenStates: negEmbeds, timestep: t, layout: negLayout,
                                     kvCache: negCache, mode: mode)
                nv = nv[0..., (nv.dim(1) - nTarget)...]
                v = nv + trueCFGScale * (v - nv)
            }
            latents = latents + (sigmas[i + 1] - sigmas[i]) * v
            eval(latents)
            if mode == .extract {
                // materialise the prefix K/V slices so the full prefill graph can be released
                eval((posCache?.arrays ?? []) + (negCache?.arrays ?? []))
            }
            progress?(i + 1, nSteps)
        }
        try Task.checkCancellation()

        // 7. Decode -> RGBA8.
        let unpacked = QwenImage21Latents.unpack(latents, pixelHeight: h, pixelWidth: w)
        let z = AutoencoderKLQwenImage21.deNormalize(unpacked.asType(vae.weightDtype))
        // Above 1024² the untiled decode dominates memory (72.9 GB peak at 2048², AB-R-0290), so
        // decode in halo-exact tiles; at or below it, keep the parity-locked untiled path.
        let decoded: MLXArray  // (1,4,1,H,W)
        if w * h > AutoencoderKLQwenImage21.untiledDecodeMaxPixels {
            let (tilesH, tilesW) = AutoencoderKLQwenImage21.suggestedTiles(outputHeight: h, outputWidth: w)
            decoded = vae.decodeTiled(z, tilesH: tilesH, tilesW: tilesW)
        } else {
            decoded = vae.decode(z)
        }
        let img8 = clip((decoded.squeezed(axis: 2) + 1) * 127.5, min: 0, max: 255).round().asType(.uint8)
        let hwc = img8[0].transposed(1, 2, 0)
        eval(hwc)
        let image = QwenImage21RGBAImage(rgba: hwc.asArray(UInt8.self), width: w, height: h)
        return Result(image: image, latentsPacked: latents, sigmas: sigmas)
    }
}
