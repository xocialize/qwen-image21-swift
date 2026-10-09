// The half of the engine package both Qwen-Image-2.1 tiers share: loading the resident DiT + VAE,
// and turning a capability request into a render. `QwenImage21Package` samples on the shifted
// linspace schedule; `QwenImage21TurboPackage` hands over the checkpoint's fixed grid.

import Foundation
import MLX
import MLXToolKit
import QwenImage21

enum QwenImage21Serving {
    /// Per-request generation defaults, resolved from a package's C9 configuration.
    struct Defaults {
        var steps: Int
        var trueCFGScale: Float
        var outputResolution: Int
        var editOutputResolution: Int
        var useKVCache: Bool
    }

    /// DiT (bf16) + VAE (fp32) stay resident; the encoder loads per request through the provider
    /// and is evicted before the denoise peak unless `keepEncoderResident`. `vaeSnapshot` may be a
    /// different root from `snapshot` — Turbo's own `vae/` is the base VAE cast to bf16, so the
    /// Turbo package points this at the base snapshot's fp32 one.
    static func load(snapshot: URL, vaeSnapshot: URL, textEncoder: URL, keepEncoderResident: Bool) throws -> QwenImage21Generator {
        let transformer = try QwenImage21Weights.loadTransformer(
            directory: snapshot.appendingPathComponent("transformer"), dtype: .bfloat16)
        let vae = try QwenImage21Weights.loadVAE(directory: vaeSnapshot.appendingPathComponent("vae"), dtype: .float32)
        let generator = QwenImage21Generator(
            encoderProvider: { try await QwenImage21PromptEncoder.load(qwenDir: textEncoder, dtype: .bfloat16) },
            transformer: transformer, vae: vae, keepEncoderResident: keepEncoderResident)
        generator.warmup()
        return generator
    }

    /// Decode the request, enforce the measured envelope, generate, encode PNG. The caller has
    /// already taken the CAN-1 entry checkpoint; the core checkpoints once per denoise step and at
    /// the post-encode / pre-decode seams. `sigmas` non-nil = a fixed grid (the request's `steps`
    /// is then ignored, as in the reference pipeline).
    static func run(
        _ request: any CapabilityRequest, generator: QwenImage21Generator, defaults: Defaults, sigmas: [Float]?,
        isolation: isolated (any Actor)? = #isolation
    ) async throws -> any CapabilityResponse {
        let prompt: String
        let negative: String?
        let images: [QwenImage21RGBAImage]
        let width: Int?, height: Int?, steps: Int, seed: UInt64
        let cfg: Float
        switch request.capability {
        case .textToImage:
            guard let t2i = request as? T2IRequest else { throw PackageError.unsupportedCapability(request.capability) }
            prompt = t2i.prompt; negative = t2i.negativePrompt; images = []
            width = t2i.width; height = t2i.height
            steps = t2i.steps ?? defaults.steps
            seed = t2i.seed ?? 0
            cfg = t2i.guidanceScale.map(Float.init) ?? defaults.trueCFGScale
        case .imageEdit:
            guard let edit = request as? IEditRequest else { throw PackageError.unsupportedCapability(request.capability) }
            guard !edit.images.isEmpty else { throw QwenImage21PackageError.imageDecode }
            prompt = edit.prompt; negative = edit.negativePrompt
            images = try edit.images.map { img in
                do { return try QwenImage21PNG.read(data: img.data) } catch { throw QwenImage21PackageError.imageDecode }
            }
            width = edit.width; height = edit.height
            steps = edit.steps ?? defaults.steps
            seed = edit.seed ?? 0
            cfg = edit.guidanceScale.map(Float.init) ?? defaults.trueCFGScale
        default:
            throw PackageError.unsupportedCapability(request.capability)
        }
        try Task.checkCancellation()

        let outputResolution = images.isEmpty ? defaults.outputResolution : defaults.editOutputResolution
        let (tw, th) = QwenImage21Latents.targetSize(
            imageSizes: images.map { ($0.width, $0.height) }, width: width, height: height,
            outputResolution: outputResolution)
        if let why = QwenImage21Envelope.violation(
            targetWidth: tw, targetHeight: th, referenceCount: images.count, outputResolution: outputResolution)
        {
            throw QwenImage21PackageError.outsideMeasuredEnvelope(why)
        }

        let result = try await generator.generate(
            prompt: prompt, images: images, negativePrompt: negative, trueCFGScale: cfg,
            width: width, height: height, outputResolution: outputResolution,
            steps: steps, sigmas: sigmas, seed: seed, useKVCache: defaults.useKVCache,
            progress: { step, total in RunProgress.report(.denoise, step: step, totalSteps: total) })

        try Task.checkCancellation()
        let png: Data
        do { png = try QwenImage21PNG.pngData(result.image) } catch { throw QwenImage21PackageError.pngEncode }
        let artifact = Image(format: .png, data: png, width: result.image.width, height: result.image.height)
        return request.capability == .textToImage ? T2IResponse(image: artifact) : IEditResponse(image: artifact)
    }
}
