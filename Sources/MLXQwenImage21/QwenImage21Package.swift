// MLXEngine package over the QwenImage21 core — `textToImage` + `imageEdit` on ONE model
// (Qwen-Image-2.1 is a unified generator/editor; the surfaces dispatch inside `run(_:)`).
//
// ⚠️ LICENCE: Qwen RESEARCH License — research / evaluation only, no commercial use. Declared
// package-locally as `LicenseRef-Qwen-Research` and deliberately NOT allowlisted, so the default
// `.permissiveOnly` policy refuses (`.blocking`) or flags (`.advisory`) it. This package is a
// research tier, never a shipping default (AB-D-0085).
//
// Core facts (PORTING-SPEC.md): 7B single-stream block-causal DiT with a prefix KV cache, 64-ch
// 16x RGBA VAE, Qwen3-VL-8B conditioner (byte-identical to Qwen/Qwen3-VL-8B-Instruct, loaded from
// that snapshot), FlowMatchEuler 256/8192 · 0.5/0.9 · terminal 0.02, no guidance by default,
// 40 steps, outputs RGBA PNG (native transparency).

import Foundation
import MLX
import MLXToolKit
import QwenImage21

extension SPDXLicense {
    /// Qwen RESEARCH LICENSE AGREEMENT (release 2026-09-20): §1(i) non-commercial = research or
    /// evaluation only; §2(b) commercial use requires a separate licence from Tongyi. Non-SPDX,
    /// `LicenseRef-` convention. Intentionally absent from `permissiveAllowlist`.
    public static let qwenResearch: SPDXLicense = "LicenseRef-Qwen-Research"
}

/// Init-time configuration (C9): the two snapshot roots and generation defaults.
public struct QwenImage21Configuration: PackageConfiguration, ModelStorable, QuantConfigured, WeightSourcing {
    /// Qwen-Image-2.1 root (`transformer/`, `vae/`, `processor/`, `scheduler/`). Empty = store.
    public var snapshotPath: String
    /// Qwen/Qwen3-VL-8B-Instruct root (weights + tokenizer). Empty = store.
    public var textEncoderPath: String
    public var defaultSteps: Int
    /// 1.0 — Qwen-Image-2.1 is meant to be sampled without guidance; > 1 with a negative prompt
    /// runs plain true CFG (two DiT forwards per step).
    public var defaultTrueCFGScale: Float
    /// `output_resolution` for text-to-image: side length whose square is the output area. The
    /// model card recommends 2048; 1024 keeps the VAE decode peak at ~31 GB (2048² decodes at 62 GB
    /// until tiled decode lands, AB-T-0021).
    public var defaultOutputResolution: Int
    /// `output_resolution` for EDITS (the area condition images are resized to and the output
    /// follows). Back to the reference's 1024 as of 2026-09-21: the 768 cap was a defensive
    /// response to what turned out to be NOISE REPLAY, not a resolution limit — see
    /// `QwenImage21Latents.noiseSeed`, which removes the cause. 1024² edits are correct once the
    /// edit noise cannot collide with the generation noise.
    public var defaultEditOutputResolution: Int
    /// Keep the ~17 GB Qwen3-VL encoder resident between requests (big-RAM tiers).
    public var keepEncoderResident: Bool
    /// Prefix KV cache across denoise steps (the reference default).
    public var useKVCache: Bool
    public var modelsRootDirectory: URL?

    /// bf16 DiT + fp32 VAE — the only tier for now.
    public var quant: Quant { .bf16 }

    /// Fleet durability policy: the DiT / VAE / pipeline config materialise from a namespace we
    /// control — an unmodified mirror of `upstreamRepo` at `upstreamRevision` (every file
    /// hash-verified; `text_encoder/` omitted, see below), redistributed under Qwen Research
    /// licence §3 with its LICENSE + NOTICE.
    public static let repo = "xocialize/Qwen-Image-2.1"
    public static let upstreamRepo = "Qwen/Qwen-Image-2.1"
    public static let upstreamRevision = "b3179ad355be050328e483a9dfdd9e60cd62adfa"
    /// The 2.1 `text_encoder/` is byte-identical to this repo (750/750 tensors); we materialise
    /// the stock Apache-2.0 snapshot instead of a second copy (the mirror leaves it out).
    public static let textEncoderRepo = "Qwen/Qwen3-VL-8B-Instruct"

    public init(
        snapshotPath: String = "",
        textEncoderPath: String = "",
        defaultSteps: Int = 40,
        defaultTrueCFGScale: Float = 1.0,
        defaultOutputResolution: Int = 1024,
        defaultEditOutputResolution: Int = 1024,
        keepEncoderResident: Bool = false,
        useKVCache: Bool = true,
        modelsRootDirectory: URL? = nil
    ) {
        self.snapshotPath = snapshotPath
        self.textEncoderPath = textEncoderPath
        self.defaultSteps = defaultSteps
        self.defaultTrueCFGScale = defaultTrueCFGScale
        self.defaultOutputResolution = defaultOutputResolution
        self.defaultEditOutputResolution = defaultEditOutputResolution
        self.keepEncoderResident = keepEncoderResident
        self.useKVCache = useKVCache
        self.modelsRootDirectory = modelsRootDirectory
    }

    /// Fresh-machine sources (MAT), split by role. The DiT/VAE/processor/scheduler come from the
    /// 2.1 repo; the conditioner from the stock Qwen3-VL-8B-Instruct repo.
    public var weightSources: [WeightSource] {
        [
            WeightSource(role: "transformer", repo: Self.repo, revision: "main", matching: ["transformer/*"]),
            WeightSource(role: "vae", repo: Self.repo, revision: "main", matching: ["vae/*"]),
            WeightSource(role: "pipeline-config", repo: Self.repo, revision: "main",
                         matching: ["model_index.json", "scheduler/*", "processor/*", "LICENSE", "NOTICE"]),
            WeightSource(role: "text-encoder", repo: Self.textEncoderRepo, revision: "main",
                         matching: ["*.safetensors", "*.json", "merges.txt"]),
        ]
    }

    static func hasTransformer(_ path: String) -> Bool {
        !path.isEmpty && FileManager.default.fileExists(atPath: URL(fileURLWithPath: path).appendingPathComponent("transformer").path)
    }

    static func hasTextEncoder(_ path: String) -> Bool {
        !path.isEmpty && FileManager.default.fileExists(atPath: URL(fileURLWithPath: path).appendingPathComponent("config.json").path)
    }

    /// Explicit paths satisfy their roles; everything else resolves through the store layout.
    public func missingWeightSources(storeRoot: URL?) -> [WeightSource] {
        var missing = defaultMissingWeightSources(storeRoot: storeRoot)
        if Self.hasTransformer(snapshotPath) { missing.removeAll { $0.repo == Self.repo } }
        if Self.hasTextEncoder(textEncoderPath) { missing.removeAll { $0.repo == Self.textEncoderRepo } }
        return missing
    }

    static func resolve(explicit: String, repo: String, storeRoot: URL?, marker: String) -> URL? {
        if !explicit.isEmpty { return URL(fileURLWithPath: explicit) }
        let store = ModelStore(root: storeRoot)
        let fm = FileManager.default
        if let flat = store.directory(for: repo), fm.fileExists(atPath: flat.appendingPathComponent(marker).path) {
            return flat
        }
        if let snap = store.snapshotDirectory(for: repo, revision: "main"),
           fm.fileExists(atPath: snap.appendingPathComponent(marker).path) {
            return snap
        }
        return store.directory(for: repo)
    }

    /// Store-resolved 2.1 snapshot root (explicit path wins, then flat store, then hub snapshot).
    public func resolvedSnapshotDirectory(storeRoot: URL?) -> URL? {
        Self.resolve(explicit: snapshotPath, repo: Self.repo, storeRoot: storeRoot, marker: "transformer")
    }

    /// Store-resolved Qwen3-VL-8B-Instruct root.
    public func resolvedTextEncoderDirectory(storeRoot: URL?) -> URL? {
        Self.resolve(explicit: textEncoderPath, repo: Self.textEncoderRepo, storeRoot: storeRoot, marker: "config.json")
    }

    private enum CodingKeys: String, CodingKey {
        case snapshotPath, textEncoderPath, defaultSteps, defaultTrueCFGScale, defaultOutputResolution,
             defaultEditOutputResolution, keepEncoderResident, useKVCache
    }
}

public enum QwenImage21PackageError: Error, LocalizedError {
    case unreadableSnapshot(String)
    case imageDecode
    case pngEncode
    /// The request would run outside the input envelope the declared footprint was MEASURED at,
    /// so the memory governor's reservation would under-state it. Refused rather than clamped:
    /// silently shrinking a requested size changes the output.
    case outsideMeasuredEnvelope(String)

    public var errorDescription: String? {
        switch self {
        case .unreadableSnapshot(let p): return "Qwen-Image-2.1 snapshot not readable at \(p)."
        case .imageDecode: return "Could not decode an input image."
        case .pngEncode: return "PNG encoding failed."
        case .outsideMeasuredEnvelope(let why): return why
        }
    }
}

/// The input envelope the manifest's footprint was measured at (`QwenImage21Gate --membench`,
/// M5 Max, fp32 VAE; AB-R-0290). `run()` refuses anything outside it so the declaration stays an
/// upper bound.
public enum QwenImage21Envelope {
    /// The model card's documented text-to-image sizes (1:1, 4:3, 3:4, 3:2, 2:3, 16:9, 9:16).
    public static let documentedTextToImageSizes: [(width: Int, height: Int)] = [
        (2048, 2048), (2400, 1792), (1792, 2400), (2528, 1696), (1696, 2528), (2752, 1536), (1536, 2752),
    ]
    /// Text-to-image output area cap = the LARGEST documented size by area — 2400×1792 (4,300,800 px),
    /// not the widest one. Reachable because the pipeline decodes above 1024² in halo-exact tiles
    /// (`AutoencoderKLQwenImage21.decodeTiled`); untiled, 2048² peaked at 72.9 GB.
    public static let maxTextToImagePixels = documentedTextToImageSizes.map { $0.width * $0.height }.max()!
    /// Edit output area. Each reference is resized to the output area, so larger edits grow the
    /// prefix (KV cache ≈ 2.1 GB per 1024² reference) — not measured above 1024², so not admitted.
    public static let maxTargetPixels = 1024 * 1024
    /// Reference images: the model's documented maximum, measured at 1024² each (peak 51.6 GB).
    public static let maxReferenceImages = 10

    /// nil when the request is inside the envelope, else a user-facing reason.
    public static func violation(targetWidth: Int, targetHeight: Int, referenceCount: Int,
                                 outputResolution: Int) -> String? {
        let areaCap = referenceCount == 0 ? maxTextToImagePixels : maxTargetPixels
        if targetWidth * targetHeight > areaCap {
            return referenceCount == 0
                ? "Qwen-Image-2.1: \(targetWidth)×\(targetHeight) exceeds the measured text-to-image envelope "
                    + "(output area ≤ 2400×1792, the largest size on the model card)."
                : "Qwen-Image-2.1: an edit output of \(targetWidth)×\(targetHeight) exceeds the measured edit "
                    + "envelope (output area ≤ 1024²)."
        }
        if referenceCount > maxReferenceImages {
            return "Qwen-Image-2.1 accepts at most \(maxReferenceImages) reference images; got \(referenceCount)."
        }
        if referenceCount > 0, outputResolution * outputResolution > maxTargetPixels {
            return "Qwen-Image-2.1: edit output_resolution \(outputResolution) exceeds the measured envelope "
                + "(≤ 1024). Each reference image is resized to that area, so larger values exceed the declared footprint."
        }
        return nil
    }
}

@InferenceActor
public final class QwenImage21Package: ModelPackage {
    public typealias Configuration = QwenImage21Configuration

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            // C7: Qwen RESEARCH License (non-commercial) — package-local LicenseRef, NOT allowlisted,
            // so `.permissiveOnly` refuses or flags this package by construction (AB-D-0085).
            // C8: port code MIT.
            license: LicenseDeclaration(weightLicense: .qwenResearch, portCodeLicense: .mit),
            // Provenance names the controlled mirror (durability policy); its card pins the upstream
            // revision (`Configuration.upstreamRevision`).
            provenance: Provenance(sourceRepo: "xocialize/Qwen-Image-2.1", revision: "main", tier: 3),
            requirements: RequirementsManifest(
                // MEASURED split (QwenImage21Gate --membench, M5 Max, AB-R-0290). Resident floor =
                // DiT bf16 + VAE fp32 after load, cache cleared: 15.58 GB. The Qwen3-VL-8B encoder
                // (~17.5 GB) is a per-request TRANSIENT evicted before the denoise peak, so it lands
                // in the activation term. Peak / activation per envelope:
                //   T2I 1024²            32.87 / 17.29 GB
                //   edit 1024², 1 ref    35.03 / 19.45
                //   edit 1024², 4 refs   41.52 / 25.94
                //   edit 1024², 10 refs  51.57 / 35.98   ← worst inside the envelope → ×1.2 = 43.2 GB
                // Text-to-image above 1024² decodes in halo-exact tiles (AB-R-0310), which is what
                // admits the model card's sizes without raising the declaration:
                //   T2I 2048²            42.03 / 26.45   (untiled it was 72.91 / 57.33)
                //   T2I 2400×1792 (max)  41.60 / 26.02
                //   T2I 2752×1536        41.15 / 25.57
                // The envelope (QwenImage21Envelope) is enforced in run() so this stays an upper bound.
                // MLX-pool numbers, not in-app phys_footprint — in-app re-baseline still owed.
                footprints: [
                    QuantFootprint(quant: .bf16, residentBytes: 15_600_000_000, peakActivationBytes: 43_200_000_000)
                ],
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0)),
                // Conservative until measured on a 36–48 GB tier; the estimate says it should fit .pro.
                chipFloor: .max
            ),
            specialties: [],
            surfaces: [
                T2IContract.descriptor(
                    name: "qwen-image-2.1",
                    summary: "Qwen-Image-2.1 text-to-image (7B single-stream block-causal DiT + Qwen3-VL-8B "
                        + "conditioning, 64-ch RGBA VAE): 40 FlowMatch Euler steps, no guidance by default, "
                        + "native transparency (RGBA PNG output; prompt 'This is an RGBA image with "
                        + "transparency. … The image has alpha channel and the background is transparent.'), "
                        + "sizes divisible by 32, 2048² native. RESEARCH LICENCE — non-commercial only.",
                    modes: []
                ),
                IEditContract.descriptor(
                    name: "qwen-image-2.1",
                    summary: "Qwen-Image-2.1 instruction editing with up to 10 reference images "
                        + "(<image1>…<imageN> in the prompt), identity-preserving edits, transparent-layer "
                        + "editing and subject extraction; output follows the LAST image's aspect at "
                        + "output_resolution² (default 1024²). Edit noise is domain-separated from the "
                        + "text-to-image draw, so a generate-then-edit flow at the same seed cannot replay "
                        + "the generation. RESEARCH LICENCE — non-commercial only.",
                    modes: []
                ),
            ]
        )
    }

    private let configuration: Configuration
    private var generator: QwenImage21Generator?

    public nonisolated init(configuration: Configuration) {
        self.configuration = configuration
    }

    public func load() async throws {
        guard generator == nil else { return }
        // Materialisation is engine-executed (contract 1.24) before load(); this is the offline
        // backstop so absent weights fail legibly.
        guard let snapshot = configuration.resolvedSnapshotDirectory(storeRoot: configuration.modelsRootDirectory),
              FileManager.default.fileExists(atPath: snapshot.appendingPathComponent("transformer").path)
        else {
            throw QwenImage21PackageError.unreadableSnapshot(
                configuration.snapshotPath.isEmpty ? Configuration.repo : configuration.snapshotPath)
        }
        guard let textEncoder = configuration.resolvedTextEncoderDirectory(storeRoot: configuration.modelsRootDirectory),
              FileManager.default.fileExists(atPath: textEncoder.appendingPathComponent("config.json").path)
        else {
            throw QwenImage21PackageError.unreadableSnapshot(
                configuration.textEncoderPath.isEmpty ? Configuration.textEncoderRepo : configuration.textEncoderPath)
        }
        // DiT (bf16) + VAE (fp32) stay resident; the encoder loads per request and is evicted
        // before the denoise peak unless `keepEncoderResident`.
        self.generator = try QwenImage21Serving.load(
            snapshot: snapshot, vaeSnapshot: snapshot, textEncoder: textEncoder,
            keepEncoderResident: configuration.keepEncoderResident)
    }

    public func unload() async {
        generator = nil
        MLX.Memory.clearCache()
    }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        // CAN-1: entry checkpoint is the FIRST act of run(), before notLoaded validation. Mid-run
        // cadence lives in the core (post-encode seam, per-denoise-step checkpoint, pre-decode
        // seam), rethrowing CancellationError unchanged.
        try Task.checkCancellation()
        guard let generator else { throw PackageError.notLoaded }
        return try await QwenImage21Serving.run(
            request, generator: generator,
            defaults: .init(steps: configuration.defaultSteps, trueCFGScale: configuration.defaultTrueCFGScale,
                            outputResolution: configuration.defaultOutputResolution,
                            editOutputResolution: configuration.defaultEditOutputResolution,
                            useKVCache: configuration.useKVCache),
            sigmas: nil)
    }
}

extension QwenImage21Package {
    /// The author one-liner the engine registers.
    public nonisolated static var registration: PackageRegistration { .of(QwenImage21Package.self) }
}
