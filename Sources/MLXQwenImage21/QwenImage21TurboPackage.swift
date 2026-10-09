// MLXEngine package for Qwen-Image-2.1-Turbo — the same core, the Turbo DiT, eight fixed steps.
//
// ⚠️ LICENCE: Qwen RESEARCH License, exactly as the base (the Turbo repo ships the same agreement):
// research / evaluation only, no commercial use. `LicenseRef-Qwen-Research` package-local, NOT
// allowlisted — `.permissiveOnly` refuses (`.blocking`) or flags (`.advisory`) it. Research tier,
// never a shipping default (AB-D-0085).
//
// What Turbo is (AB-R-0437, 2026-10-09): the 2.1 DiT re-weighted for 8 steps, plus the schedule it
// was distilled on — `sample_sigmas` in model_index.json (diffusers PR #14950) with the scheduler's
// shift pipeline switched off. Everything else is the base model: its `vae/` is the base VAE cast
// to bf16 and its `text_encoder/` is byte-identical to Qwen/Qwen3-VL-8B-Instruct. So this package
// loads the DiT + pipeline config from the Turbo mirror, the fp32 VAE from the base mirror and the
// conditioner from the stock Instruct snapshot — 14.2 GB of new weights, not 33. Same shapes as the
// base package, so the footprint envelope is the base's (QwenImage21Envelope).
//
// Sampling: the checkpoint's grid IS the schedule. A request's `steps` is ignored, as in the
// reference pipeline, and guidance stays at 1 — the model card has evaluated no other grid.

import Foundation
import MLX
import MLXToolKit
import QwenImage21

/// Init-time configuration (C9): three snapshot roots and generation defaults. No `defaultSteps`
/// — the step count is the checkpoint's `sample_sigmas` length (8).
public struct QwenImage21TurboConfiguration: PackageConfiguration, ModelStorable, QuantConfigured, WeightSourcing {
    /// Qwen-Image-2.1-Turbo root (`transformer/`, `model_index.json`, `scheduler/`). Empty = store.
    public var snapshotPath: String
    /// Base Qwen-Image-2.1 root for `vae/` (fp32 — Turbo's own copy is the same weights in bf16).
    /// Empty = store.
    public var vaeSnapshotPath: String
    /// Qwen/Qwen3-VL-8B-Instruct root (weights + tokenizer). Empty = store.
    public var textEncoderPath: String
    /// 1.0 — the distillation runs without guidance; > 1 with a negative prompt re-applies true CFG
    /// (two DiT forwards per step) on a model that was not evaluated with it.
    public var defaultTrueCFGScale: Float
    /// Same meaning and envelope as the base package.
    public var defaultOutputResolution: Int
    public var defaultEditOutputResolution: Int
    public var keepEncoderResident: Bool
    public var useKVCache: Bool
    public var modelsRootDirectory: URL?

    /// bf16 DiT + fp32 VAE — the only tier for now.
    public var quant: Quant { .bf16 }

    /// Fleet durability policy: the DiT + pipeline config materialise from a namespace we control —
    /// an unmodified, hash-verified mirror of `upstreamRepo` at `upstreamRevision` (`transformer/`,
    /// `model_index.json`, `scheduler/`, LICENSE + NOTICE; no `text_encoder/`, no `vae/`), under
    /// Qwen Research licence §3.
    public static let repo = "xocialize/Qwen-Image-2.1-Turbo"
    public static let upstreamRepo = "Qwen/Qwen-Image-2.1-Turbo"
    public static let upstreamRevision = "d65dbc9a7e8f6b5479e33dee6030eaab2a906509"
    /// The fp32 VAE comes from the base mirror; the conditioner from the stock Instruct repo.
    public static let vaeRepo = QwenImage21Configuration.repo
    public static let textEncoderRepo = QwenImage21Configuration.textEncoderRepo

    public init(
        snapshotPath: String = "",
        vaeSnapshotPath: String = "",
        textEncoderPath: String = "",
        defaultTrueCFGScale: Float = 1.0,
        defaultOutputResolution: Int = 1024,
        defaultEditOutputResolution: Int = 1024,
        keepEncoderResident: Bool = false,
        useKVCache: Bool = true,
        modelsRootDirectory: URL? = nil
    ) {
        self.snapshotPath = snapshotPath
        self.vaeSnapshotPath = vaeSnapshotPath
        self.textEncoderPath = textEncoderPath
        self.defaultTrueCFGScale = defaultTrueCFGScale
        self.defaultOutputResolution = defaultOutputResolution
        self.defaultEditOutputResolution = defaultEditOutputResolution
        self.keepEncoderResident = keepEncoderResident
        self.useKVCache = useKVCache
        self.modelsRootDirectory = modelsRootDirectory
    }

    /// Fresh-machine sources (MAT), split by role across three repos: the Turbo mirror for what
    /// Turbo actually changed, the base mirror for the VAE, the stock Instruct repo for the
    /// conditioner. A machine that already serves the base package downloads only the DiT.
    public var weightSources: [WeightSource] {
        [
            WeightSource(role: "transformer", repo: Self.repo, revision: "main", matching: ["transformer/*"]),
            WeightSource(role: "pipeline-config", repo: Self.repo, revision: "main",
                         matching: ["model_index.json", "scheduler/*", "LICENSE", "NOTICE"]),
            WeightSource(role: "vae", repo: Self.vaeRepo, revision: "main", matching: ["vae/*"]),
            WeightSource(role: "text-encoder", repo: Self.textEncoderRepo, revision: "main",
                         matching: ["*.safetensors", "*.json", "merges.txt"]),
        ]
    }

    static func hasVAE(_ path: String) -> Bool {
        !path.isEmpty && FileManager.default.fileExists(atPath: URL(fileURLWithPath: path).appendingPathComponent("vae").path)
    }

    /// Explicit paths satisfy their own repo's roles; everything else resolves through the store.
    public func missingWeightSources(storeRoot: URL?) -> [WeightSource] {
        var missing = defaultMissingWeightSources(storeRoot: storeRoot)
        if QwenImage21Configuration.hasTransformer(snapshotPath) { missing.removeAll { $0.repo == Self.repo } }
        if Self.hasVAE(vaeSnapshotPath) { missing.removeAll { $0.repo == Self.vaeRepo } }
        if QwenImage21Configuration.hasTextEncoder(textEncoderPath) { missing.removeAll { $0.repo == Self.textEncoderRepo } }
        return missing
    }

    /// Store-resolved Turbo root (explicit path wins, then flat store, then hub snapshot).
    public func resolvedSnapshotDirectory(storeRoot: URL?) -> URL? {
        QwenImage21Configuration.resolve(explicit: snapshotPath, repo: Self.repo, storeRoot: storeRoot, marker: "transformer")
    }

    /// Store-resolved base root carrying `vae/`.
    public func resolvedVAEDirectory(storeRoot: URL?) -> URL? {
        QwenImage21Configuration.resolve(explicit: vaeSnapshotPath, repo: Self.vaeRepo, storeRoot: storeRoot, marker: "vae")
    }

    /// Store-resolved Qwen3-VL-8B-Instruct root.
    public func resolvedTextEncoderDirectory(storeRoot: URL?) -> URL? {
        QwenImage21Configuration.resolve(explicit: textEncoderPath, repo: Self.textEncoderRepo, storeRoot: storeRoot, marker: "config.json")
    }

    private enum CodingKeys: String, CodingKey {
        case snapshotPath, vaeSnapshotPath, textEncoderPath, defaultTrueCFGScale, defaultOutputResolution,
             defaultEditOutputResolution, keepEncoderResident, useKVCache
    }
}

@InferenceActor
public final class QwenImage21TurboPackage: ModelPackage {
    public typealias Configuration = QwenImage21TurboConfiguration

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            // C7: Qwen RESEARCH License (non-commercial) — the base's package-local LicenseRef, NOT
            // allowlisted (AB-D-0085). C8: port code MIT.
            license: LicenseDeclaration(weightLicense: .qwenResearch, portCodeLicense: .mit),
            // Provenance names the controlled Turbo mirror; its card pins `upstreamRevision`.
            provenance: Provenance(sourceRepo: QwenImage21TurboConfiguration.repo, revision: "main", tier: 3),
            requirements: RequirementsManifest(
                // The base package's MEASURED split (AB-R-0290): Turbo has the same tensor set, the
                // same resident DiT bf16 + VAE fp32, the same per-step graph at every envelope size
                // and the same decode — only the step count differs, which the footprint does not
                // depend on. The envelope (QwenImage21Envelope) is enforced in run() as for the base.
                // A Turbo `--membench` re-run is owed for the record (AB-T-0212).
                footprints: QwenImage21Package.manifest.requirements.footprints,
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0)),
                chipFloor: .max
            ),
            specialties: [],
            surfaces: [
                T2IContract.descriptor(
                    name: "qwen-image-2.1-turbo",
                    summary: "Qwen-Image-2.1-Turbo text-to-image: the 2.1 DiT (7B single-stream block-causal, "
                        + "Qwen3-VL-8B conditioning, 64-ch RGBA VAE) distilled to EIGHT fixed FlowMatch Euler steps "
                        + "— the checkpoint's own sampling grid; `steps` is ignored — no guidance, native "
                        + "transparency (RGBA PNG), sizes divisible by 32, 2048² native. About 5x fewer DiT "
                        + "forwards than qwen-image-2.1 at the same memory. RESEARCH LICENCE — non-commercial only.",
                    modes: []
                ),
                IEditContract.descriptor(
                    name: "qwen-image-2.1-turbo",
                    summary: "Qwen-Image-2.1-Turbo instruction editing with up to 10 reference images "
                        + "(<image1>…<imageN> in the prompt) in eight fixed steps (`steps` is ignored, no guidance); "
                        + "output follows the LAST image's aspect at output_resolution² (default 1024²). Edit noise "
                        + "is domain-separated from the text-to-image draw as in qwen-image-2.1. RESEARCH LICENCE "
                        + "— non-commercial only.",
                    modes: []
                ),
            ]
        )
    }

    private let configuration: Configuration
    private var generator: QwenImage21Generator?
    /// The checkpoint's `sample_sigmas` (terminal excluded), read at load.
    private var sigmas: [Float] = []

    public nonisolated init(configuration: Configuration) {
        self.configuration = configuration
    }

    public func load() async throws {
        guard generator == nil else { return }
        // Materialisation is engine-executed (contract 1.24) before load(); these are the offline
        // backstops so absent weights fail legibly.
        let storeRoot = configuration.modelsRootDirectory
        guard let snapshot = configuration.resolvedSnapshotDirectory(storeRoot: storeRoot),
              FileManager.default.fileExists(atPath: snapshot.appendingPathComponent("transformer").path)
        else {
            throw QwenImage21PackageError.unreadableSnapshot(
                configuration.snapshotPath.isEmpty ? Configuration.repo : configuration.snapshotPath)
        }
        guard let vaeSnapshot = configuration.resolvedVAEDirectory(storeRoot: storeRoot),
              FileManager.default.fileExists(atPath: vaeSnapshot.appendingPathComponent("vae").path)
        else {
            throw QwenImage21PackageError.unreadableSnapshot(
                configuration.vaeSnapshotPath.isEmpty ? Configuration.vaeRepo : configuration.vaeSnapshotPath)
        }
        guard let textEncoder = configuration.resolvedTextEncoderDirectory(storeRoot: storeRoot),
              FileManager.default.fileExists(atPath: textEncoder.appendingPathComponent("config.json").path)
        else {
            throw QwenImage21PackageError.unreadableSnapshot(
                configuration.textEncoderPath.isEmpty ? Configuration.textEncoderRepo : configuration.textEncoderPath)
        }
        // The grid is what makes this snapshot a Turbo one; a base snapshot here would sample a
        // 40-step model on nothing at all.
        guard let grid = try QwenImage21Scheduler.loadSampleSigmas(snapshot: snapshot) else {
            throw QwenImage21PackageError.unreadableSnapshot(
                "\(snapshot.path): model_index.json has no sample_sigmas — not a Qwen-Image-2.1-Turbo snapshot")
        }
        sigmas = grid
        generator = try QwenImage21Serving.load(
            snapshot: snapshot, vaeSnapshot: vaeSnapshot, textEncoder: textEncoder,
            keepEncoderResident: configuration.keepEncoderResident)
    }

    public func unload() async {
        generator = nil
        MLX.Memory.clearCache()
    }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        // CAN-1: entry checkpoint first, before notLoaded validation (cadence lives in the core).
        try Task.checkCancellation()
        guard let generator else { throw PackageError.notLoaded }
        return try await QwenImage21Serving.run(
            request, generator: generator,
            defaults: .init(steps: sigmas.count, trueCFGScale: configuration.defaultTrueCFGScale,
                            outputResolution: configuration.defaultOutputResolution,
                            editOutputResolution: configuration.defaultEditOutputResolution,
                            useKVCache: configuration.useKVCache),
            sigmas: sigmas)
    }
}

extension QwenImage21TurboPackage {
    /// The author one-liner the engine registers.
    public nonisolated static var registration: PackageRegistration { .of(QwenImage21TurboPackage.self) }
}
