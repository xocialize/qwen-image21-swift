// Offline manifest / configuration tests for the Qwen-Image-2.1-Turbo package (no weights).
import Foundation
import MLXToolKit
import XCTest

@testable import MLXQwenImage21

final class QwenImage21TurboPackageTests: XCTestCase {
    func testManifestIsTheResearchTierToo() {
        let m = QwenImage21TurboPackage.manifest
        XCTAssertEqual(m.license.weightLicense, .qwenResearch)
        XCTAssertFalse(SPDXLicense.permissiveAllowlist.contains(m.license.weightLicense))
        XCTAssertEqual(m.license.portCodeLicense, .mit)
        XCTAssertEqual(m.contractVersion, ContractVersion.current)
    }

    func testTwoSurfacesNamedTurboOnTheBaseFootprint() {
        let m = QwenImage21TurboPackage.manifest
        XCTAssertEqual(Set(m.surfaces.map(\.capability)), [.textToImage, .imageEdit])
        XCTAssertEqual(Set(m.surfaces.map(\.name)), ["qwen-image-2.1-turbo"])
        XCTAssertEqual(m.provenance.sourceRepo, "xocialize/Qwen-Image-2.1-Turbo")
        // Same tensor set, same per-step graph, same decode: the base's measured split carries.
        XCTAssertEqual(m.requirements.footprints.map(\.residentBytes),
                       QwenImage21Package.manifest.requirements.footprints.map(\.residentBytes))
        XCTAssertEqual(m.requirements.footprints.map(\.peakActivationBytes),
                       QwenImage21Package.manifest.requirements.footprints.map(\.peakActivationBytes))
        // Every surface says the step count is the checkpoint's.
        XCTAssertTrue(m.surfaces.allSatisfy { $0.summary.contains("`steps` is ignored") })
    }

    /// The DiT + pipeline config come from the Turbo mirror, the VAE from the base mirror, the
    /// conditioner from the stock Instruct repo — never the Turbo repo's bf16 VAE or 17.5 GB encoder.
    func testWeightSourcesSplitAcrossThreeRepos() {
        let cfg = QwenImage21TurboConfiguration()
        let byRole = Dictionary(uniqueKeysWithValues: cfg.weightSources.map { ($0.role, $0) })
        XCTAssertEqual(Set(byRole.keys), ["transformer", "vae", "pipeline-config", "text-encoder"])
        XCTAssertEqual(byRole["transformer"]?.repo, QwenImage21TurboConfiguration.repo)
        XCTAssertEqual(byRole["pipeline-config"]?.repo, QwenImage21TurboConfiguration.repo)
        XCTAssertEqual(byRole["vae"]?.repo, QwenImage21Configuration.repo)
        XCTAssertEqual(byRole["text-encoder"]?.repo, QwenImage21Configuration.textEncoderRepo)
        XCTAssertEqual(byRole["vae"]?.matching, ["vae/*"])
        let config = byRole["pipeline-config"]?.matching ?? []
        XCTAssertTrue(config.contains("model_index.json"))  // the grid lives there
        XCTAssertTrue(config.contains("NOTICE"))            // §3(c) attribution travels with the weights
        XCTAssertFalse(cfg.weightSources.contains { $0.repo == QwenImage21TurboConfiguration.repo && ($0.matching ?? []).contains("vae/*") })
    }

    func testExplicitPathsSatisfyTheirOwnRepoOnly() throws {
        let tmp = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("qi21-turbo-mat-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: tmp) }
        try FileManager.default.createDirectory(at: tmp.appendingPathComponent("turbo/transformer"), withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: tmp.appendingPathComponent("base/vae"), withIntermediateDirectories: true)
        let turboOnly = QwenImage21TurboConfiguration(snapshotPath: tmp.appendingPathComponent("turbo").path)
        var missing = turboOnly.missingWeightSources(storeRoot: nil)
        XCTAssertFalse(missing.contains { $0.repo == QwenImage21TurboConfiguration.repo })
        XCTAssertTrue(missing.contains { $0.repo == QwenImage21TurboConfiguration.vaeRepo })
        XCTAssertTrue(missing.contains { $0.repo == QwenImage21TurboConfiguration.textEncoderRepo })
        let both = QwenImage21TurboConfiguration(
            snapshotPath: tmp.appendingPathComponent("turbo").path, vaeSnapshotPath: tmp.appendingPathComponent("base").path)
        missing = both.missingWeightSources(storeRoot: nil)
        XCTAssertFalse(missing.contains { $0.repo == QwenImage21TurboConfiguration.vaeRepo })
        XCTAssertTrue(missing.contains { $0.repo == QwenImage21TurboConfiguration.textEncoderRepo })
        XCTAssertEqual(both.resolvedSnapshotDirectory(storeRoot: nil)?.lastPathComponent, "turbo")
        XCTAssertEqual(both.resolvedVAEDirectory(storeRoot: nil)?.lastPathComponent, "base")
        XCTAssertFalse(QwenImage21TurboConfiguration().missingWeightSources(storeRoot: nil).isEmpty)
    }

    func testConfigurationRoundTripsWithoutAStepCount() throws {
        let cfg = QwenImage21TurboConfiguration(
            snapshotPath: "/tmp/qi21-turbo", vaeSnapshotPath: "/tmp/qi21", textEncoderPath: "/tmp/qwen3vl")
        let data = try JSONEncoder().encode(cfg)
        let back = try JSONDecoder().decode(QwenImage21TurboConfiguration.self, from: data)
        XCTAssertEqual(back.snapshotPath, "/tmp/qi21-turbo")
        XCTAssertEqual(back.vaeSnapshotPath, "/tmp/qi21")
        XCTAssertEqual(back.textEncoderPath, "/tmp/qwen3vl")
        XCTAssertEqual(back.defaultTrueCFGScale, 1.0)
        XCTAssertEqual(back.defaultOutputResolution, 1024)
        XCTAssertTrue(back.useKVCache)
        XCTAssertEqual(QwenImage21TurboConfiguration().quant, .bf16)
        // The schedule is the checkpoint's: no steps knob exists to serialise.
        let keys = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any]).keys
        XCTAssertFalse(keys.contains("defaultSteps"))
    }
}
