// MAT gate (offline) — two repos: the 2.1 snapshot and the stock Qwen3-VL-8B-Instruct conditioner.
import Foundation
import MLXServeConformance
import MLXToolKit
import XCTest

@testable import MLXQwenImage21

final class QwenImage21MaterializationTests: XCTestCase {
    static let localSnapshot = "/Volumes/Satechi/Development/mlxengine-image/weights/Qwen-Image-2.1"
    static let localTextEncoder = "/Volumes/Satechi/Development/mlxengine-image/weights/Qwen3-VL-8B-Instruct"

    func testMATGate() {
        let fresh = QwenImage21Configuration()
        let haveLocal = FileManager.default.fileExists(atPath: Self.localSnapshot + "/transformer")
            && FileManager.default.fileExists(atPath: Self.localTextEncoder + "/config.json")
        let satisfied = QwenImage21Configuration(snapshotPath: Self.localSnapshot, textEncoderPath: Self.localTextEncoder)
        let report = MaterializationConformance.check(
            freshConfiguration: fresh, satisfiedConfiguration: haveLocal ? satisfied : nil)
        XCTAssertTrue(report.passed, report.summary)
    }

    func testWeightSourcesCoverBothRepos() {
        let cfg = QwenImage21Configuration()
        XCTAssertEqual(Set(cfg.weightSources.map(\.role)), ["transformer", "vae", "pipeline-config", "text-encoder"])
        XCTAssertEqual(Set(cfg.weightSources.map(\.repo)),
                       [QwenImage21Configuration.repo, QwenImage21Configuration.textEncoderRepo])
        let globs = cfg.weightSources.flatMap { $0.matching ?? [] }
        XCTAssertTrue(globs.contains("processor/*"))  // tokenizer + preprocessor config for the prompt encoder
        XCTAssertTrue(globs.contains("transformer/*"))
        XCTAssertTrue(globs.contains("vae/*"))
        XCTAssertTrue(globs.contains("*.safetensors"))
        XCTAssertTrue(globs.contains("NOTICE"))  // Qwen Research licence §3(c) attribution travels with the weights
    }

    func testDiTSourcesFromTheControlledMirror() {
        let sources = QwenImage21Configuration().weightSources
        XCTAssertEqual(QwenImage21Configuration.repo, "xocialize/Qwen-Image-2.1")
        XCTAssertTrue(sources.filter { $0.role != "text-encoder" }.allSatisfy { $0.repo == QwenImage21Configuration.repo })
        XCTAssertEqual(QwenImage21Package.manifest.provenance.sourceRepo, QwenImage21Configuration.repo)
    }

    /// Turbo: three repos — the Turbo mirror (DiT + pipeline config), the base mirror (VAE) and
    /// the stock conditioner. A local Turbo download carries only the first.
    func testTurboMATGate() {
        let localTurbo = "/Volumes/Satechi/Development/mlxengine-image/weights/Qwen-Image-2.1-Turbo"
        let fresh = QwenImage21TurboConfiguration()
        let haveLocal = FileManager.default.fileExists(atPath: localTurbo + "/transformer")
            && FileManager.default.fileExists(atPath: Self.localSnapshot + "/vae")
            && FileManager.default.fileExists(atPath: Self.localTextEncoder + "/config.json")
        let satisfied = QwenImage21TurboConfiguration(
            snapshotPath: localTurbo, vaeSnapshotPath: Self.localSnapshot, textEncoderPath: Self.localTextEncoder)
        let report = MaterializationConformance.check(
            freshConfiguration: fresh, satisfiedConfiguration: haveLocal ? satisfied : nil)
        XCTAssertTrue(report.passed, report.summary)
    }

    func testExplicitPathsSatisfyTheirOwnRepoOnly() {
        let tmp = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("qi21-mat-probe")
        try? FileManager.default.createDirectory(at: tmp.appendingPathComponent("transformer"), withIntermediateDirectories: true)
        let pinned = QwenImage21Configuration(snapshotPath: tmp.path)
        let missing = pinned.missingWeightSources(storeRoot: nil)
        XCTAssertFalse(missing.contains { $0.repo == QwenImage21Configuration.repo })
        XCTAssertTrue(missing.contains { $0.repo == QwenImage21Configuration.textEncoderRepo })
        XCTAssertEqual(pinned.resolvedSnapshotDirectory(storeRoot: nil)?.lastPathComponent, "qi21-mat-probe")
        XCTAssertFalse(QwenImage21Configuration().missingWeightSources(storeRoot: nil).isEmpty)
    }
}
