// CAN gate (offline, no kernels, no weights). CAN-1/2: run() pre-cancelled on BOTH surfaces —
// the entry checkpoint is the first act of run(), before notLoaded. CAN-3: cadence of record —
// the core's denoise loop checks once per step; post-encode and pre-decode seams bracket it.
import Foundation
import MLXServeConformance
import MLXToolKit
import XCTest

@testable import MLXQwenImage21

final class QwenImage21CancellationTests: XCTestCase {
    func testCANGatePreCancelledTextToImage() async {
        let package = QwenImage21Package(configuration: QwenImage21Configuration())
        let report = await CancellationConformance.checkRun(package: package, request: T2IRequest(prompt: "probe"))
        XCTAssertTrue(report.passed, report.summary)
    }

    func testCANGatePreCancelledImageEdit() async {
        let package = QwenImage21Package(configuration: QwenImage21Configuration())
        let report = await CancellationConformance.checkRun(
            package: package, request: IEditRequest(images: [], prompt: "probe"))
        XCTAssertTrue(report.passed, report.summary)
    }

    func testCANCadenceDeclaration() {
        XCTAssertTrue(CancellationConformance.longRunImplied(by: QwenImage21Package.manifest))
        let report = CancellationConformance.checkCadence(
            manifest: QwenImage21Package.manifest,
            posture: .cadence([.init(phase: .denoise, unit: .step)]))
        XCTAssertTrue(report.passed, report.summary)
    }

    // The Turbo package shares the serving path, so the same gates hold.

    func testTurboCANGatePreCancelledTextToImage() async {
        let package = QwenImage21TurboPackage(configuration: QwenImage21TurboConfiguration())
        let report = await CancellationConformance.checkRun(package: package, request: T2IRequest(prompt: "probe"))
        XCTAssertTrue(report.passed, report.summary)
    }

    func testTurboCANGatePreCancelledImageEdit() async {
        let package = QwenImage21TurboPackage(configuration: QwenImage21TurboConfiguration())
        let report = await CancellationConformance.checkRun(
            package: package, request: IEditRequest(images: [], prompt: "probe"))
        XCTAssertTrue(report.passed, report.summary)
    }

    func testTurboCANCadenceDeclaration() {
        XCTAssertTrue(CancellationConformance.longRunImplied(by: QwenImage21TurboPackage.manifest))
        let report = CancellationConformance.checkCadence(
            manifest: QwenImage21TurboPackage.manifest,
            posture: .cadence([.init(phase: .denoise, unit: .step)]))
        XCTAssertTrue(report.passed, report.summary)
    }
}
