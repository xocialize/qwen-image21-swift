import XCTest
@testable import QwenImage21

final class SchedulerTests: XCTestCase {
    func testShiftAndTerminal() {
        // mu at 4096 target tokens with the 2.1 config (256/8192, 0.5/0.9)
        let mu = QwenImage21Scheduler.calculateShift(imageSeqLen: 4096)
        XCTAssertEqual(mu, 0.5 + (0.9 - 0.5) / Float(8192 - 256) * Float(4096 - 256), accuracy: 1e-6)
        let s = QwenImage21Scheduler.sigmas(steps: 4, mu: mu)
        XCTAssertEqual(s.count, 5)
        XCTAssertEqual(s[0], 1, accuracy: 1e-7)
        XCTAssertEqual(s[3], 0.02, accuracy: 1e-6)  // shift_terminal
        XCTAssertEqual(s[4], 0)
        XCTAssertTrue(zip(s, s.dropFirst()).allSatisfy { $0 > $1 })
    }

    /// Qwen-Image-2.1-Turbo ships its schedule as `sample_sigmas` (diffusers PR #14950) with the
    /// shift pipeline off: the grid is used verbatim, 0 appended, whatever the token count.
    func testFixedGridIsVerbatimPlusTerminal() throws {
        let turbo: [Float] = [1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568]
        let s = try QwenImage21Scheduler.fixedGrid(turbo)
        XCTAssertEqual(s.count, 9)
        XCTAssertEqual(Array(s.dropLast()), turbo)
        XCTAssertEqual(s.last, 0)
        XCTAssertThrowsError(try QwenImage21Scheduler.fixedGrid([]))
        XCTAssertThrowsError(try QwenImage21Scheduler.fixedGrid([1.0, 0.5, 0.5]))  // not strictly decreasing
        XCTAssertThrowsError(try QwenImage21Scheduler.fixedGrid([1.0, 0.5, 0.0]))  // the terminal is the scheduler's
        XCTAssertThrowsError(try QwenImage21Scheduler.fixedGrid([1.2, 0.5]))       // outside (0, 1]
    }

    /// `sample_sigmas` comes from `model_index.json`; a scheduler config that would still shift
    /// the grid is refused, not silently ignored.
    func testSampleSigmasReadFromModelIndex() throws {
        let tmp = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("qi21-sample-sigmas-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmp.appendingPathComponent("scheduler"), withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmp) }
        let index = tmp.appendingPathComponent("model_index.json")
        let sched = tmp.appendingPathComponent("scheduler/scheduler_config.json")
        // the base 2.1: no key -> nil (the shifted linspace schedule applies)
        try #"{"_class_name": "QwenImage21Pipeline"}"#.write(to: index, atomically: true, encoding: .utf8)
        XCTAssertNil(try QwenImage21Scheduler.loadSampleSigmas(snapshot: tmp))
        // Turbo: the grid + the identity scheduler
        try #"{"sample_sigmas": [1.0, 0.5, 0.25]}"#.write(to: index, atomically: true, encoding: .utf8)
        try #"{"use_dynamic_shifting": false, "shift": 1.0, "shift_terminal": null}"#.write(to: sched, atomically: true, encoding: .utf8)
        XCTAssertEqual(try QwenImage21Scheduler.loadSampleSigmas(snapshot: tmp), [1.0, 0.5, 0.25])
        // a shifting scheduler would process the grid — refused
        try #"{"use_dynamic_shifting": true, "shift": 1.0, "shift_terminal": 0.02}"#.write(to: sched, atomically: true, encoding: .utf8)
        XCTAssertThrowsError(try QwenImage21Scheduler.loadSampleSigmas(snapshot: tmp))
    }

    /// Noise replay (diffusers#14824): an edit must never draw the text-to-image noise for the
    /// same seed, or it re-runs the generation instead of editing.
    func testEditNoiseIsDomainSeparatedFromTextToImage() {
        for seed: UInt64 in [0, 1, 42, 4242, UInt64.max] {
            XCTAssertEqual(QwenImage21Latents.noiseSeed(seed, isEdit: false), seed)
            XCTAssertNotEqual(QwenImage21Latents.noiseSeed(seed, isEdit: true), seed)
            // deterministic: same inputs, same draw
            XCTAssertEqual(QwenImage21Latents.noiseSeed(seed, isEdit: true),
                           QwenImage21Latents.noiseSeed(seed, isEdit: true))
        }
        // and no edit seed collides with the T2I seed of another ordinary seed nearby
        let edits = Set((0..<64).map { QwenImage21Latents.noiseSeed(UInt64($0), isEdit: true) })
        XCTAssertTrue(edits.isDisjoint(with: Set((0..<64).map { UInt64($0) })))
    }

    /// The single target-size rule shared by the generator and the package's envelope guard.
    func testTargetSizeRule() {
        // T2I: explicit size wins, floored to /32
        XCTAssertTrue(QwenImage21Latents.targetSize(imageSizes: [], width: 1000, height: 1030, outputResolution: 1024) == (992, 1024))
        // T2I: no size → output_resolution square
        XCTAssertTrue(QwenImage21Latents.targetSize(imageSizes: [], width: nil, height: nil, outputResolution: 1024) == (1024, 1024))
        // edit: follows the LAST image's aspect at output_resolution² area (3:4 → 896×1184 at 1024²)
        let t = QwenImage21Latents.targetSize(imageSizes: [(1000, 1000), (300, 400)], width: nil, height: nil, outputResolution: 1024)
        XCTAssertEqual(t.width * 4, t.height * 3, accuracy: 256)
        XCTAssertLessThanOrEqual(abs(t.width * t.height - 1024 * 1024), 1024 * 64)
    }

    func testCalculateDimensions() {
        let (w, h) = QwenImage21Latents.calculateDimensions(targetArea: 1024 * 1024, ratio: 1)
        XCTAssertEqual(w, 1024); XCTAssertEqual(h, 1024)
        let (w2, h2) = QwenImage21Latents.calculateDimensions(targetArea: 320 * 320, ratio: 0.75)
        XCTAssertEqual(w2, 288); XCTAssertEqual(h2, 384)
    }
}
