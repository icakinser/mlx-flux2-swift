import CoreGraphics
import Foundation
import ImageIO
import Testing
@testable import Flux2Kit

// PERMANENT — quality harness replaces the hard Python parity lock.
//
// These tests are deliberately opt-in: they load the full model, require the MLX metallib, and
// perform real generation. Run with:
//   FLUX2_RUN_IMAGE_TESTS=1 FLUX2_REPO=/path/to/snapshot swift test --filter imageQuality
private let imageTestsEnabled: Bool = {
    guard ProcessInfo.processInfo.environment["FLUX2_RUN_IMAGE_TESTS"] == "1",
        let repo = ProcessInfo.processInfo.environment["FLUX2_REPO"]
    else {
        return false
    }
    return FileManager.default.fileExists(atPath: repo)
}()

private let strictMeanThreshold = 0.5
private let softMeanThreshold = 5.0

private struct PixelDiff {
    let mean: Double
    let maximum: Int
    let differingChannels: Int
    let channelCount: Int
}

private final class ProgressRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var values: [GenerationProgress] = []

    func append(_ progress: GenerationProgress) {
        lock.withLock { values.append(progress) }
    }

    var snapshot: [GenerationProgress] {
        lock.withLock { values }
    }
}

private func fixtureURL(_ name: String) -> URL {
    URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .appendingPathComponent("Fixtures")
        .appendingPathComponent(name)
}

private func rgbaBytes(_ image: CGImage) throws -> [UInt8] {
    let width = image.width
    let height = image.height
    var bytes = [UInt8](repeating: 0, count: width * height * 4)
    guard let context = CGContext(
        data: &bytes,
        width: width,
        height: height,
        bitsPerComponent: 8,
        bytesPerRow: width * 4,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    ) else {
        throw Flux2Error.generationFailed("Could not create quality-harness image context")
    }
    context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
    return bytes
}

private func pixelDiff(_ lhs: CGImage, _ rhs: CGImage) throws -> PixelDiff {
    guard lhs.width == rhs.width, lhs.height == rhs.height else {
        throw Flux2Error.generationFailed(
            "Quality images have different sizes: \(lhs.width)x\(lhs.height) vs "
                + "\(rhs.width)x\(rhs.height)")
    }
    let a = try rgbaBytes(lhs)
    let b = try rgbaBytes(rhs)
    var sum = 0
    var maximum = 0
    var differing = 0
    var channels = 0
    for pixel in stride(from: 0, to: a.count, by: 4) {
        for channel in 0 ..< 3 {
            let delta = abs(Int(a[pixel + channel]) - Int(b[pixel + channel]))
            sum += delta
            maximum = max(maximum, delta)
            differing += delta == 0 ? 0 : 1
            channels += 1
        }
    }
    return PixelDiff(
        mean: Double(sum) / Double(channels),
        maximum: maximum,
        differingChannels: differing,
        channelCount: channels)
}

private func makeQualityPipeline(compile: Bool) async throws -> Flux2Pipeline {
    guard let repo = ProcessInfo.processInfo.environment["FLUX2_REPO"] else {
        throw Flux2Error.configMissing("FLUX2_REPO is required for image quality tests")
    }
    return try await Flux2Pipeline(
        repoPath: URL(fileURLWithPath: repo),
        compile: compile,
        residency: .keepResident)
}

private func generateBike(
    _ pipeline: Flux2Pipeline,
    sampler: Sampler = .euler,
    progress: (@Sendable (GenerationProgress) -> Void)? = nil
) throws -> CGImage {
    try pipeline.generate(
        prompt: "a red bicycle leaning against a stone wall, golden hour",
        width: 512,
        height: 512,
        numSteps: 4,
        guidance: 1.0,
        seed: 42,
        sampler: sampler,
        progress: progress)
}

// MLXRandom is process-global, so real-generation tests must never overlap.
@Suite(.serialized)
struct ImageQualityTests {
    @Test(.enabled(if: imageTestsEnabled))
    func imageQualityStrictEulerGolden() async throws {
        let golden = try #require(try loadImages([fixtureURL("ref_bike_s42.png")]).first)
        let recorder = ProgressRecorder()
        let generated = try generateBike(
            await makeQualityPipeline(compile: false),
            progress: { recorder.append($0) })
        let diff = try pixelDiff(generated, golden)
        print(
            "quality[strict/euler] mean=\(diff.mean) max=\(diff.maximum) "
                + "different=\(diff.differingChannels)/\(diff.channelCount)")
        #expect(diff.mean <= strictMeanThreshold)
        #expect(recorder.snapshot.map(\.step) == [0, 1, 2, 3])
        #expect(recorder.snapshot.allSatisfy { $0.totalSteps == 4 })
    }

    @Test(.enabled(if: imageTestsEnabled))
    func imageQualitySoftCompile() async throws {
        let eager = try generateBike(await makeQualityPipeline(compile: false))
        let compiled = try generateBike(await makeQualityPipeline(compile: true))
        let diff = try pixelDiff(compiled, eager)
        print(
            "quality[soft/compile] mean=\(diff.mean) max=\(diff.maximum) "
                + "different=\(diff.differingChannels)/\(diff.channelCount)")
        #expect(diff.mean <= softMeanThreshold)
    }

    @Test(.enabled(if: imageTestsEnabled))
    func imageQualityStrictHeunGolden() async throws {
        let golden = try #require(try loadImages([fixtureURL("ref_bike_heun_s42.png")]).first)
        let generated = try generateBike(
            await makeQualityPipeline(compile: false),
            sampler: .heun)
        let diff = try pixelDiff(generated, golden)
        print(
            "quality[strict/heun] mean=\(diff.mean) max=\(diff.maximum) "
                + "different=\(diff.differingChannels)/\(diff.channelCount)")
        #expect(diff.mean <= strictMeanThreshold)
    }
}
