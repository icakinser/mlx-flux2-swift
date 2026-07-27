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

private func expectGolden(_ generated: CGImage, _ name: String, label: String) throws {
    let golden = try #require(try loadImages([fixtureURL(name)]).first)
    let diff = try pixelDiff(generated, golden)
    print(
        "quality[strict/\(label)] mean=\(diff.mean) max=\(diff.maximum) "
            + "different=\(diff.differingChannels)/\(diff.channelCount)")
    #expect(diff.mean <= strictMeanThreshold)
}

@Test func imageFixtureManifestIsComplete() throws {
    let manifestURL = fixtureURL("manifest.json")
    let data = try Data(contentsOf: manifestURL)
    let json = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    let cases = try #require(json["cases"] as? [[String: Any]])
    let expected = Set([
        "t2i-euler",
        "t2i-heun",
        "img2img",
        "inpaint",
        "outpaint",
        "reference",
        "multi-reference",
    ])
    #expect(Set(cases.compactMap { $0["id"] as? String }) == expected)
    for fixture in cases {
        let file = try #require(fixture["file"] as? String)
        #expect(FileManager.default.fileExists(atPath: fixtureURL(file).path))
        #expect(fixture["threshold"] as? String == "strict")
    }
}

// MLXRandom is process-global, so real-generation tests must never overlap.
@Suite(.serialized)
struct ImageQualityTests {
    @Test(.enabled(if: imageTestsEnabled))
    func imageQualityStrictEulerGolden() async throws {
        let golden = try #require(try loadImages([fixtureURL("ref_bike_s42.png")]).first)
        let recorder = ProgressRecorder()
        let pipeline = try await makeQualityPipeline(compile: false)
        let generated = try generateBike(
            pipeline,
            progress: { recorder.append($0) })
        let diff = try pixelDiff(generated, golden)
        print(
            "quality[strict/euler] mean=\(diff.mean) max=\(diff.maximum) "
                + "different=\(diff.differingChannels)/\(diff.channelCount)")
        #expect(diff.mean <= strictMeanThreshold)
        #expect(recorder.snapshot.map(\.step) == [0, 1, 2, 3])
        #expect(recorder.snapshot.allSatisfy { $0.totalSteps == 4 })
        let repeated = try generateBike(pipeline)
        let repeatedDiff = try pixelDiff(repeated, golden)
        #expect(repeatedDiff.mean <= strictMeanThreshold)
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

    @Test(.enabled(if: imageTestsEnabled))
    func imageQualityStrictImg2ImgGolden() async throws {
        let source = try #require(try loadImages([fixtureURL("ref_bike_s42.png")]).first)
        let options = ImageToImageOptions(
            prompt: "turn the bicycle blue, preserve the stone wall",
            strength: 0.6,
            width: 512,
            height: 512,
            denoise: DenoiseOptions(numSteps: 4, guidance: 1.0, seed: 43))
        let generated = try await makeQualityPipeline(compile: false)
            .generateImg2Img(options, source: source)
        try expectGolden(generated, "edit_img2img_s43.png", label: "img2img")
    }

    @Test(.enabled(if: imageTestsEnabled))
    func imageQualityStrictInpaintGolden() async throws {
        let source = try #require(try loadImages([fixtureURL("ref_bike_s42.png")]).first)
        let mask = try makeBoxMask(
            width: 512, height: 512, x: 136, y: 152, boxWidth: 240, boxHeight: 224)
        let options = InpaintOptions(
            prompt: "a wooden crate against the stone wall",
            strength: 0.85,
            width: 512,
            height: 512,
            denoise: DenoiseOptions(numSteps: 4, guidance: 1.0, seed: 44))
        let generated = try await makeQualityPipeline(compile: false)
            .generateInpaint(options, source: source, mask: mask)
        try expectGolden(generated, "edit_inpaint_s44.png", label: "inpaint")
    }

    @Test(.enabled(if: imageTestsEnabled))
    func imageQualityStrictOutpaintGolden() async throws {
        let source = try #require(try loadImages([fixtureURL("ref_bike_s42.png")]).first)
        let options = OutpaintOptions(
            prompt: "continue the stone wall and golden-hour scene",
            left: 32,
            right: 32,
            top: 32,
            bottom: 32,
            strength: 0.95,
            denoise: DenoiseOptions(numSteps: 4, guidance: 1.0, seed: 45))
        let generated = try await makeQualityPipeline(compile: false)
            .generateOutpaint(options, source: source)
        #expect(generated.width == 576)
        #expect(generated.height == 576)
        try expectGolden(generated, "edit_outpaint_s45.png", label: "outpaint")
    }

    @Test(.enabled(if: imageTestsEnabled))
    func imageQualityStrictReferenceGolden() async throws {
        let reference = try #require(try loadImages([fixtureURL("ref_bike_s42.png")]).first)
        let options = GenerationOptions(
            prompt: "a bicycle product photograph inspired by the reference",
            width: 512,
            height: 512,
            numSteps: 4,
            guidance: 1.0,
            seed: 46)
        let generated = try await makeQualityPipeline(compile: false)
            .generate(options, inputImages: [reference])
        try expectGolden(generated, "ref_condition_s46.png", label: "reference")
    }

    @Test(.enabled(if: imageTestsEnabled))
    func imageQualityStrictMultiReferenceGolden() async throws {
        let references = try loadImages([
            fixtureURL("ref_bike_s42.png"),
            fixtureURL("edit_img2img_s43.png"),
        ])
        #expect(references.count == 2)
        let options = GenerationOptions(
            prompt: "a studio bicycle scene combining both references",
            width: 512,
            height: 512,
            numSteps: 4,
            guidance: 1.0,
            seed: 48)
        let generated = try await makeQualityPipeline(compile: false)
            .generate(options, inputImages: references)
        try expectGolden(generated, "ref_multi_condition_s48.png", label: "multi-reference")
    }

    @Test(.enabled(if: imageTestsEnabled))
    func imageGenerationCancelsBetweenSteps() async throws {
        let cancellation = GenerationCancellation()
        var options = GenerationOptions(
            prompt: "cancellation quality test",
            width: 256,
            height: 256,
            numSteps: 4,
            guidance: 1.0,
            seed: 47)
        options.cancellation = cancellation
        options.progress = { progress in
            if progress.step == 0 { cancellation.cancel() }
        }
        let pipeline = try await makeQualityPipeline(compile: false)
        #expect(throws: Flux2Error.self) {
            _ = try pipeline.generate(options)
        }
    }
}
