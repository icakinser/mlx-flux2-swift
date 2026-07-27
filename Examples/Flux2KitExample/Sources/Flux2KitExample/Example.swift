// Flux2Kit library-first usage example.
//
//   swift run Flux2KitExample process <image.png>
//   FLUX2_REPO=/path/to/FLUX-2 swift run Flux2KitExample generate
//   FLUX2_REPO=... swift run Flux2KitExample img2img <source.png> "prompt"
//   FLUX2_REPO=... swift run Flux2KitExample remove <source.png> <mask.png>

import CoreGraphics
import Flux2Kit
import Foundation

@main
struct Example {
    static func main() async throws {
        let args = Array(CommandLine.arguments.dropFirst())

        // Model-free image processing — no model, no weights, no metallib (pure CoreGraphics).
        if args.count >= 2, args[0] == "process" {
            guard let src = try loadImages([URL(fileURLWithPath: args[1])]).first else {
                print("Could not load \(args[1])")
                return
            }
            let out = try applyImageOps(
                src, [.resize(384, 384), .rotate(90), .vignette(0.25)])
            let url = URL(fileURLWithPath: "example-processed.png")
            try savePNG(out, to: url)
            print("Wrote \(url.lastPathComponent)")
            return
        }

        guard let repo = ProcessInfo.processInfo.environment["FLUX2_REPO"] else {
            print("""
            usage:
              swift run Flux2KitExample process <image.png>
              FLUX2_REPO=/path/to/FLUX-2 swift run Flux2KitExample generate
              FLUX2_REPO=... swift run Flux2KitExample img2img <source.png> "prompt"
              FLUX2_REPO=... swift run Flux2KitExample remove <source.png> <mask.png>

            knobs:
              FLUX2_COMPILE=1      enable the quality-gated compile path
              FLUX2_LOW_MEMORY=1   unload each model after its stage
            """)
            return
        }

        print("Loading FLUX.2 from \(repo) …")
        let useCompile = ProcessInfo.processInfo.environment["FLUX2_COMPILE"] == "1"
        let lowMemory = ProcessInfo.processInfo.environment["FLUX2_LOW_MEMORY"] == "1"
        let pipeline = try await Flux2Pipeline(
            repoPath: URL(fileURLWithPath: repo),
            compile: useCompile,
            residency: lowMemory ? .unloadAfterUse : .keepResident)

        let image: CGImage
        switch args.first ?? "generate" {
        case "generate":
            var options = GenerationOptions(
                prompt: "a red bicycle leaning against a stone wall, golden hour",
                width: 512, height: 512, numSteps: 4, guidance: 1.0, seed: 42)
            options.progress = { progress in
                print(
                    "step \(progress.step + 1)/\(progress.totalSteps) "
                        + "(\(String(format: "%.0f", progress.elapsedMilliseconds)) ms)")
            }
            image = try pipeline.generate(options)
        case "img2img":
            guard args.count >= 2 else { throw Flux2Error.configMissing("img2img needs a source") }
            let source = try requireImage(args[1])
            let prompt = args.dropFirst(2).joined(separator: " ")
            image = try pipeline.generateImg2Img(
                prompt: prompt.isEmpty ? "make it winter" : prompt,
                source: source,
                strength: 0.6,
                width: aligned(source.width),
                height: aligned(source.height),
                seed: 42)
        case "remove":
            guard args.count >= 3 else {
                throw Flux2Error.configMissing("remove needs a source and mask")
            }
            let source = try requireImage(args[1])
            image = try pipeline.removeObject(
                source: source,
                mask: try requireImage(args[2]),
                width: aligned(source.width),
                height: aligned(source.height),
                seed: 42)
        default:
            throw Flux2Error.configMissing("Unknown example command: \(args[0])")
        }

        let url = URL(fileURLWithPath: "example-generated.png")
        try savePNG(image, to: url)
        print("Wrote \(url.lastPathComponent)")
    }

    private static func aligned(_ value: Int) -> Int {
        max(16, value / 16 * 16)
    }

    private static func requireImage(_ path: String) throws -> CGImage {
        guard let image = try loadImages([URL(fileURLWithPath: path)]).first else {
            throw Flux2Error.loadFailed("Could not load \(path)")
        }
        return image
    }
}
