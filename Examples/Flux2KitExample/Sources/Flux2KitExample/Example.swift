// Minimal Flux2Kit usage example.
//
//   swift run Flux2KitExample process <image.png>       # model-free ops — no weights, instant
//   FLUX2_REPO=/path/to/FLUX-2 swift run Flux2KitExample # text-to-image (needs weights + metallib)

import Flux2Kit
import Foundation

@main
struct Example {
    static func main() async throws {
        let args = CommandLine.arguments

        // 1) Model-free image processing — no model, no weights, no metallib (pure CoreGraphics).
        if args.count >= 3, args[1] == "process" {
            guard let src = try loadImages([URL(fileURLWithPath: args[2])]).first else {
                print("Could not load \(args[2])")
                return
            }
            // Ops run in order. Add color/effects too (e.g. .grayscale, .vignette(0.4)) — those use
            // MLX and need the Metal shader library at runtime; geometry ops need nothing.
            let out = try applyImageOps(src, [.resize(384, 384), .rotate(90)])
            let url = URL(fileURLWithPath: "example-processed.png")
            try savePNG(out, to: url)
            print("Wrote \(url.lastPathComponent) (resized to 384x384, rotated 90°)")
            return
        }

        // 2) Text-to-image — needs a FLUX.2 [klein] diffusers snapshot at $FLUX2_REPO.
        guard let repo = ProcessInfo.processInfo.environment["FLUX2_REPO"] else {
            print("""
            usage:
              swift run Flux2KitExample process <image.png>        # model-free, no weights needed
              FLUX2_REPO=/path/to/FLUX-2 swift run Flux2KitExample # text-to-image
            """)
            return
        }

        print("Loading FLUX.2 from \(repo) …")
        let pipeline = try await Flux2Pipeline(repoPath: URL(fileURLWithPath: repo))
        let image = try pipeline.generate(
            prompt: "a red bicycle leaning on a brick wall, golden hour",
            width: 512, height: 512, numSteps: 4, seed: 42)
        let url = URL(fileURLWithPath: "example-generated.png")
        try savePNG(image, to: url)
        print("Wrote \(url.lastPathComponent)")
    }
}
