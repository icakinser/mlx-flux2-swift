// Flux2Kit — native MLX Swift port of FLUX.2 [klein], derived from scf4/mlx-flux2 (MIT).
// Headless CLI harness for the text-to-image / image-to-image pipeline.
// Requires the MLX metallib (run via build.sh-style metallib copy or DEVELOPER_DIR xcodebuild flow).

import CoreGraphics
import Flux2Kit
import Foundation

@main
struct Flux2KitCLI {
    static func main() async {
        var prompt: String?
        var width = defaultWidth
        var height = defaultHeight
        var steps = defaultSteps
        var guidance = Double(defaultGuidance)
        var seed: UInt64?
        var output = defaultOutput
        // Path to the FLUX.2 diffusers snapshot. Override with --repo or the
        // FLUX2_REPO environment variable; defaults to ./Models/FLUX-2.
        var repo = ProcessInfo.processInfo.environment["FLUX2_REPO"] ?? "./Models/FLUX-2"
        var inputs: [String] = []
        var quantize: String?
        var dtype = defaultDtype
        var vaeFp16 = false
        var safeAttn = false
        var verbose = false
        var evalFreq = 1

        var args = Array(CommandLine.arguments.dropFirst())
        while !args.isEmpty {
            let arg = args.removeFirst()
            func next(_ flag: String) -> String? {
                guard !args.isEmpty else {
                    FileHandle.standardError.write(Data("missing value for \(flag)\n".utf8))
                    exit(2)
                }
                return args.removeFirst()
            }
            switch arg {
            case "-p", "--prompt": prompt = next(arg)
            case "-w", "--width": width = Int(next(arg) ?? "") ?? width
            case "-h", "--height": height = Int(next(arg) ?? "") ?? height
            case "-t", "--steps": steps = Int(next(arg) ?? "") ?? steps
            case "--guidance": guidance = Double(next(arg) ?? "") ?? guidance
            case "-s", "--seed": seed = UInt64(next(arg) ?? "")
            case "--output": output = next(arg) ?? output
            case "--repo": repo = next(arg) ?? repo
            case "--input":
                while let first = args.first, !first.hasPrefix("-") {
                    inputs.append(args.removeFirst())
                }
            case "-q", "--quantize":
                let q = next(arg)
                quantize = (q == "none") ? nil : q
            case "--dtype": dtype = next(arg) ?? dtype
            case "--vae-fp16": vaeFp16 = true
            case "--safe-attn": safeAttn = true
            case "-v", "--verbose": verbose = true
            case "--eval-freq": evalFreq = Int(next(arg) ?? "") ?? evalFreq
            case "--help":
                print("""
                usage: flux2kit-cli -p PROMPT [-w WIDTH] [-h HEIGHT] [-t STEPS] [--guidance G]
                                    [-s SEED] [--output OUT.png] [--repo LOCAL_PATH]
                                    [--input REF.png ...] [-q none|int8|int4]
                                    [--dtype float16|bfloat16] [--vae-fp16] [--safe-attn]
                                    [-v] [--eval-freq N]
                """)
                exit(0)
            default:
                FileHandle.standardError.write(Data("unknown argument: \(arg)\n".utf8))
                exit(2)
            }
        }

        guard let prompt else {
            FileHandle.standardError.write(Data("error: --prompt is required (see --help)\n".utf8))
            exit(2)
        }

        do {
            let loadStart = ProcessInfo.processInfo.systemUptime
            let pipeline = try await Flux2Pipeline(
                repoPath: URL(fileURLWithPath: repo),
                dtype: dtype,
                quantize: quantize,
                safeAttn: safeAttn,
                vaeFp16: vaeFp16)
            if verbose {
                let ms = (ProcessInfo.processInfo.systemUptime - loadStart) * 1000
                print(String(format: "[%7.1fms] Pipeline load", ms))
            }

            let refImages = inputs.isEmpty
                ? nil
                : try loadImages(inputs.map { URL(fileURLWithPath: $0) })

            let image = try pipeline.generate(
                prompt: prompt,
                width: width,
                height: height,
                numSteps: steps,
                guidance: guidance,
                seed: seed,
                inputImages: refImages,
                verbose: verbose,
                evalFreq: evalFreq)

            let outputURL = URL(fileURLWithPath: output)
            try savePNG(image, to: outputURL)
            print("Saved \(outputURL.path)")
        } catch {
            FileHandle.standardError.write(Data("error: \(error)\n".utf8))
            exit(1)
        }
    }
}
