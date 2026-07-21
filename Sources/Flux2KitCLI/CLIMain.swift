// Flux2Kit — native MLX Swift port of FLUX.2 [klein], derived from scf4/mlx-flux2 (MIT).
// Headless CLI harness for text-to-image, image-to-image, and the editing operations.
// Requires the MLX metallib (run via build.sh-style metallib copy or DEVELOPER_DIR xcodebuild flow).

import CoreGraphics
import Flux2Kit
import Foundation

/// Parse a `--recolor` spec like "hue=0.2,sat=1.1,exp=0.3,contrast=1.1,gamma=1.0".
private func parseRecolor(_ s: String)
    -> (hue: Float, sat: Float, exp: Float, contrast: Float, gamma: Float)
{
    var hue: Float = 0, sat: Float = 1, exp: Float = 0, contrast: Float = 1, gamma: Float = 1
    for part in s.split(separator: ",") {
        let kv = part.split(separator: "=")
        guard kv.count == 2,
            let val = Float(kv[1].trimmingCharacters(in: .whitespaces))
        else { continue }
        switch kv[0].trimmingCharacters(in: .whitespaces).lowercased() {
        case "hue": hue = val
        case "sat", "saturation": sat = val
        case "exp", "exposure": exp = val
        case "contrast": contrast = val
        case "gamma": gamma = val
        default: break
        }
    }
    return (hue, sat, exp, contrast, gamma)
}

private func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data("\(message)\n".utf8))
    exit(2)
}

@main
struct Flux2KitCLI {
    static func main() async {
        var prompt: String?
        var width = defaultWidth
        var height = defaultHeight
        var widthSet = false
        var heightSet = false
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

        // Editing options.
        var sourcePath: String?
        var maskPath: String?
        var strength: Double?
        var invertMask = false
        var maskFeather: Int?
        var doRemove = false
        var addObjectPrompt: String?
        var replaceBgPrompt: String?
        var editPrompt: String?
        var recolorSpec: String?
        var experimentalLatentColor = false

        var args = Array(CommandLine.arguments.dropFirst())
        while !args.isEmpty {
            let arg = args.removeFirst()
            func next(_ flag: String) -> String? {
                guard !args.isEmpty else { fail("missing value for \(flag)") }
                return args.removeFirst()
            }
            switch arg {
            case "-p", "--prompt": prompt = next(arg)
            case "-w", "--width": width = Int(next(arg) ?? "") ?? width; widthSet = true
            case "-h", "--height": height = Int(next(arg) ?? "") ?? height; heightSet = true
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
            // Editing flags.
            case "--source": sourcePath = next(arg)
            case "--mask": maskPath = next(arg)
            case "--strength": strength = Double(next(arg) ?? "")
            case "--invert-mask": invertMask = true
            case "--mask-feather": maskFeather = Int(next(arg) ?? "")
            case "--remove": doRemove = true
            case "--add-object": addObjectPrompt = next(arg)
            case "--replace-background": replaceBgPrompt = next(arg)
            case "--edit": editPrompt = next(arg)
            case "--recolor": recolorSpec = next(arg)
            case "--experimental-latent-color": experimentalLatentColor = true
            case "--help":
                print("""
                usage:
                  text-to-image:
                    flux2kit-cli -p PROMPT [-w W] [-h H] [-t STEPS] [--guidance G] [-s SEED]
                                 [--output OUT.png] [--repo PATH] [--input REF.png ...]
                                 [-q none|int8|int4] [--dtype float16|bfloat16]
                                 [--vae-fp16] [--safe-attn] [-v] [--eval-freq N]

                  editing (require --source; inpaint modes also require --mask;
                           mask convention: white = region to edit, black = keep):
                    --remove                       remove the masked object, fill background
                    --add-object "PROMPT"          synthesize an object in the masked region
                    --replace-background "PROMPT"  keep masked subject, regenerate the rest
                    --edit "PROMPT"                general masked edit (also: semantic recolor)
                    --recolor "hue=..,sat=..,exp=..,contrast=..,gamma=.."
                                                   exact pixel-space grade (masked if --mask given)
                    --experimental-latent-color    with --recolor: latent-space A/B (unreliable)

                  editing options: --strength F  --invert-mask  --mask-feather N  [-s SEED]
                """)
                exit(0)
            default:
                fail("unknown argument: \(arg)")
            }
        }

        let editActive = doRemove || addObjectPrompt != nil || replaceBgPrompt != nil
            || editPrompt != nil || recolorSpec != nil || experimentalLatentColor

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

            let outputURL = URL(fileURLWithPath: output)

            if editActive {
                guard let sourcePath else { fail("editing requires --source PATH") }
                guard let srcImg = try loadImages([URL(fileURLWithPath: sourcePath)]).first else {
                    fail("could not load --source image")
                }
                if !widthSet { width = max(16, (srcImg.width / 16) * 16) }
                if !heightSet { height = max(16, (srcImg.height / 16) * 16) }
                let refImages = inputs.isEmpty
                    ? nil : try loadImages(inputs.map { URL(fileURLWithPath: $0) })
                let feather = maskFeather ?? 1

                func loadMask() throws -> CGImage {
                    guard let maskPath else { fail("this mode requires --mask PATH") }
                    guard let m = try loadImages([URL(fileURLWithPath: maskPath)]).first else {
                        fail("could not load --mask image")
                    }
                    return m
                }

                let result: CGImage
                if let recolorSpec {
                    let rc = parseRecolor(recolorSpec)
                    if experimentalLatentColor {
                        result = try pipeline.experimentalLatentColor(
                            source: srcImg, width: width, height: height,
                            exposure: rc.exp, contrast: rc.contrast, gamma: rc.gamma)
                    } else {
                        let maskImg = maskPath == nil ? nil : try loadMask()
                        result = try pipeline.recolor(
                            source: srcImg, mask: maskImg,
                            hue: rc.hue, saturation: rc.sat, exposure: rc.exp,
                            contrast: rc.contrast, gamma: rc.gamma,
                            invertMask: invertMask, maskFeather: feather, verbose: verbose)
                    }
                } else if experimentalLatentColor {
                    fail("--experimental-latent-color requires --recolor \"exp=..,contrast=..,gamma=..\"")
                } else if doRemove {
                    result = try pipeline.removeObject(
                        source: srcImg, mask: try loadMask(),
                        strength: strength ?? 0.9, width: width, height: height,
                        numSteps: steps, guidance: guidance, seed: seed,
                        maskFeather: feather, verbose: verbose, evalFreq: evalFreq)
                } else if let addObjectPrompt {
                    result = try pipeline.addObject(
                        source: srcImg, mask: try loadMask(), prompt: addObjectPrompt,
                        referenceImage: refImages?.first,
                        strength: strength ?? 0.85, width: width, height: height,
                        numSteps: steps, guidance: guidance, seed: seed,
                        maskFeather: feather, verbose: verbose, evalFreq: evalFreq)
                } else if let replaceBgPrompt {
                    result = try pipeline.replaceBackground(
                        source: srcImg, subjectMask: try loadMask(), prompt: replaceBgPrompt,
                        strength: strength ?? 0.9, width: width, height: height,
                        numSteps: steps, guidance: guidance, seed: seed,
                        maskFeather: feather, verbose: verbose, evalFreq: evalFreq)
                } else {
                    // editPrompt
                    result = try pipeline.editRegion(
                        source: srcImg, mask: try loadMask(), prompt: editPrompt ?? "",
                        strength: strength ?? 0.85, width: width, height: height,
                        numSteps: steps, guidance: guidance, seed: seed,
                        invertMask: invertMask, maskFeather: feather,
                        verbose: verbose, evalFreq: evalFreq)
                }

                try savePNG(result, to: outputURL)
                print("Saved \(outputURL.path)")
                return
            }

            // Text-to-image (default path).
            guard let prompt else {
                fail("error: --prompt is required for text-to-image (see --help)")
            }
            let refImages = inputs.isEmpty
                ? nil : try loadImages(inputs.map { URL(fileURLWithPath: $0) })
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
            try savePNG(image, to: outputURL)
            print("Saved \(outputURL.path)")
        } catch {
            FileHandle.standardError.write(Data("error: \(error)\n".utf8))
            exit(1)
        }
    }
}
