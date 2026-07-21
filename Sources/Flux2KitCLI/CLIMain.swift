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

/// Parse "a,b,c,d" into four ints.
private func parse4(_ s: String) -> (Int, Int, Int, Int) {
    let p = s.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
    guard p.count == 4 else { fail("expected 4 comma-separated integers, got: \(s)") }
    return (p[0], p[1], p[2], p[3])
}

/// Parse outpaint margins: "L,R,T,B" or a single value applied to all sides.
private func parseOutpaint(_ s: String) -> (Int, Int, Int, Int) {
    let p = s.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
    if p.count == 1 { return (p[0], p[0], p[0], p[0]) }
    guard p.count == 4 else { fail("--outpaint expects L,R,T,B or a single value, got: \(s)") }
    return (p[0], p[1], p[2], p[3])
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

        // Memory system.
        var lowMemory = false
        var memReport = false
        var cacheLimitMB: Int?
        var memoryLimitMB: Int?
        var vaeTile: Int?
        var residency: ResidencyPolicy = .keepResident

        // More editing + CLI expansion.
        var doImg2Img = false
        var maskBox: String?
        var maskEllipse: String?
        var maskDilate: Int?
        var maskErode: Int?
        var outpaintSpec: String?
        var pixelFilter: String?
        var sharpenAmount: Float = 1.0
        var matchColorRef: String?
        var numImages = 1
        var seedsList: [UInt64]?
        var format = "png"

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
            // Memory system.
            case "--low-memory": lowMemory = true
            case "--mem-report": memReport = true
            case "--cache-limit": cacheLimitMB = Int(next(arg) ?? "")
            case "--memory-limit": memoryLimitMB = Int(next(arg) ?? "")
            case "--vae-tile": vaeTile = Int(next(arg) ?? "")
            // More editing + CLI expansion.
            case "--img2img": doImg2Img = true
            case "--mask-box": maskBox = next(arg)
            case "--mask-ellipse": maskEllipse = next(arg)
            case "--mask-dilate": maskDilate = Int(next(arg) ?? "")
            case "--mask-erode": maskErode = Int(next(arg) ?? "")
            case "--outpaint": outpaintSpec = next(arg)
            case "--grayscale": pixelFilter = "grayscale"
            case "--sepia": pixelFilter = "sepia"
            case "--invert": pixelFilter = "invert"
            case "--sharpen": pixelFilter = "sharpen"
            case "--sharpen-amount": sharpenAmount = Float(next(arg) ?? "") ?? sharpenAmount
            case "--match-color": pixelFilter = "match-color"; matchColorRef = next(arg)
            case "--num": numImages = max(1, Int(next(arg) ?? "") ?? 1)
            case "--seeds":
                seedsList = (next(arg) ?? "").split(separator: ",").compactMap {
                    UInt64($0.trimmingCharacters(in: .whitespaces))
                }
            case "--format": format = next(arg) ?? format
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
                    --img2img                      regenerate --source from -p at --strength
                    --outpaint L,R,T,B             extend the canvas and fill the new border
                                                   (single value applies to all sides)

                  masks (any inpaint mode; no external file needed):
                    --mask FILE | --mask-box x,y,w,h | --mask-ellipse x,y,w,h   (top-left origin)
                    --mask-dilate N  --mask-erode N  --mask-feather N  --invert-mask

                  pixel filters (model-free, instant; masked if a mask is given):
                    --grayscale  --sepia  --invert  --sharpen [--sharpen-amount F]
                    --match-color REF.png          transfer REF's palette/tone to --source

                  batch / output:
                    --num N            emit N variations (seeds SEED, SEED+1, …)
                    --seeds a,b,c      explicit seed list
                    --format png|jpg   output format

                  editing options: --strength F  --invert-mask  --mask-feather N  [-s SEED]

                  memory:
                    -q int8|int4          quantize the transformer + text encoder
                    --low-memory          preset: int4 + free each model after its stage +
                                          fp16 VAE + tiled decode + 512MB cache cap
                    --mem-report          print per-stage active/cache/peak memory
                    --cache-limit MB      cap the MLX buffer cache
                    --memory-limit MB     soft memory limit (MLX evicts under pressure)
                    --vae-tile N          tiled VAE decode at latent tile size N
                """)
                exit(0)
            default:
                fail("unknown argument: \(arg)")
            }
        }

        let editActive = doRemove || addObjectPrompt != nil || replaceBgPrompt != nil
            || editPrompt != nil || recolorSpec != nil || experimentalLatentColor
            || doImg2Img || outpaintSpec != nil || pixelFilter != nil

        // --low-memory preset: int4 + staged unload + cache cap + fp16 VAE + tiled decode.
        if lowMemory {
            residency = .unloadAfterUse
            if quantize == nil { quantize = "int4" }
            vaeFp16 = true
            if cacheLimitMB == nil { cacheLimitMB = 512 }
            if vaeTile == nil { vaeTile = 64 }
        }

        // Seeds to run (multi-seed batch). Deterministic ops collapse to one below.
        let seeds: [UInt64?]
        if let seedsList { seeds = seedsList.map { Optional($0) } }
        else if numImages > 1 {
            let base = seed ?? 0
            seeds = (0 ..< numImages).map { Optional(base + UInt64($0)) }
        } else { seeds = [seed] }

        do {
            let loadStart = ProcessInfo.processInfo.systemUptime
            let pipeline = try await Flux2Pipeline(
                repoPath: URL(fileURLWithPath: repo),
                dtype: dtype,
                quantize: quantize,
                safeAttn: safeAttn,
                vaeFp16: vaeFp16,
                residency: residency,
                cacheLimitMB: cacheLimitMB,
                memoryLimitMB: memoryLimitMB,
                memReport: memReport)
            pipeline.vaeTileLatent = vaeTile
            if verbose {
                let ms = (ProcessInfo.processInfo.systemUptime - loadStart) * 1000
                print(String(format: "[%7.1fms] Pipeline load", ms))
            }

            // Load source (required by every editing operation) and derive geometry.
            var srcImg: CGImage?
            if let sourcePath {
                guard let s = try loadImages([URL(fileURLWithPath: sourcePath)]).first else {
                    fail("could not load --source image")
                }
                srcImg = s
            }
            if editActive {
                guard let s = srcImg else { fail("this operation requires --source PATH") }
                if !widthSet { width = max(16, (s.width / 16) * 16) }
                if !heightSet { height = max(16, (s.height / 16) * 16) }
            }
            let refImages = inputs.isEmpty
                ? nil : try loadImages(inputs.map { URL(fileURLWithPath: $0) })
            let feather = maskFeather ?? 1

            // Resolve the edit mask from a file, a generated box/ellipse, then dilate/erode.
            var resolvedMask: CGImage?
            if let s = srcImg {
                if let maskPath {
                    resolvedMask = try loadImages([URL(fileURLWithPath: maskPath)]).first
                } else if let spec = maskBox {
                    let r = parse4(spec)
                    resolvedMask = try makeBoxMask(
                        width: s.width, height: s.height, x: r.0, y: r.1, boxWidth: r.2, boxHeight: r.3)
                } else if let spec = maskEllipse {
                    let r = parse4(spec)
                    resolvedMask = try makeEllipseMask(
                        width: s.width, height: s.height, x: r.0, y: r.1, boxWidth: r.2, boxHeight: r.3)
                }
                if let m = resolvedMask, let d = maskDilate {
                    resolvedMask = try dilateMask(m, iterations: d)
                }
                if let m = resolvedMask, let e = maskErode {
                    resolvedMask = try erodeMask(m, iterations: e)
                }
            }
            func requireMask() -> CGImage {
                if let resolvedMask { return resolvedMask }
                fail("this mode requires --mask FILE, --mask-box x,y,w,h, or --mask-ellipse x,y,w,h")
            }

            func runOnce(_ curSeed: UInt64?) throws -> CGImage {
                if editActive {
                    guard let src = srcImg else { fail("this operation requires --source PATH") }
                    if let pixelFilter {
                        var ref: CGImage?
                        if let rp = matchColorRef {
                            ref = try loadImages([URL(fileURLWithPath: rp)]).first
                        }
                        return try pipeline.applyPixelFilter(
                            source: src, filter: pixelFilter, amount: sharpenAmount, reference: ref,
                            mask: resolvedMask, invertMask: invertMask, maskFeather: maskFeather ?? 2)
                    }
                    if let spec = outpaintSpec {
                        let (l, r, t, b) = parseOutpaint(spec)
                        return try pipeline.generateOutpaint(
                            source: src, prompt: prompt ?? "", left: l, right: r, top: t, bottom: b,
                            strength: strength ?? 0.95, numSteps: steps, guidance: guidance,
                            seed: curSeed, verbose: verbose, evalFreq: evalFreq)
                    }
                    if doImg2Img {
                        guard let p = prompt else { fail("--img2img requires -p PROMPT") }
                        return try pipeline.generateImg2Img(
                            prompt: p, source: src, strength: strength ?? 0.6,
                            width: width, height: height, numSteps: steps, guidance: guidance,
                            seed: curSeed, inputImages: refImages, verbose: verbose, evalFreq: evalFreq)
                    }
                    if let recolorSpec {
                        let rc = parseRecolor(recolorSpec)
                        if experimentalLatentColor {
                            return try pipeline.experimentalLatentColor(
                                source: src, width: width, height: height,
                                exposure: rc.exp, contrast: rc.contrast, gamma: rc.gamma)
                        }
                        return try pipeline.recolor(
                            source: src, mask: resolvedMask, hue: rc.hue, saturation: rc.sat,
                            exposure: rc.exp, contrast: rc.contrast, gamma: rc.gamma,
                            invertMask: invertMask, maskFeather: feather, verbose: verbose)
                    }
                    if experimentalLatentColor {
                        fail("--experimental-latent-color requires --recolor \"exp=..,contrast=..,gamma=..\"")
                    }
                    if doRemove {
                        return try pipeline.removeObject(
                            source: src, mask: requireMask(), strength: strength ?? 0.9,
                            width: width, height: height, numSteps: steps, guidance: guidance,
                            seed: curSeed, maskFeather: feather, verbose: verbose, evalFreq: evalFreq)
                    }
                    if let addObjectPrompt {
                        return try pipeline.addObject(
                            source: src, mask: requireMask(), prompt: addObjectPrompt,
                            referenceImage: refImages?.first, strength: strength ?? 0.85,
                            width: width, height: height, numSteps: steps, guidance: guidance,
                            seed: curSeed, maskFeather: feather, verbose: verbose, evalFreq: evalFreq)
                    }
                    if let replaceBgPrompt {
                        return try pipeline.replaceBackground(
                            source: src, subjectMask: requireMask(), prompt: replaceBgPrompt,
                            strength: strength ?? 0.9, width: width, height: height, numSteps: steps,
                            guidance: guidance, seed: curSeed, maskFeather: feather,
                            verbose: verbose, evalFreq: evalFreq)
                    }
                    if let editPrompt {
                        return try pipeline.editRegion(
                            source: src, mask: requireMask(), prompt: editPrompt,
                            strength: strength ?? 0.85, width: width, height: height, numSteps: steps,
                            guidance: guidance, seed: curSeed, invertMask: invertMask,
                            maskFeather: feather, verbose: verbose, evalFreq: evalFreq)
                    }
                    fail("no editing operation matched")
                }
                guard let p = prompt else {
                    fail("--prompt is required for text-to-image (see --help)")
                }
                return try pipeline.generate(
                    prompt: p, width: width, height: height, numSteps: steps, guidance: guidance,
                    seed: curSeed, inputImages: refImages, verbose: verbose, evalFreq: evalFreq)
            }

            // Pixel filters / experimental-latent are deterministic → a single output.
            let deterministic = pixelFilter != nil || experimentalLatentColor
            let runSeeds = deterministic ? [seeds.first ?? nil] : seeds
            let base = (output as NSString).deletingPathExtension
            for (i, s) in runSeeds.enumerated() {
                let img = try runOnce(s)
                let name = runSeeds.count > 1 ? "\(base)_\(i).\(format)" : "\(base).\(format)"
                let url = URL(fileURLWithPath: name)
                try saveImage(img, to: url, format: format)
                print("Saved \(url.path)")
            }
        } catch {
            FileHandle.standardError.write(Data("error: \(error)\n".utf8))
            exit(1)
        }
    }
}
