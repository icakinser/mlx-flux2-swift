// Flux2Kit — native MLX Swift port of FLUX.2 [klein], derived from scf4/mlx-flux2 (MIT).
// Headless CLI harness for text-to-image, image-to-image, and the editing operations.
// Requires the MLX metallib (run via build.sh-style metallib copy or DEVELOPER_DIR xcodebuild flow).

import CoreGraphics
import Flux2Kit
import Foundation

private func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data("\(message)\n".utf8))
    exit(2)
}

/// Sendable box for throttling download-progress prints across the async boundary.
private final class ProgressBox: @unchecked Sendable { var last = -1 }

// Thin wrappers over the testable library parsers: convert a thrown `CLIParseError` into a clean
// `fail()` (exit 2) so the executable keeps its existing single-line error behavior.
private func parse4(_ flag: String, _ s: String) -> (Int, Int, Int, Int) {
    do { return try parse4Arg(flag, s) } catch { fail("\(error)") }
}
private func parseWxH(_ s: String) -> (Int, Int) {
    do { return try parseWxHArg(s) } catch { fail("\(error)") }
}
private func parseOutpaint(_ s: String) -> (Int, Int, Int, Int) {
    do { return try parseOutpaintArg(s) } catch { fail("\(error)") }
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
        var guidanceEnd: Double?
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
        var useCompileFlag = false
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
        var ops: [ImageOp] = []  // ordered model-free image ops (geometry + color + effects)
        var numImages = 1
        var seedsList: [UInt64]?
        var format = "png"
        var upscale = 1
        var sampler = Sampler.euler
        var quality = 0.92

        // Weight download.
        var doDownload = false
        var downloadRepoId = defaultRepoId
        var hfToken: String?

        var args = Array(CommandLine.arguments.dropFirst())
        while !args.isEmpty {
            let arg = args.removeFirst()
            func next(_ flag: String) -> String? {
                guard !args.isEmpty else { fail("missing value for \(flag)") }
                return args.removeFirst()
            }
            // Strict numeric parsing: a malformed value is a hard error, not a silent fallback.
            func intArg() -> Int {
                do { return try parseIntArg(arg, next(arg) ?? "") } catch { fail("\(error)") }
            }
            func doubleArg() -> Double {
                do { return try parseDoubleArg(arg, next(arg) ?? "") } catch { fail("\(error)") }
            }
            func floatArg() -> Float {
                do { return try parseFloatArg(arg, next(arg) ?? "") } catch { fail("\(error)") }
            }
            func uintArg() -> UInt64 {
                do { return try parseUInt64Arg(arg, next(arg) ?? "") } catch { fail("\(error)") }
            }
            switch arg {
            case "-p", "--prompt": prompt = next(arg)
            case "-w", "--width": width = intArg(); widthSet = true
            case "-H", "--height": height = intArg(); heightSet = true
            case "-t", "--steps": steps = intArg()
            case "--guidance": guidance = doubleArg()
            case "--guidance-end": guidanceEnd = doubleArg()
            case "-s", "--seed": seed = uintArg()
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
            case "--compile": useCompileFlag = true
            case "-v", "--verbose": verbose = true
            case "--eval-freq": evalFreq = intArg()
            // Memory system.
            case "--low-memory": lowMemory = true
            case "--mem-report": memReport = true
            case "--cache-limit": cacheLimitMB = intArg()
            case "--memory-limit": memoryLimitMB = intArg()
            case "--vae-tile": vaeTile = intArg()
            // More editing + CLI expansion.
            case "--img2img": doImg2Img = true
            case "--mask-box": maskBox = next(arg)
            case "--mask-ellipse": maskEllipse = next(arg)
            case "--mask-dilate": maskDilate = intArg()
            case "--mask-erode": maskErode = intArg()
            case "--outpaint": outpaintSpec = next(arg)
            // Model-free image ops (applied in the order given; no model load).
            case "--resize": let r = parseWxH(next(arg) ?? ""); ops.append(.resize(r.0, r.1))
            case "--scale": ops.append(.scale(floatArg()))
            case "--crop": let r = parse4(arg, next(arg) ?? ""); ops.append(.crop(r.0, r.1, r.2, r.3))
            case "--rotate": ops.append(.rotate(intArg()))
            case "--flip":
                let raw = next(arg) ?? "h"
                guard let mode = FlipMode(parsing: raw) else {
                    fail("--flip must be h, v, or hv, got: \(raw)")
                }
                ops.append(.flip(mode))
            case "--fit-16": ops.append(.fit16)
            case "--pixelate": ops.append(.pixelate(intArg()))
            case "--grayscale": ops.append(.grayscale)
            case "--sepia": ops.append(.sepia)
            case "--invert": ops.append(.invert)
            case "--auto-contrast": ops.append(.autoContrast)
            case "--sharpen": ops.append(.sharpen(floatArg()))
            case "--blur": ops.append(.blur(intArg()))
            case "--brightness": ops.append(.brightness(floatArg()))
            case "--saturation": ops.append(.saturation(floatArg()))
            case "--temperature": ops.append(.temperature(floatArg()))
            case "--posterize": ops.append(.posterize(intArg()))
            case "--threshold": ops.append(.threshold(floatArg()))
            case "--vignette": ops.append(.vignette(floatArg()))
            case "--match-color": ops.append(.matchColor(next(arg) ?? ""))
            case "--num": numImages = max(1, intArg())
            case "--seeds":
                // 2026-07-26 EDT | PERMANENT — strict seed parsing, no silent drops
                let rawSeeds = (next(arg) ?? "").split(separator: ",").map {
                    $0.trimmingCharacters(in: .whitespaces)
                }
                var parsed: [UInt64] = []
                for entry in rawSeeds {
                    do {
                        parsed.append(try parseUInt64Arg("--seeds", entry))
                    } catch {
                        fail("--seeds: invalid seed value '\(entry)'")
                    }
                }
                seedsList = parsed
            case "--format": format = next(arg) ?? format
            case "--upscale": upscale = intArg()
            case "--sampler":
                let raw = next(arg) ?? "euler"
                guard let s = Sampler(rawValue: raw) else {
                    fail("--sampler must be euler or heun, got: \(raw)")
                }
                sampler = s
            case "--quality": quality = doubleArg()
            // Weight download.
            case "--download": doDownload = true
            case "--download-repo": downloadRepoId = next(arg) ?? downloadRepoId
            case "--hf-token": hfToken = next(arg)
            // Editing flags.
            case "--source": sourcePath = next(arg)
            case "--mask": maskPath = next(arg)
            case "--strength": strength = doubleArg()
            case "--invert-mask": invertMask = true
            case "--mask-feather": maskFeather = intArg()
            case "--remove": doRemove = true
            case "--add-object": addObjectPrompt = next(arg)
            case "--replace-background": replaceBgPrompt = next(arg)
            case "--edit": editPrompt = next(arg)
            case "--recolor": recolorSpec = next(arg)
            case "--experimental-latent-color": experimentalLatentColor = true
            case "-h", "--help":
                print("""
                usage:
                  text-to-image:
                    flux2kit-cli -p PROMPT [-w W] [-H H] [-t STEPS] [--guidance G]
                                 [--guidance-end G] [-s SEED]
                                 [--output OUT.png] [--repo PATH] [--input REF.png ...]
                                 [-q none|int8|int4] [--dtype bfloat16]
                                 [--vae-fp16] [--safe-attn] [--compile] [-v] [--eval-freq N]
                                 [--sampler euler|heun]

                  editing (require --source; inpaint modes also require --mask;
                           mask convention: white = region to edit, black = keep):
                    --remove                       remove the masked object, fill background
                    --add-object "PROMPT"          synthesize an object in the masked region
                    --replace-background "PROMPT"  keep masked subject, regenerate the rest
                    --edit "PROMPT"                general masked edit (also: semantic recolor)
                    --experimental-latent-color    with --recolor: latent-space A/B (unreliable)
                    --img2img                      regenerate --source from -p at --strength
                    --outpaint L,R,T,B             extend the canvas and fill the new border
                                                   (single value applies to all sides)

                  masks (any inpaint mode; no external file needed):
                    --mask FILE | --mask-box x,y,w,h | --mask-ellipse x,y,w,h   (top-left origin)
                    --mask-dilate N  --mask-erode N  --mask-feather N  --invert-mask

                  model-free image ops (NO model load — instant; applied in the order given, either
                  standalone on --source or as post-processing after a generate/edit):
                    geometry: --resize WxH  --scale F  --crop x,y,w,h  --rotate 90|180|270
                              --flip h|v|hv  --fit-16  --pixelate N
                    color:    --brightness F  --saturation F  --temperature F  --auto-contrast
                              --recolor "hue=..,sat=..,exp=..,contrast=..,gamma=.."
                              --match-color REF.png
                    effects:  --grayscale  --sepia  --invert  --sharpen F  --blur N
                              --posterize N  --threshold F  --vignette F

                  batch / output:
                    --num N            emit N variations (seeds SEED, SEED+1, …)
                    --seeds a,b,c      explicit seed list
                    --format png|jpg   output format
                    --upscale N        upscale output by integer factor N (1-8, default 1)
                    --sampler S        denoising integrator: euler (default) or heun (2x passes,
                                       smoother low-step output)
                    --guidance-end F   linearly decay/ramp guidance to F (experimental)
                    --quality F        JPEG quality 0.0-1.0 (default 0.92; ignored for PNG)

                  editing options: --strength F  --invert-mask  --mask-feather N  [-s SEED]

                  weights:
                    --download            fetch the FLUX.2 [klein] snapshot from Hugging Face
                                          (~15 GB; opt-in). Use alone to just download.
                    --download-repo ID    override the Hub repo id
                    --hf-token TOKEN      HF token for gated repos (or set HF_TOKEN)

                  memory:
                    -q int8|int4          quantize the transformer + text encoder
                    --compile             enable mx.compile for 30-50% per-step speedup after warmup
                    --low-memory          preset: int4 + free each model after its stage + fp16 VAE
                    --mem-report          print per-stage RSS / MLX active / peak memory
                    --cache-limit MB      cap the MLX buffer cache
                    --memory-limit MB     soft memory limit (MLX evicts under pressure)
                    --vae-tile N          tiled VAE decode at latent tile size N (lossy; large images)
                """)
                exit(0)
            default:
                fail("unknown argument: \(arg)")
            }
        }

        let diffusionActive = doRemove || addObjectPrompt != nil || replaceBgPrompt != nil
            || editPrompt != nil || experimentalLatentColor || doImg2Img || outpaintSpec != nil

        // --low-memory preset: int4 + staged unload + fp16 VAE + cache cap. Tiling is NOT auto-
        // enabled (it is lossy — FLUX's VAE has global attention); opt in with --vae-tile.
        if lowMemory {
            residency = .unloadAfterUse
            if quantize == nil { quantize = "int4" }
            vaeFp16 = true
            if cacheLimitMB == nil { cacheLimitMB = 512 }
        }

        // Validate enum-like flags up front with clear errors (rather than deep in the pipeline or,
        // for quantize, silently ignoring an unknown mode).
        guard ["png", "jpg", "jpeg"].contains(format.lowercased()) else {
            fail("--format must be png or jpg, got: \(format)")
        }
        if let q = quantize, q != "int8", q != "int4" {
            fail("--quantize must be none, int8, or int4, got: \(q)")
        }
        guard upscale >= 1, upscale <= 8 else {
            fail("--upscale must be 1-8, got: \(upscale)")
        }
        guard quality >= 0.0, quality <= 1.0 else {
            fail("--quality must be 0.0-1.0, got: \(quality)")
        }
        let outputDir = URL(fileURLWithPath: output).deletingLastPathComponent()
        if !outputDir.path.isEmpty,
            !FileManager.default.fileExists(atPath: outputDir.path)
        {
            fail("output directory does not exist: \(outputDir.path)")
        }

        // Non-experimental --recolor is a model-free pixel op; fold it into the op chain.
        if let spec = recolorSpec, !experimentalLatentColor {
            let rc = parseRecolorArg(spec)
            for w in rc.warnings {
                FileHandle.standardError.write(Data("warning: \(w)\n".utf8))
            }
            ops.append(
                .recolor(hue: rc.hue, sat: rc.sat, exp: rc.exp, contrast: rc.contrast, gamma: rc.gamma))
            recolorSpec = nil
        }

        // Warnings for ignored/conflicting flags.
        if seedsList != nil && numImages > 1 {
            FileHandle.standardError.write(
                Data("warning: --num ignored because --seeds was given\n".utf8))
        }

        // Opt-in weight download from the Hugging Face Hub.
        if doDownload {
            do {
                print("Downloading \(downloadRepoId) from Hugging Face (~15 GB; this can take a while)…")
                let tracker = ProgressBox()
                let url = try await downloadFluxSnapshot(
                    repoId: downloadRepoId,
                    hfToken: hfToken ?? ProcessInfo.processInfo.environment["HF_TOKEN"]
                ) { frac in
                    let pct = Int(frac * 100)
                    if pct != tracker.last, pct % 5 == 0 {
                        tracker.last = pct
                        FileHandle.standardError.write(Data("  \(pct)%\n".utf8))
                    }
                }
                repo = url.path
                print("Downloaded to \(url.path)")
            } catch {
                FileHandle.standardError.write(Data("error: \(error)\n".utf8))
                exit(1)
            }
            // If download was the only request, we're done.
            if ops.isEmpty && !diffusionActive && prompt == nil { return }
        }

        // MODEL-FREE FAST PATH: only geometry/color/effect ops on a source → no pipeline, no model.
        if !ops.isEmpty && !diffusionActive && prompt == nil {
            guard let sourcePath else { fail("image ops require --source PATH") }
            do {
                guard let src = try loadImages([URL(fileURLWithPath: sourcePath)]).first else {
                    fail("could not load --source image")
                }
                let out = try applyImageOps(src, ops)
                let url = URL(
                    fileURLWithPath: "\((output as NSString).deletingPathExtension).\(format)")
                try saveImage(out, to: url, format: format, quality: quality)
                print("Saved \(url.path)")
            } catch {
                FileHandle.standardError.write(Data("error: \(error)\n".utf8))
                exit(1)
            }
            return
        }

        // Seeds to run (multi-seed batch). Deterministic ops collapse to one below.
        // 2026-07-26 EDT | PERMANENT — deterministic-per-run seeding for reproducibility
        let seeds: [UInt64?]
        if let seedsList {
            seeds = seedsList.map { Optional($0) }
        } else if let seed {
            // Explicit base seed: derive a contiguous run.
            seeds = (0 ..< numImages).map { Optional(seed + UInt64($0)) }
        } else {
            // No seed supplied: draw a fresh random base so each unseeded run is independent,
            // then surface it so the result is reproducible by re-passing -s.
            let base = UInt64.random(in: 0 ... UInt64.max)
            if verbose {
                if numImages > 1 {
                    print("Using random base seed: \(base) (seeds \(base)...\(base + UInt64(numImages - 1)))")
                } else {
                    print("Using seed: \(base)")
                }
            }
            seeds = (0 ..< numImages).map { Optional(base + UInt64($0)) }
        }

        do {
            let loadStart = ProcessInfo.processInfo.systemUptime
            let pipeline = try await Flux2Pipeline(
                repoPath: URL(fileURLWithPath: repo),
                dtype: dtype,
                quantize: quantize,
                safeAttn: safeAttn,
                vaeFp16: vaeFp16,
                compile: useCompileFlag,
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
            if diffusionActive {
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
                    let r = parse4("--mask-box", spec)
                    resolvedMask = try makeBoxMask(
                        width: s.width, height: s.height, x: r.0, y: r.1, boxWidth: r.2, boxHeight: r.3)
                } else if let spec = maskEllipse {
                    let r = parse4("--mask-ellipse", spec)
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
                if diffusionActive {
                    guard let src = srcImg else { fail("this operation requires --source PATH") }
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
                    if experimentalLatentColor {
                        guard let spec = recolorSpec else {
                            fail("--experimental-latent-color requires --recolor \"exp=..,contrast=..,gamma=..\"")
                        }
                        let rc = parseRecolorArg(spec)
                        for w in rc.warnings {
                            FileHandle.standardError.write(Data("warning: \(w)\n".utf8))
                        }
                        return try pipeline.experimentalLatentColor(
                            source: src, width: width, height: height,
                            exposure: rc.exp, contrast: rc.contrast, gamma: rc.gamma)
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
                    seed: curSeed, inputImages: refImages, verbose: verbose, evalFreq: evalFreq,
                    sampler: sampler,
                    guidanceSchedule: guidanceEnd.map { .linear(end: $0) } ?? .constant)
            }

            // experimental-latent is deterministic → a single output.
            let runSeeds = experimentalLatentColor ? [seeds.first ?? nil] : seeds
            let base = (output as NSString).deletingPathExtension
            for (i, s) in runSeeds.enumerated() {
                var img = try runOnce(s)
                if !ops.isEmpty { img = try applyImageOps(img, ops) }  // model-free post-processing
                // 2026-07-26 EDT | PERMANENT — Lanczos-equivalent upscale for game asset workflows
                if upscale > 1 {
                    img = try resizeHighQuality(img, width: img.width * upscale, height: img.height * upscale)
                }
                let name = runSeeds.count > 1 ? "\(base)_\(i).\(format)" : "\(base).\(format)"
                let url = URL(fileURLWithPath: name)
                try saveImage(img, to: url, format: format, quality: quality)
                print("Saved \(url.path)")
            }
        } catch {
            FileHandle.standardError.write(Data("error: \(error)\n".utf8))
            // Point the user at how to obtain the weights when they can't be located.
            if case Flux2Error.loadFailed = error {
                FileHandle.standardError.write(Data("\n\(weightsHelpMessage())\n".utf8))
            }
            exit(1)
        }
    }
}
