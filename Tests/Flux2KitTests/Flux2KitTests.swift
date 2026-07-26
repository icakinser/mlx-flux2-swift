// Flux2Kit — golden parity tests against the reference implementation.
// Golden values were captured once from the reference and are checked in here (see
// Fixtures/README.md). CPU-only: no MLX GPU ops (swift test runs without the Cmlx
// metallib), so these cover the tokenizer/template contract, not tensors.

import CoreGraphics
import Foundation
import MLX
import MLXNN
import Testing
@testable import Flux2Kit

// Path to the FLUX.2 diffusers snapshot. Set FLUX2_REPO to run the tokenizer
// parity tests; otherwise they self-skip via the `tokenizerAvailable` guard.
private let repoPath = URL(fileURLWithPath:
    ProcessInfo.processInfo.environment["FLUX2_REPO"] ?? "./Models/FLUX-2")

private var tokenizerAvailable: Bool {
    FileManager.default.fileExists(
        atPath: repoPath.appendingPathComponent("tokenizer/tokenizer.json").path)
}

// The editing tests below exercise MLX array math, which requires the Metal shader library
// (`default.metallib`). A plain `swift build`/`swift test` does not produce it, so those tests are
// opt-in: set FLUX2_RUN_MLX_TESTS=1 after building through the xcodebuild/metallib flow. Without it,
// they skip cleanly (a fresh `swift test` stays green). The tokenizer tests likewise skip unless
// FLUX2_REPO points at a snapshot on disk.
private let mlxTestsEnabled = ProcessInfo.processInfo.environment["FLUX2_RUN_MLX_TESTS"] != nil

@Test(.enabled(if: tokenizerAvailable)) func chatTemplateMatchesReferenceRender() async throws {
    // Golden: repr of the reference Qwen3 chat template applied to 'TESTPROMPT'.
    let expected = "<|im_start|>user\nTESTPROMPT<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    let tok = try await Qwen3Tokenizer.fromRepo(repoPath)
    #expect(tok.applyChatTemplate("TESTPROMPT") == expected)
}

@Test(.enabled(if: tokenizerAvailable)) func specialTokenIdsMatchGolden() async throws {
    let tok = try await Qwen3Tokenizer.fromRepo(repoPath)
    // Golden: pad_id/eos_id from the reference tokenizer.
    #expect(tok.padId == 151643)
    #expect(tok.eosId == 151645)
}

@Test(.enabled(if: tokenizerAvailable)) func tokenIdsMatchGolden() async throws {
    let tok = try await Qwen3Tokenizer.fromRepo(repoPath)
    // Golden: reference encode_batch(['a red bicycle'], max_length=512) unpadded prefix.
    let goldenPrefix = [
        151644, 872, 198, 64, 2518, 34986, 151645, 198, 151644, 77091, 198, 151667, 271, 151668,
        271,
    ]
    let text = tok.applyChatTemplate("a red bicycle")
    let ids = tok.tokenizer.encode(text: text, addSpecialTokens: false)
    #expect(Array(ids.prefix(goldenPrefix.count)) == goldenPrefix)
    // The reference pads to ceil(15/64)*64 = 64 with pad_id; mirror the arithmetic (host-side check).
    let targetLen = min(512, ((ids.count + 63) / 64) * 64)
    #expect(targetLen == 64)
}

// MARK: - Editing: latent + color unit tests (CPU-only, no model weights)

// These exercise MLX array math, which dispatches to Metal by default and would need the metallib
// under `swift test`. Pin the device to CPU (ops bind their stream at creation, so build AND eval
// inside the scope) so the tests run standalone without the metallib.
private func onCPU(_ body: () -> Void) {
    Device.withDefaultDevice(Device(.cpu), body)
}

private func onCPUThrows(_ body: () throws -> Void) throws {
    try Device.withDefaultDevice(Device(.cpu), body)
}

/// The mask is tokenized with the same `prcImg` raster order as the source latent, so a mask token
/// at sequence index i corresponds to source token i. This alignment is the whole basis of the blend.
@Test(.enabled(if: mlxTestsEnabled)) func maskTokensFollowRasterOrder() {
    onCPU {
        let h = 3, w = 4
        let n = h * w
        let grid = MLXArray((0 ..< n).map { Float($0) }, [1, h, w])
        let (tok, _) = prcImg(grid)  // (N, 1)
        MLX.eval(tok)
        let vals = tok.reshaped([n]).asArray(Float.self)
        #expect(vals == (0 ..< n).map { Float($0) })
    }
}

/// The inpaint blend is `img * editMask + keep * (1 - editMask)` with editMask `(1,N,1)`, img the
/// CFG-doubled `(2,N,C)` batch, and keep `(1,N,C)`. Verify the broadcast keeps the edit region equal
/// to `img` and forces the keep region to `keep` identically across both CFG halves.
@Test(.enabled(if: mlxTestsEnabled)) func blendBroadcastsOverCfgBatch() {
    onCPU {
        let n = 5, c = 2
        let editMask = MLXArray([1, 1, 0, 0, 0].map { Float($0) }, [1, n, 1])
        let img = MLXArray((0 ..< (2 * n * c)).map { Float($0) }, [2, n, c])
        let keep = MLXArray((0 ..< (n * c)).map { Float(100 + $0) }, [1, n, c])
        let blended = img * editMask + keep * (1 - editMask)
        MLX.eval(blended)
        let arr = blended.asArray(Float.self)
        for b in 0 ..< 2 {
            for t in 0 ..< n {
                for ch in 0 ..< c {
                    let idx = ((b * n) + t) * c + ch
                    if t < 2 {
                        #expect(arr[idx] == Float(idx))  // edit region: untouched img value
                    } else {
                        #expect(arr[idx] == Float(100 + (t * c + ch)))  // keep region: broadcast
                    }
                }
            }
        }
    }
}

/// The full step schedule is rescaled into `[strength, 0]` (no truncation), preserving monotonicity.
@Test(.enabled(if: mlxTestsEnabled)) func scheduleRescaleIntoStrengthWindow() throws {
    try onCPUThrows {
        let full = getSchedule(4, 256)
        #expect(full.count == 5)
        let firstFull = try #require(full.first)
        #expect(firstFull > 0)
        for i in 1 ..< full.count { #expect(full[i] <= full[i - 1]) }
        let s = 0.6
        let rescaled = full.map { $0 * s }
        let firstRescaled = try #require(rescaled.first)
        #expect(abs(firstRescaled - firstFull * s) < 1e-12)
        #expect(firstRescaled < firstFull)
        for i in 1 ..< rescaled.count { #expect(rescaled[i] <= rescaled[i - 1]) }
    }
}

/// RGB → HSV → RGB is an identity (within float tolerance), including gray/black/white edge cases.
@Test(.enabled(if: mlxTestsEnabled)) func hsvRoundTrip() {
    onCPU {
        let rgb = MLXArray(
            [0.2, 0.5, 0.8, 0.9, 0.1, 0.3, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0].map { Float($0) }, [2, 2, 3])
        let (h, s, v) = rgbToHsv(rgb)
        let back = hsvToRgb(h, s, v)
        MLX.eval(back)
        let a = rgb.asArray(Float.self)
        let b = back.asArray(Float.self)
        var maxDiff: Float = 0
        for i in 0 ..< a.count { maxDiff = max(maxDiff, abs(a[i] - b[i])) }
        #expect(maxDiff < 1e-4)
    }
}

/// Pixel-space curves match closed-form values and are identity at their neutral parameters.
@Test(.enabled(if: mlxTestsEnabled)) func colorCurvesClosedForm() {
    onCPU {
        let x = MLXArray([0.25, 0.75, 0.25].map { Float($0) }, [1, 1, 3])

        let ev = applyExposure(x, stops: 1).asArray(Float.self)  // *2, clipped
        #expect(abs(ev[0] - 0.5) < 1e-5)
        #expect(abs(ev[1] - 1.0) < 1e-5)

        let gv = applyGamma(x, 2).asArray(Float.self)  // ^(1/2): sqrt(0.25)=0.5
        #expect(abs(gv[0] - 0.5) < 1e-5)

        let cv = applyContrast(x, 2).asArray(Float.self)  // (x-0.5)*2+0.5
        #expect(abs(cv[0] - 0.0) < 1e-5)
        #expect(abs(cv[1] - 1.0) < 1e-5)

        let idv = adjustColor(x, exposure: 0, contrast: 1, gamma: 1, hue: 0, saturation: 1)
            .asArray(Float.self)
        let xv = x.asArray(Float.self)
        for i in 0 ..< xv.count { #expect(abs(idv[i] - xv[i]) < 1e-6) }
    }
}

/// Feathering: a uniform mask is unchanged; a step edge blurs into a monotonic ramp bounded in [0,1].
@Test(.enabled(if: mlxTestsEnabled)) func boxBlurFeatherProperties() {
    onCPU {
        let ones = MLXArray([Float](repeating: 1, count: 16), [4, 4])
        let blurredOnes = boxBlur(ones, passes: 3)
        MLX.eval(blurredOnes)
        for v in blurredOnes.asArray(Float.self) { #expect(abs(v - 1) < 1e-5) }

        var vals = [Float](repeating: 0, count: 16)
        for r in 0 ..< 4 { for c in 0 ..< 4 { vals[r * 4 + c] = c < 2 ? 0 : 1 } }
        let step = MLXArray(vals, [4, 4])
        let bv = boxBlur(step, passes: 1).asArray(Float.self)
        for v in bv { #expect(v >= -1e-6 && v <= 1 + 1e-6) }
        #expect(bv[0] <= bv[1] + 1e-6)
        #expect(bv[1] <= bv[2] + 1e-6)
        #expect(bv[2] <= bv[3] + 1e-6)
        #expect(bv[1] > 1e-4)  // boundary actually feathered
    }
}

// MARK: - Memory system unit tests

/// The quantization filter quantizes big group-aligned Linear matmuls and skips the adaLN
/// `[SiLU, Linear]` container (which crashes MLXNN.quantize) and dim-misaligned / non-Linear layers.
@Test(.enabled(if: mlxTestsEnabled)) func quantFilterSelectsBigLinears() {
    onCPU {
        let f = flux2QuantFilter(groupSize: 64)
        #expect(f("transformer_blocks.0.attn.to_q", Linear(64, 128)))  // aligned Linear -> yes
        #expect(!f("last_layer.adaLN_modulation.1", Linear(3072, 6144)))  // adaLN -> no
        #expect(!f("x.proj", Linear(10, 20)))  // dims not %64 -> no
        #expect(!f("x.act", SiLU()))  // not a Linear -> no
    }
}

/// The tile feather ramp is flat 1 in the middle, ramps 0→1 at interior edges, and stays flat 1 at
/// borders (edgeStart/edgeEnd false) so image boundaries are not darkened.
@Test(.enabled(if: mlxTestsEnabled)) func featherRampShape() {
    onCPU {
        let r = featherRamp(10, 3, edgeStart: true, edgeEnd: true).asArray(Float.self)
        #expect(abs(r[5] - 1) < 1e-6)  // middle flat
        #expect(r[0] < r[1])
        #expect(r[1] < r[2])
        #expect(abs(r[3] - 1) < 1e-6)  // ramp ends at 1
        #expect(r[9] < r[8])
        #expect(r[8] < r[7])

        let flat = featherRamp(10, 3, edgeStart: false, edgeEnd: false).asArray(Float.self)
        for v in flat { #expect(abs(v - 1) < 1e-6) }  // no feather at borders
    }
}

// MARK: - More editing features

/// A generated box mask is white inside the box, black outside, at the right coverage fraction.
@Test(.enabled(if: mlxTestsEnabled)) func boxMaskCoverage() throws {
    let img = try makeBoxMask(width: 100, height: 100, x: 20, y: 20, boxWidth: 40, boxHeight: 40)
    try onCPUThrows {
        let grid = try maskGridFromCGImage(img, width: 100, height: 100)  // (100,100) in [0,1]
        MLX.eval(grid)
        #expect(grid[40, 40].item(Float.self) > 0.5)  // inside the box
        #expect(grid[5, 5].item(Float.self) < 0.5)  // outside
        let mean = MLX.mean(grid).item(Float.self)
        #expect(abs(mean - 0.16) < 0.02)  // 40*40 / 100*100
    }
}

/// Dilation grows the white region; erosion shrinks it.
@Test(.enabled(if: mlxTestsEnabled)) func dilateErodeChangeArea() throws {
    let box = try makeBoxMask(width: 100, height: 100, x: 30, y: 30, boxWidth: 40, boxHeight: 40)
    let dil = try dilateMask(box, iterations: 3)
    let ero = try erodeMask(box, iterations: 3)
    try onCPUThrows {
        func mean(_ img: CGImage) throws -> Float {
            let g = try maskGridFromCGImage(img, width: 100, height: 100)
            return MLX.mean(g).item(Float.self)
        }
        let base = try mean(box)
        #expect(try mean(dil) > base)
        #expect(try mean(ero) < base)
    }
}

/// Reinhard color match moves the source's per-channel mean onto the reference's.
@Test(.enabled(if: mlxTestsEnabled)) func matchColorMovesMean() {
    onCPU {
        let src = MLXArray((0 ..< 12).map { Float($0) / 40 + 0.1 }, [2, 2, 3])
        let ref = MLXArray((0 ..< 12).map { Float($0) / 40 + 0.5 }, [2, 2, 3])
        let matched = matchColor(src, reference: ref)
        let mm = MLX.mean(matched, axes: [0, 1]).asArray(Float.self)
        let rm = MLX.mean(ref, axes: [0, 1]).asArray(Float.self)
        for i in 0 ..< 3 { #expect(abs(mm[i] - rm[i]) < 1e-3) }
    }
}

/// Sharpen at amount 0 is identity; grayscale collapses the channels.
@Test(.enabled(if: mlxTestsEnabled)) func pixelFilterBasics() {
    onCPU {
        let rgb = MLXArray([0.2, 0.5, 0.8, 0.9, 0.1, 0.3].map { Float($0) }, [1, 2, 3])
        let same = sharpen(rgb, amount: 0).asArray(Float.self)
        let orig = rgb.asArray(Float.self)
        for i in 0 ..< orig.count { #expect(abs(same[i] - orig[i]) < 1e-6) }

        let gray = toGrayscale(rgb).asArray(Float.self)  // (1,2,3)
        #expect(abs(gray[0] - gray[1]) < 1e-6 && abs(gray[1] - gray[2]) < 1e-6)  // pixel 0 r=g=b
        #expect(abs(gray[3] - gray[4]) < 1e-6 && abs(gray[4] - gray[5]) < 1e-6)  // pixel 1 r=g=b
    }
}

/// The missing-weights guidance names the download flag and the default repo.
@Test func weightsHelpMentionsDownload() {
    let msg = weightsHelpMessage()
    #expect(msg.contains("--download"))
    #expect(msg.contains(defaultRepoId))
}

/// Geometric ops produce the expected output dimensions (pure CoreGraphics, no MLX).
@Test func imageOpsGeometry() throws {
    let img = try makeBoxMask(width: 100, height: 80, x: 10, y: 10, boxWidth: 20, boxHeight: 20)
    let resized = try applyImageOps(img, [.resize(50, 60)])
    #expect(resized.width == 50 && resized.height == 60)
    let cropped = try applyImageOps(img, [.crop(10, 5, 40, 30)])
    #expect(cropped.width == 40 && cropped.height == 30)
    let rotated = try applyImageOps(img, [.rotate(90)])
    #expect(rotated.width == 80 && rotated.height == 100)  // 90° swaps dims
    let fit = try applyImageOps(img, [.fit16])  // 100x80 -> 96x80
    #expect(fit.width % 16 == 0 && fit.height % 16 == 0 && fit.width == 96)
    // Flip preserves dimensions; FlipMode parses the documented spellings.
    let flipped = try applyImageOps(img, [.flip(.both)])
    #expect(flipped.width == 100 && flipped.height == 80)
    #expect(FlipMode(parsing: "H") == .horizontal)
    #expect(FlipMode(parsing: "vh") == .both)
    #expect(FlipMode(parsing: "diagonal") == nil)
}

/// Fused elementwise ops in `applyImageOps` behave correctly: a chain of consecutive MLX ops runs
/// without a per-op CGImage round-trip, and inverting twice recovers the input within uint8 error.
@Test(.enabled(if: mlxTestsEnabled)) func applyImageOpsFusedChain() throws {
    let w = 8, h = 6
    var rgba = [UInt8](repeating: 255, count: h * w * 4)
    for i in 0 ..< (h * w) {
        rgba[i * 4] = UInt8((i * 3) % 256)
        rgba[i * 4 + 1] = UInt8((i * 5) % 256)
        rgba[i * 4 + 2] = UInt8((i * 7) % 256)
    }
    let provider = CGDataProvider(data: Data(rgba) as CFData)!
    let img = CGImage(
        width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: w * 4,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
        provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
    try onCPUThrows {
        // Two inverts fuse into one round-trip and cancel out.
        let out = try applyImageOps(img, [.invert, .invert])
        #expect(out.width == w && out.height == h)
        let a = try cgImageToArray(img)
        let b = try cgImageToArray(out)
        MLX.eval(a, b)
        let av = a.asArray(Float.self), bv = b.asArray(Float.self)
        for i in 0 ..< av.count { #expect(abs(av[i] - bv[i]) < 2.0 / 255.0 + 1e-4) }

        // A geometric op interleaved with fused ops still yields correct dimensions.
        let mixed = try applyImageOps(img, [.brightness(0.1), .rotate(90), .invert])
        #expect(mixed.width == h && mixed.height == w)
    }
}

/// Effect filters: posterize/threshold quantize, brightness offsets, auto-contrast stretches.
@Test(.enabled(if: mlxTestsEnabled)) func effectsBasics() {
    onCPU {
        let rgb = MLXArray([0.2, 0.4, 0.6, 0.8, 0.1, 0.9].map { Float($0) }, [1, 2, 3])
        let orig = rgb.asArray(Float.self)

        for v in posterize(rgb, levels: 2).asArray(Float.self) {  // -> {0,1}
            #expect(v < 1e-6 || abs(v - 1) < 1e-6)
        }
        for v in threshold(rgb, 0.5).asArray(Float.self) {  // -> {0,1}
            #expect(v < 1e-6 || abs(v - 1) < 1e-6)
        }
        let br = adjustBrightness(rgb, 0.1).asArray(Float.self)
        #expect(abs(br[0] - min(1, orig[0] + 0.1)) < 1e-5)

        let ac = autoContrast(rgb).asArray(Float.self)  // min 0.1 -> 0, max 0.9 -> 1
        #expect((ac.min() ?? 1) < 1e-5 && abs((ac.max() ?? 0) - 1) < 1e-5)

        let warm = adjustTemperature(rgb, 0.5).asArray(Float.self)  // red up
        #expect(warm[0] >= orig[0] - 1e-6)
    }
}

// MARK: - CLI parsing (Q3 / V1 / V3)

/// Strict numeric parsers accept valid input and throw a descriptive `CLIParseError` on garbage,
/// rather than silently falling back to a default value.
@Test func cliNumericParsersStrict() throws {
    #expect(try parseIntArg("--steps", "8") == 8)
    #expect(try parseDoubleArg("--guidance", "3.5") == 3.5)
    #expect(try parseUInt64Arg("--seed", "42") == 42)
    #expect(try parseFloatArg("--sharpen", "1.5") == Float(1.5))

    for bad in ["", "abc", "3.5x", "--", "0x10"] {
        do {
            _ = try parseIntArg("--steps", bad)
            Issue.record("expected throw for '\(bad)'")
        } catch { #expect(error is CLIParseError) }
    }
    do {
        _ = try parseUInt64Arg("--seed", "-1")
        Issue.record("expected throw for negative seed")
    } catch { #expect(error is CLIParseError) }
}

/// WxH / 4-tuple / outpaint parsers accept valid specs and reject malformed ones.
@Test func cliTupleParsers() throws {
    let (w, h) = try parseWxHArg("512x768")
    #expect(w == 512 && h == 768)
    do { _ = try parseWxHArg("512"); Issue.record("expected throw") } catch { #expect(error is CLIParseError) }
    do { _ = try parseWxHArg("axb"); Issue.record("expected throw") } catch { #expect(error is CLIParseError) }

    let four = try parse4Arg("--crop", "1,2,3,4")
    #expect(four == (1, 2, 3, 4))
    do { _ = try parse4Arg("--crop", "1,2,3"); Issue.record("expected throw") } catch { #expect(error is CLIParseError) }

    #expect(try parseOutpaintArg("7") == (7, 7, 7, 7))
    #expect(try parseOutpaintArg("1,2,3,4") == (1, 2, 3, 4))
    do { _ = try parseOutpaintArg("1,2"); Issue.record("expected throw") } catch { #expect(error is CLIParseError) }
}

/// `parseRecolorArg` maps known keys, and collects warnings for unknown keys / bad values instead of
/// silently dropping them.
@Test func cliRecolorParserWarnings() {
    let ok = parseRecolorArg("hue=0.2,sat=1.1,exp=0.3,contrast=1.2,gamma=0.9")
    #expect(ok.hue == Float(0.2))
    #expect(ok.sat == Float(1.1))
    #expect(ok.gamma == Float(0.9))
    #expect(ok.warnings.isEmpty)

    let bad = parseRecolorArg("hue=0.2,bogus=1,contrast=notnum,malformed")
    #expect(bad.hue == Float(0.2))
    #expect(bad.warnings.count == 3)  // unknown key, non-number, malformed term
}

// MARK: - Negative paths / edge cases (Q5)

/// `applyImageOps` rejects an unsupported rotation angle rather than silently no-op'ing.
@Test func rotateInvalidDegreesThrows() throws {
    let img = try makeBoxMask(width: 32, height: 32, x: 4, y: 4, boxWidth: 8, boxHeight: 8)
    do {
        _ = try applyImageOps(img, [.rotate(45)])
        Issue.record("expected rotate(45) to throw")
    } catch {
        #expect(error is Flux2Error)
    }
    // Supported angles and 0 (identity) do not throw.
    _ = try applyImageOps(img, [.rotate(0)])
    _ = try applyImageOps(img, [.rotate(270)])
}

/// `capMinPixels` rejects too-small images and extreme aspect ratios.
@Test func capMinPixelsRejectsBadImages() throws {
    let tiny = try makeBoxMask(width: 16, height: 16, x: 0, y: 0, boxWidth: 4, boxHeight: 4)
    do {
        _ = try capMinPixels(tiny)
        Issue.record("expected capMinPixels to reject a 16x16 image")
    } catch { #expect(error is Flux2Error) }

    let wide = try makeBoxMask(width: 900, height: 64, x: 0, y: 0, boxWidth: 10, boxHeight: 10)
    do {
        _ = try capMinPixels(wide)  // 900/64 ≈ 14 > maxAr 8
        Issue.record("expected capMinPixels to reject extreme aspect ratio")
    } catch { #expect(error is Flux2Error) }
}

// MARK: - Weight conversion validation (C1/C2)

/// An empty weight dict makes the VAE conversion throw `missingWeights` naming the first required
/// tensor, instead of silently returning a partial/empty dictionary.
@Test func vaeConversionThrowsOnEmpty() {
    do {
        _ = try convertVaeDiffusersWeights([:])
        Issue.record("expected convertVaeDiffusersWeights to throw on empty input")
    } catch let e as Flux2Error {
        if case .missingWeights = e {} else {
            Issue.record("expected .missingWeights, got \(e)")
        }
    } catch {
        Issue.record("expected Flux2Error, got \(error)")
    }
}

/// A dict that supplies one bn stat but omits the other still fails fast at conversion.
@Test(.enabled(if: mlxTestsEnabled)) func vaeConversionThrowsOnMissingKey() {
    onCPU {
        let one = MLXArray([Float(0)], [1])
        // Only running_mean present; running_var missing.
        let partial: [String: MLXArray] = ["bn.running_mean": one]
        do {
            _ = try convertVaeDiffusersWeights(partial)
            Issue.record("expected throw on missing bn.running_var")
        } catch let e as Flux2Error {
            if case .missingWeights = e {} else { Issue.record("expected .missingWeights, got \(e)") }
        } catch {
            Issue.record("expected Flux2Error, got \(error)")
        }
    }
}

private func tinyFluxConfig(depth: Int, single: Int) -> Flux2Config {
    Flux2Config(
        inChannels: 128, contextInDim: 16, hiddenSize: 32, numHeads: 4, depth: depth,
        depthSingleBlocks: single, axesDim: [8, 8], theta: 10000, mlpRatio: 4, useGuidanceEmbed: false)
}

/// A checkpoint missing a required top-level rename tensor throws (previously silently skipped).
@Test func transformerConversionThrowsOnMissingRename() {
    let cfg = tinyFluxConfig(depth: 0, single: 0)
    do {
        _ = try convertFlux2DiffusersWeights([:], cfg)
        Issue.record("expected convertFlux2DiffusersWeights to throw on empty input")
    } catch let e as Flux2Error {
        if case .missingWeights = e {} else { Issue.record("expected .missingWeights, got \(e)") }
    } catch {
        Issue.record("expected Flux2Error, got \(error)")
    }
}

/// With depth 0 (no blocks) and all top-level rename tensors present, conversion succeeds and maps
/// every rename entry to its native destination key.
@Test(.enabled(if: mlxTestsEnabled)) func transformerConversionMapsRenameKeys() throws {
    try onCPUThrows {
        let one = MLXArray([Float(0)], [1])
        // norm_out needs an even row count: the converter swaps its scale/shift halves.
        let two = MLXArray([Float(0), Float(0)], [2, 1])
        let srcKeys = [
            "x_embedder.weight", "context_embedder.weight",
            "time_guidance_embed.timestep_embedder.linear_1.weight",
            "time_guidance_embed.timestep_embedder.linear_2.weight",
            "double_stream_modulation_img.linear.weight",
            "double_stream_modulation_txt.linear.weight",
            "single_stream_modulation.linear.weight",
            "proj_out.weight",
        ]
        var weights: [String: MLXArray] = [:]
        for k in srcKeys { weights[k] = one }
        weights["norm_out.linear.weight"] = two
        let out = try convertFlux2DiffusersWeights(weights, tinyFluxConfig(depth: 0, single: 0))
        #expect(out["img_in.weight"] != nil)
        #expect(out["txt_in.weight"] != nil)
        #expect(out["final_layer.linear.weight"] != nil)
        #expect(out.count == srcKeys.count + 1)
    }
}

/// Parity regression (2026-07-26): diffusers' AdaLayerNormContinuous stores the final-layer
/// modulation as [scale, shift] while the BFL LastLayer splits [shift, scale]. The converter must
/// swap the two row-halves of `norm_out.linear.weight`, or every image decodes with transposed
/// scale/shift (mottled artifacts; see Fixtures/ref_bike_s42.png parity check).
@Test(.enabled(if: mlxTestsEnabled)) func transformerConversionSwapsNormOutHalves() throws {
    try onCPUThrows {
        let one = MLXArray([Float(0)], [1])
        var weights: [String: MLXArray] = [:]
        for k in [
            "x_embedder.weight", "context_embedder.weight",
            "time_guidance_embed.timestep_embedder.linear_1.weight",
            "time_guidance_embed.timestep_embedder.linear_2.weight",
            "double_stream_modulation_img.linear.weight",
            "double_stream_modulation_txt.linear.weight",
            "single_stream_modulation.linear.weight",
            "proj_out.weight",
        ] { weights[k] = one }
        // Rows 0-1 = diffusers scale rows, rows 2-3 = diffusers shift rows.
        weights["norm_out.linear.weight"] = MLXArray((0 ..< 8).map { Float($0) }, [4, 2])
        let out = try convertFlux2DiffusersWeights(weights, tinyFluxConfig(depth: 0, single: 0))
        let converted = try #require(out["final_layer.adaLN_modulation.1.weight"])
        // Expect [shift rows; scale rows]: [[4,5],[6,7],[0,1],[2,3]]
        let expected: [Float] = [4, 5, 6, 7, 0, 1, 2, 3]
        #expect(converted.asArray(Float.self) == expected)

        // Odd row count cannot be split into halves and must throw.
        weights["norm_out.linear.weight"] = MLXArray([Float(0), 0, 0], [3, 1])
        #expect(throws: Flux2Error.self) {
            _ = try convertFlux2DiffusersWeights(weights, tinyFluxConfig(depth: 0, single: 0))
        }
    }
}

// MARK: - Image I/O (vectorized RGBA<->RGB path)

/// The vectorized `cgImageToArray` must reproduce the exact values the scalar luminance/normalize
/// loop produced: uint8→float via `v/127.5 - 1`. Uses a known solid-color image so the expected
/// values are hand-computable, guarding the parity-critical conversion.
@Test(.enabled(if: mlxTestsEnabled)) func cgImageToArrayKnownValues() throws {
    // Solid RGB (200, 100, 50) opaque image.
    let w = 4, h = 3
    var rgba = [UInt8](repeating: 0, count: h * w * 4)
    for i in 0 ..< (h * w) {
        rgba[i * 4] = 200
        rgba[i * 4 + 1] = 100
        rgba[i * 4 + 2] = 50
        rgba[i * 4 + 3] = 255
    }
    let provider = CGDataProvider(data: Data(rgba) as CFData)!
    let img = CGImage(
        width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: w * 4,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
        provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
    try onCPUThrows {
        let arr = try cgImageToArray(img)  // (h, w, 3) in [-1, 1]
        #expect(arr.shape == [h, w, 3])
        MLX.eval(arr)
        let expectedR = Float(200) / 127.5 - 1.0
        let expectedG = Float(100) / 127.5 - 1.0
        let expectedB = Float(50) / 127.5 - 1.0
        #expect(abs(arr[0, 0, 0].item(Float.self) - expectedR) < 1e-6)
        #expect(abs(arr[1, 2, 1].item(Float.self) - expectedG) < 1e-6)
        #expect(abs(arr[2, 3, 2].item(Float.self) - expectedB) < 1e-6)
    }
}

/// `arrayToCGImage` → `cgImageToArray` is a stable round-trip: the vectorized RGB→RGBA repack loses
/// no channel data, so re-reading the encoded image recovers the (uint8-quantized) input.
@Test(.enabled(if: mlxTestsEnabled)) func imageRoundTripStable() throws {
    try onCPUThrows {
        // Values chosen to sit near uint8 bucket centers so quantization is stable.
        let src = MLXArray(
            [Float(-1.0), -0.5, 0.0, 0.5, 1.0, -0.2, 0.2, -0.8, 0.8].map { $0 },
            [1, 3, 3])
        let img = try arrayToCGImage(src)
        #expect(img.width == 3 && img.height == 1)
        let back = try cgImageToArray(img)
        MLX.eval(back)
        let a = src.asArray(Float.self)
        let b = back.asArray(Float.self)
        // uint8 quantization step in [-1,1] space is 2/255 ≈ 0.0078; allow one step.
        for i in 0 ..< a.count { #expect(abs(a[i] - b[i]) < 2.0 / 255.0 + 1e-4) }
    }
}

// MARK: - Sharded weight loading (9B diffusers snapshot)

/// `resolveShardPaths` reads the supplied index json, dedupes + sorts the shard names,
/// and verifies every referenced shard exists on disk.
@Test func resolveShardPathsWithIndex() throws {
    let tmp = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: tmp) }

    // Two shards with an index that references them (unsorted, duplicates).
    let shardA = tmp.appendingPathComponent("diffusion_pytorch_model-00002-of-00002.safetensors")
    let shardB = tmp.appendingPathComponent("diffusion_pytorch_model-00001-of-00002.safetensors")
    try Data().write(to: shardA)
    try Data().write(to: shardB)
    let index: [String: Any] = [
        "weight_map": [
            "a": shardB.lastPathComponent,
            "b": shardA.lastPathComponent,
            "c": shardA.lastPathComponent,
        ]
    ]
    let indexData = try JSONSerialization.data(withJSONObject: index)
    try indexData.write(to: tmp.appendingPathComponent("diffusion_pytorch_model.safetensors.index.json"))

    let resolved = try resolveShardPaths(tmp, indexFileName: "diffusion_pytorch_model.safetensors.index.json")
    #expect(resolved.map(\.lastPathComponent) == [shardB.lastPathComponent, shardA.lastPathComponent])
}

/// When the index json references a missing shard, `resolveShardPaths` throws immediately
/// rather than producing a partial load.
@Test func resolveShardPathsMissingShard() throws {
    let tmp = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: tmp) }

    let index: [String: Any] = [
        "weight_map": ["a": "diffusion_pytorch_model-00001-of-00002.safetensors"]
    ]
    let indexData = try JSONSerialization.data(withJSONObject: index)
    try indexData.write(to: tmp.appendingPathComponent("diffusion_pytorch_model.safetensors.index.json"))

    do {
        _ = try resolveShardPaths(tmp, indexFileName: "diffusion_pytorch_model.safetensors.index.json")
        Issue.record("Expected resolveShardPaths to throw on missing shard")
    } catch {
        #expect(error is Flux2Error)
    }
}

/// Without an index json, `resolveShardPaths` falls back to the sorted glob order
/// (all *.safetensors, then model-*.safetensors if none found).
@Test func resolveShardPathsFallbackGlob() throws {
    let tmp = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: tmp) }

    // No index — just loose files in non-alphabetical creation order.
    let f2 = tmp.appendingPathComponent("diffusion_pytorch_model-00002-of-00002.safetensors")
    let f1 = tmp.appendingPathComponent("diffusion_pytorch_model-00001-of-00002.safetensors")
    try Data().write(to: f2)
    try Data().write(to: f1)

    // Without an index file, the default fallback glob is used.
    let resolved = try resolveShardPaths(tmp)
    #expect(resolved.map(\.lastPathComponent) == [f1.lastPathComponent, f2.lastPathComponent])
}
