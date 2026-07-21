// Flux2Kit — mask-guided inpainting (editing foundation). Not parity-locked: no counterpart in
// the reference. Reuses the img2img re-noising, the prcImg/scatterIds token mapping, and the
// denoise postStep hook. One mechanism underlies removal, background replacement, region editing,
// and prompt-conditioned object addition (see Editing.swift).
//
// Approach: encode the source to latents, initialize the EDIT region from noise (like img2img),
// and after every denoise step re-blend the KEEP region with the source latent re-noised to the
// current step's noise level — the flow-matching analogue of blended/RePaint inpainting:
//     keep_t = tPrev * noise + (1 - tPrev) * srcTokens
//     img    = img * editMask + keep_t * (1 - editMask)

import CoreGraphics
import Foundation
import MLX
import MLXRandom

extension Flux2Pipeline {

    /// Mask-guided inpainting. `mask` is a grayscale image at any resolution; by convention
    /// **white (1) = the region to regenerate, black (0) = the region to preserve** (flip with
    /// `invertMask`). `strength` controls how far the edit region is regenerated (higher = freer);
    /// the full step schedule is rescaled into `[strength, 0]` (see Img2Img.swift for the rationale).
    /// Output geometry must be divisible by 16.
    public func generateInpaint(
        prompt: String,
        source: CGImage,
        mask: CGImage,
        strength: Double = 0.85,
        width: Int,
        height: Int,
        numSteps: Int = defaultSteps,
        guidance: Double = Double(defaultGuidance),
        seed: UInt64? = nil,
        inputImages: [CGImage]? = nil,
        invertMask: Bool = false,
        maskFeather: Int = 1,
        verbose: Bool = false,
        evalFreq: Int = 1
    ) throws -> CGImage {
        guard width % 16 == 0, height % 16 == 0 else {
            throw Flux2Error.generationFailed("width and height must be divisible by 16")
        }
        let strength = min(max(strength, 0.0), 1.0)

        if let seed {
            MLXRandom.seed(seed)
        }

        let guidanceDistilled = isDistilled
        let (ctx, ctxIds, _) = try encodePrompt(
            prompt, guidanceDistilled: guidanceDistilled, verbose: verbose)

        var imgCondSeq: MLXArray?
        var imgCondSeqIds: MLXArray?
        if let inputImages, !inputImages.isEmpty {
            (imgCondSeq, imgCondSeqIds) = try encodeImageRefs(vae, inputImages)
        }

        // Source latents at the output geometry (identical to img2img).
        let resized = try resizeExactRGB(source, width: width, height: height)
        let sourceArray = try cgImageToArray(resized)  // (H, W, 3) in [-1, 1]
        let sourceLatents = vae.encode(expandedDimensions(sourceArray, axis: 0))
            .transposed(0, 3, 1, 2)
        let (srcTokens, xIds) = batchedPrcImg(sourceLatents.asType(dtype))

        // One fixed noise sample, reused for init and for re-noising the KEEP region each step.
        let noise = MLXRandom.normal(srcTokens.shape, dtype: dtype)

        // Edit mask → per-token weights (1, N, 1), aligned with srcTokens' raster order.
        let hL = height / 16
        let wL = width / 16
        var maskGrid = try maskGridFromCGImage(mask, width: wL, height: hL)  // (hL, wL) in [0,1]
        if maskFeather > 0 {
            maskGrid = boxBlur(maskGrid, passes: maskFeather)
        }
        if invertMask {
            maskGrid = 1.0 - maskGrid
        }
        let (maskTok, _) = prcImg(maskGrid.reshaped([1, hL, wL]).asType(dtype))  // (N, 1)
        let editMask = expandedDimensions(maskTok, axis: 0)  // (1, N, 1)

        // Rescaled schedule into [strength, 0] (no truncation — see Img2Img.swift).
        let fullSchedule = getSchedule(numSteps, srcTokens.dim(1))
        let timesteps = fullSchedule.map { $0 * strength }
        guard let tStart = timesteps.first else {
            throw Flux2Error.generationFailed("empty inpaint schedule")
        }

        let tS = MLXArray(Float(tStart)).asType(dtype)
        var x = tS * noise + (1 - tS) * srcTokens

        // Per-step blend: re-noise the KEEP region to the step's target level, then composite.
        // editMask (1,N,1) and the (1,N,C) source/noise broadcast over the CFG-doubled batch.
        let blend: (Int, Double, MLXArray) -> MLXArray = { _, tPrev, img in
            let tp = MLXArray(Float(tPrev)).asType(self.dtype)
            let keep = tp * noise + (1 - tp) * srcTokens
            return img * editMask + keep * (1 - editMask)
        }

        var imgInputIds = xIds
        if let imgCondSeqIds {
            imgInputIds = concatenated([xIds, imgCondSeqIds], axis: 1)
        }
        if !guidanceDistilled {
            imgInputIds = concatenated([imgInputIds, imgInputIds], axis: 0)
        }
        let peX = model.peEmbedder(imgInputIds)
        let peCtx = model.peEmbedder(ctxIds)

        let modelFn: Flux2ModelFn = { [model] x, xIds, t, ctx, ctxIds, g, peX, peCtx, txtEmb, gEmb in
            model(x, xIds, t, ctx, ctxIds, g, peX, peCtx, txtEmb, gEmb)
        }
        let modelFnCfg: Flux2ModelCfgFn = { [model] x, xIds, t, ctx, ctxIds, peX, peCtx, txtEmb in
            model(x, xIds, t, ctx, ctxIds, nil, peX, peCtx, txtEmb, nil)
        }

        if guidanceDistilled {
            x = denoise(
                model, x, xIds, ctx, ctxIds,
                timesteps: timesteps, guidance: guidance,
                imgCondSeq: imgCondSeq, imgCondSeqIds: imgCondSeqIds,
                peX: peX, peCtx: peCtx, modelFn: modelFn, evalFreq: evalFreq, postStep: blend)
        } else {
            x = denoiseCfg(
                model, x, xIds, ctx, ctxIds,
                timesteps: timesteps, guidance: guidance,
                imgCondSeq: imgCondSeq, imgCondSeqIds: imgCondSeqIds,
                peX: peX, peCtx: peCtx, modelFn: modelFn, modelFnCfg: modelFnCfg,
                evalFreq: evalFreq, postStep: blend)
        }

        x = concatenated(scatterIds(x, xIds), axis: 0)
        if x.dim(2) == 1 {
            x = x.squeezed(axis: 2)
        } else {
            x = x[0..., 0..., 0, 0..., 0...]
        }
        x = x.transposed(0, 2, 3, 1)
        eval(x)

        let decoded = vae.decode(x)
        eval(decoded)
        return try arrayToCGImage(decoded[0])
    }
}

// MARK: - Mask helpers (internal so tests can reach them via @testable)

/// Rasterize a grayscale mask into a `(height, width)` MLXArray in `[0, 1]`, using the same bitmap
/// draw/read convention as `cgImageToArray` so the grid aligns with the source latent grid. The
/// CGContext downsamples the full-resolution mask to the latent grid directly.
func maskGridFromCGImage(_ img: CGImage, width w: Int, height h: Int) throws -> MLXArray {
    let bytesPerRow = w * 4
    var pixels = [UInt8](repeating: 0, count: h * bytesPerRow)
    try pixels.withUnsafeMutableBytes { buffer in
        guard
            let context = CGContext(
                data: buffer.baseAddress, width: w, height: h, bitsPerComponent: 8,
                bytesPerRow: bytesPerRow, space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        else {
            throw Flux2Error.generationFailed("Could not create mask bitmap context")
        }
        context.interpolationQuality = .high
        context.draw(img, in: CGRect(x: 0, y: 0, width: w, height: h))
    }
    var gray = [Float](repeating: 0, count: h * w)
    for y in 0 ..< h {
        for x in 0 ..< w {
            let src = y * bytesPerRow + x * 4
            // Luminance (masks are usually grayscale; this is robust if they are not).
            let r = Float(pixels[src])
            let g = Float(pixels[src + 1])
            let b = Float(pixels[src + 2])
            gray[y * w + x] = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
        }
    }
    return MLXArray(gray, [h, w])
}

/// Edge-replicating 3x3 box blur, applied `passes` times. Feathers a `(H, W)` mask so edit
/// boundaries blend instead of hard-cutting. Monotonic and shape-preserving.
func boxBlur(_ grid: MLXArray, passes: Int) -> MLXArray {
    guard passes > 0, grid.ndim == 2 else { return grid }
    var m = grid
    let h = grid.dim(0)
    let w = grid.dim(1)
    for _ in 0 ..< passes {
        // Replicate-pad by 1 on each side.
        let top = m[0 ..< 1, 0...]
        let bot = m[(h - 1) ..< h, 0...]
        var p = concatenated([top, m, bot], axis: 0)  // (h+2, w)
        let left = p[0..., 0 ..< 1]
        let right = p[0..., (w - 1) ..< w]
        p = concatenated([left, p, right], axis: 1)  // (h+2, w+2)
        var sum = p[0 ..< h, 0 ..< w]
        for dy in 0 ..< 3 {
            for dx in 0 ..< 3 {
                if dy == 0 && dx == 0 { continue }
                sum = sum + p[dy ..< (dy + h), dx ..< (dx + w)]
            }
        }
        m = sum / 9.0
    }
    return m
}

/// Resize a CGImage to an exact RGB size (high-quality). Mirrors the img2img resize.
func resizeExactRGB(_ img: CGImage, width: Int, height: Int) throws -> CGImage {
    if img.width == width && img.height == height {
        return img
    }
    guard
        let context = CGContext(
            data: nil, width: width, height: height, bitsPerComponent: 8,
            bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
    else {
        throw Flux2Error.generationFailed("Could not create resize context")
    }
    context.interpolationQuality = .high
    context.draw(img, in: CGRect(x: 0, y: 0, width: width, height: height))
    guard let resized = context.makeImage() else {
        throw Flux2Error.generationFailed("resize failed")
    }
    return resized
}
