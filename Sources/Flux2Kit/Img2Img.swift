// Flux2Kit — an img2img extension beyond the scf4/mlx-flux2 reference (which has no img2img path).
// 2026-07-20 EDT | PERMANENT (img2img strength) — kontext-from-noise regenerates text/fine
// structure imperfectly; initializing from noised source latents (diffusers-style strength)
// preserves glyphs and applies instructions more faithfully (A/B evidence vs the Draw Things
// engine, which honors strength the same way). This file is NOT parity-locked: it has no
// counterpart in the reference. It reuses only public parity-locked building blocks.

import CoreGraphics
import Foundation
import MLX
import MLXRandom

extension Flux2Pipeline {

    /// Image-to-image generation: the source image initializes the latents at an intermediate
    /// timestep chosen by `strength` (1.0 = pure noise / full regeneration, low = stay close
    /// to the source), while `inputImages` still condition as kontext reference tokens.
    /// The output dimensions must match the prepared source dimensions (both /16).
    public func generateImg2Img(
        prompt: String,
        source: CGImage,
        strength: Double,
        width: Int,
        height: Int,
        numSteps: Int = defaultSteps,
        guidance: Double = Double(defaultGuidance),
        seed: UInt64? = nil,
        inputImages: [CGImage]? = nil,
        verbose: Bool = false,
        evalFreq: Int = 1
    ) throws -> CGImage {
        guard width % 16 == 0, height % 16 == 0 else {
            throw Flux2Error.generationFailed("width and height must be divisible by 16")
        }
        let strength = min(max(strength, 0.0), 1.0)
        // Full-strength requests are exactly the reference path.
        if strength >= 1.0 {
            return try generate(
                prompt: prompt, width: width, height: height, numSteps: numSteps,
                guidance: guidance, seed: seed, inputImages: inputImages, verbose: verbose,
                evalFreq: evalFreq)
        }

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

        // Source latents at exactly the output geometry, tokenized with the same position
        // ids the noise path would produce.
        let resized = try renderExact(source, width: width, height: height)
        let sourceArray = try cgImageToArray(resized)  // (H, W, 3) in [-1, 1]
        // Consistent with encodeImageRefs: VAE encode returns NHWC-patchified
        // latents; tokenization expects channels-first (b, 128, h/16, w/16).
        let sourceLatents = vae.encode(expandedDimensions(sourceArray, axis: 0))
            .transposed(0, 3, 1, 2)
        let (srcTokens, xIds) = batchedPrcImg(sourceLatents.asType(dtype))

        let noise = MLXRandom.normal(srcTokens.shape, dtype: dtype)

        // 2026-07-20 EDT | PERMANENT (rescaled schedule) — do NOT truncate the schedule:
        // with a 4-step distilled model, diffusers-style truncation leaves 1-2 steps for
        // any strength below ~0.88, and one step cannot execute a semantic edit (verified:
        // remove/recolor edits at s=0.5-0.7 reproduced the source unchanged). Instead,
        // rescale the FULL numSteps schedule into the [strength, 0] window — all steps
        // execute, entry noise level still honors strength.
        let fullSchedule = getSchedule(numSteps, srcTokens.dim(1))
        let timesteps = fullSchedule.map { $0 * strength }
        guard let tStart = timesteps.first else {
            throw Flux2Error.generationFailed("empty img2img schedule")
        }

        let tS = MLXArray(Float(tStart)).asType(dtype)
        var x = tS * noise + (1 - tS) * srcTokens

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
                peX: peX, peCtx: peCtx, modelFn: modelFn, evalFreq: evalFreq)
        } else {
            x = denoiseCfg(
                model, x, xIds, ctx, ctxIds,
                timesteps: timesteps, guidance: guidance,
                imgCondSeq: imgCondSeq, imgCondSeqIds: imgCondSeqIds,
                peX: peX, peCtx: peCtx, modelFn: modelFn, modelFnCfg: modelFnCfg,
                evalFreq: evalFreq)
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

    private func renderExact(_ img: CGImage, width: Int, height: Int) throws -> CGImage {
        if img.width == width && img.height == height {
            return img
        }
        guard
            let context = CGContext(
                data: nil, width: width, height: height, bitsPerComponent: 8,
                bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        else {
            throw Flux2Error.generationFailed("Could not create img2img resize context")
        }
        context.interpolationQuality = .high
        context.draw(img, in: CGRect(x: 0, y: 0, width: width, height: height))
        guard let resized = context.makeImage() else {
            throw Flux2Error.generationFailed("img2img resize failed")
        }
        return resized
    }
}
