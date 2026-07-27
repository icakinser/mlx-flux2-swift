// Flux2Kit — an img2img extension beyond the scf4/mlx-flux2 reference (which has no img2img path).
// 2026-07-20 EDT | PERMANENT (img2img strength) — kontext-from-noise regenerates text/fine
// structure imperfectly; initializing from noised source latents (diffusers-style strength)
// preserves glyphs and applies instructions more faithfully (A/B evidence vs the Draw Things
// engine, which honors strength the same way). This path has no direct Python counterpart and is
// governed by Flux2Kit's quality gates.

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
        evalFreq: Int = 1,
        sampler: Sampler = .euler,
        guidanceSchedule: GuidanceSchedule = .constant,
        progress: (@Sendable (GenerationProgress) -> Void)? = nil,
        cancellation: GenerationCancellation? = nil
    ) throws -> CGImage {
        generationLock.lock()
        defer { generationLock.unlock() }
        if cancellation?.isCancelled == true { throw Flux2Error.cancelled }
        guard width > 0, height > 0, width % 16 == 0, height % 16 == 0 else {
            throw Flux2Error.generationFailed(
                "width and height must be positive multiples of 16 (got \(width)x\(height))")
        }
        guard numSteps > 0 else {
            throw Flux2Error.generationFailed("numSteps must be positive (got \(numSteps))")
        }
        let strength = min(max(strength, 0.0), 1.0)
        // Full-strength requests are exactly the reference path.
        if strength >= 1.0 {
            return try generate(
                prompt: prompt, width: width, height: height, numSteps: numSteps,
                guidance: guidance, seed: seed, inputImages: inputImages, verbose: verbose,
                evalFreq: evalFreq, sampler: sampler, guidanceSchedule: guidanceSchedule,
                progress: progress, cancellation: cancellation)
        }
        // Zero strength injects no noise, so every denoise step is a no-op and the token
        // patchify/scatter is a lossless round-trip: the result is just the VAE reconstruction of
        // the source at the requested geometry. Short-circuit to avoid loading the transformer and
        // running a pointless denoise loop.
        if strength <= 0.0 {
            try ensureVAE()
            let vae = try requireVAE()
            let resized = try resizeHighQuality(source, width: width, height: height)
            let sourceArray = try cgImageToArray(resized)
            let latents = try vae.encode(expandedDimensions(sourceArray, axis: 0)).asType(dtype)
            let decoded = try decodeMaybeTiled(latents)
            eval(decoded)
            return try arrayToCGImage(decoded[0])
        }

        if let seed {
            MLXRandom.seed(seed)
        }

        try ensureTextEncoder()
        try ensureVAE()
        let vae = try requireVAE()
        reportMemory("pre-encode")
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
        let resized = try resizeHighQuality(source, width: width, height: height)
        let sourceArray = try cgImageToArray(resized)  // (H, W, 3) in [-1, 1]
        // Consistent with encodeImageRefs: VAE encode returns NHWC-patchified
        // latents; tokenization expects channels-first (b, 128, h/16, w/16).
        let sourceLatents = try vae.encode(expandedDimensions(sourceArray, axis: 0))
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
        unloadTextEncoder()
        try ensureTransformer()
        let model = try requireTransformer()
        reportMemory("pre-denoise")
        let peX = model.peEmbedder(imgInputIds)
        let peCtx = model.peEmbedder(ctxIds)

        x = try executeDenoise(
            model: model,
            request: DenoiseRequest(
                image: x,
                imageIds: xIds,
                text: ctx,
                textIds: ctxIds,
                timesteps: timesteps,
                guidance: guidance,
                guidanceDistilled: guidanceDistilled,
                imageCondition: imgCondSeq,
                imageConditionIds: imgCondSeqIds,
                positionImage: peX,
                positionText: peCtx),
            options: DenoiseExecutionOptions(
                sampler: sampler,
                guidanceSchedule: guidanceSchedule,
                evalFrequency: evalFreq,
                verbose: verbose,
                progress: progress,
                cancellation: cancellation))
        if cancellation?.isCancelled == true { throw Flux2Error.cancelled }

        return try scatterAndDecodeToImage(x, xIds: xIds)
    }

}
