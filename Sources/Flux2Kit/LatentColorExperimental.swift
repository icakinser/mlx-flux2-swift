// Flux2Kit — EXPERIMENTAL latent-space color ops. Opt-in only (never a default).
//
// WARNING: FLUX.2 latents are a 128-dim learned, patchified representation — NOT a color space.
// Elementwise "curves" on latent channels do not map to perceptual color/exposure edits; they push
// latents off the VAE manifold and the decoder returns artifacts as often as clean edits. This file
// exists purely so the difference vs the correct pixel-space path (ColorAdjust.swift) can be A/B'd.
// For real coloring/exposure work, use recolor()/adjustColor() instead.

import CoreGraphics
import Foundation
import MLX

/// Naive elementwise "curve" on VAE latents. Layout-agnostic (works on NHWC or NCHW).
/// - exposure: scale by 2^exposure. - contrast: expand around the global mean.
/// - gamma: sign-preserving magnitude gamma. All are meaningless as *color* operations; see header.
public func applyLatentCurve(
    _ latents: MLXArray, exposure: Float = 0, contrast: Float = 1, gamma: Float = 1
) -> MLXArray {
    var x = latents
    if exposure != 0 {
        x = x * MLXArray(Float(pow(2.0, Double(exposure))))
    }
    if contrast != 1 {
        let m = MLX.mean(x)
        x = (x - m) * contrast + m
    }
    if gamma != 1 {
        let g = max(gamma, 1e-4)
        x = MLX.sign(x) * MLX.pow(MLX.abs(x), MLXArray(Float(1.0 / g)))
    }
    return x
}

extension Flux2Pipeline {
    /// EXPERIMENTAL: encode → apply a latent-space curve → decode, with NO denoising. Demonstrates
    /// (usually poorly) what latent color ops do. Provided for A/B against the pixel-space path.
    public func experimentalLatentColor(
        source: CGImage, width: Int, height: Int,
        exposure: Float = 0, contrast: Float = 1, gamma: Float = 1
    ) throws -> CGImage {
        guard width % 16 == 0, height % 16 == 0 else {
            throw Flux2Error.generationFailed("width and height must be divisible by 16")
        }
        try ensureVAE()
        let resized = try resizeExactRGB(source, width: width, height: height)
        let srcArray = try cgImageToArray(resized)
        let latents = vae.encode(expandedDimensions(srcArray, axis: 0)).asType(dtype)
        let adjusted = applyLatentCurve(
            latents, exposure: exposure, contrast: contrast, gamma: gamma)
        let decoded = vae.decode(adjusted)
        eval(decoded)
        return try arrayToCGImage(decoded[0])
    }
}
