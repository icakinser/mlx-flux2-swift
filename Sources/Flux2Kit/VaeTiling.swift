// Flux2Kit — tiled VAE decode. Caps the decode-stage activation peak for large images by decoding
// the latent in overlapping spatial tiles and feather-blending them in pixel space. Opt-in via
// `vaeTileLatent`; when unset (or the latent is small) decode is a single bit-identical pass.
//
// NOTE: tiling is a memory/quality trade, not free. FLUX's VAE mid-block uses GLOBAL attention, so a
// tile decoded in isolation differs from the full decode (measured ~7/255 mean, visible seams at
// small tiles). Use only when a large image would otherwise exhaust memory; larger tiles + overlap
// reduce the artifacts. It is never auto-enabled — the caller must set `vaeTileLatent`.

import CoreGraphics
import Foundation
import MLX

extension Flux2Pipeline {

    /// Shared final stage for the editing entry points (img2img / inpaint): reassemble the denoised
    /// canvas tokens into an NHWC latent, free the transformer, ensure the VAE is resident, and decode
    /// to an image. `generate` keeps its own instrumented copy that records per-stage timings.
    ///
    /// The MLX ops here are identical to the previously-inlined blocks, so decode output is unchanged.
    func scatterAndDecodeToImage(_ x: MLXArray, xIds: MLXArray) throws -> CGImage {
        var latent = concatenated(scatterIds(x, xIds), axis: 0)
        if latent.dim(2) == 1 {
            latent = latent.squeezed(axis: 2)
        } else {
            latent = latent[0..., 0..., 0, 0..., 0...]
        }
        latent = latent.transposed(0, 2, 3, 1)
        eval(latent)

        unloadTransformer()
        try ensureVAE()
        reportMemory("pre-decode")
        let decoded = try decodeMaybeTiled(latent)
        eval(decoded)
        return try arrayToCGImage(decoded[0])
    }

    /// Decode `x` (NHWC latent) — tiled when `vaeTileLatent` is set (and positive) and the latent
    /// exceeds it, otherwise a single `vae.decode`.
    func decodeMaybeTiled(_ x: MLXArray) throws -> MLXArray {
        let vae = try requireVAE()
        // Parenthesize the size test so a future single-expression refactor can't accidentally drop
        // the ndim guard from the `||` branch.
        if let tile = vaeTileLatent, tile > 0, x.ndim == 4, (x.dim(1) > tile || x.dim(2) > tile) {
            return try decodeTiled(x, tileLatent: tile, overlap: max(2, tile / 4))
        }
        return try vae.decode(x)
    }

    /// Overlapping-tile decode of a `(1, HL, WL, C)` latent with feather blending.
    func decodeTiled(_ latents: MLXArray, tileLatent: Int, overlap: Int) throws -> MLXArray {
        let vae = try requireVAE()
        let hL = latents.dim(1)
        let wL = latents.dim(2)
        // Degenerate or non-positive-tile inputs cannot be tiled; fall back to a single decode
        // (a zero/negative tile would make the stride 1 with empty slices → infinite loop).
        guard hL > 0, wL > 0, tileLatent > 0 else {
            return try vae.decode(latents)
        }
        let step = max(1, tileLatent - overlap)

        var ff = 0
        var accum: MLXArray? = nil  // (1, H, W, 3)
        var wsum: MLXArray? = nil  // (1, H, W, 1)

        var ly = 0
        while ly < hL {
            let ly1 = min(ly + tileLatent, hL)
            var lx = 0
            while lx < wL {
                let lx1 = min(lx + tileLatent, wL)

                let tile = latents[0..., ly ..< ly1, lx ..< lx1, 0...]
                let dec = try vae.decode(tile)  // (1, th, tw, 3)
                eval(dec)

                if ff == 0 {
                    ff = dec.dim(1) / (ly1 - ly)
                    accum = MLX.zeros([1, hL * ff, wL * ff, 3], dtype: dec.dtype)
                    wsum = MLX.zeros([1, hL * ff, wL * ff, 1], dtype: dec.dtype)
                }
                let th = dec.dim(1)
                let tw = dec.dim(2)
                let py0 = ly * ff
                let px0 = lx * ff

                let wy = featherRamp(th, overlap * ff, edgeStart: ly > 0, edgeEnd: ly1 < hL)
                let wx = featherRamp(tw, overlap * ff, edgeStart: lx > 0, edgeEnd: lx1 < wL)
                let wtile = wy.reshaped([1, th, 1, 1]) * wx.reshaped([1, 1, tw, 1])

                // Accumulate weighted contribution into the canvas region (read-modify-write).
                let a = accum![0..., py0 ..< (py0 + th), px0 ..< (px0 + tw), 0...]
                accum![0..., py0 ..< (py0 + th), px0 ..< (px0 + tw), 0...] = a + dec * wtile
                let s = wsum![0..., py0 ..< (py0 + th), px0 ..< (px0 + tw), 0...]
                wsum![0..., py0 ..< (py0 + th), px0 ..< (px0 + tw), 0...] = s + wtile

                lx += step
            }
            ly += step
        }
        // accum/wsum are assigned on the first tile; the guards above ensure at least one tile runs.
        guard let accum, let wsum else {
            throw Flux2Error.generationFailed("tiled decode produced no tiles")
        }
        return accum / MLX.maximum(wsum, MLXArray(Float(1e-6)))
    }
}

/// 1-D blend ramp of length `n`: linear 0→1 over `ramp` samples at an interior edge, flat 1 in the
/// middle and at image borders (so tiles overlap-blend but the outer boundary is not darkened).
func featherRamp(_ n: Int, _ ramp: Int, edgeStart: Bool, edgeEnd: Bool) -> MLXArray {
    var w = [Float](repeating: 1, count: n)
    let r = max(1, min(ramp, n / 2))
    if edgeStart {
        for i in 0 ..< r { w[i] = Float(i + 1) / Float(r + 1) }
    }
    if edgeEnd {
        for i in 0 ..< r { w[n - 1 - i] = Float(i + 1) / Float(r + 1) }
    }
    return MLXArray(w, [n])
}
