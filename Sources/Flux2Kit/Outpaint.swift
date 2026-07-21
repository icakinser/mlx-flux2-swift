// Flux2Kit — outpainting (canvas extension). Places the source on a larger canvas, seeds the new
// border from a stretched copy of the source, masks the border as the edit region, and delegates to
// generateInpaint so the model fills the extension in context. Reuses the whole inpaint machinery.

import CoreGraphics
import Foundation

extension Flux2Pipeline {

    /// Extend `source` by the given pixel margins on each side and fill the new area from `prompt`.
    /// New canvas dims are rounded up to a multiple of 16 (extra pixels added to right/bottom).
    public func generateOutpaint(
        source: CGImage, prompt: String,
        left: Int, right: Int, top: Int, bottom: Int,
        strength: Double = 0.95,
        numSteps: Int = defaultSteps, guidance: Double = Double(defaultGuidance),
        seed: UInt64? = nil, maskFeather: Int = 2,
        verbose: Bool = false, evalFreq: Int = 1
    ) throws -> CGImage {
        let sw = source.width
        let sh = source.height
        let l = max(0, left), t = max(0, top)
        func roundUp16(_ n: Int) -> Int { ((n + 15) / 16) * 16 }
        let nw = roundUp16(sw + l + max(0, right))
        let nh = roundUp16(sh + t + max(0, bottom))
        guard nw > sw || nh > sh else {
            throw Flux2Error.generationFailed("outpaint margins must extend the canvas")
        }

        // Extended canvas: blurry stretched source as background, crisp source composited at (l, t).
        guard
            let ctx = CGContext(
                data: nil, width: nw, height: nh, bitsPerComponent: 8, bytesPerRow: nw * 4,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        else {
            throw Flux2Error.generationFailed("Could not create outpaint canvas")
        }
        ctx.interpolationQuality = .low
        ctx.draw(source, in: CGRect(x: 0, y: 0, width: nw, height: nh))  // stretched fill
        ctx.interpolationQuality = .high
        ctx.draw(source, in: CGRect(x: l, y: nh - t - sh, width: sw, height: sh))  // crisp original
        guard let canvas = ctx.makeImage() else {
            throw Flux2Error.generationFailed("Could not render outpaint canvas")
        }

        // Mask: white (edit) everywhere except the kept interior of the original (inset by overlap).
        guard
            let mctx = CGContext(
                data: nil, width: nw, height: nh, bitsPerComponent: 8, bytesPerRow: nw * 4,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        else {
            throw Flux2Error.generationFailed("Could not create outpaint mask")
        }
        mctx.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
        mctx.fill(CGRect(x: 0, y: 0, width: nw, height: nh))
        let ov = max(1, min(sw, sh) / 32)
        let kw = sw - 2 * ov
        let kh = sh - 2 * ov
        if kw > 0 && kh > 0 {
            mctx.setFillColor(CGColor(red: 0, green: 0, blue: 0, alpha: 1))
            mctx.fill(CGRect(x: l + ov, y: nh - (t + ov) - kh, width: kw, height: kh))
        }
        guard let mask = mctx.makeImage() else {
            throw Flux2Error.generationFailed("Could not render outpaint mask")
        }

        return try generateInpaint(
            prompt: prompt, source: canvas, mask: mask, strength: strength,
            width: nw, height: nh, numSteps: numSteps, guidance: guidance, seed: seed,
            maskFeather: maskFeather, verbose: verbose, evalFreq: evalFreq)
    }
}
