// Flux2Kit — built-in mask generation and morphology, so callers don't need an external mask PNG.
// Coordinates use the image convention: (0,0) is the top-left corner. Masks are white = edit region,
// black = keep (matching generateInpaint). Morphology reuses the boxBlur neighbor-slice pattern.

import CoreGraphics
import Foundation
import MLX

/// White filled rectangle on a black `width`×`height` canvas. `(x, y)` is the top-left of the box.
public func makeBoxMask(width: Int, height: Int, x: Int, y: Int, boxWidth: Int, boxHeight: Int) throws
    -> CGImage
{
    try drawMask(width: width, height: height) { ctx in
        ctx.fill(CGRect(x: x, y: height - y - boxHeight, width: boxWidth, height: boxHeight))
    }
}

/// White filled ellipse inscribed in the given rect (top-left origin) on a black canvas.
public func makeEllipseMask(
    width: Int, height: Int, x: Int, y: Int, boxWidth: Int, boxHeight: Int
) throws -> CGImage {
    try drawMask(width: width, height: height) { ctx in
        ctx.fillEllipse(in: CGRect(x: x, y: height - y - boxHeight, width: boxWidth, height: boxHeight))
    }
}

private func drawMask(width w: Int, height h: Int, _ draw: (CGContext) -> Void) throws -> CGImage {
    guard
        let ctx = CGContext(
            data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
            space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
    else {
        throw Flux2Error.generationFailed("Could not create mask context")
    }
    ctx.setFillColor(CGColor(red: 0, green: 0, blue: 0, alpha: 1))
    ctx.fill(CGRect(x: 0, y: 0, width: w, height: h))
    ctx.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
    draw(ctx)
    guard let img = ctx.makeImage() else {
        throw Flux2Error.generationFailed("Could not render mask")
    }
    return img
}

/// Grow the white region by `iterations` (3×3 max-pool per iteration).
public func dilateMask(_ img: CGImage, iterations: Int) throws -> CGImage {
    try morphMask(img, iterations: iterations, dilate: true)
}

/// Shrink the white region by `iterations` (3×3 min-pool per iteration).
public func erodeMask(_ img: CGImage, iterations: Int) throws -> CGImage {
    try morphMask(img, iterations: iterations, dilate: false)
}

private func morphMask(_ img: CGImage, iterations: Int, dilate: Bool) throws -> CGImage {
    guard iterations > 0 else { return img }
    var grid = try maskGridFromCGImage(img, width: img.width, height: img.height)  // (H,W) [0,1]
    let h = grid.dim(0)
    let w = grid.dim(1)
    for _ in 0 ..< iterations {
        let top = grid[0 ..< 1, 0...]
        let bot = grid[(h - 1) ..< h, 0...]
        var p = concatenated([top, grid, bot], axis: 0)
        let left = p[0..., 0 ..< 1]
        let right = p[0..., (w - 1) ..< w]
        p = concatenated([left, p, right], axis: 1)
        var acc = p[0 ..< h, 0 ..< w]
        for dy in 0 ..< 3 {
            for dx in 0 ..< 3 {
                if dy == 0 && dx == 0 { continue }
                let n = p[dy ..< (dy + h), dx ..< (dx + w)]
                acc = dilate ? MLX.maximum(acc, n) : MLX.minimum(acc, n)
            }
        }
        grid = acc
    }
    return try maskGridToCGImage(grid)
}

/// Convert a `(H, W)` mask grid in `[0, 1]` back into a grayscale CGImage.
public func maskGridToCGImage(_ grid: MLXArray) throws -> CGImage {
    let scaled = (clip(grid, min: 0, max: 1) * 255).asType(.uint8)
    eval(scaled)
    let h = scaled.dim(0)
    let w = scaled.dim(1)
    let vals: [UInt8] = scaled.asArray(UInt8.self)
    var rgba = [UInt8](repeating: 255, count: h * w * 4)
    for i in 0 ..< (h * w) {
        let v = vals[i]
        rgba[i * 4] = v
        rgba[i * 4 + 1] = v
        rgba[i * 4 + 2] = v
    }
    guard
        let provider = CGDataProvider(data: Data(rgba) as CFData),
        let image = CGImage(
            width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: w * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
            provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)
    else {
        throw Flux2Error.generationFailed("Could not create mask CGImage")
    }
    return image
}
