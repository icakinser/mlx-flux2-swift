// Flux2Kit — model-free image operations: geometric transforms + a composable op pipeline that
// covers every instant, no-model edit (geometry + color + effects). Run these without constructing
// Flux2Pipeline at all — no weights load, no VAE, no waiting on a big model.

import CoreGraphics
import Foundation
import MLX

/// Axis (or axes) to mirror across in a flip op.
public enum FlipMode: String, Sendable, CaseIterable {
    case horizontal = "h"
    case vertical = "v"
    case both = "hv"

    /// Parse a user-supplied flip spec ("h"/"v"/"hv" and long forms). Returns nil on anything else.
    public init?(parsing raw: String) {
        switch raw.lowercased() {
        case "h", "horizontal": self = .horizontal
        case "v", "vertical": self = .vertical
        case "hv", "vh", "both": self = .both
        default: return nil
        }
    }

    var flipsHorizontally: Bool { self == .horizontal || self == .both }
    var flipsVertically: Bool { self == .vertical || self == .both }
}

/// A single model-free operation. `applyImageOps` runs a list in order.
public enum ImageOp {
    case resize(Int, Int)
    case scale(Float)
    case crop(Int, Int, Int, Int)
    case rotate(Int)  // 90 / 180 / 270 (clockwise)
    case flip(FlipMode)
    case fit16  // center-crop to a multiple of 16
    case pixelate(Int)
    case grayscale
    case sepia
    case invert
    case autoContrast
    case sharpen(Float)
    case blur(Int)
    case brightness(Float)
    case saturation(Float)
    case temperature(Float)
    case posterize(Int)
    case threshold(Float)
    case vignette(Float)
    case recolor(hue: Float, sat: Float, exp: Float, contrast: Float, gamma: Float)
    case matchColor(String)  // reference image path
}

/// Apply model-free ops to a CGImage in order. No model, no VAE — pure CoreGraphics + elementwise MLX.
///
/// Consecutive elementwise (MLX) ops are fused: the working image is held as an rgb01 `MLXArray` and
/// only round-tripped through `CGImage` when a geometric op needs CoreGraphics, or at the very end.
/// This avoids a `cgImageToArray`/`arrayToCGImage` pair (and a full-frame uint8 re-quantization) per
/// op. To preserve the previous path's clamp semantics, the fused value is clipped to `[0,1]` between
/// ops (matching the old `arrayToCGImage` clip); the only intentional difference is that lossy uint8
/// re-quantization no longer happens between fused ops (a precision improvement, not a change in
/// intent — these ops are model-free creative effects with their own unit-tested contracts).
///
/// When `mask` is provided, ops run on the full frame and the result is composited back over the
/// source inside the (optionally feathered / inverted) mask — same convention as
/// ``Flux2Pipeline/recolor`` / ``Flux2Pipeline/applyPixelFilter`` (white = apply). The final size
/// must match `source` (geometry ops that change dimensions are rejected when a mask is set).
public func applyImageOps(
    _ source: CGImage,
    _ ops: [ImageOp],
    mask: CGImage? = nil,
    invertMask: Bool = false,
    maskFeather: Int = 2
) throws -> CGImage {
    var img = source
    var pending: MLXArray? = nil  // rgb01 in [0,1], shape (H, W, 3), when a fused run is in flight

    func loadPending() throws -> MLXArray {
        if let p = pending { return p }
        return (try cgImageToArray(img) + 1) / 2
    }
    func flush() throws {
        if let p = pending {
            img = try arrayToCGImage(p * 2 - 1)
            pending = nil
        }
    }
    func fuse(_ f: (MLXArray) -> MLXArray) throws {
        pending = clip(f(try loadPending()), min: 0.0, max: 1.0)
    }

    for op in ops {
        switch op {
        case .resize(let w, let h):
            try flush()
            img = try resizeHighQuality(img, width: max(1, w), height: max(1, h))
        case .scale(let f):
            try flush()
            img = try resizeHighQuality(
                img, width: max(1, Int(Float(img.width) * f)),
                height: max(1, Int(Float(img.height) * f)))
        case .crop(let x, let y, let w, let h):
            try flush()
            img = try cropImage(img, x: x, y: y, width: w, height: h)
        case .rotate(let d):
            try flush()
            img = try rotateImage(img, degrees: d)
        case .flip(let m):
            try flush()
            img = try flipImage(img, mode: m)
        case .fit16:
            try flush()
            img = try centerCropToMultiple(img, 16)
        case .pixelate(let b):
            try flush()
            img = try pixelateImage(img, block: b)
        case .grayscale: try fuse { toGrayscale($0) }
        case .sepia: try fuse { toSepia($0) }
        case .invert: try fuse { invertColor($0) }
        case .autoContrast: try fuse { autoContrast($0) }
        case .sharpen(let a): try fuse { sharpen($0, amount: a) }
        case .blur(let p): try fuse { blurRGB($0, passes: p) }
        case .brightness(let b): try fuse { adjustBrightness($0, b) }
        case .saturation(let s):
            try fuse { applyHueSaturation($0, hue: 0, saturation: s) }
        case .temperature(let t): try fuse { adjustTemperature($0, t) }
        case .posterize(let n): try fuse { posterize($0, levels: n) }
        case .threshold(let t): try fuse { threshold($0, t) }
        case .vignette(let a): try fuse { vignette($0, amount: a) }
        case .recolor(let h, let s, let e, let c, let g):
            try fuse {
                adjustColor($0, exposure: e, contrast: c, gamma: g, hue: h, saturation: s)
            }
        case .matchColor(let path):
            guard let ref = try loadImages([URL(fileURLWithPath: path)]).first else {
                throw Flux2Error.generationFailed("could not load match-color reference: \(path)")
            }
            let ref01 = (try cgImageToArray(ref) + 1) / 2
            try fuse { matchColor($0, reference: ref01) }
        }
    }
    try flush()

    guard let mask else { return img }
    guard img.width == source.width, img.height == source.height else {
        throw Flux2Error.generationFailed(
            "masked image ops require size-preserving ops (source \(source.width)x\(source.height), "
                + "result \(img.width)x\(img.height))")
    }
    let base = (try cgImageToArray(source) + 1) / 2
    let adjusted = (try cgImageToArray(img) + 1) / 2
    var m = try maskGridFromCGImage(mask, width: source.width, height: source.height)
    if maskFeather > 0 { m = boxBlur(m, passes: maskFeather) }
    if invertMask { m = 1 - m }
    let composited = compositeMasked(base: base, adjusted: adjusted, mask: m)
    return try arrayToCGImage(composited * 2 - 1)
}

// MARK: - Geometric (CoreGraphics)

func cropImage(_ img: CGImage, x: Int, y: Int, width w: Int, height h: Int) throws -> CGImage {
    let cx = max(0, min(x, img.width - 1))
    let cy = max(0, min(y, img.height - 1))
    let cw = max(1, min(w, img.width - cx))
    let ch = max(1, min(h, img.height - cy))
    guard let cropped = img.cropping(to: CGRect(x: cx, y: cy, width: cw, height: ch)) else {
        throw Flux2Error.generationFailed("crop failed")
    }
    return cropped
}

func rotateImage(_ img: CGImage, degrees: Int) throws -> CGImage {
    let d = ((degrees % 360) + 360) % 360
    guard d == 90 || d == 180 || d == 270 else {
        if d == 0 { return img }
        throw Flux2Error.generationFailed("rotate supports 90/180/270, got \(degrees)")
    }
    let w = img.width
    let h = img.height
    let (nw, nh) = d == 180 ? (w, h) : (h, w)
    guard
        let ctx = CGContext(
            data: nil, width: nw, height: nh, bitsPerComponent: 8, bytesPerRow: nw * 4,
            space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
    else { throw Flux2Error.generationFailed("rotate context failed") }
    ctx.translateBy(x: CGFloat(nw) / 2, y: CGFloat(nh) / 2)
    ctx.rotate(by: CGFloat(-Double(d) * .pi / 180))
    ctx.draw(
        img, in: CGRect(x: -CGFloat(w) / 2, y: -CGFloat(h) / 2, width: CGFloat(w), height: CGFloat(h)))
    guard let out = ctx.makeImage() else { throw Flux2Error.generationFailed("rotate failed") }
    return out
}

func flipImage(_ img: CGImage, mode: FlipMode) throws -> CGImage {
    let w = img.width
    let h = img.height
    guard
        let ctx = CGContext(
            data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
            space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
    else { throw Flux2Error.generationFailed("flip context failed") }
    if mode.flipsHorizontally { ctx.translateBy(x: CGFloat(w), y: 0); ctx.scaleBy(x: -1, y: 1) }
    if mode.flipsVertically { ctx.translateBy(x: 0, y: CGFloat(h)); ctx.scaleBy(x: 1, y: -1) }
    ctx.draw(img, in: CGRect(x: 0, y: 0, width: w, height: h))
    guard let out = ctx.makeImage() else { throw Flux2Error.generationFailed("flip failed") }
    return out
}

func pixelateImage(_ img: CGImage, block: Int) throws -> CGImage {
    let b = max(2, block)
    let w = img.width
    let h = img.height
    let small = try resizeHighQuality(img, width: max(1, w / b), height: max(1, h / b))
    // Upscale back with nearest-neighbor for the blocky look.
    guard
        let ctx = CGContext(
            data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
            space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
    else { throw Flux2Error.generationFailed("pixelate context failed") }
    ctx.interpolationQuality = .none
    ctx.draw(small, in: CGRect(x: 0, y: 0, width: w, height: h))
    guard let out = ctx.makeImage() else { throw Flux2Error.generationFailed("pixelate failed") }
    return out
}
