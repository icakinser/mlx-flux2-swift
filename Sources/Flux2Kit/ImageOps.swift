// Flux2Kit — model-free image operations: geometric transforms + a composable op pipeline that
// covers every instant, no-model edit (geometry + color + effects). Run these without constructing
// Flux2Pipeline at all — no weights load, no VAE, no waiting on a big model.

import CoreGraphics
import Foundation
import MLX

/// A single model-free operation. `applyImageOps` runs a list in order.
public enum ImageOp {
    case resize(Int, Int)
    case scale(Float)
    case crop(Int, Int, Int, Int)
    case rotate(Int)  // 90 / 180 / 270 (clockwise)
    case flip(String)  // "h" / "v" / "hv"
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
public func applyImageOps(_ source: CGImage, _ ops: [ImageOp]) throws -> CGImage {
    var img = source
    for op in ops {
        switch op {
        case .resize(let w, let h):
            img = try resizeExactRGB(img, width: max(1, w), height: max(1, h))
        case .scale(let f):
            img = try resizeExactRGB(
                img, width: max(1, Int(Float(img.width) * f)),
                height: max(1, Int(Float(img.height) * f)))
        case .crop(let x, let y, let w, let h):
            img = try cropImage(img, x: x, y: y, width: w, height: h)
        case .rotate(let d): img = try rotateImage(img, degrees: d)
        case .flip(let m): img = try flipImage(img, mode: m)
        case .fit16: img = try centerCropToMultiple(img, 16)
        case .pixelate(let b): img = try pixelateImage(img, block: b)
        case .grayscale: img = try mapRGB(img) { toGrayscale($0) }
        case .sepia: img = try mapRGB(img) { toSepia($0) }
        case .invert: img = try mapRGB(img) { invertColor($0) }
        case .autoContrast: img = try mapRGB(img) { autoContrast($0) }
        case .sharpen(let a): img = try mapRGB(img) { sharpen($0, amount: a) }
        case .blur(let p): img = try mapRGB(img) { blurRGB($0, passes: p) }
        case .brightness(let b): img = try mapRGB(img) { adjustBrightness($0, b) }
        case .saturation(let s):
            img = try mapRGB(img) { applyHueSaturation($0, hue: 0, saturation: s) }
        case .temperature(let t): img = try mapRGB(img) { adjustTemperature($0, t) }
        case .posterize(let n): img = try mapRGB(img) { posterize($0, levels: n) }
        case .threshold(let t): img = try mapRGB(img) { threshold($0, t) }
        case .vignette(let a): img = try mapRGB(img) { vignette($0, amount: a) }
        case .recolor(let h, let s, let e, let c, let g):
            img = try mapRGB(img) {
                adjustColor($0, exposure: e, contrast: c, gamma: g, hue: h, saturation: s)
            }
        case .matchColor(let path):
            guard let ref = try loadImages([URL(fileURLWithPath: path)]).first else {
                throw Flux2Error.generationFailed("could not load match-color reference: \(path)")
            }
            let ref01 = (try cgImageToArray(ref) + 1) / 2
            img = try mapRGB(img) { matchColor($0, reference: ref01) }
        }
    }
    return img
}

/// Convert a CGImage to RGB [0,1], apply `f`, convert back.
private func mapRGB(_ img: CGImage, _ f: (MLXArray) -> MLXArray) throws -> CGImage {
    let rgb01 = (try cgImageToArray(img) + 1) / 2
    return try arrayToCGImage(f(rgb01) * 2 - 1)
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

func flipImage(_ img: CGImage, mode: String) throws -> CGImage {
    let w = img.width
    let h = img.height
    guard
        let ctx = CGContext(
            data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
            space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
    else { throw Flux2Error.generationFailed("flip context failed") }
    let m = mode.lowercased()
    if m.contains("h") { ctx.translateBy(x: CGFloat(w), y: 0); ctx.scaleBy(x: -1, y: 1) }
    if m.contains("v") { ctx.translateBy(x: 0, y: CGFloat(h)); ctx.scaleBy(x: 1, y: -1) }
    ctx.draw(img, in: CGRect(x: 0, y: 0, width: w, height: h))
    guard let out = ctx.makeImage() else { throw Flux2Error.generationFailed("flip failed") }
    return out
}

func pixelateImage(_ img: CGImage, block: Int) throws -> CGImage {
    let b = max(2, block)
    let w = img.width
    let h = img.height
    let small = try resizeExactRGB(img, width: max(1, w / b), height: max(1, h / b))
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
