// Flux2Kit — native MLX Swift port of FLUX.2 [klein], derived from scf4/mlx-flux2 (MIT). Image I/O via CoreGraphics.
// 2026-07-19 EDT | PERMANENT (Flux2Kit t2i port) — numerical parity with the reference implementation is the
// contract; do not refactor without re-running the parity harness.

import CoreGraphics
import Foundation
import ImageIO
import MLX
import UniformTypeIdentifiers

// High-quality CoreGraphics resampling (LANCZOS-equivalent downscaling) is NOT bit-identical to the
// reference — this only affects the reference-image (kontext) path, never text-to-image. Flagged for
// the parity harness.
public func capPixels(_ img: CGImage, _ k: Int) throws -> CGImage {
    let w = img.width
    let h = img.height
    if w * h <= k {
        return img
    }
    let scale = (Double(k) / Double(w * h)).squareRoot()
    let newW = Int(Double(w) * scale)
    let newH = Int(Double(h) * scale)
    return try renderResized(img, width: newW, height: newH)
}

// Rejects images that are too small or have an extreme aspect ratio.
public func capMinPixels(_ img: CGImage, maxAr: Double = 8.0, minSidelength: Int = 64) throws -> CGImage {
    let w = img.width
    let h = img.height
    if w < minSidelength || h < minSidelength {
        throw Flux2Error.generationFailed("Image too small: \(w)x\(h)")
    }
    if Double(w) / Double(h) > maxAr || Double(h) / Double(w) > maxAr {
        throw Flux2Error.generationFailed("Image aspect ratio too extreme: \(w)x\(h)")
    }
    return img
}

// Center-crops the image down to the nearest multiple of `mult` on each side.
public func centerCropToMultiple(_ img: CGImage, _ mult: Int) throws -> CGImage {
    let w = img.width
    let h = img.height
    let newW = (w / mult) * mult
    let newH = (h / mult) * mult
    let left = (w - newW) / 2
    let top = (h - newH) / 2
    guard let cropped = img.cropping(to: CGRect(x: left, y: top, width: newW, height: newH)) else {
        throw Flux2Error.generationFailed("Center crop failed for \(w)x\(h) -> \(newW)x\(newH)")
    }
    return cropped
}

// RGB float32 in [-1, 1]
public func cgImageToArray(_ img: CGImage) throws -> MLXArray {
    let w = img.width
    let h = img.height
    let bytesPerRow = w * 4
    var pixels = [UInt8](repeating: 0, count: h * bytesPerRow)
    // 2026-07-20 EDT | PERMANENT (review finding) — the context must not outlive the
    // scoped pointer: &pixels bridging is only guaranteed for the initializer call, so
    // create AND draw inside withUnsafeMutableBytes (no numeric change).
    try pixels.withUnsafeMutableBytes { buffer in
        guard
            let context = CGContext(
                data: buffer.baseAddress, width: w, height: h, bitsPerComponent: 8,
                bytesPerRow: bytesPerRow, space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        else {
            throw Flux2Error.generationFailed("Could not create bitmap context")
        }
        context.draw(img, in: CGRect(x: 0, y: 0, width: w, height: h))
    }

    var rgb = [Float](repeating: 0, count: h * w * 3)
    for y in 0 ..< h {
        for x in 0 ..< w {
            let src = y * bytesPerRow + x * 4
            let dst = (y * w + x) * 3
            rgb[dst] = Float(pixels[src])
            rgb[dst + 1] = Float(pixels[src + 1])
            rgb[dst + 2] = Float(pixels[src + 2])
        }
    }
    return MLXArray(rgb, [h, w, 3]) / 127.5 - 1.0
}

// Clip to [-1, 1], scale to uint8 RGB
public func arrayToCGImage(_ arr: MLXArray) throws -> CGImage {
    let clipped = clip(arr, min: -1.0, max: 1.0)
    let scaled = ((clipped + 1.0) * 127.5).asType(.uint8)
    eval(scaled)
    let h = scaled.dim(0)
    let w = scaled.dim(1)
    let rgb: [UInt8] = scaled.asArray(UInt8.self)

    var rgba = [UInt8](repeating: 255, count: h * w * 4)
    for i in 0 ..< (h * w) {
        rgba[i * 4] = rgb[i * 3]
        rgba[i * 4 + 1] = rgb[i * 3 + 1]
        rgba[i * 4 + 2] = rgb[i * 3 + 2]
    }
    let bytesPerRow = w * 4
    guard
        let provider = CGDataProvider(data: Data(rgba) as CFData),
        let image = CGImage(
            width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: bytesPerRow,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
            provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)
    else {
        throw Flux2Error.generationFailed("Could not create CGImage from array")
    }
    return image
}

// Default image preprocessing pipeline: validate size/aspect ratio, cap max pixels, center-crop to
// a multiple of `ensureMultiple`, then convert to a normalized MLXArray.
public func defaultPrep(_ img: CGImage, limitPixels: Int?, ensureMultiple: Int = 16) throws -> MLXArray {
    // EXIF transpose happens at load time in loadImages (CGImageSource applies orientation);
    // a CGImage in memory carries no orientation tag once that transpose has been applied.
    var image = try capMinPixels(img)
    if let limitPixels {
        image = try capPixels(image, limitPixels)
    }
    image = try centerCropToMultiple(image, ensureMultiple)
    return try cgImageToArray(image)
}

// EXIF orientation is applied at load
public func loadImages(_ paths: [URL]) throws -> [CGImage] {
    var images: [CGImage] = []
    for p in paths {
        guard let source = CGImageSourceCreateWithURL(p as CFURL, nil) else {
            throw Flux2Error.loadFailed("Could not open image at \(p.path)")
        }
        let thumbOptions: [CFString: Any] = [
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceThumbnailMaxPixelSize: max(
                CGImageSourceCopyPropertiesAtIndex(source, 0, nil).flatMap {
                    ($0 as? [CFString: Any])?[kCGImagePropertyPixelWidth] as? Int
                } ?? 16384,
                CGImageSourceCopyPropertiesAtIndex(source, 0, nil).flatMap {
                    ($0 as? [CFString: Any])?[kCGImagePropertyPixelHeight] as? Int
                } ?? 16384),
        ]
        guard
            let image = CGImageSourceCreateThumbnailAtIndex(source, 0, thumbOptions as CFDictionary)
        else {
            throw Flux2Error.loadFailed("Could not decode image at \(p.path)")
        }
        images.append(image)
    }
    return images
}

/// PNG save helper for the CLI.
public func savePNG(_ img: CGImage, to url: URL) throws {
    guard
        let destination = CGImageDestinationCreateWithURL(
            url as CFURL, UTType.png.identifier as CFString, 1, nil)
    else {
        throw Flux2Error.generationFailed("Could not create PNG destination at \(url.path)")
    }
    CGImageDestinationAddImage(destination, img, nil)
    guard CGImageDestinationFinalize(destination) else {
        throw Flux2Error.generationFailed("Could not write PNG at \(url.path)")
    }
}

private func renderResized(_ img: CGImage, width: Int, height: Int) throws -> CGImage {
    guard
        let context = CGContext(
            data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
    else {
        throw Flux2Error.generationFailed("Could not create resize context")
    }
    context.interpolationQuality = .high
    context.draw(img, in: CGRect(x: 0, y: 0, width: width, height: height))
    guard let resized = context.makeImage() else {
        throw Flux2Error.generationFailed("Resize failed")
    }
    return resized
}
