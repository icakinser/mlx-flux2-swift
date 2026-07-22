// Flux2Kit — additional model-free photo effects (elementwise on RGB in [0,1], shape (H,W,3)).
// Instant, no model, no VAE. Composed by applyImageOps (ImageOps.swift).

import Foundation
import MLX

/// Add a brightness offset in [-1, 1].
public func adjustBrightness(_ rgb: MLXArray, _ b: Float) -> MLXArray {
    if b == 0 { return rgb }
    return clip(rgb + b, min: 0, max: 1)
}

/// Warm/cool white balance. `t` in [-1, 1]: positive warms (more red, less blue).
public func adjustTemperature(_ rgb: MLXArray, _ t: Float) -> MLXArray {
    if t == 0 { return rgb }
    let scale = MLXArray([1 + 0.3 * t, 1, 1 - 0.3 * t], [3])
    return clip(rgb * scale, min: 0, max: 1)
}

/// Reduce to `levels` tonal steps per channel (levels >= 2).
public func posterize(_ rgb: MLXArray, levels: Int) -> MLXArray {
    let n = Float(max(2, levels) - 1)
    return MLX.round(clip(rgb, min: 0, max: 1) * n) / n
}

/// Binarize on luminance at `t` (0..1): pixels brighter than `t` become white, else black.
public func threshold(_ rgb: MLXArray, _ t: Float) -> MLXArray {
    let r = rgb[0..., 0..., 0]
    let g = rgb[0..., 0..., 1]
    let b = rgb[0..., 0..., 2]
    let lum = 0.299 * r + 0.587 * g + 0.114 * b
    let on = MLX.greater(lum, MLXArray(t)).asType(.float32)
    return MLX.stacked([on, on, on], axis: -1)
}

/// Radial vignette: darken toward the corners by up to `amount` (0..1).
public func vignette(_ rgb: MLXArray, amount: Float) -> MLXArray {
    if amount == 0 { return rgb }
    let h = rgb.dim(0)
    let w = rgb.dim(1)
    let ys = (MLXArray(0 ..< h).asType(.float32) / Float(max(1, h - 1)) - 0.5).reshaped([h, 1])
    let xs = (MLXArray(0 ..< w).asType(.float32) / Float(max(1, w - 1)) - 0.5).reshaped([1, w])
    let dist = MLX.sqrt(ys * ys + xs * xs) / 0.7071  // 0 at center, ~1 at corners
    let factor = clip(1 - amount * clip(dist, min: 0, max: 1), min: 0, max: 1)
    return rgb * expandedDimensions(factor, axis: -1)
}

/// Stretch the global min/max to [0,1] (auto-levels).
public func autoContrast(_ rgb: MLXArray) -> MLXArray {
    let mn = MLX.min(rgb)
    let mx = MLX.max(rgb)
    return clip((rgb - mn) / MLX.maximum(mx - mn, MLXArray(Float(1e-5))), min: 0, max: 1)
}
