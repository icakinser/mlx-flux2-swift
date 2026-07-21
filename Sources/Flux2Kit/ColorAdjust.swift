// Flux2Kit — pixel-space color adjustment (the correct domain for coloring/exposure edits).
// Operates on decoded RGB in [0, 1]; exact, instant, no model. Used by recolor() when no prompt
// is given, and for global grade of any generated/decoded image. For a latent-space A/B, see
// LatentColorExperimental.swift (opt-in, unreliable by design).

import Foundation
import MLX

// MARK: - Curves (elementwise, RGB in [0,1])

/// Exposure in stops: multiply by 2^stops.
public func applyExposure(_ rgb: MLXArray, stops: Float) -> MLXArray {
    if stops == 0 { return rgb }
    return clip(rgb * MLXArray(Float(pow(2.0, Double(stops)))), min: 0, max: 1)
}

/// Contrast around mid-gray 0.5. `c == 1` is identity.
public func applyContrast(_ rgb: MLXArray, _ c: Float) -> MLXArray {
    if c == 1 { return rgb }
    return clip((rgb - 0.5) * c + 0.5, min: 0, max: 1)
}

/// Gamma: output = input^(1/gamma). `gamma > 1` brightens mid-tones. `gamma == 1` is identity.
public func applyGamma(_ rgb: MLXArray, _ gamma: Float) -> MLXArray {
    if gamma == 1 { return rgb }
    let g = max(gamma, 1e-4)
    return MLX.pow(clip(rgb, min: 0, max: 1), MLXArray(Float(1.0 / g)))
}

/// Rotate hue by `hue` (fraction of the wheel, [0,1)) and scale saturation by `saturation`.
public func applyHueSaturation(_ rgb: MLXArray, hue: Float, saturation: Float) -> MLXArray {
    if hue == 0 && saturation == 1 { return rgb }
    var (h, s, v) = rgbToHsv(rgb)
    if hue != 0 {
        h = h + hue
        h = h - MLX.floor(h)  // wrap to [0,1)
    }
    if saturation != 1 {
        s = clip(s * saturation, min: 0, max: 1)
    }
    return hsvToRgb(h, s, v)
}

/// Apply the full grade in a fixed order: exposure → contrast → gamma → hue/saturation.
public func adjustColor(
    _ rgb: MLXArray, exposure: Float, contrast: Float, gamma: Float, hue: Float, saturation: Float
) -> MLXArray {
    var x = rgb
    x = applyExposure(x, stops: exposure)
    x = applyContrast(x, contrast)
    x = applyGamma(x, gamma)
    x = applyHueSaturation(x, hue: hue, saturation: saturation)
    return clip(x, min: 0, max: 1)
}

/// Composite `adjusted` over `base` within a (H,W) mask (1 = fully adjusted). `base`/`adjusted`
/// are (H,W,3); the mask broadcasts over the channel axis.
public func compositeMasked(base: MLXArray, adjusted: MLXArray, mask: MLXArray) -> MLXArray {
    let m = expandedDimensions(mask, axis: -1)  // (H,W,1)
    return base * (1 - m) + adjusted * m
}

// MARK: - RGB <-> HSV (elementwise; rgb is (H,W,3) in [0,1])

/// Returns (h, s, v), each (H,W), all in [0,1].
public func rgbToHsv(_ rgb: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
    let r = rgb[0..., 0..., 0]
    let g = rgb[0..., 0..., 1]
    let b = rgb[0..., 0..., 2]
    let cmax = MLX.maximum(MLX.maximum(r, g), b)
    let cmin = MLX.minimum(MLX.minimum(r, g), b)
    let delta = cmax - cmin
    let deltaSafe = MLX.maximum(delta, MLXArray(Float(1e-8)))
    // Mutually exclusive "which channel is max" selectors (r has priority on ties).
    let fr = MLX.equal(cmax, r).asType(.float32)
    let fg = MLX.equal(cmax, g).asType(.float32) * (1 - fr)
    let fb = (1 - fr) * (1 - fg)
    let hr = (g - b) / deltaSafe
    let hg = (b - r) / deltaSafe + 2
    let hb = (r - g) / deltaSafe + 4
    var h = (fr * hr + fg * hg + fb * hb) / 6
    h = h - MLX.floor(h)  // mod 1 (handles negatives)
    let deltaZero = MLX.equal(delta, MLXArray(Float(0))).asType(.float32)
    h = h * (1 - deltaZero)  // undefined hue -> 0
    let s = delta / MLX.maximum(cmax, MLXArray(Float(1e-8)))
    let v = cmax
    return (h, s, v)
}

// MARK: - Model-free filters (rgb is (H,W,3) in [0,1])

/// Per-channel mean/std over the spatial axes, shape (3,).
private func channelStats(_ rgb: MLXArray) -> (mean: MLXArray, std: MLXArray) {
    let m = MLX.mean(rgb, axes: [0, 1])
    let v = MLX.mean(MLX.square(rgb - m), axes: [0, 1])
    return (m, MLX.sqrt(v))
}

/// Reinhard color transfer: shift `source` to the per-channel mean/std of `reference`.
public func matchColor(_ source: MLXArray, reference: MLXArray) -> MLXArray {
    let (sm, ss) = channelStats(source)
    let (rm, rs) = channelStats(reference)
    let out = (source - sm) / MLX.maximum(ss, MLXArray(Float(1e-5))) * rs + rm
    return clip(out, min: 0, max: 1)
}

/// Luminance grayscale (replicated to 3 channels).
public func toGrayscale(_ rgb: MLXArray) -> MLXArray {
    let r = rgb[0..., 0..., 0]
    let g = rgb[0..., 0..., 1]
    let b = rgb[0..., 0..., 2]
    let lum = 0.299 * r + 0.587 * g + 0.114 * b
    return MLX.stacked([lum, lum, lum], axis: -1)
}

/// Classic sepia tone matrix.
public func toSepia(_ rgb: MLXArray) -> MLXArray {
    let r = rgb[0..., 0..., 0]
    let g = rgb[0..., 0..., 1]
    let b = rgb[0..., 0..., 2]
    let nr = clip(0.393 * r + 0.769 * g + 0.189 * b, min: 0, max: 1)
    let ng = clip(0.349 * r + 0.686 * g + 0.168 * b, min: 0, max: 1)
    let nb = clip(0.272 * r + 0.534 * g + 0.131 * b, min: 0, max: 1)
    return MLX.stacked([nr, ng, nb], axis: -1)
}

/// Invert.
public func invertColor(_ rgb: MLXArray) -> MLXArray { 1 - rgb }

/// Per-channel box blur (reuses the 2-D `boxBlur`).
public func blurRGB(_ rgb: MLXArray, passes: Int) -> MLXArray {
    let r = boxBlur(rgb[0..., 0..., 0], passes: passes)
    let g = boxBlur(rgb[0..., 0..., 1], passes: passes)
    let b = boxBlur(rgb[0..., 0..., 2], passes: passes)
    return MLX.stacked([r, g, b], axis: -1)
}

/// Unsharp-mask sharpen: `rgb + amount * (rgb - blur(rgb))`. `amount == 0` is identity.
public func sharpen(_ rgb: MLXArray, amount: Float) -> MLXArray {
    if amount == 0 { return rgb }
    return clip(rgb + amount * (rgb - blurRGB(rgb, passes: 1)), min: 0, max: 1)
}

/// Inverse of `rgbToHsv`. h,s,v are (H,W) in [0,1]; returns (H,W,3).
public func hsvToRgb(_ h: MLXArray, _ s: MLXArray, _ v: MLXArray) -> MLXArray {
    let h6 = h * 6
    let i = MLX.floor(h6)
    let f = h6 - i
    let p = v * (1 - s)
    let q = v * (1 - f * s)
    let t = v * (1 - (1 - f) * s)
    let im = i - MLX.floor(i / 6) * 6  // safety: fold into [0,6)
    func sel(_ k: Int) -> MLXArray { MLX.equal(im, MLXArray(Float(k))).asType(.float32) }
    let m0 = sel(0), m1 = sel(1), m2 = sel(2), m3 = sel(3), m4 = sel(4), m5 = sel(5)
    let r = m0 * v + m1 * q + m2 * p + m3 * p + m4 * t + m5 * v
    let g = m0 * t + m1 * v + m2 * v + m3 * q + m4 * p + m5 * p
    let b = m0 * p + m1 * p + m2 * t + m3 * v + m4 * v + m5 * q
    return MLX.stacked([r, g, b], axis: -1)
}
