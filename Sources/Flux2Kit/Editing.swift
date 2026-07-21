// Flux2Kit — contributor-friendly editing API. Thin wrappers over generateInpaint (Inpaint.swift)
// and the pixel-space color ops (ColorAdjust.swift). Removal, background replacement, region edit,
// and object addition are all the same mask-guided inpaint with different prompts / mask polarity.

import CoreGraphics
import Foundation
import MLX

extension Flux2Pipeline {

    /// Default prompt used by `removeObject` — asks the model to continue the surroundings.
    public static let removalPrompt =
        "clean, seamless background that naturally continues the surrounding scene, no object"

    /// General mask-guided edit: regenerate the white region of `mask` conditioned on `prompt`.
    public func editRegion(
        source: CGImage, mask: CGImage, prompt: String,
        strength: Double = 0.85, width: Int, height: Int,
        numSteps: Int = defaultSteps, guidance: Double = Double(defaultGuidance),
        seed: UInt64? = nil, invertMask: Bool = false, maskFeather: Int = 1,
        verbose: Bool = false, evalFreq: Int = 1
    ) throws -> CGImage {
        try generateInpaint(
            prompt: prompt, source: source, mask: mask, strength: strength,
            width: width, height: height, numSteps: numSteps, guidance: guidance, seed: seed,
            invertMask: invertMask, maskFeather: maskFeather, verbose: verbose, evalFreq: evalFreq)
    }

    /// Remove whatever is under the white region of `mask` and fill with continued background.
    public func removeObject(
        source: CGImage, mask: CGImage,
        strength: Double = 0.9, width: Int, height: Int,
        numSteps: Int = defaultSteps, guidance: Double = Double(defaultGuidance),
        seed: UInt64? = nil, maskFeather: Int = 2,
        verbose: Bool = false, evalFreq: Int = 1
    ) throws -> CGImage {
        try generateInpaint(
            prompt: Self.removalPrompt, source: source, mask: mask, strength: strength,
            width: width, height: height, numSteps: numSteps, guidance: guidance, seed: seed,
            invertMask: false, maskFeather: maskFeather, verbose: verbose, evalFreq: evalFreq)
    }

    /// Replace the background (everything OUTSIDE the white subject region) with `prompt`.
    /// The mask marks the SUBJECT to keep; `invertMask` is applied so the background is edited.
    public func replaceBackground(
        source: CGImage, subjectMask: CGImage, prompt: String,
        strength: Double = 0.9, width: Int, height: Int,
        numSteps: Int = defaultSteps, guidance: Double = Double(defaultGuidance),
        seed: UInt64? = nil, maskFeather: Int = 2,
        verbose: Bool = false, evalFreq: Int = 1
    ) throws -> CGImage {
        try generateInpaint(
            prompt: prompt, source: source, mask: subjectMask, strength: strength,
            width: width, height: height, numSteps: numSteps, guidance: guidance, seed: seed,
            invertMask: true, maskFeather: maskFeather, verbose: verbose, evalFreq: evalFreq)
    }

    /// Add an object described by `prompt` into the white region of `mask`. The model synthesizes it
    /// in context (lighting, shadows, perspective). An optional `referenceImage` is passed as a
    /// kontext reference token to steer appearance.
    public func addObject(
        source: CGImage, mask: CGImage, prompt: String, referenceImage: CGImage? = nil,
        strength: Double = 0.85, width: Int, height: Int,
        numSteps: Int = defaultSteps, guidance: Double = Double(defaultGuidance),
        seed: UInt64? = nil, maskFeather: Int = 1,
        verbose: Bool = false, evalFreq: Int = 1
    ) throws -> CGImage {
        let refs = referenceImage.map { [$0] }
        return try generateInpaint(
            prompt: prompt, source: source, mask: mask, strength: strength,
            width: width, height: height, numSteps: numSteps, guidance: guidance, seed: seed,
            inputImages: refs, invertMask: false, maskFeather: maskFeather,
            verbose: verbose, evalFreq: evalFreq)
    }

    /// Recolor. With a `prompt`, does a mask-guided diffusion recolor (respects lighting/material).
    /// Without a prompt, does an exact pixel-space color grade — globally, or within `mask` if given.
    /// The pixel path runs at the source's native resolution; `width`/`height` are used only for the
    /// diffusion (prompt) path and must be divisible by 16 there.
    public func recolor(
        source: CGImage, mask: CGImage? = nil, prompt: String? = nil,
        hue: Float = 0, saturation: Float = 1, exposure: Float = 0,
        contrast: Float = 1, gamma: Float = 1,
        strength: Double = 0.7, width: Int = 0, height: Int = 0,
        numSteps: Int = defaultSteps, guidance: Double = Double(defaultGuidance),
        seed: UInt64? = nil, invertMask: Bool = false, maskFeather: Int = 2,
        verbose: Bool = false, evalFreq: Int = 1
    ) throws -> CGImage {
        // Semantic recolor via diffusion.
        if let prompt {
            guard let mask else {
                throw Flux2Error.generationFailed("recolor(prompt:) requires a mask")
            }
            return try generateInpaint(
                prompt: prompt, source: source, mask: mask, strength: strength,
                width: width, height: height, numSteps: numSteps, guidance: guidance, seed: seed,
                invertMask: invertMask, maskFeather: maskFeather, verbose: verbose,
                evalFreq: evalFreq)
        }

        // Exact pixel-space grade at native resolution.
        let srcArray = try cgImageToArray(source)  // (H,W,3) in [-1,1]
        let rgb01 = (srcArray + 1) / 2
        var adjusted = adjustColor(
            rgb01, exposure: exposure, contrast: contrast, gamma: gamma,
            hue: hue, saturation: saturation)
        if let mask {
            var m = try maskGridFromCGImage(mask, width: source.width, height: source.height)
            if maskFeather > 0 { m = boxBlur(m, passes: maskFeather) }
            if invertMask { m = 1 - m }
            adjusted = compositeMasked(base: rgb01, adjusted: adjusted, mask: m)
        }
        let out = adjusted * 2 - 1
        return try arrayToCGImage(out)
    }

    /// Model-free pixel filter: grayscale / sepia / invert / sharpen / match-color. Optionally scoped
    /// to `mask`. Runs at the source's native resolution and loads no models.
    public func applyPixelFilter(
        source: CGImage, filter: String, amount: Float = 1.0, reference: CGImage? = nil,
        mask: CGImage? = nil, invertMask: Bool = false, maskFeather: Int = 2
    ) throws -> CGImage {
        let rgb01 = (try cgImageToArray(source) + 1) / 2  // (H,W,3) in [0,1]
        let out: MLXArray
        switch filter.lowercased() {
        case "grayscale", "gray": out = toGrayscale(rgb01)
        case "sepia": out = toSepia(rgb01)
        case "invert": out = invertColor(rgb01)
        case "sharpen": out = sharpen(rgb01, amount: amount)
        case "match-color", "matchcolor":
            guard let reference else {
                throw Flux2Error.generationFailed("match-color requires a reference image")
            }
            out = matchColor(rgb01, reference: (try cgImageToArray(reference) + 1) / 2)
        default:
            throw Flux2Error.generationFailed("unknown filter: \(filter)")
        }
        var result = out
        if let mask {
            var m = try maskGridFromCGImage(mask, width: source.width, height: source.height)
            if maskFeather > 0 { m = boxBlur(m, passes: maskFeather) }
            if invertMask { m = 1 - m }
            result = compositeMasked(base: rgb01, adjusted: out, mask: m)
        }
        return try arrayToCGImage(result * 2 - 1)
    }
}
