# Flux2Kit — Next Release

## Highlights

- One shared generation engine now powers text-to-image, img2img, inpaint, outpaint, and editing.
- Sampler, guidance schedule, compile policy, progress, evaluation frequency, and cancellation behave
  consistently across generation modes.
- New `PipelineConfiguration`, `DenoiseOptions`, `ImageToImageOptions`, `InpaintOptions`, and
  `OutpaintOptions` provide a compact application API.
- `GenerationCancellation` supports cooperative cancellation between denoise steps.
- The image quality matrix now covers Euler, Heun, compile, img2img, inpaint, outpaint, and single-
  and multi-reference conditioning.
- Points-of-interest signposts expose generation stages and individual denoise steps in Instruments.
- External SwiftPM consumer smoke checks and a SwiftUI application example validate integration.
- The deployment target is macOS 14, matching the pinned mlx-swift package.

## API compatibility

Transformer, VAE, text-encoder, weight conversion, CLI parsing, and low-level color math symbols are
implementation details and are now package-scoped. Applications should use `Flux2Pipeline`, the
public option types, image operations, masks, and image I/O APIs.

The existing positional generation methods remain available. Typed option overloads are additive.

## Performance policy

Compile remains opt-in until a supported MLX revision produces a repeatable release-build win.
Optimizations must pass the declared image-quality threshold and improve their measured workload by
at least 5%; reference-conditioning optimizations require at least 10%.

## Migration

Prefer:

```swift
let pipeline = try await Flux2Pipeline(
    configuration: PipelineConfiguration(
        repoPath: modelURL,
        residency: .keepResident))

let options = ImageToImageOptions(
    prompt: "make it winter",
    strength: 0.6,
    denoise: DenoiseOptions(seed: 42))

let output = try pipeline.generateImg2Img(options, source: source)
```

Code that directly constructed transformer, VAE, or text-encoder modules must migrate to the
pipeline API.
