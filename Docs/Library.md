# Flux2Kit Library Guide

`Flux2Kit` is the product. `flux2kit-cli` is a thin executable that demonstrates and tests the same
public API an application uses.

## Stable public surface

- `Flux2Pipeline`, `PipelineConfiguration`, `GenerationOptions`, `GenerationProgress`
- `DenoiseOptions`, `ImageToImageOptions`, `InpaintOptions`, `OutpaintOptions`
- `GenerationCancellation`
- `Sampler`, `GuidanceSchedule`, `ResidencyPolicy`
- Text-to-image, img2img, inpaint, outpaint, and editing methods on `Flux2Pipeline`
- `ImageOp`, `FlipMode`, `applyImageOps`
- Image loading, saving, and `resizeHighQuality`
- `Flux2Error`
- Model download helpers

Transformer, VAE, text-encoder, MLX conversion/math, weight conversion, denoise, memory, and CLI
implementation details are package-scoped.

## Basic generation

```swift
import Flux2Kit

let pipeline = try await Flux2Pipeline(
    configuration: PipelineConfiguration(
        repoPath: modelURL,
        compile: false,
        residency: .keepResident))

var options = GenerationOptions(
    prompt: "a red bicycle leaning against a stone wall, golden hour",
    width: 512,
    height: 512,
    numSteps: 4,
    guidance: 1.0,
    seed: 42,
    sampler: .euler)

options.progress = { progress in
    print("step \(progress.step + 1)/\(progress.totalSteps)")
}
let cancellation = GenerationCancellation()
options.cancellation = cancellation

let image = try pipeline.generate(options)
try savePNG(image, to: outputURL)
```

## Performance and memory policy

- `.keepResident`: recommended for interactive apps and repeated generation. It avoids reloading the
  text encoder, transformer, and VAE between calls.
- `.unloadAfterUse`: recommended for single-shot or memory-constrained processes. Each heavy stage is
  released after use and reloaded lazily on the next call.
- `compile: true`: quality-gated fast path. Its benefit depends on MLX and workload; benchmark your
  release build with `Scripts/bench.sh`. The checked-in baseline keeps it opt-in because it did not
  beat warm eager denoising on the measured host.
- `vaeFp16`: limits float16 to the VAE. The transformer stays bfloat16 because float16 activations
  overflow on this model.

## Threading

One `Flux2Pipeline` serializes all generation and lifecycle calls with a recursive lock. This makes a
shared pipeline safe to call from multiple threads, but calls run one at a time because MLX random
state is process-global. For true parallel generation, use separate processes.

The progress callback is `@Sendable` and is invoked synchronously after each completed denoise step.
Dispatch UI work to the main actor from the callback when integrating with AppKit or SwiftUI.

Cancellation is cooperative: call `GenerationCancellation.cancel()` from any thread. Flux2Kit checks
the token before each denoise step and throws `Flux2Error.cancelled`; a model forward already in
flight is allowed to finish.

All generation modes use the same internal execution engine. Sampler, guidance schedule, evaluation
frequency, progress, cancellation, and pipeline-level compile policy therefore behave consistently
for text-to-image, img2img, inpaint, outpaint, and editing wrappers.

## Quality contract

The Python project is a baseline, not a permanent feature ceiling. Changes are judged by:

- strict deterministic Euler golden: mean absolute channel difference ≤ 0.5/255
- optimized/alternative paths: mean difference ≤ 5/255
- release performance measurements from `Scripts/bench.sh`

Run the real-model gates with:

```sh
FLUX2_RUN_IMAGE_TESTS=1 FLUX2_REPO=/path/to/model swift test --filter ImageQualityTests
```
