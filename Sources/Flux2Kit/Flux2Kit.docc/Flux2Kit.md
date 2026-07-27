# ``Flux2Kit``

Generate and edit FLUX.2 images natively on Apple Silicon with Swift and MLX.

## Overview

Create one ``Flux2Pipeline`` and reuse it for generation. The pipeline serializes calls because MLX
random state is process-global, while still making shared use safe across application tasks.

```swift
let pipeline = try await Flux2Pipeline(
    configuration: PipelineConfiguration(
        repoPath: modelURL,
        residency: .keepResident))

var options = GenerationOptions(
    prompt: "a red bicycle at golden hour",
    seed: 42)
options.progress = { progress in
    print(progress.step + 1, progress.totalSteps)
}

let image = try pipeline.generate(options)
```

Use ``GenerationCancellation`` for cooperative cancellation between denoise steps. Editing modes
accept ``DenoiseOptions`` through ``ImageToImageOptions``, ``InpaintOptions``, and
``OutpaintOptions`` so sampler, guidance schedule, progress, and cancellation behave consistently.

## Topics

### Pipeline

- ``Flux2Pipeline``
- ``PipelineConfiguration``
- ``ResidencyPolicy``

### Generation

- ``GenerationOptions``
- ``DenoiseOptions``
- ``GenerationProgress``
- ``GenerationCancellation``
- ``Sampler``
- ``GuidanceSchedule``

### Editing

- ``ImageToImageOptions``
- ``InpaintOptions``
- ``OutpaintOptions``

### Image utilities

- ``ImageOp``
- ``applyImageOps(_:_:)``
- ``loadImages(_:)``
- ``saveImage(_:to:format:quality:)``
