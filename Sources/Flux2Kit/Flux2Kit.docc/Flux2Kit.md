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
- ``Flux2Error``

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
- ``Flux2Pipeline/recolor``
- ``Flux2Pipeline/applyPixelFilter``

### Image utilities

- ``ImageOp``
- ``FlipMode``
- ``applyImageOps(_:_:mask:invertMask:maskFeather:)``
- ``loadImages(_:)``
- ``savePNG(_:to:)``
- ``saveImage(_:to:format:quality:)``
- ``resizeHighQuality(_:width:height:)``

### Masks

- ``makeBoxMask(width:height:x:y:boxWidth:boxHeight:)``
- ``makeEllipseMask(width:height:x:y:boxWidth:boxHeight:)``
- ``dilateMask(_:iterations:)``
- ``erodeMask(_:iterations:)``

### Weights

- ``downloadFluxSnapshot(repoId:revision:hfToken:progress:)``
- ``weightsHelpMessage(repoId:)``
- ``defaultRepoId``
- ``WeightDownloadError``
