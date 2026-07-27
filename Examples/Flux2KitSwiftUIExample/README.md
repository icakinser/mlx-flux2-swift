# Flux2Kit SwiftUI Example

This standalone SwiftPM macOS application demonstrates:

- a reusable `Flux2Pipeline`
- `PipelineConfiguration`
- per-step progress
- cooperative cancellation
- repeated generation
- optional reference-image import

Build and stage MLX's Metal library:

```sh
swift build
../../Scripts/setup_metallib.sh
../../Scripts/stage_metallib.sh .build/arm64-apple-macosx/debug
FLUX2_REPO=/path/to/FLUX.2-klein-4B swift run
```

The example targets macOS 14 and requires the Swift 6.2/Xcode 26 toolchain used by the parent package.
