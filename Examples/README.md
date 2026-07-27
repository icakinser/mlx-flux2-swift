# Flux2Kit example

A standalone SwiftPM project that depends on `Flux2Kit` by relative path. It demonstrates the
library API directly; the CLI is not involved.

```sh
cd Examples/Flux2KitExample

# 1) Model-free image processing — no weights, no metallib, instant.
swift run Flux2KitExample process /path/to/any.png
# -> writes example-processed.png (resized + rotated via applyImageOps)

# 2) Text-to-image with per-step progress.
FLUX2_REPO=/path/to/Models/FLUX-2 swift run Flux2KitExample generate
# -> writes example-generated.png

# 3) Image-to-image.
FLUX2_REPO=/path/to/Models/FLUX-2 \
  swift run Flux2KitExample img2img source.png "make it winter"

# 4) Mask-guided removal (white mask region is replaced).
FLUX2_REPO=/path/to/Models/FLUX-2 \
  swift run Flux2KitExample remove source.png mask.png

# Library-level performance/memory knobs.
FLUX2_COMPILE=1 FLUX2_REPO=/path/to/Models/FLUX-2 swift run Flux2KitExample generate
FLUX2_LOW_MEMORY=1 FLUX2_REPO=/path/to/Models/FLUX-2 swift run Flux2KitExample generate
```

The example is a separate package, so building it also demonstrates consuming Flux2Kit as a
dependency. See `Sources/Flux2KitExample/Example.swift`. To depend on this library from your own
package, point `.package(path:)` at this repo (or use the GitHub URL). The stable API and threading
contract are documented in [`../Docs/Library.md`](../Docs/Library.md).

## SwiftUI application example

`Flux2KitSwiftUIExample` demonstrates the application-facing API: pipeline configuration, repeated
generation on a shared thread-safe pipeline, progress updates, cooperative cancellation, and
reference-image import.

```sh
cd Examples/Flux2KitSwiftUIExample
swift build
../../Scripts/setup_metallib.sh
../../Scripts/stage_metallib.sh .build/arm64-apple-macosx/debug
FLUX2_REPO=/path/to/Models/FLUX-2 swift run
```
