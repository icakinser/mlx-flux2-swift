# Flux2Kit example

A tiny standalone SwiftPM project that depends on `Flux2Kit` (by relative path) and shows the two
main ways to use the library.

```sh
cd Examples/Flux2KitExample

# 1) Model-free image processing — no weights, no metallib, instant.
swift run Flux2KitExample process /path/to/any.png
# -> writes example-processed.png (resized + rotated via applyImageOps)

# 2) Text-to-image — needs a FLUX.2 [klein] snapshot and the MLX metallib (Xcode toolchain).
FLUX2_REPO=/path/to/Models/FLUX-2 swift run Flux2KitExample
# -> writes example-generated.png
```

The example is a separate package, so building it also demonstrates consuming Flux2Kit as a
dependency. See `Sources/Flux2KitExample/Example.swift` — it's ~40 lines. To depend on this library
from your own package, point `.package(path:)` at this repo (or use the GitHub URL).
