# SwiftPM Consumer Setup

## 1. Add the package

Pin to a released tag (recommended):

```swift
dependencies: [
    .package(
        url: "https://github.com/icakinser/mlx-flux2-swift.git",
        .upToNextMinor(from: "0.1.0"))
],
targets: [
    .executableTarget(
        name: "MyApp",
        dependencies: [.product(name: "Flux2Kit", package: "mlx-flux2-swift")])
]
```

For local development, use `.package(path: "../mlx-flux2-swift")`. Prefer a version range over
`branch: "main"` so apps do not track an unstable tip.

## 2. Build the consumer

```sh
swift build -c release
```

## 3. Stage MLX Metal shaders

MLX requires `mlx.metallib` beside the executable. Generate/cache it once in this repository, then
stage it into the consumer's build-product directory:

```sh
cd /path/to/mlx-flux2-swift
Scripts/setup_metallib.sh
Scripts/stage_metallib.sh /path/to/MyApp/.build/arm64-apple-macosx/release
```

Xcode app targets normally receive package resources through Xcode's build system; the explicit
staging step is primarily for command-line SwiftPM consumers.

## 4. Supply weights and generate

Point your application at a local FLUX.2 [klein] diffusers snapshot and initialize
`Flux2Pipeline`. See [Library.md](Library.md) and the standalone
[`Examples/Flux2KitExample`](../Examples/Flux2KitExample).

## Compatibility

- Apple Silicon macOS
- macOS 14+
- Xcode 26 / Swift 6.2 (required to build the MLX Metal shader library)
- Pinned `mlx-swift` and `swift-transformers` revisions in `Package.swift`

Dependency revisions are intentionally pinned. Upgrade them only with the full unit suite, image
quality gates, and release benchmark.
