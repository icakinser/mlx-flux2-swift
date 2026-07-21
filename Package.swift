// swift-tools-version: 6.2
// flux2-swift — a native MLX Swift port of FLUX.2 [klein] 4B.
//
// Transliterated from scf4/mlx-flux2 (MIT). Runs text-to-image and image-to-image
// on Apple Silicon via mlx-swift, consuming the diffusers snapshot of
// black-forest-labs/FLUX.2-klein (Apache-2.0) directly.

import PackageDescription

let package = Package(
    name: "Flux2Kit",
    platforms: [
        .macOS(.v26)
    ],
    products: [
        .library(
            name: "Flux2Kit",
            targets: ["Flux2Kit"]
        ),
        .executable(
            name: "flux2kit-cli",
            targets: ["Flux2KitCLI"]
        ),
    ],
    dependencies: [
        // Pinned to the exact upstream revisions this port was validated against
        // (seed-42 parity vs the mlx-flux2 Python reference). Bump as needed.
        .package(
            url: "https://github.com/ml-explore/mlx-swift.git",
            revision: "de3342cefe687116afbdd4a422d5bc8a19d21506"
        ),
        .package(
            url: "https://github.com/huggingface/swift-transformers.git",
            revision: "eec09bdffb04a6f13eaed57c7b4acca3f6729a6c"
        ),
    ],
    targets: [
        .target(
            name: "Flux2Kit",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            path: "Sources/Flux2Kit"
        ),
        .executableTarget(
            name: "Flux2KitCLI",
            dependencies: ["Flux2Kit"],
            path: "Sources/Flux2KitCLI"
        ),
        .testTarget(
            name: "Flux2KitTests",
            dependencies: ["Flux2Kit"],
            path: "Tests/Flux2KitTests"
        ),
    ]
)
