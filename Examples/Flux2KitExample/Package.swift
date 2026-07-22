// swift-tools-version: 6.2
// A tiny standalone project showing how to depend on and use Flux2Kit as a library.
// Depends on the parent package by relative path.

import PackageDescription

let package = Package(
    name: "Flux2KitExample",
    platforms: [
        .macOS(.v26)
    ],
    dependencies: [
        .package(path: "../..")
    ],
    targets: [
        .executableTarget(
            name: "Flux2KitExample",
            dependencies: [
                // Path dependencies are referenced by directory name (the package identity).
                .product(name: "Flux2Kit", package: "mlx-flux2-swift")
            ]
        )
    ]
)
