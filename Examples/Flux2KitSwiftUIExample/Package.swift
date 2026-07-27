// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "Flux2KitSwiftUIExample",
    platforms: [.macOS(.v14)],
    dependencies: [
        .package(path: "../..")
    ],
    targets: [
        .executableTarget(
            name: "Flux2KitSwiftUIExample",
            dependencies: [
                .product(name: "Flux2Kit", package: "mlx-flux2-swift")
            ])
    ])
