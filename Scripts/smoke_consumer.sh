#!/usr/bin/env bash
#
# Build a fresh external SwiftPM consumer against this repository.
#
# Local working tree:
#   Scripts/smoke_consumer.sh --local
#
# Remote revision (used by CI/release checks):
#   Scripts/smoke_consumer.sh https://github.com/icakinser/mlx-flux2-swift.git <revision>

set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/flux2kit-consumer.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

if [[ "${1:-}" == "--local" ]]; then
    DEPENDENCY=".package(path: \"$ROOT\")"
else
    URL="${1:-https://github.com/icakinser/mlx-flux2-swift.git}"
    REVISION="${2:-$(git rev-parse HEAD)}"
    DEPENDENCY=".package(url: \"$URL\", revision: \"$REVISION\")"
fi

mkdir -p "$WORK/Sources/ConsumerSmoke"
cat > "$WORK/Package.swift" <<EOF
// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "ConsumerSmoke",
    platforms: [.macOS(.v14)],
    dependencies: [$DEPENDENCY],
    targets: [
        .executableTarget(
            name: "ConsumerSmoke",
            dependencies: [
                .product(name: "Flux2Kit", package: "mlx-flux2-swift")
            ])
    ])
EOF

cat > "$WORK/Sources/ConsumerSmoke/main.swift" <<'EOF'
import Flux2Kit

let configuration = PipelineConfiguration(
    compile: false,
    residency: .keepResident)
let denoise = DenoiseOptions(
    numSteps: 4,
    guidance: 1.0,
    sampler: .euler)
let options = ImageToImageOptions(
    prompt: "consumer smoke test",
    strength: 0.6,
    denoise: denoise)

print(configuration.repoId, options.prompt)
EOF

swift build -c release --package-path "$WORK"
echo "External consumer smoke build passed."
