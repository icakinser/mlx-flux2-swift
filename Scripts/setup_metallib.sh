#!/usr/bin/env bash
#
# setup_metallib.sh — make MLX's `default.metallib` available so `flux2kit-cli` and the tests run.
#
# Why: `swift build` does NOT compile MLX's Metal shaders — only Xcode's build system does. This
# script generates the metallib once with `xcodebuild`, then copies it next to the built binaries
# where MLX looks for it. Run it once after cloning (and again only after a full clean).
#
# Usage:
#   Scripts/setup_metallib.sh
#
# Requires: a full Xcode install (not just the Command Line Tools).

set -euo pipefail

cd "$(dirname "$0")/.."
REPO="$PWD"
DEV="${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}"
DD="$REPO/.xcode-metallib"

say() { printf '\033[1m==>\033[0m %s\n' "$*"; }

# 1) Reuse an existing metallib if one is already around.
FOUND="$(find "$DD" "$REPO/.build" -name default.metallib 2>/dev/null | head -1 || true)"

# 2) Otherwise compile it with xcodebuild (first run is slow — it builds MLX's Metal kernels).
if [[ -z "$FOUND" ]]; then
    if [[ ! -d "$DEV" ]]; then
        echo "Xcode not found at: $DEV" >&2
        echo "Install Xcode from the App Store, then run:" >&2
        echo "  sudo xcode-select -s /Applications/Xcode.app/Contents/Developer" >&2
        exit 1
    fi
    say "Generating default.metallib with xcodebuild (first run takes a few minutes)…"
    # -skipPackagePluginValidation / -skipMacroValidation: mlx-swift ships a build plugin and Swift
    # macros that xcodebuild otherwise blocks on the command line (the Xcode GUI would prompt "Trust").
    DEVELOPER_DIR="$DEV" xcodebuild -scheme flux2kit-cli \
        -destination 'platform=macOS,arch=arm64' -derivedDataPath "$DD" \
        -skipPackagePluginValidation -skipMacroValidation build >/dev/null
    FOUND="$(find "$DD" -name default.metallib 2>/dev/null | head -1 || true)"
fi

if [[ -z "$FOUND" ]]; then
    echo "Could not produce default.metallib. Ensure a full Xcode (not only Command Line Tools) is" >&2
    echo "installed and selected (xcode-select -p should point inside Xcode.app)." >&2
    exit 1
fi
say "Using: $FOUND"

# Cache a copy in .build so future runs are instant even after you delete .xcode-metallib.
mkdir -p "$REPO/.build"
cp -f "$FOUND" "$REPO/.build/default.metallib"

# 3) Build the SPM products if they aren't there yet.
if ! ls "$REPO"/.build/*/release/flux2kit-cli >/dev/null 2>&1; then
    say "Building the CLI (swift build -c release)…"
    DEVELOPER_DIR="$DEV" swift build -c release >/dev/null
fi

# 4) Stage the metallib as `mlx.metallib` next to every built binary (that's where MLX searches).
staged=0
for d in "$REPO"/.build/*/release "$REPO"/.build/*/debug; do
    [[ -d "$d" ]] || continue
    cp -f "$FOUND" "$d/mlx.metallib"
    say "staged -> $d/mlx.metallib"
    staged=1
done
# Test bundles look for it next to the xctest executable too.
for x in "$REPO"/.build/*/debug/*.xctest/Contents/MacOS; do
    [[ -d "$x" ]] || continue
    cp -f "$FOUND" "$x/mlx.metallib"
    say "staged -> $x/mlx.metallib"
done

[[ "$staged" == 1 ]] || {
    echo "No swift build products found. Run 'swift build -c release', then re-run this script." >&2
    exit 1
}

say "Done. Test it:"
echo "    FLUX2_REPO=/path/to/Models/FLUX-2 .build/release/flux2kit-cli -p 'a red bicycle' --output out.png"
