#!/usr/bin/env bash
#
# Copy the cached MLX metallib into one or more consumer build-product directories.
#
# Usage:
#   Scripts/setup_metallib.sh
#   Scripts/stage_metallib.sh /path/to/app/.build/arm64-apple-macosx/debug

set -euo pipefail

CALLER="$PWD"
DESTINATIONS=()
for destination in "$@"; do
    if [[ "$destination" != /* ]]; then
        destination="$CALLER/$destination"
    fi
    DESTINATIONS+=("$destination")
done

cd "$(dirname "$0")/.."

if [[ "$#" -eq 0 ]]; then
    echo "usage: Scripts/stage_metallib.sh BUILD_PRODUCT_DIR [...]" >&2
    exit 2
fi

SOURCE=".build/default.metallib"
if [[ ! -f "$SOURCE" ]]; then
    echo "$SOURCE is missing; run Scripts/setup_metallib.sh first" >&2
    exit 1
fi

for destination in "${DESTINATIONS[@]}"; do
    if [[ ! -d "$destination" ]]; then
        echo "build-product directory does not exist: $destination" >&2
        exit 1
    fi
    cp -f "$SOURCE" "$destination/mlx.metallib"
    echo "staged -> $destination/mlx.metallib"
done
