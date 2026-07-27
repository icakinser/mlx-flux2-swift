#!/usr/bin/env bash
#
# Intentionally refresh the checked-in image-quality goldens.
#
# Usage:
#   FLUX2_REPO=/path/to/FLUX.2-klein-4B Scripts/refresh_goldens.sh
#
# Review generated image diffs before committing them. This script never runs from normal tests.

set -euo pipefail

cd "$(dirname "$0")/.."

: "${FLUX2_REPO:?Set FLUX2_REPO to a FLUX.2-klein-4B snapshot}"

BIN=".build/release/flux2kit-cli"
FIXTURE="Tests/Flux2KitTests/Fixtures/ref_bike_s42.png"
HEUN_FIXTURE="Tests/Flux2KitTests/Fixtures/ref_bike_heun_s42.png"

# Always rebuild so the golden is produced by the source revision being reviewed, not a stale binary.
swift build -c release

if [[ ! -f ".build/default.metallib" ]]; then
    Scripts/setup_metallib.sh
fi

echo "Refreshing $FIXTURE"
"$BIN" \
    --repo "$FLUX2_REPO" \
    --prompt "a red bicycle leaning against a stone wall, golden hour" \
    --width 512 \
    --height 512 \
    --steps 4 \
    --guidance 1.0 \
    --seed 42 \
    --sampler euler \
    --output "$FIXTURE"

echo "Refreshing $HEUN_FIXTURE"
"$BIN" \
    --repo "$FLUX2_REPO" \
    --prompt "a red bicycle leaning against a stone wall, golden hour" \
    --width 512 \
    --height 512 \
    --steps 4 \
    --guidance 1.0 \
    --seed 42 \
    --sampler heun \
    --output "$HEUN_FIXTURE"

echo "Golden refreshed. Run the opt-in image suite before committing:"
echo "  FLUX2_RUN_IMAGE_TESTS=1 FLUX2_REPO=\"$FLUX2_REPO\" swift test --filter imageQuality"
