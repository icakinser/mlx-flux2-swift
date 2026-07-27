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
IMG2IMG_FIXTURE="Tests/Flux2KitTests/Fixtures/edit_img2img_s43.png"
INPAINT_FIXTURE="Tests/Flux2KitTests/Fixtures/edit_inpaint_s44.png"
OUTPAINT_FIXTURE="Tests/Flux2KitTests/Fixtures/edit_outpaint_s45.png"
REFERENCE_FIXTURE="Tests/Flux2KitTests/Fixtures/ref_condition_s46.png"
MULTI_REFERENCE_FIXTURE="Tests/Flux2KitTests/Fixtures/ref_multi_condition_s48.png"
MANIFEST="Tests/Flux2KitTests/Fixtures/manifest.json"

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

echo "Refreshing $IMG2IMG_FIXTURE"
"$BIN" \
    --repo "$FLUX2_REPO" \
    --img2img \
    --source "$FIXTURE" \
    --prompt "turn the bicycle blue, preserve the stone wall" \
    --strength 0.6 \
    --width 512 \
    --height 512 \
    --steps 4 \
    --guidance 1.0 \
    --seed 43 \
    --sampler euler \
    --output "$IMG2IMG_FIXTURE"

echo "Refreshing $INPAINT_FIXTURE"
"$BIN" \
    --repo "$FLUX2_REPO" \
    --source "$FIXTURE" \
    --edit "a wooden crate against the stone wall" \
    --mask-box "136,152,240,224" \
    --strength 0.85 \
    --width 512 \
    --height 512 \
    --steps 4 \
    --guidance 1.0 \
    --seed 44 \
    --sampler euler \
    --output "$INPAINT_FIXTURE"

echo "Refreshing $OUTPAINT_FIXTURE"
"$BIN" \
    --repo "$FLUX2_REPO" \
    --source "$FIXTURE" \
    --outpaint "32,32,32,32" \
    --prompt "continue the stone wall and golden-hour scene" \
    --strength 0.95 \
    --steps 4 \
    --guidance 1.0 \
    --seed 45 \
    --sampler euler \
    --output "$OUTPAINT_FIXTURE"

echo "Refreshing $REFERENCE_FIXTURE"
"$BIN" \
    --repo "$FLUX2_REPO" \
    --input "$FIXTURE" \
    --prompt "a bicycle product photograph inspired by the reference" \
    --width 512 \
    --height 512 \
    --steps 4 \
    --guidance 1.0 \
    --seed 46 \
    --sampler euler \
    --output "$REFERENCE_FIXTURE"

echo "Refreshing $MULTI_REFERENCE_FIXTURE"
"$BIN" \
    --repo "$FLUX2_REPO" \
    --input "$FIXTURE" \
    --input "$IMG2IMG_FIXTURE" \
    --prompt "a studio bicycle scene combining both references" \
    --width 512 \
    --height 512 \
    --steps 4 \
    --guidance 1.0 \
    --seed 48 \
    --sampler euler \
    --output "$MULTI_REFERENCE_FIXTURE"

cat > "$MANIFEST" <<EOF
{
  "modelRevision": "$(basename "$FLUX2_REPO")",
  "generatedBy": "Scripts/refresh_goldens.sh",
  "cases": [
    {"id":"t2i-euler","file":"ref_bike_s42.png","seed":42,"size":[512,512],"steps":4,"guidance":1.0,"sampler":"euler","threshold":"strict"},
    {"id":"t2i-heun","file":"ref_bike_heun_s42.png","seed":42,"size":[512,512],"steps":4,"guidance":1.0,"sampler":"heun","threshold":"strict"},
    {"id":"img2img","file":"edit_img2img_s43.png","seed":43,"size":[512,512],"steps":4,"guidance":1.0,"sampler":"euler","strength":0.6,"threshold":"strict"},
    {"id":"inpaint","file":"edit_inpaint_s44.png","seed":44,"size":[512,512],"steps":4,"guidance":1.0,"sampler":"euler","strength":0.85,"threshold":"strict"},
    {"id":"outpaint","file":"edit_outpaint_s45.png","seed":45,"size":[576,576],"steps":4,"guidance":1.0,"sampler":"euler","strength":0.95,"threshold":"strict"},
    {"id":"reference","file":"ref_condition_s46.png","seed":46,"size":[512,512],"steps":4,"guidance":1.0,"sampler":"euler","threshold":"strict"},
    {"id":"multi-reference","file":"ref_multi_condition_s48.png","seed":48,"size":[512,512],"steps":4,"guidance":1.0,"sampler":"euler","threshold":"strict"}
  ]
}
EOF

echo "Golden refreshed. Run the opt-in image suite before committing:"
echo "  FLUX2_RUN_IMAGE_TESTS=1 FLUX2_REPO=\"$FLUX2_REPO\" swift test --filter imageQuality"
