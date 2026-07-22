# mlx-flux2-swift

A native **Swift + MLX** port of **FLUX.2 [klein] 4B** for Apple Silicon.

Runs text-to-image and image-to-image entirely on-device via
[`mlx-swift`](https://github.com/ml-explore/mlx-swift), with no Python runtime.
Transliterated from the [`scf4/mlx-flux2`](https://github.com/scf4/mlx-flux2)
reference (MIT) and validated for **seed-42 parity** against it.

> **Module name:** the SwiftPM library target is `Flux2Kit` (you `import Flux2Kit`).
> The repository / product is `mlx-flux2-swift`, mirroring the `mlx-swift` naming convention.

## Features

- **Text-to-image** — FLUX.2 [klein] 4B, validated for seed-42 parity with the reference.
- **Image-to-image** — `--img2img` at any strength, plus reference-image (kontext) conditioning.
- **Mask-guided editing** — object removal / addition, background replacement, region edits, and
  semantic recolor. Built-in mask generation (`--mask-box`, `--mask-ellipse`) + dilate/erode, so no
  external mask file is required.
- **Outpainting** — `--outpaint` extends the canvas and fills the new border in context.
- **Model-free image toolkit** — geometry (resize/scale/crop/rotate/flip/fit-16/pixelate) and
  color/effects (brightness, temperature, saturation, grayscale, sepia, invert, sharpen, blur,
  posterize, threshold, vignette, auto-contrast, match-color). **Instant (~50 ms), no model load** —
  run standalone or as post-processing.
- **Memory system** — quantization + staged residency drop peak RAM from ~12.6 GB to ~1.65 GB
  (`--low-memory`), all opt-in.
- **Batch & formats** — `--num N` / `--seeds`, PNG or JPEG.
- **Weights** — optional `--download` from Hugging Face, or bring your own.
- **Library + sample** — use `Flux2Kit` in your own package; a runnable example lives in
  [`Examples/Flux2KitExample`](Examples/Flux2KitExample).
- **ComfyUI nodes** — Generate and Edit/Inpaint nodes (Apple Silicon macOS only) in
  [`ComfyUI/`](ComfyUI), wrapping the CLI.

## Quickstart

The fastest way to see it working needs **no weights and no 15 GB download** — the sample project
applies model-free image ops:

```sh
git clone https://github.com/icakinser/mlx-flux2-swift
cd mlx-flux2-swift/Examples/Flux2KitExample
swift run Flux2KitExample process /path/to/any.png     # → example-processed.png
```

For generation, grab the weights (`flux2kit-cli --download`, or bring your own) and read on.

## Requirements

- Apple Silicon Mac (M-series)
- macOS 26+
- Xcode 26 toolchain (needed to build the MLX Metal shader library — see note below)
- A FLUX.2 [klein] snapshot from [`black-forest-labs/FLUX.2-klein-4B`](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) (Apache-2.0) — fetch it with `--download` or bring your own

## Model weights

Download the FLUX.2 [klein] diffusers snapshot and point the tools at it via the
`FLUX2_REPO` environment variable (default: `./Models/FLUX-2`):

```
Models/FLUX-2/
├── transformer/
├── vae/
├── text_encoder/        # Qwen3
└── tokenizer/
```

The weights are **not** included in this repo (they are gitignored). Get them by either:

```sh
# let the CLI fetch them (opt-in, ~15 GB; set HF_TOKEN if the repo is license-gated)
flux2kit-cli --download

# or download yourself, then point FLUX2_REPO at the snapshot
huggingface-cli download black-forest-labs/FLUX.2-klein-4B
export FLUX2_REPO=/path/to/snapshot
```

Accept the model license at <https://huggingface.co/black-forest-labs/FLUX.2-klein-4B> first. If the
weights can't be found, the CLI prints this guidance automatically.

## Build

```sh
swift build -c release
```

### MLX Metal library note

MLX GPU ops require the compiled `default.metallib` shipped by the MLX Metal
backend. A plain `swift build` compiles the Swift sources fine, but *running*
GPU generation needs that metallib on the library search path. If you hit a
Metal-not-found error at runtime, build through the Xcode toolchain (which
bundles the metallib) or copy `default.metallib` next to the executable. The
tokenizer/template parity tests run CPU-only and do not need it.

## CLI usage

```sh
export FLUX2_REPO=/path/to/Models/FLUX-2

# text-to-image
swift run -c release flux2kit-cli \
  -p "a red bicycle leaning on a brick wall" \
  -w 512 -h 512 -t 4 -s 42 --output out.png

# image-to-image (one or more reference images)
swift run -c release flux2kit-cli \
  -p "make it winter" --input ref.png -s 42 --output out.png
```

Flags:

| Flag | Meaning | Default |
|------|---------|---------|
| `-p, --prompt` | text prompt (required) | — |
| `-w, --width` / `-h, --height` | output size | config default |
| `-t, --steps` | sampling steps | config default |
| `--guidance` | guidance scale | config default |
| `-s, --seed` | RNG seed | random |
| `--input REF ...` | reference image(s) for img2img | none |
| `--repo PATH` | model snapshot path (overrides `FLUX2_REPO`) | `./Models/FLUX-2` |
| `-q, --quantize` | `none` \| `int8` \| `int4` | none |
| `--dtype` | `float16` \| `bfloat16` | config default |
| `--vae-fp16` | run the VAE in fp16 | off |
| `--safe-attn` | numerically safer attention | off |
| `-v, --verbose` | per-stage timing | off |

## Library usage

```swift
import Flux2Kit

let pipeline = try await Flux2Pipeline(
    repoPath: URL(fileURLWithPath: "/path/to/Models/FLUX-2"),
    dtype: "bfloat16")

let image = try pipeline.generate(
    prompt: "a red bicycle leaning on a brick wall",
    width: 512, height: 512,
    numSteps: 4, guidance: 4.0, seed: 42)
```

A runnable sample project lives in [`Examples/Flux2KitExample`](Examples/Flux2KitExample) — it depends
on this package by path and shows both the model-free ops (no weights needed) and text-to-image:

```sh
cd Examples/Flux2KitExample
swift run Flux2KitExample process /path/to/any.png          # instant, no weights
FLUX2_REPO=/path/to/Models/FLUX-2 swift run Flux2KitExample # text-to-image
```

## ComfyUI

Custom nodes in [`ComfyUI/`](ComfyUI) wrap the CLI: **Flux2Kit Generate** (text-to-image) and
**Flux2Kit Edit / Inpaint** (`IMAGE` + `MASK` → `IMAGE`, with edit / remove / add-object /
replace-background modes). **Apple Silicon macOS only** — the nodes shell out to `flux2kit-cli`, so
they need the built CLI, the metallib, and the weights.

```sh
swift build -c release                                        # build the CLI first
ln -s "$PWD/ComfyUI" /path/to/ComfyUI/custom_nodes/flux2kit   # install
export FLUX2_REPO=/path/to/Models/FLUX-2                      # then start ComfyUI
```

Full setup, per-node parameters, mask conventions, and troubleshooting are in
[`ComfyUI/README.md`](ComfyUI/README.md).

## Editing

Mask-guided editing built on the same pipeline. Bring a grayscale **mask** the size of your image;
by convention **white = the region to change, black = the region to keep** (flip with `--invert-mask`).
One inpainting mechanism underlies removal, background replacement, region editing, and object
addition — the model regenerates the masked region in context (lighting, shadows, perspective) while
the rest is preserved by re-blending the source at each denoise step.

```sh
export FLUX2_REPO=/path/to/Models/FLUX-2

# remove the masked object and fill with continued background
swift run -c release flux2kit-cli --source scene.png --mask obj.png --remove -s 42 --output out.png

# add an object into the masked region (optionally steer with --input ref.png)
swift run -c release flux2kit-cli --source scene.png --mask spot.png \
  --add-object "a red bicycle" -s 42 --output out.png

# keep the masked subject, regenerate everything else
swift run -c release flux2kit-cli --source portrait.png --mask subject.png \
  --replace-background "sunset beach" -s 42 --output out.png

# general masked edit / semantic recolor
swift run -c release flux2kit-cli --source car.png --mask car.png \
  --edit "make the car red" -s 42 --output out.png

# exact pixel-space color grade (global, or within --mask); no model call
swift run -c release flux2kit-cli --source photo.png \
  --recolor "exp=0.3,contrast=1.1,sat=1.2,hue=0.02" --output out.png
```

More modes and options:

```sh
# img2img — regenerate the source from a prompt at a given strength
flux2kit-cli --img2img -p "an orange on a plate" --source in.png --strength 0.6 --output out.png

# outpainting — extend the canvas and fill the new border (L,R,T,B, or one value for all sides)
flux2kit-cli --outpaint 128 -p "wooden table, plain background" --source in.png --output out.png

# generated masks — no external mask file needed (top-left origin)
flux2kit-cli --source in.png --mask-box 176,150,170,200 --edit "a green apple" --output out.png
flux2kit-cli --source in.png --mask-ellipse 180,160,150,180 --mask-dilate 3 --edit "…" --output out.png

# model-free image ops — NO model load (instant, ~50 ms, ~50 MB). Run standalone on a --source,
# or chain after a generate/edit. Applied in the order given.
flux2kit-cli --source in.png --resize 768x512 --output out.png       # geometry: also --scale --crop
flux2kit-cli --source in.png --rotate 90 --flip h --fit-16 --output out.png
flux2kit-cli --source in.png --grayscale --output out.png            # effects: --sepia --invert
flux2kit-cli --source in.png --posterize 4 --threshold 0.5 --pixelate 8 --vignette 0.5 --output out.png
flux2kit-cli --source in.png --brightness 0.1 --temperature 0.3 --saturation 1.2 --auto-contrast --output out.png
flux2kit-cli --source in.png --sharpen 1.5 --blur 2 --match-color ref.png --output out.png
flux2kit-cli -p "a red bicycle" --grayscale --output out.png         # generate, then post-process

# batch / output format
flux2kit-cli -p "a red bicycle" --num 4 -s 100 --output out.png      # out_0.png … out_3.png
flux2kit-cli -p "a red bicycle" --format jpg --output out.jpg
```

Editing options: `--strength F` (how freely the region regenerates), `--invert-mask`,
`--mask-feather N`, `--mask-dilate N` / `--mask-erode N` (grow/shrink the mask), `-s SEED`.
Pair any editing mode with `--low-memory` to run at ~1.5 GB.

> **Why pixel-space color?** FLUX latents are a learned 128-dim representation, not a color space, so
> HSV/gamma applied to latents is unreliable. `--recolor` grades in pixel space (exact). A latent
> A/B path exists behind `--experimental-latent-color` (with `--recolor`) purely for comparison.

The same operations are available as a Swift API: `removeObject`, `addObject`, `replaceBackground`,
`editRegion`, `recolor`, and the underlying `generateInpaint` (see `Sources/Flux2Kit/Editing.swift`).

## Memory

Inference is memory-bandwidth bound and the three sub-models (Qwen3 text encoder, transformer, VAE)
run sequentially, so two levers dominate: **quantization** (fewer weight bytes to store *and* read)
and **staged residency** (free each model once its stage is done). Both are opt-in; the default keeps
everything resident in bf16.

Measured peak RSS, 512² / 4 steps (M-series), same prompt & seed:

| Config | Peak RSS | vs bf16 |
|--------|---------:|--------:|
| default (bf16, all resident) | ~12.6 GB | 1.0× |
| `-q int4` | ~3.8 GB | **3.3×** |
| `--low-memory` | ~1.65 GB | **7.6×** |

```sh
# quantize (int8 ≈ half, int4 ≈ quarter of weight memory + bandwidth)
flux2kit-cli -p "…" -q int4 --output out.png

# one-flag minimum-footprint preset: int4 + free each model after its stage +
# fp16 VAE + a 512MB buffer-cache cap
flux2kit-cli -p "…" --low-memory --output out.png

# see where the memory goes, per stage
flux2kit-cli -p "…" --low-memory --mem-report --output out.png
```

Individual knobs: `--mem-report`, `--cache-limit MB`, `--memory-limit MB`, `--vae-fp16`, and
`--vae-tile N` (opt-in tiled VAE decode for very large images — lossy, since FLUX's VAE has global
attention, so it is never auto-enabled).
The `Flux2Pipeline` init exposes `residency: .keepResident | .unloadAfterUse`, `cacheLimitMB`,
`memoryLimitMB`, `memReport`, and `vaeTileLatent`. (Quantization skips the small `adaLN` modulation
layers — the standard FLUX recipe — and the transformer/text-encoder big matmuls carry the savings.)

## Tests

```sh
# tokenizer parity (needs a snapshot on disk)
FLUX2_REPO=/path/to/Models/FLUX-2 swift test

# also run the editing/latent/color unit tests (need the MLX metallib — see note)
FLUX2_RUN_MLX_TESTS=1 FLUX2_REPO=/path/to/Models/FLUX-2 swift test
```

A bare `swift test` passes with everything skipped. Tokenizer parity tests self-skip unless
`FLUX2_REPO` points at a snapshot. The editing/color tests exercise MLX array math, which needs the
Metal shader library (`default.metallib`) that a plain `swift build` does not produce; they are
gated behind `FLUX2_RUN_MLX_TESTS=1` and expect the metallib to be present (build through the Xcode
toolchain, then place `mlx.metallib` next to the test binary).

## Credits & licensing

- Port of [`scf4/mlx-flux2`](https://github.com/scf4/mlx-flux2) (MIT).
- Runs on [`mlx-swift`](https://github.com/ml-explore/mlx-swift) (MIT) and
  [`swift-transformers`](https://github.com/huggingface/swift-transformers) (Apache-2.0).
- FLUX.2 [klein] weights: © Black Forest Labs, released under Apache-2.0.

This project is licensed under the **PolyForm Noncommercial License 1.0.0** — see [LICENSE](LICENSE).
Noncommercial use only. Portions derived from `scf4/mlx-flux2` remain under the MIT License (retained
in the LICENSE file), and the FLUX.2 weights and dependencies keep their own licenses (Apache-2.0 / MIT).
