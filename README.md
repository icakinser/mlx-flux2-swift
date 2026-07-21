# mlx-flux2-swift

A native **Swift + MLX** port of **FLUX.2 [klein] 4B** for Apple Silicon.

Runs text-to-image and image-to-image entirely on-device via
[`mlx-swift`](https://github.com/ml-explore/mlx-swift), with no Python runtime.
Transliterated from the [`scf4/mlx-flux2`](https://github.com/scf4/mlx-flux2)
reference (MIT) and validated for **seed-42 parity** against it.

> **Module name:** the SwiftPM library target is `Flux2Kit` (you `import Flux2Kit`).
> The repository / product is `mlx-flux2-swift`, mirroring the `mlx-swift` naming convention.

## Requirements

- Apple Silicon Mac (M-series)
- macOS 26+
- Xcode 26 toolchain (needed to build the MLX Metal shader library — see note below)
- A local diffusers snapshot of [`black-forest-labs/FLUX.2-klein`](https://huggingface.co/black-forest-labs/FLUX.2-klein) (Apache-2.0)

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

The weights are **not** included in this repo (they are gitignored).

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

## Tests

```sh
FLUX2_REPO=/path/to/Models/FLUX-2 swift test
```

Parity tests self-skip if `FLUX2_REPO` is unset or the tokenizer isn't on disk.

## Credits & licensing

- Port of [`scf4/mlx-flux2`](https://github.com/scf4/mlx-flux2) (MIT).
- Runs on [`mlx-swift`](https://github.com/ml-explore/mlx-swift) (MIT) and
  [`swift-transformers`](https://github.com/huggingface/swift-transformers) (Apache-2.0).
- FLUX.2 [klein] weights: © Black Forest Labs, released under Apache-2.0.

This project is MIT-licensed — see [LICENSE](LICENSE).
