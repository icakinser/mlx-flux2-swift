# Flux2Kit ComfyUI nodes

Custom [ComfyUI](https://github.com/comfyanonymous/ComfyUI) nodes that run **FLUX.2 [klein]** through
this repo's `flux2kit-cli`. Two nodes: **Generate** (text-to-image) and **Edit / Inpaint**.

> ### ⚠️ Apple Silicon macOS only
> These nodes shell out to `flux2kit-cli`, which runs on **MLX/Metal**. They do **not** work on
> NVIDIA/AMD, Windows, or Linux. You need an M-series Mac with the CLI built, the MLX metallib
> present, and the FLUX.2 weights on disk — exactly the same requirements as the standalone CLI.

There is no direct Python binding for the Swift/MLX library, so the nodes launch the CLI as a
subprocess and exchange images as PNG files. This is a thin (~200-line) adapter.

---

## 1. Prerequisites

| Requirement | How |
|-------------|-----|
| Apple Silicon Mac, macOS 26+, Xcode 26 | — |
| The `flux2kit-cli` binary | `swift build -c release` in the repo root (see step 2) |
| MLX metallib next to the binary | build via the Xcode toolchain, or copy `default.metallib`/`mlx.metallib` next to the executable (see the repo README's *MLX Metal library note*) |
| FLUX.2 [klein] weights | `flux2kit-cli --download`, or `huggingface-cli download black-forest-labs/FLUX.2-klein-4B` |
| ComfyUI with its Python env | ships `torch`, `numpy`, `Pillow` — no extra pip installs needed |

---

## 2. Install (step by step)

**a. Build the CLI** (from the repo root):

```sh
cd /path/to/mlx-flux2-swift
swift build -c release
# confirm it runs (prints usage):
.build/release/flux2kit-cli --help
```

If you get a "failed to load the default metallib" error at run time, the Metal shader library isn't
next to the binary — build through Xcode's toolchain or copy the metallib beside
`.build/release/flux2kit-cli` (repo README explains this).

**b. Link the nodes into ComfyUI:**

```sh
ln -s /path/to/mlx-flux2-swift/ComfyUI \
      /path/to/ComfyUI/custom_nodes/flux2kit
```

(A symlink is preferred so the nodes track the repo; a copy works too.)

**c. Tell the nodes where the weights are** — either:

- set `FLUX2_REPO` in the environment ComfyUI runs in:
  ```sh
  export FLUX2_REPO=/path/to/FLUX-2   # then start ComfyUI from that shell
  ```
- or leave it unset and type the path into each node's **`repo`** field.

**d. (optional) Pin the CLI path** if auto-detection fails:

```sh
export FLUX2KIT_CLI=/path/to/mlx-flux2-swift/.build/release/flux2kit-cli
```

**e. Restart ComfyUI.** The nodes appear under the **Flux2Kit** category (right-click → Add Node →
Flux2Kit, or double-click the canvas and search "Flux2Kit").

---

## 3. Nodes & parameters

### Flux2Kit Generate (FLUX.2 klein)

Text-to-image. Output: `IMAGE`.

| Input | Type | Default | Notes |
|-------|------|---------|-------|
| `prompt` | STRING (multiline) | — | the text prompt |
| `width` / `height` | INT | 512 | multiples of 16 |
| `steps` | INT | 4 | klein is distilled; 4 is the sweet spot |
| `seed` | INT | 42 | RNG seed |
| `guidance` | FLOAT | 4.0 | guidance scale |
| `quantize` | none / int8 / int4 | none | int4 ≈ ¼ the memory & bandwidth |
| `low_memory` | BOOLEAN | false | one-flag minimum footprint (~1.6 GB) |
| `repo` | STRING | "" | overrides `$FLUX2_REPO` |

### Flux2Kit Edit / Inpaint

Mask-guided editing. Inputs: `IMAGE` + `MASK`. Output: `IMAGE`.

| Input | Type | Default | Notes |
|-------|------|---------|-------|
| `image` | IMAGE | — | the source image |
| `mask` | MASK | — | see mask conventions below |
| `mode` | edit / remove / add-object / replace-background | edit | what to do |
| `prompt` | STRING (multiline) | "" | used by edit / add-object / replace-background (ignored by remove) |
| `strength` | FLOAT | 0.85 | how freely the region regenerates (remove works well at ~0.9) |
| `steps` | INT | 4 | |
| `seed` | INT | 42 | |
| `invert_mask` | BOOLEAN | false | flip the mask polarity |
| `mask_feather` | INT | 1 | blur the mask edge for a softer seam |
| `low_memory` | BOOLEAN | false | run at ~1.5 GB |
| `repo` | STRING | "" | overrides `$FLUX2_REPO` |

**Mask conventions** (this trips people up):

| Mode | White (1) in the mask means… |
|------|------------------------------|
| `edit` | the region to **change** |
| `remove` | the object to **remove** |
| `add-object` | where to **place** the new object |
| `replace-background` | the **subject to keep** (everything else is regenerated) |

If your mask is the other way round, toggle **`invert_mask`**. ComfyUI's mask editors and the
"Convert Image to Mask" node produce white = selected, which matches the first three modes.

---

## 4. Example workflows

- **Text-to-image:** `Flux2Kit Generate` → `Save Image`.
- **Inpaint an object out:** `Load Image` → (mask it with the ComfyUI mask editor / `Load Image (as
  Mask)`) → `Flux2Kit Edit` (mode `remove`) → `Save Image`.
- **Add an object:** paint a mask where it should go → `Flux2Kit Edit` (mode `add-object`, prompt
  "a red bicycle") → `Save Image`.
- **Swap the background:** mask the subject → `Flux2Kit Edit` (mode `replace-background`, prompt
  "sunset beach") → `Save Image`.

You can chain the output `IMAGE` into any standard ComfyUI node (upscalers, savers, previews, etc.).

---

## 5. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `flux2kit-cli not found` | build it (`swift build -c release`) or set `FLUX2KIT_CLI` to the binary path |
| `failed to load the default metallib` | put the MLX metallib next to the CLI binary (repo README) |
| `FLUX.2 weights were not found` (in the node error) | set `FLUX2_REPO` / the `repo` field, or run `flux2kit-cli --download` |
| Gated-repo / 401 on `--download` | accept the license at huggingface.co/black-forest-labs/FLUX.2-klein-4B and set `HF_TOKEN` |
| The edit changed the wrong area | toggle `invert_mask`, or check the per-mode mask table above |
| Slow / reloads each run | expected — each node run spawns the CLI and reloads the model (~0.5 s mmap + generation). There is no warm server yet |
| Node doesn't appear | confirm the symlink is in `custom_nodes/`, then fully restart ComfyUI and check its console for import errors |

---

## 6. Limitations

- **macOS/Apple Silicon only** (see the banner above).
- **No persistent server** — the model reloads per node execution. Fine for iterating; not ideal for
  huge batches.
- Only **Generate** and **Edit/Inpaint** are wrapped so far. img2img, outpainting, and the model-free
  image ops are all available in the CLI and could be added as nodes on request.
- The bridge exchanges images as PNG temp files; conversions live in `bridge.py`.
