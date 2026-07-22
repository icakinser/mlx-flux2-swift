# Flux2Kit ComfyUI nodes

Custom ComfyUI nodes that run FLUX.2 [klein] through this repo's `flux2kit-cli`.

> **Apple Silicon macOS only.** The nodes shell out to `flux2kit-cli`, which runs on MLX/Metal — it
> does **not** run on NVIDIA/Windows/Linux. You need a working CLI (built + metallib) and the FLUX.2
> weights, exactly as for the standalone CLI.

## Nodes

- **Flux2Kit Generate (FLUX.2 klein)** — text-to-image. Prompt, size, steps, seed, guidance, plus
  `quantize` (none/int8/int4) and `low_memory`. Output: `IMAGE`.
- **Flux2Kit Edit / Inpaint** — `IMAGE` + `MASK` + `mode` (edit / remove / add-object /
  replace-background) + prompt → `IMAGE`.
  - Mask: **white = region to change**, except **replace-background** where white = the subject to
    **keep**. Toggle `invert_mask` if yours is reversed.

## Install

1. Build the CLI in the repo root: `swift build -c release` (and ensure `mlx.metallib` is next to the
   binary — build via the Xcode toolchain or copy it; see the repo README's Metal note).
2. Symlink this folder into ComfyUI's `custom_nodes`:
   ```sh
   ln -s /path/to/mlx-flux2-swift/ComfyUI \
         /path/to/ComfyUI/custom_nodes/flux2kit
   ```
3. Point the nodes at the weights: set `FLUX2_REPO` in ComfyUI's environment, or fill the node's
   `repo` field. (Or fetch once with `flux2kit-cli --download`.)
4. Restart ComfyUI. The nodes appear under the **Flux2Kit** category.

Optional: set `FLUX2KIT_CLI` to an explicit binary path if it isn't found automatically.

## Notes / limitations

- Each node run spawns the CLI and reloads the model (~0.5 s mmap load + generation); there is no
  warm/persistent server yet.
- Requires `numpy` and `Pillow` (both ship with ComfyUI) plus `torch` (ComfyUI provides it).
- The bridge exchanges images as PNG temp files; ComfyUI `IMAGE`/`MASK` tensors are converted in
  `bridge.py`.
