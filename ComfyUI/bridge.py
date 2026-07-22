"""Bridge between ComfyUI (Python) and flux2kit-cli (Swift/MLX).

There is no direct Python binding for the Swift library, so the nodes shell out to the compiled
`flux2kit-cli` and exchange images as PNG files. Apple Silicon macOS only — the CLI runs on MLX/Metal.
"""

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image


def find_cli() -> str:
    """Locate the flux2kit-cli binary: $FLUX2KIT_CLI, then the repo's .build, then PATH."""
    override = os.environ.get("FLUX2KIT_CLI")
    if override and Path(override).exists():
        return override
    here = Path(__file__).resolve().parent  # <repo>/ComfyUI (symlinks resolved)
    for base in (here.parent, here):
        for config in ("release", "debug"):
            cand = base / ".build" / config / "flux2kit-cli"
            if cand.exists():
                return str(cand)
    found = shutil.which("flux2kit-cli")
    if found:
        return found
    raise RuntimeError(
        "flux2kit-cli not found. Build it with `swift build -c release` in the repo root, "
        "or set the FLUX2KIT_CLI environment variable to the binary path."
    )


def run_cli(args, repo: str | None = None) -> str:
    """Run flux2kit-cli with args; raise with stderr on failure. `repo` sets FLUX2_REPO."""
    cli = find_cli()
    env = dict(os.environ)
    if repo:
        env["FLUX2_REPO"] = repo
    proc = subprocess.run([cli, *args], capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(
            "flux2kit-cli failed:\n" + (proc.stderr.strip() or proc.stdout.strip() or "(no output)")
        )
    return proc.stdout


# --- image <-> PNG helpers (numpy float [0,1] <-> file) ---

def save_image_png(np_img: np.ndarray, path: str) -> None:
    """np_img: HxWx3 float [0,1] (or uint8)."""
    if np_img.dtype != np.uint8:
        np_img = (np.clip(np_img, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    Image.fromarray(np_img, mode="RGB").save(path)


def save_mask_png(np_mask: np.ndarray, path: str) -> None:
    """np_mask: HxW float [0,1] -> grayscale-in-RGB PNG (white = 1)."""
    m = (np.clip(np_mask, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    Image.fromarray(m, mode="L").convert("RGB").save(path)


def load_image_np(path: str) -> np.ndarray:
    """Load a PNG/JPEG -> HxWx3 float [0,1]."""
    img = Image.open(path).convert("RGB")
    return np.asarray(img).astype(np.float32) / 255.0
