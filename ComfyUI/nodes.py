"""ComfyUI nodes wrapping flux2kit-cli (FLUX.2 [klein] via Swift/MLX). Apple Silicon macOS only."""

import os
import tempfile

import numpy as np
import torch

from . import bridge

_SEED_MAX = 0xFFFF_FFFF_FFFF_FFFF


class Flux2KitGenerate:
    """Text-to-image with FLUX.2 [klein]."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "a red bicycle on a brick wall"}),
                "width": ("INT", {"default": 512, "min": 128, "max": 2048, "step": 16}),
                "height": ("INT", {"default": 512, "min": 128, "max": 2048, "step": 16}),
                "steps": ("INT", {"default": 4, "min": 1, "max": 50}),
                "seed": ("INT", {"default": 42, "min": 0, "max": _SEED_MAX}),
                "guidance": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 20.0, "step": 0.1}),
            },
            "optional": {
                "quantize": (["none", "int8", "int4"],),
                "low_memory": ("BOOLEAN", {"default": False}),
                "repo": ("STRING", {"default": ""}),  # overrides $FLUX2_REPO
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "Flux2Kit"

    def generate(self, prompt, width, height, steps, seed, guidance,
                 quantize="none", low_memory=False, repo=""):
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "out.png")
            args = ["-p", prompt, "-w", str(width), "-h", str(height),
                    "-t", str(steps), "-s", str(seed), "--guidance", str(guidance),
                    "--output", out]
            if quantize and quantize != "none":
                args += ["-q", quantize]
            if low_memory:
                args += ["--low-memory"]
            bridge.run_cli(args, repo=repo or None)
            img = bridge.load_image_np(out)  # HxWx3 [0,1]
        return (torch.from_numpy(img)[None, ...],)  # [1,H,W,C]


class Flux2KitEdit:
    """Mask-guided editing: edit / remove / add-object / replace-background.

    Mask convention: white (1) marks the region to change, EXCEPT for `replace-background`
    where white marks the subject to KEEP. Use invert_mask if your mask is the other way round.
    """

    MODES = ["edit", "remove", "add-object", "replace-background"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "mode": (cls.MODES,),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "strength": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.05}),
                "steps": ("INT", {"default": 4, "min": 1, "max": 50}),
                "seed": ("INT", {"default": 42, "min": 0, "max": _SEED_MAX}),
            },
            "optional": {
                "invert_mask": ("BOOLEAN", {"default": False}),
                "mask_feather": ("INT", {"default": 1, "min": 0, "max": 32}),
                "low_memory": ("BOOLEAN", {"default": False}),
                "repo": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "edit"
    CATEGORY = "Flux2Kit"

    def edit(self, image, mask, mode, prompt, strength, steps, seed,
             invert_mask=False, mask_feather=1, low_memory=False, repo=""):
        img_np = image[0].cpu().numpy()  # HxWx3 [0,1]
        mask_np = mask[0].cpu().numpy()  # HxW [0,1]
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "src.png")
            msk = os.path.join(td, "mask.png")
            out = os.path.join(td, "out.png")
            bridge.save_image_png(img_np, src)
            bridge.save_mask_png(mask_np, msk)
            args = ["--source", src, "--mask", msk, "--strength", str(strength),
                    "-t", str(steps), "-s", str(seed), "--mask-feather", str(mask_feather),
                    "--output", out]
            if invert_mask:
                args += ["--invert-mask"]
            if low_memory:
                args += ["--low-memory"]
            if mode == "remove":
                args += ["--remove"]
            elif mode == "add-object":
                args += ["--add-object", prompt]
            elif mode == "replace-background":
                args += ["--replace-background", prompt]
            else:
                args += ["--edit", prompt]
            bridge.run_cli(args, repo=repo or None)
            result = bridge.load_image_np(out)
        return (torch.from_numpy(result)[None, ...],)
