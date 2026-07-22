"""ComfyUI custom nodes for Flux2Kit (FLUX.2 [klein] via Swift/MLX). Apple Silicon macOS only."""

from .nodes import Flux2KitEdit, Flux2KitGenerate

NODE_CLASS_MAPPINGS = {
    "Flux2KitGenerate": Flux2KitGenerate,
    "Flux2KitEdit": Flux2KitEdit,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Flux2KitGenerate": "Flux2Kit Generate (FLUX.2 klein)",
    "Flux2KitEdit": "Flux2Kit Edit / Inpaint",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
