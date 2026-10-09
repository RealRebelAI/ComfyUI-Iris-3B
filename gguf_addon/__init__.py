"""Optional GGUF-only extension for ComfyUI-Iris3B; never modifies base nodes."""
from .gguf_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
