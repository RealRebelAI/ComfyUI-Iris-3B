"""Iris-3B ComfyUI nodes. The GGUF addon is optional and isolated."""
import logging
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# GGUF registration never changes or wraps the original Iris sampler.
try:
    from .gguf_addon import (
        NODE_CLASS_MAPPINGS as GGUF_NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as GGUF_NODE_DISPLAY_NAME_MAPPINGS,
    )
    conflicts = set(NODE_CLASS_MAPPINGS).intersection(GGUF_NODE_CLASS_MAPPINGS)
    if conflicts:
        raise RuntimeError(f"GGUF addon has duplicate node IDs: {sorted(conflicts)}")
except Exception:
    logging.exception("[Iris3B] GGUF addon could not register; FP32/W4A8 base nodes remain available")
else:
    NODE_CLASS_MAPPINGS.update(GGUF_NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(GGUF_NODE_DISPLAY_NAME_MAPPINGS)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
