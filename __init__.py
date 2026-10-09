"""Rebel AI Iris-3B custom nodes.

Register GGUF node UIs without importing the optional GGUF implementation.
Existing Iris safetensors sampler in nodes.py remains completely unchanged.
"""
from .nodes import NODE_CLASS_MAPPINGS as _BASE_NODE_MAPPINGS
from .nodes import NODE_DISPLAY_NAME_MAPPINGS as _BASE_DISPLAY_NAMES


class Iris3BGGUFCheckpoint:
    """Independent GGUF-only checkpoint node; import its implementation on execution."""

    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        files = [f for f in folder_paths.get_filename_list("checkpoints")
                 if f.lower().endswith(".gguf")]
        return {"required": {"checkpoint": (files or ["<no GGUF files>"],)}}

    RETURN_TYPES = ("IRIS3B_CHECKPOINT",)
    RETURN_NAMES = ("iris_checkpoint",)
    FUNCTION = "load"
    CATEGORY = "Rebel AI/Iris 3B GGUF"

    def load(self, checkpoint):
        from .gguf_addon.gguf_nodes import Iris3BGGUFCheckpoint as Implementation
        return Implementation().load(checkpoint)


class Iris3BGGUFSampler:
    """Independent GGUF-only sampler; the working W4A8/FP32 sampler is untouched."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "iris_checkpoint": ("IRIS3B_CHECKPOINT",),
            "conditioning": ("IRIS3B_CONDITIONING",),
            "width": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 16}),
            "height": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 16}),
            "steps": ("INT", {"default": 100, "min": 1, "max": 200}),
            "order": ([1, 2], {"default": 2}),
            "cfg": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 20.0, "step": 0.1}),
            "shift": ("FLOAT", {"default": 4.0, "min": 0.1, "max": 20.0, "step": 0.1}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "dequant_dtype": (["bfloat16", "float16"], {"default": "bfloat16"}),
            "unload_after": ("BOOLEAN", {"default": True}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "sample"
    CATEGORY = "Rebel AI/Iris 3B GGUF"

    def sample(self, iris_checkpoint, conditioning, width, height, steps, order, cfg,
               shift, seed, dequant_dtype, unload_after):
        from .gguf_addon.gguf_nodes import Iris3BGGUFSampler as Implementation
        return Implementation().sample(
            iris_checkpoint, conditioning, width, height, steps, order, cfg,
            shift, seed, dequant_dtype, unload_after,
        )


NODE_CLASS_MAPPINGS = dict(_BASE_NODE_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS = dict(_BASE_DISPLAY_NAMES)
NODE_CLASS_MAPPINGS.update({
    "Iris3BGGUFCheckpoint": Iris3BGGUFCheckpoint,
    "Iris3BGGUFSampler": Iris3BGGUFSampler,
})
NODE_DISPLAY_NAME_MAPPINGS.update({
    "Iris3BGGUFCheckpoint": "Iris 3B GGUF Checkpoint",
    "Iris3BGGUFSampler": "Iris 3B GGUF Sampler (Experimental)",
})

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
