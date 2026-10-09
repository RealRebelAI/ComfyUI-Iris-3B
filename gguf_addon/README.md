# Iris-3B: optional GGUF nodes

These nodes are automatically registered from `ComfyUI-Iris3B/__init__.py`.
**Do not copy `gguf_addon/` into `custom_nodes/` as a second package.**

The base `nodes.py` is **not** modified. The new nodes have unique IDs:
`Iris3BGGUFCheckpoint` and `Iris3BGGUFSampler`.

## Requirements

- Main [ComfyUI-Iris-3B](https://github.com/RealRebelAI/ComfyUI-Iris-3B) installation, including upstream Iris source in `third_party/iris-3b/src`.
- [city96/ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF), in its own sibling folder under `custom_nodes`.
- Iris `.gguf` files under `ComfyUI/models/checkpoints/Iris-3B/`.

## Graph

`Iris 3B GGUF Checkpoint` feeds BOTH the existing `Iris 3B Qwen3-VL Encode` and the `Iris 3B GGUF Sampler (Experimental)`. Feed the encoder's conditioning into the GGUF sampler, then connect the sampler to `Save Image`.

Use 1024×1024, 100 steps, CFG 3.0, shift 4.0, order 2 as a reference baseline.

**Experimental:** This GGUF path uses city96's GGML tensor reader/dequantization with a narrow exception for `iris` architecture tags. Real-GPU output quality and memory usage have not yet been verified. FP32/W4A8 still use the original, untouched base sampler. Prompt-based image editing is not implemented; Iris's upscaler and depth are separate official checkpoints.
