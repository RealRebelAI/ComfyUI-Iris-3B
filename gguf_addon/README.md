# ComfyUI-Iris-3B

**Local Iris-3B image generation in ComfyUI — with FP32, W4A8/INT8 safetensors and an optional experimental GGUF addon.**

Created and maintained by **[Rebel AI](https://huggingface.co/realrebelai)**.

Iris-3B is an open-weights, 3-billion-parameter pixel-space diffusion transformer from [SperiLabs](https://huggingface.co/speridlabs/iris-3b). It generates RGB pixels directly without a VAE. These ComfyUI nodes adapt Iris's original architecture for local workflows, including memory-reduced checkpoints.

> **Status:** The A/B Diagnostic safetensors pipeline has produced clean images at 1024×1024. GGUF support is **experimental** and has not yet completed a real-GPU quality test. Keep a backup of any known-good node installation before trying development versions.

## Model formats

| Format | Support | Notes |
|---|---|---|
| Original Iris FP32 `.safetensors` | Base nodes | Large memory footprint. |
| Rebel AI W4A8 `.safetensors` | Base nodes | Uses native Comfy quantized Linear operations; odd-width INT8 layers use a safe per-layer dequantized matmul fallback. |
| GGUF Q3_K_S / Q4_K_S / Q5_K_S / Q6_K | Optional `gguf_addon/` | Isolated sampler; quality and CUDA compatibility still under validation. |

## Requirements

- ComfyUI and Python 3.11+ with a CUDA-capable NVIDIA GPU.
- `torch`, `transformers`, `omegaconf`, `safetensors`, `accelerate`, and Iris's upstream source code.
- A working **Qwen3-VL-4B-Instruct** text encoder installation (downloaded from Hugging Face on first use).
- For GGUF only: [city96/ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) installed separately.

## Install base Iris nodes

```bash
git clone https://github.com/RealRebelAI/ComfyUI-Iris-3B.git ComfyUI-Iris3B
```

Clone into `ComfyUI/custom_nodes/`, install dependencies into **ComfyUI's own Python environment**, and run the included `install_iris_source.py` once if you do not have the Iris source yet. This places upstream code in `third_party/iris-3b/src/` within the node folder, independent of machine-specific paths.

Place the original Iris checkpoint or your W4A8 file under `ComfyUI/models/checkpoints/Iris-3B/`. The base nodes use ComfyUI's checkpoint dropdown; they do not require a typed filesystem path.

The original author's model/config files are available at [speridlabs/iris-3b](https://huggingface.co/speridlabs/iris-3b). Use the checkpoint's own matching config when applicable.

## Working text-to-image workflow

```text
Iris 3B Checkpoint ─────┬─> Iris 3B Qwen3-VL Encode
                        └─> Iris 3B Sampler <── conditioning
                                        │
                                        └─> Save Image
```

**Reference settings:** 1024×1024 (or a native ~1MP aspect ratio), 100 steps, CFG 3, shift 4, order 2, BF16 autocast. The author showcases generation at approximately one megapixel; reducing the resolution drastically produced visible block/grid artifacts during our early tests.

## GGUF — optional isolated addon

The `gguf_addon/` directory is **not part of the base node implementation**. Install its contents as a separate ComfyUI custom-node folder:

```text
ComfyUI/custom_nodes/
  ComfyUI-Iris3B/             # Existing base nodes, leave unchanged
  ComfyUI-GGUF/               # city96 backend
  ComfyUI-Iris3B-GGUF-Addon/  # Copy the files from gguf_addon/ here
```

Then choose the **Iris 3B GGUF Checkpoint** and feed its output to the **existing Iris 3B Qwen3-VL Encode** node and to the **Iris 3B GGUF Sampler (Experimental)**. The addon uses unique node IDs and doesn't replace or rewrite the W4A8/FP32 sampler.

GGUF files belong in `ComfyUI/models/checkpoints/Iris-3B/` and can be downloaded from [Rebel AI's Iris GGUF repository](https://huggingface.co/realrebelai/iris-3b_GGUFs). A custom Iris-specific ComfyUI backend is required — these GGUFs are not llama.cpp chat models.

For installation steps, use [gguf_addon/README.md](gguf_addon/README.md).

## Image restoration and editing

Iris's public release contains a **separate fine-tuned image restoration/upscaling model** (`upscaler/model.safetensors`, `empty_prompt.safetensors`, and `config.yaml`) and a monocular depth model. Those are **not interchangeable** with the base text-to-image checkpoint. An instruction-based img2img editor is **not implemented** in the current ComfyUI nodes; the existence of an upscaler should not be mistaken for an editing sampler.

See the [original Iris model card](https://huggingface.co/speridlabs/iris-3b) for the official task definitions.

## Known limitations

- The W4A8 code is sensitive to model-specific quantization metadata; use tested converters and compatible ComfyUI releases.
- Qwen3-VL is a separate, memory-intensive text encoder; `.gguf` here refers to **the diffusion model only**.
- GGUF loading is experimental; checkpoint schema, tensor shapes, CUDA memory requirements, and final image quality should be tested on each release.
- Native 1024px generation uses considerably more memory than 512px. Quantization does not eliminate activation memory usage.
- Keep known-good versions backed up before installing updates.

## Links and credits

- **Model:** [speridlabs/iris-3b](https://huggingface.co/speridlabs/iris-3b)
- **Source:** [speridlabs/iris-3b](https://github.com/speridlabs/iris-3b)
- **Quantized weights:** [realrebelai/iris-3b_GGUFs](https://huggingface.co/realrebelai/iris-3b_GGUFs)
- **GGUF integration backend:** [city96/ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF)

The original Iris code and weights are released under their respective license terms. Credit to SperiLabs for the model architecture and training, and city96 for the ComfyUI-GGUF infrastructure. This repository provides the ComfyUI integration and experimental quantization loaders.