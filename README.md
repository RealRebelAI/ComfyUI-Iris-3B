# ComfyUI-Iris3B — W4A8 build

## Model dropdown

Uses the normal ComfyUI folder:

```text
ComfyUI/models/checkpoints/
```

No path boxes.

## Supported diffusion checkpoints in this build

- Standard Iris safetensors
- Native Comfy **W4A8** (`asym_w4a8_int8`)
- Native Comfy **INT8** when the checkpoint contains compatible Comfy quant metadata
- GGUF is listed in the dropdown but the GGUF execution backend is still separate / not wired here

The W4A8 loader uses ComfyUI's `mixed_precision_ops` / `QuantizedTensor` path.
Packed 4-bit matrices stay packed instead of being loaded into ordinary
`torch.nn.Linear`.

The loader accepts both:
1. standard `_quantization_metadata` in the safetensors header, and
2. Rebel W4A8 files containing `weight_s_rel` scales even if metadata has to be reconstructed.

The fallback reconstruction matches this Iris recipe:

```text
group_size = 16
convrot_groupsize = 256
format = asym_w4a8_int8
```

## Iris source

Keep the upstream source inside the node package:

```text
ComfyUI-Iris3B/third_party/iris-3b/src/
```

## Terminal progress

```text
[Iris3B] quantized checkpoint detected: ... quantized Linear layers
[Iris3B] step 1/50
[Iris3B] step 2/50
...
```

## First W4A8 test

```text
512 x 512
50 steps
order 2
CFG 3.0
shift 4.0
bfloat16
unload_after = true
```

This build fixes the diffusion-model W4A8 loader. The Qwen3-VL encoder is still
the HF baseline path for now; local quantized Qwen3-VL is a separate next step.


## Iris odd-width INT8 fallback

Iris uses an odd SwiGLU intermediate width of `6826`.

Some of those layers are stored as `int8_tensorwise` fallback weights by the
W4A8 converter. NVIDIA/Comfy Kitchen INT8 GEMM requires matrix dimensions
divisible by 4, while `6826 % 4 == 2`.

This build detects incompatible INT8 layers and sets:

```text
full_precision_matrix_mult = true
```

for those layers only. The weights remain quantized in the checkpoint, but
Comfy dequantizes them for that individual matmul instead of calling the
unsupported INT8 CUDA kernel.

Compatible W4A8/INT8 layers still use their normal quantized paths.


## W4A8 pixel-space precision fix

This build changes Iris inference precision handling:

- W4A8/INT8 weights remain packed and quantized.
- The evolving pixel-space sample `x` remains FP32.
- Iris conditioning is passed to the diffusion model as FP32.
- Quantized Iris `mixed_precision_ops` are constructed with FP32 compute dtype.
- Global CUDA BF16 autocast is disabled for the W4A8 diffusion pass.
- Standard unquantized Iris keeps FP32 model weights and uses autocast, matching
  the upstream execution pattern more closely.
- The previous odd-width INT8 GEMM fallback remains included.

This targets structured 16x16/block/checkerboard artifacts caused by applying a
global low-precision path to Iris's pixel-space residual/output stages.


## Upstream autocast execution fix

The official Iris sampler keeps model/solver state in FP32 and wraps generation
in CUDA BF16 autocast. Previous prototype builds deviated from this in two ways:

- an early build cast the whole diffusion model to BF16;
- the later W4A8 precision build disabled autocast for quantized inference.

This build matches upstream sampling semantics:

- integrator/noise state: FP32
- conditioning passed to Iris: FP32
- unquantized model weights: FP32
- CUDA inference context: BF16 autocast
- W4A8 packed weights: preserved
- odd-width INT8 fallback: preserved
- PixelFP checkpoints remain supported
