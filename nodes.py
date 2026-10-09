import gc
import json
import sys
from contextlib import contextmanager
from pathlib import Path

import torch
import folder_paths


IRIS_HIDDEN_LAYERS = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)
IRIS_PROMPT_PREFIX = (
    "<|im_start|>system\n"
    "Describe the image by detailing the color, shape, size, texture, quantity, text, spatial "
    "relationships of the objects and background:<|im_end|>\n"
    "<|im_start|>user\n"
)
IRIS_PROMPT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"

THIS_DIR = Path(__file__).resolve().parent
THIRD_PARTY_SRC = THIS_DIR / "third_party" / "iris-3b" / "src"

TEXT_ENCODER_PRESETS = {
    "Qwen3-VL-4B-Instruct (HF baseline)": "Qwen/Qwen3-VL-4B-Instruct",
}


def _add_iris_source():
    if not THIRD_PARTY_SRC.is_dir():
        raise FileNotFoundError(
            "Iris source is missing from the node package.\n"
            f"Expected: {THIRD_PARTY_SRC}"
        )
    src = str(THIRD_PARTY_SRC)
    if src not in sys.path:
        sys.path.insert(0, src)


def _cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _checkpoint_choices():
    # Native ComfyUI checkpoint folder. No custom model directory.
    choices = folder_paths.get_filename_list("checkpoints")
    iris_like = [
        x for x in choices
        if x.lower().endswith((".safetensors", ".ckpt", ".pt", ".pth", ".gguf"))
    ]
    return iris_like or ["<no checkpoints found>"]


def _resolve_checkpoint(name):
    if name.startswith("<no checkpoints"):
        raise RuntimeError(
            "No files were found in ComfyUI/models/checkpoints.\n"
            "Put the Iris checkpoint there and refresh/restart ComfyUI."
        )
    return folder_paths.get_full_path_or_raise("checkpoints", name)


def _quant_metadata(path):
    if not str(path).lower().endswith(".safetensors"):
        return None
    try:
        from safetensors import safe_open
        with safe_open(str(path), framework="pt", device="cpu") as f:
            md = f.metadata() or {}
        raw = md.get("_quantization_metadata")
        if raw:
            return json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        pass
    return None



@contextmanager
def _temporary_linear_class(linear_cls):
    """Temporarily make Iris construct Comfy quant-aware Linear modules."""
    original = torch.nn.Linear
    torch.nn.Linear = linear_cls
    try:
        yield
    finally:
        torch.nn.Linear = original


def _prepare_quantized_state_dict(path):
    import comfy.utils

    state_dict, metadata = comfy.utils.load_torch_file(
        path, safe_load=True, device=torch.device("cpu"), return_metadata=True
    )
    metadata = metadata or {}

    # Normal Comfy quant format: metadata in safetensors header -> comfy_quant tensors.
    if not any(k.endswith(".comfy_quant") for k in state_dict):
        state_dict, metadata = comfy.utils.convert_old_quants(
            state_dict, model_prefix="", metadata=metadata
        )

    # Rebel W4A8 converter fallback: if packed W4 scales are present but the
    # file has no header metadata, reconstruct the per-layer Comfy config.
    # These values match the Iris W4A8 conversion recipe:
    # group_size=16, convrot_groupsize=256.
    if not any(k.endswith(".comfy_quant") for k in state_dict):
        w4_layers = sorted(
            k[:-len(".weight_s_rel")]
            for k in state_dict
            if k.endswith(".weight_s_rel")
        )
        for layer in w4_layers:
            conf = {
                "format": "asym_w4a8_int8",
                "group_size": 16,
                "convrot_groupsize": 256,
            }
            state_dict[f"{layer}.comfy_quant"] = torch.tensor(
                list(json.dumps(conf).encode("utf-8")), dtype=torch.uint8
            )

    # Iris has SwiGLU width 6826. The converter intentionally falls back to
    # tensorwise INT8 for matrices that cannot use grouped W4A8. CUDA INT8 GEMM
    # requires matrix dimensions divisible by 4, but 6826 % 4 == 2.
    #
    # Keep those weights quantized on disk/in memory, but force Comfy's
    # full-precision matmul fallback for only the incompatible INT8 layers.
    # Comfy will dequantize the QuantizedTensor for the operation instead of
    # dispatching comfy_kitchen.int8_linear.
    for key in list(state_dict.keys()):
        if not key.endswith(".comfy_quant"):
            continue

        prefix = key[:-len(".comfy_quant")]
        raw = state_dict[key]
        try:
            conf = json.loads(raw.numpy().tobytes())
        except Exception:
            continue

        if conf.get("format") != "int8_tensorwise":
            continue

        weight = state_dict.get(prefix + ".weight")
        if weight is None or weight.ndim < 2:
            continue

        out_features = int(weight.shape[-2])
        in_features = int(weight.shape[-1])

        if (in_features % 4) != 0 or (out_features % 4) != 0:
            conf["full_precision_matrix_mult"] = True
            state_dict[key] = torch.tensor(
                list(json.dumps(conf).encode("utf-8")), dtype=torch.uint8
            )
            print(
                f"[Iris3B] INT8 GEMM fallback enabled for {prefix} "
                f"shape={tuple(weight.shape)}",
                flush=True,
            )

    quant_layers = [k for k in state_dict if k.endswith(".comfy_quant")]
    return state_dict, metadata, quant_layers


def _construct_quant_iris(IrisDiT, cfg_obj, compute_dtype, full_precision_mm=False):
    import comfy.ops

    operations = comfy.ops.mixed_precision_ops({}, compute_dtype=compute_dtype, full_precision_mm=full_precision_mm)

    # Iris initializes weights in __init__. Quant-aware Linear modules intentionally
    # have no normal floating weight until state_dict loading, so skip init entirely.
    original_init = IrisDiT.initialize_weights
    IrisDiT.initialize_weights = lambda self: None
    try:
        with torch.device("meta"), _temporary_linear_class(operations.Linear):
            model = IrisDiT(cfg_obj.model)
    finally:
        IrisDiT.initialize_weights = original_init

    return model


def _build_prompt_batch(tokenizer, prompts, max_length=300):
    prefix_ids = tokenizer.encode(IRIS_PROMPT_PREFIX, add_special_tokens=False)
    suffix_ids = tokenizer.encode(IRIS_PROMPT_SUFFIX, add_special_tokens=False)

    caption_budget = max_length - len(suffix_ids)
    if caption_budget < 1:
        raise RuntimeError("max_length leaves no room for caption tokens.")

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise RuntimeError("Tokenizer has neither pad_token_id nor eos_token_id.")

    captions = tokenizer(prompts, add_special_tokens=False)["input_ids"]
    start = len(prefix_ids)
    stop = start + max_length

    input_ids = torch.full((len(prompts), stop), int(pad_id), dtype=torch.long)
    attention_mask = torch.zeros((len(prompts), stop), dtype=torch.long)

    input_ids[:, :start] = torch.tensor(prefix_ids, dtype=torch.long)
    suffix = torch.tensor(suffix_ids, dtype=torch.long)

    for row, ids in enumerate(captions):
        n = min(len(ids), caption_budget)
        if n:
            input_ids[row, start:start+n] = torch.tensor(ids[:n], dtype=torch.long)
        input_ids[row, start+n:start+n+len(suffix_ids)] = suffix
        attention_mask[row, :start+n+len(suffix_ids)] = 1

    return input_ids, attention_mask, start, stop


class Iris3BCheckpoint:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "checkpoint": (_checkpoint_choices(),),
            }
        }

    RETURN_TYPES = ("IRIS3B_CHECKPOINT",)
    RETURN_NAMES = ("iris_checkpoint",)
    FUNCTION = "load"
    CATEGORY = "Rebel AI/Iris 3B"

    BUILD = "w4a8-matmul-A-B-test-v3"

    def load(self, checkpoint):
        _add_iris_source()
        path = _resolve_checkpoint(checkpoint)
        quant = _quant_metadata(path)

        return ({
            "name": checkpoint,
            "path": path,
            "quantization_metadata": quant,
        },)


class Iris3BQwenEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "iris_checkpoint": ("IRIS3B_CHECKPOINT",),
                "prompt": ("STRING", {"multiline": True, "default": "a red fox sleeping in fresh snow, realistic wildlife photography"}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "text_encoder": (list(TEXT_ENCODER_PRESETS.keys()),),
                "load_mode": (["auto", "cpu", "cuda"], {"default": "auto"}),
                "gpu_memory_mb": ("INT", {"default": 6200, "min": 1024, "max": 65536, "step": 128}),
                "cpu_memory_mb": ("INT", {"default": 8192, "min": 2048, "max": 131072, "step": 256}),
            }
        }

    RETURN_TYPES = ("IRIS3B_CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = "Rebel AI/Iris 3B"

    def encode(self, iris_checkpoint, prompt, negative_prompt, text_encoder, load_mode, gpu_memory_mb, cpu_memory_mb):
        _add_iris_source()
        from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration

        model_id = TEXT_ENCODER_PRESETS[text_encoder]
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        kwargs = {
            "dtype": torch.bfloat16,
            "attn_implementation": "sdpa",
            "low_cpu_mem_usage": True,
        }

        if load_mode == "auto" and torch.cuda.is_available():
            kwargs["device_map"] = "auto"
            kwargs["max_memory"] = {
                0: f"{int(gpu_memory_mb)}MiB",
                "cpu": f"{int(cpu_memory_mb)}MiB",
            }

        model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, **kwargs)

        if load_mode == "cpu":
            model = model.to("cpu")
        elif load_mode == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA selected but CUDA is not available.")
            model = model.to("cuda")

        model.eval()
        decoder = model.get_decoder()
        decoder.eval()
        decoder.requires_grad_(False)

        input_ids, attention_mask, start, stop = _build_prompt_batch(
            tokenizer, [prompt, negative_prompt], max_length=300
        )

        try:
            embed_device = decoder.get_input_embeddings().weight.device
        except Exception:
            embed_device = next(decoder.parameters()).device

        input_ids = input_ids.to(embed_device)
        attention_mask = attention_mask.to(embed_device)

        try:
            with torch.inference_mode():
                output = decoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                    output_hidden_states=True,
                )

            if output.hidden_states is None or len(output.hidden_states) <= IRIS_HIDDEN_LAYERS[-1]:
                raise RuntimeError(
                    f"Qwen3-VL returned only {0 if output.hidden_states is None else len(output.hidden_states)} "
                    f"hidden-state entries; Iris needs layer {IRIS_HIDDEN_LAYERS[-1]}."
                )

            mask = attention_mask[:, start:stop].to("cpu", dtype=torch.int64)

            selected = [
                output.hidden_states[layer][:, start:stop].detach().to("cpu", dtype=torch.bfloat16)
                for layer in IRIS_HIDDEN_LAYERS
            ]
            embeddings = torch.stack(selected, dim=2)
            embeddings = embeddings * mask[:, :, None, None].to(embeddings.dtype)

            conditioning = {
                "positive": embeddings[0:1].contiguous(),
                "positive_mask": mask[0:1].contiguous(),
                "negative": embeddings[1:2].contiguous(),
                "negative_mask": mask[1:2].contiguous(),
            }
        finally:
            try:
                del output
            except Exception:
                pass
            del decoder
            del model
            _cleanup_cuda()

        return (conditioning,)


class _ProgressFlowDPMSolver:
    def __init__(self, base_solver):
        self.base = base_solver

    @torch.no_grad()
    def sample(self, z, cond, uncond=None, steps=50, order=2, shift=1.0):
        try:
            from comfy.utils import ProgressBar
            pbar = ProgressBar(steps)
        except Exception:
            pbar = None

        grid = self.base.time_grid(steps, shift)
        x = z
        history = []

        for i in range(1, steps + 1):
            print(f"[Iris3B] step {i}/{steps}", flush=True)

            s, t = grid[i - 1], grid[i]
            x0 = self.base._pred_x0(x, s, cond, uncond)
            history.append((s, x0))

            step_order = min(i, order, steps + 1 - i)
            if step_order == 1:
                x = self.base._first_order(x, s, t, x0)
            else:
                x = self.base._second_order(x, history[-2], history[-1], t)

            if len(history) > 2:
                history.pop(0)

            if pbar is not None:
                pbar.update_absolute(i, steps)

        return x


class Iris3BSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "iris_checkpoint": ("IRIS3B_CHECKPOINT",),
                "conditioning": ("IRIS3B_CONDITIONING",),
                "width": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 16}),
                "height": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 16}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 200}),
                "order": ([1, 2], {"default": 2}),
                "cfg": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 20.0, "step": 0.1}),
                "shift": ("FLOAT", {"default": 4.0, "min": 0.1, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "model_dtype": (["bfloat16", "float16"], {"default": "bfloat16"}),
                "w4a8_matmul": (["dequantized_A_B_test", "native_quantized"], {"default": "dequantized_A_B_test"}),
                "unload_after": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "sample"
    CATEGORY = "Rebel AI/Iris 3B"

    def sample(self, iris_checkpoint, conditioning, width, height, steps, order, cfg, shift, seed, model_dtype, w4a8_matmul, unload_after):
        if width % 16 or height % 16:
            raise ValueError("Iris width and height must be divisible by 16.")
        if not torch.cuda.is_available():
            raise RuntimeError("This prototype sampler currently requires CUDA.")

        path = iris_checkpoint["path"]
        quant = iris_checkpoint.get("quantization_metadata")

        if str(path).lower().endswith(".gguf"):
            raise RuntimeError(
                "The GGUF Iris backend is not wired yet. "
                "The checkpoint dropdown is fixed, but this sampler currently handles safetensors only."
            )


        _add_iris_source()

        from safetensors.torch import load_file
        from iris3b.config import Config
        from iris3b.flow.solver import FlowDPMSolver
        from iris3b.models.dit import IrisDiT

        cfg_obj = Config()
        dtype = torch.bfloat16 if model_dtype == "bfloat16" else torch.float16
        device = torch.device("cuda")

        _cleanup_cuda()

        # W4A8/INT8 checkpoints need Comfy's mixed-precision Linear modules.
        # Standard checkpoints keep the original upstream torch.nn.Linear path.
        weights_q, metadata_q, quant_layers = _prepare_quantized_state_dict(path)
        is_quantized = len(quant_layers) > 0

        if is_quantized:
            print(
                f"[Iris3B] quantized checkpoint detected: {len(quant_layers)} quantized Linear layers",
                flush=True,
            )
            model = _construct_quant_iris(
                IrisDiT, cfg_obj, torch.bfloat16,
                full_precision_mm=(w4a8_matmul == "dequantized_A_B_test")
            )
            print(f"[Iris3B] W4A8 matmul mode: {w4a8_matmul}", flush=True)
            weights = weights_q
        else:
            # _prepare_quantized_state_dict already loaded it; reuse that state dict.
            weights = weights_q
            with torch.device("meta"):
                model = IrisDiT(cfg_obj.model)

        missing, unexpected = model.load_state_dict(weights, strict=False, assign=True)
        if missing or unexpected:
            raise RuntimeError(
                "Iris state_dict mismatch.\n"
                f"Missing ({len(missing)}): {missing[:30]}\n"
                f"Unexpected ({len(unexpected)}): {unexpected[:30]}"
            )

        del weights
        del weights_q
        model.eval()
        model.requires_grad_(False)

        try:
            # IMPORTANT: never dtype-cast packed QuantizedTensor weights.
            # Comfy quant Linear handles W4A8 on-device; only move them.
            if is_quantized:
                model = model.to(device=device)
            else:
                model = model.to(device=device, dtype=torch.float32)

            cond = conditioning["positive"].to(device=device, dtype=torch.float32)
            cond_mask = conditioning["positive_mask"].to(device=device)

            uncond = None
            cfg_mask = None
            if float(cfg) != 1.0:
                uncond = conditioning["negative"].to(device=device, dtype=torch.float32)
                uncond_mask = conditioning["negative_mask"].to(device=device)
                cfg_mask = torch.cat([uncond_mask, cond_mask], dim=0)

            generator = torch.Generator(device=device).manual_seed(int(seed))
            z = torch.randn(
                1, cfg_obj.model.in_channels, int(height), int(width),
                generator=generator, device=device, dtype=torch.float32
            )

            def model_fn(x, t_model, y):
                y_mask = cond_mask if y.shape[0] == cond.shape[0] else cfg_mask

                # Match upstream Iris generate(): solver/input state stays FP32,
                # while the outer CUDA autocast controls heavy-op precision.
                x_in = x.to(device=device, dtype=torch.float32)
                t_in = t_model.to(device=device, dtype=torch.float32)
                y_in = y.to(device=device, dtype=torch.float32)

                return model(x_in, t_in, y_in, y_mask=y_mask).x.float()

            base_solver = FlowDPMSolver(
                model_fn,
                num_timesteps=cfg_obj.flow.num_train_timesteps,
                cfg_scale=float(cfg),
                cfg_interval=tuple(cfg_obj.sample.cfg_interval),
                prediction=cfg_obj.flow.prediction,
            )

            solver = _ProgressFlowDPMSolver(base_solver)

            print(
                "[Iris3B] upstream execution mode: FP32 solver/model state + "
                "BF16 CUDA autocast",
                flush=True,
            )
            with torch.inference_mode(), torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=True
            ):
                sample = solver.sample(
                    z, cond, uncond,
                    steps=int(steps), order=int(order), shift=float(shift)
                )

            sample = sample.clamp(-1, 1)
            image = sample.add(1.0).div(2.0).permute(0, 2, 3, 1).float().cpu().contiguous()

        finally:
            if unload_after:
                try:
                    del model
                except Exception:
                    pass
                _cleanup_cuda()

        return (image,)


NODE_CLASS_MAPPINGS = {
    "Iris3BCheckpoint": Iris3BCheckpoint,
    "Iris3BQwenEncode": Iris3BQwenEncode,
    "Iris3BSampler": Iris3BSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Iris3BCheckpoint": "Iris 3B Checkpoint",
    "Iris3BQwenEncode": "Iris 3B Qwen3-VL Encode",
    "Iris3BSampler": "Iris 3B Sampler",
}
