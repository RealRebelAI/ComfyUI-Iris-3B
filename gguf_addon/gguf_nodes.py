"""Isolated Iris-3B GGUF loader/sampler.

Does not import or modify the W4A8/FP32 nodes. Share the original Iris
conditioning node via the existing IRIS3B_CONDITIONING interface.
"""
import gc
import importlib
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import torch
import folder_paths

HERE = Path(__file__).resolve().parent


def _iris_source():
    """Find upstream Iris already installed with the user's base Iris node."""
    base = HERE.parent
    candidates = []
    for sibling in base.iterdir():
        if sibling.is_dir() and sibling.resolve() != HERE and "iris" in sibling.name.lower():
            candidates.append(sibling / "third_party" / "iris-3b" / "src")
    candidates += [HERE / "third_party" / "iris-3b" / "src"]
    for src in candidates:
        if (src / "iris3b" / "config.py").is_file():
            if str(src) not in sys.path:
                sys.path.insert(0, str(src))
            return
    try:
        import iris3b.config  # noqa: F401
        return
    except ImportError as exc:
        raise RuntimeError(
            "Upstream Iris source not found. Install the original Iris node and "
            "run its install_iris_source.py first."
        ) from exc


def _backend():
    """Load city96 modules from a sibling install without re-registering its nodes."""
    if "_iris_gguf_city96" in sys.modules:
        return sys.modules["_iris_gguf_city96"]
    root = HERE.parent
    candidates = [root / "ComfyUI-GGUF", root / "comfyui-gguf"]
    candidates += [p for p in root.glob("*GGUF*") if p.is_dir()]
    folder = next((p for p in candidates if (p / "loader.py").is_file() and (p / "ops.py").is_file() and (p / "dequant.py").is_file()), None)
    if folder is None:
        raise RuntimeError("Install city96/ComfyUI-GGUF under ComfyUI/custom_nodes before using Iris GGUF.")
    package = types.ModuleType("_iris_gguf_city96")
    package.__path__ = [str(folder)]
    sys.modules[package.__name__] = package
    package.loader = importlib.import_module(package.__name__ + ".loader")
    package.ops = importlib.import_module(package.__name__ + ".ops")
    package.dequant = importlib.import_module(package.__name__ + ".dequant")
    return package


def _load_weights(path):
    backend = _backend()
    # city96 currently does not list 'iris' as a supported diffusion arch.
    # Admit only this one tag, and restore the backend whitelist immediately.
    allowed = backend.loader.IMG_ARCH_LIST
    was_present = "iris" in allowed
    try:
        allowed.add("iris")
        sd, extra = backend.loader.gguf_sd_loader(str(path), handle_prefix="")
    finally:
        if not was_present:
            allowed.remove("iris")
    arch = extra.get("arch_str")
    if arch not in {"iris", "wan"}:
        raise ValueError(f"Expected Iris GGUF (iris or legacy wan tag); found {arch!r}.")
    return sd, backend


@contextmanager
def _linear_override(linear_class):
    original = torch.nn.Linear
    try:
        torch.nn.Linear = linear_class
        yield
    finally:
        torch.nn.Linear = original


def _build_model(cfg, IrisDiT, linear_class):
    original_initialize = IrisDiT.initialize_weights
    try:
        IrisDiT.initialize_weights = lambda self: None
        with torch.device("meta"), _linear_override(linear_class):
            model = IrisDiT(cfg.model)
    finally:
        IrisDiT.initialize_weights = original_initialize
    return model


def _load_into_model(model, state_dict, backend):
    """Check keys; keep packed GGML data only inside GGUF Linear modules."""
    ggml_linear = backend.ops.GGMLOps.Linear
    expected = set(model.state_dict().keys())
    # GGMLLinear creates no weight during __init__, so its keys are missing
    # from state_dict on meta. Collect those explicitly.
    for name, mod in model.named_modules():
        if isinstance(mod, ggml_linear):
            expected.add(f"{name}.weight")
            if mod.bias is not None:
                expected.add(f"{name}.bias")
    incoming = set(state_dict)
    missing = sorted(expected - incoming)
    unexpected = sorted(incoming - expected)
    if missing or unexpected:
        raise RuntimeError(
            f"Iris GGUF tensor keys mismatch: missing {len(missing)} {missing[:12]}; "
            f"unexpected {len(unexpected)} {unexpected[:12]}"
        )

    for key, tensor in list(state_dict.items()):
        parent, _, leaf = key.rpartition(".")
        mod = model.get_submodule(parent) if parent else model
        if isinstance(mod, ggml_linear) and leaf in ("weight", "bias"):
            continue
        if backend.dequant.is_quantized(tensor):
            raise RuntimeError(f"Quantized non-Linear parameter {key} is unsupported; no silent corruption.")
        # Plain torch parameters should not retain a custom GGMLTensor subclass.
        # BF16 GGUF entries need their special bit-pattern dequantization first.
        qtype = getattr(tensor, "tensor_type", None)
        if getattr(qtype, "name", "") == "BF16":
            tensor = backend.dequant.dequantize_tensor(tensor, dtype=torch.float32)
        state_dict[key] = tensor.as_subclass(torch.Tensor) if type(tensor) is not torch.Tensor else tensor

    incompatible = model.load_state_dict(state_dict, strict=False, assign=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Iris GGUF model load mismatch: "
            f"missing={incompatible.missing_keys[:12]}, unexpected={incompatible.unexpected_keys[:12]}"
        )
    for name, module in model.named_modules():
        if isinstance(module, ggml_linear) and module.weight is None:
            raise RuntimeError(f"GGUF Linear weight not loaded: {name}")
    return model


def _progress_solver(base_solver, z, cond, uncond, steps, order, shift):
    """Same step loop as the working Iris A/B Diagnostic build."""
    try:
        from comfy.utils import ProgressBar
        progress = ProgressBar(steps)
    except ImportError:
        progress = None
    grid = base_solver.time_grid(steps, shift)
    x = z
    history = []
    for i in range(1, steps + 1):
        print(f"[Iris3B GGUF] step {i}/{steps}", flush=True)
        s, t = grid[i - 1], grid[i]
        x0 = base_solver._pred_x0(x, s, cond, uncond)
        history.append((s, x0))
        step_order = min(i, order, steps + 1 - i)
        x = (base_solver._first_order(x, s, t, x0) if step_order == 1
             else base_solver._second_order(x, history[-2], history[-1], t))
        if len(history) > 2:
            history.pop(0)
        if progress is not None:
            progress.update_absolute(i, steps)
    return x


class Iris3BGGUFCheckpoint:
    """Drop-in checkpoint for the ORIGINAL Iris Qwen encoder."""
    @classmethod
    def INPUT_TYPES(cls):
        names = [name for name in folder_paths.get_filename_list("checkpoints") if name.lower().endswith(".gguf")]
        return {"required": {"checkpoint": (names or ["<no GGUF files>"],)}}

    RETURN_TYPES = ("IRIS3B_CHECKPOINT",)
    RETURN_NAMES = ("iris_checkpoint",)
    FUNCTION = "load"
    CATEGORY = "Rebel AI/Iris 3B GGUF"

    def load(self, checkpoint):
        if checkpoint.startswith("<no "):
            raise RuntimeError("Put the Iris GGUF under ComfyUI/models/checkpoints/Iris-3B and refresh.")
        path = folder_paths.get_full_path_or_raise("checkpoints", checkpoint)
        if not str(path).lower().endswith(".gguf"):
            raise ValueError("GGUF checkpoint node only accepts .gguf")
        return ({"path": path, "name": checkpoint, "quantization_metadata": None},)


class Iris3BGGUFSampler:
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

    @torch.no_grad()
    def sample(self, iris_checkpoint, conditioning, width, height, steps, order, cfg, shift, seed, dequant_dtype, unload_after):
        path = iris_checkpoint["path"]
        if not str(path).lower().endswith(".gguf"):
            raise ValueError("This sampler requires an Iris GGUF. W4A8 and FP32 should use the original sampler.")
        if width % 16 or height % 16:
            raise ValueError("Width and height must be divisible by 16.")
        if not torch.cuda.is_available():
            raise RuntimeError("Iris GGUF sampler currently requires CUDA.")
        _iris_source()
        from iris3b.config import Config
        from iris3b.flow.solver import FlowDPMSolver
        from iris3b.models.dit import IrisDiT

        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[dequant_dtype]
        device = torch.device("cuda")
        gc.collect()
        torch.cuda.empty_cache()

        sd, backend = _load_weights(path)
        ops = backend.ops.GGMLOps()
        ops.Linear.dequant_dtype = dtype
        ops.Linear.patch_dtype = None
        cfg_obj = Config()  # Same as working A/B Diagnostic build.
        model = _build_model(cfg_obj, IrisDiT, ops.Linear)
        model = _load_into_model(model, sd, backend)
        del sd
        model.eval().requires_grad_(False)
        try:
            model = model.to(device=device)  # Never cast packed GGUF weights.
            cond = conditioning["positive"].to(device=device, dtype=torch.float32)
            cond_mask = conditioning["positive_mask"].to(device=device)
            uncond = None
            cfg_mask = None
            if float(cfg) != 1.0:
                uncond = conditioning["negative"].to(device=device, dtype=torch.float32)
                uncond_mask = conditioning["negative_mask"].to(device=device)
                cfg_mask = torch.cat([uncond_mask, cond_mask], dim=0)
            generator = torch.Generator(device=device).manual_seed(int(seed))
            z = torch.randn(1, cfg_obj.model.in_channels, int(height), int(width),
                            generator=generator, device=device, dtype=torch.float32)

            def model_fn(x, t, y):
                mask = cond_mask if y.shape[0] == cond.shape[0] else cfg_mask
                return model(x.float(), t.float(), y.float(), y_mask=mask).x.float()

            base = FlowDPMSolver(
                model_fn,
                num_timesteps=cfg_obj.flow.num_train_timesteps,
                cfg_scale=float(cfg),
                cfg_interval=tuple(cfg_obj.sample.cfg_interval),
                prediction=cfg_obj.flow.prediction,
            )
            print(f"[Iris3B GGUF] Using GGUF weights; upstream Iris solver settings; BF16/FP16 autocast={dtype}")
            with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
                sample = _progress_solver(base, z, cond, uncond, int(steps), int(order), float(shift))
            image = sample.clamp(-1, 1).add(1).div(2).permute(0, 2, 3, 1).float().cpu().contiguous()
            return (image,)
        finally:
            if unload_after:
                del model
                gc.collect()
                torch.cuda.empty_cache()


NODE_CLASS_MAPPINGS = {
    "Iris3BGGUFCheckpoint": Iris3BGGUFCheckpoint,
    "Iris3BGGUFSampler": Iris3BGGUFSampler,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Iris3BGGUFCheckpoint": "Iris 3B GGUF Checkpoint",
    "Iris3BGGUFSampler": "Iris 3B GGUF Sampler (Experimental)",
}