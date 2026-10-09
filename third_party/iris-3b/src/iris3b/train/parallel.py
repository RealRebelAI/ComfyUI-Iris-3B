"""Data-parallel strategy construction: DDP and FSDP2 from one config.

Nothing here holds run state. Every entry point is a pure function of
``train.dist`` (plus the model config where the topology is model-derived), so
the layer is unit-testable without a process group; ``torch.distributed`` is
imported only inside the functions that need a live world.

The strategies differ in what crosses the network per step. DDP moves
``2M(W-1)/W`` bytes of gradient, FSDP full sharding ``3M(W-1)/W`` of parameter
and gradient, and hybrid sharding keeps the all-gathers inside one shard group
and sends only ``~2(M/G)(R-1)/R`` between groups, which is why the mesh exists.
"""

import inspect
from contextlib import contextmanager
from functools import lru_cache
from types import MethodType

import torch
from torch import nn

import iris3b.models  # noqa: F401  (registers the block classes in BLOCKS)
from iris3b.registry import BLOCKS

_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


# (config attribute, DDP kwarg, the value that omitting the kwarg leaves in force)
_OPTIONAL_DDP = (
    ("batched_grad_copy", "batched_grad_copy", False),
    ("skip_all_reduce_unused_params", "skip_all_reduce_unused_params", False),
    ("forward_sync_buffers", "broadcast_buffers", True),
)


def wrap_class_names(model_cfg) -> set[str]:
    """Module class names that become their own shard/all-gather unit."""
    names = {BLOCKS.get(model_cfg.block).__name__}
    if model_cfg.dual_depth > 0:
        # a hybrid stack runs two block families; naming only one leaves the
        # other unsharded and silently replicated on every rank
        names.add("MMDiTBlock")
    if model_cfg.pixel.enabled:
        names.add("PiTBlock")
    if model_cfg.text_adapter == "blocks2":
        names.add("TextAdapterBlock")
    elif model_cfg.text_adapter == "lap_blocks2":
        names.update({"LayerwiseAttentionBlock", "TextAdapterBlock"})
    return names


def resolve_mesh(dist_cfg, world_size: int, local_world_size: int) -> tuple[int, int]:
    """``(dp_replicate, dp_shard)`` for ``world_size`` ranks.

    ``mesh.dp_shard == 0`` means shard over the whole world. A hybrid
    factorization must tile the world exactly, and its shard group should sit
    on one side of a node boundary or the sharding collectives it was chosen
    to keep local go back onto the network.
    """
    mesh = dist_cfg.mesh
    if not dist_cfg.hybrid:
        if (mesh.dp_shard not in (0, world_size)) or (mesh.dp_replicate not in (0, 1)):
            raise ValueError(
                f"train.dist.sharding={dist_cfg.sharding!r} shards over every rank, so "
                f"mesh.dp_shard must be 0 or {world_size} (got {mesh.dp_shard}) and "
                f"mesh.dp_replicate must be 0 or 1 (got {mesh.dp_replicate}); "
                "use sharding=hybrid or hybrid_grad_op to replicate across groups"
            )
        return 1, world_size

    dp_shard = mesh.dp_shard
    if dp_shard <= 0:
        raise ValueError(
            "hybrid sharding needs train.dist.mesh.dp_shard (the ranks one copy is "
            "sharded over, normally the GPU count of one node)"
        )
    dp_replicate = mesh.dp_replicate or world_size // dp_shard
    if dp_replicate * dp_shard != world_size:
        raise ValueError(
            f"train.dist.mesh does not tile the world: dp_replicate={dp_replicate} x "
            f"dp_shard={dp_shard} = {dp_replicate * dp_shard}, but world_size={world_size}"
        )
    if local_world_size > 1 and dp_shard % local_world_size and local_world_size % dp_shard:
        raise ValueError(
            f"train.dist.mesh.dp_shard={dp_shard} straddles a node boundary at "
            f"{local_world_size} ranks per node: a shard group that is neither inside one "
            "node nor a whole number of nodes puts the all-gather back on the network"
        )
    return dp_replicate, dp_shard


def build_device_mesh(dist_cfg, world_size: int, local_world_size: int):
    """Device mesh for a sharded strategy: 2D for hybrid, 1D otherwise."""
    from torch.distributed.device_mesh import init_device_mesh

    dp_replicate, dp_shard = resolve_mesh(dist_cfg, world_size, local_world_size)
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    if dist_cfg.hybrid:
        return init_device_mesh(
            device_type, (dp_replicate, dp_shard), mesh_dim_names=("dp_replicate", "dp_shard")
        )
    return init_device_mesh(device_type, (dp_shard,), mesh_dim_names=("dp_shard",))


def mixed_precision_policy(dist_cfg) -> tuple[torch.dtype, torch.dtype]:
    """``(param_dtype, reduce_dtype)`` as torch dtypes.

    ``reduce_dtype`` defaults to fp32 because bf16 reduction error grows with
    the number of ranks reduced over; left unset, FSDP inherits the bf16
    parameter dtype for the reduce as well.
    """
    try:
        return _DTYPES[dist_cfg.param_dtype], _DTYPES[dist_cfg.reduce_dtype]
    except KeyError as exc:
        raise ValueError(
            f"train.dist param_dtype/reduce_dtype must be one of {sorted(_DTYPES)}, got "
            f"param_dtype={dist_cfg.param_dtype!r} reduce_dtype={dist_cfg.reduce_dtype!r}"
        ) from exc


@lru_cache(maxsize=1)
def _ddp_supported() -> frozenset[str]:
    """DDP kwargs the installed torch accepts and accelerate's handler carries.

    accelerate transports DDP kwargs through a fixed dataclass, so a kwarg torch
    understands but the handler has no field for cannot reach the reducer.
    """
    from dataclasses import fields

    from accelerate import DistributedDataParallelKwargs

    accepted = set(inspect.signature(nn.parallel.DistributedDataParallel.__init__).parameters)
    return frozenset(accepted & {f.name for f in fields(DistributedDataParallelKwargs)})


def ddp_kwargs(dist_cfg, model_cfg) -> dict:
    """Keyword arguments for ``DistributedDataParallelKwargs``.

    ``static_graph`` is absent and must stay absent: measured against this
    model it corrupts the gradient allreduce in the presence of the unused text
    tail (+22% grad norm on step 1 with a bitwise-identical loss).
    """
    ddp = dist_cfg.ddp
    if ddp.find_unused_parameters == "auto":
        # Only a dual-stream final block leaves parameters gradient-free: its
        # text tail (norm_y2, attn.proj_y, mlp_y) is computed and then
        # discarded. A single-stream block shares every weight between the two
        # halves, so the image rows keep all of them alive whatever
        # final_block_text says. The trunk is dual at the tail when every block
        # is dual or when the tail family itself is mmdit.
        last_is_dual = model_cfg.dual_depth == model_cfg.depth or model_cfg.block == "mmdit"
        find_unused = last_is_dual and model_cfg.final_block_text == "keep"
    else:
        find_unused = ddp.find_unused_parameters == "true"
    kwargs: dict = {
        "find_unused_parameters": find_unused,
        "gradient_as_bucket_view": ddp.gradient_as_bucket_view,
    }
    if ddp.bucket_cap_mb > 0:
        kwargs["bucket_cap_mb"] = ddp.bucket_cap_mb
    supported = _ddp_supported()
    for attr, kwarg, _omitted in _OPTIONAL_DDP:
        if kwarg in supported:
            kwargs[kwarg] = getattr(ddp, attr)
    return kwargs


def unsupported_ddp_options(dist_cfg) -> list[str]:
    """Config knobs the installed torch/accelerate pair silently cannot apply."""
    supported = _ddp_supported()
    return [
        f"train.dist.ddp.{attr}"
        for attr, kwarg, omitted in _OPTIONAL_DDP
        if kwarg not in supported and getattr(dist_cfg.ddp, attr) != omitted
    ]


@contextmanager
def _fsdp2_no_sync(model):
    """Gradient accumulation without a reduce-scatter per micro-batch.

    fsdp2 replaced ``no_sync`` with ``set_requires_gradient_sync``, but
    accelerate's ``accumulate`` still probes the module for ``no_sync`` and
    silently reduces every micro-batch when it is absent. Unsharded gradients
    accumulate for the duration.
    """
    model.set_requires_gradient_sync(False)
    try:
        yield
    finally:
        model.set_requires_gradient_sync(True)


def shard_model(model: nn.Module, dist_cfg, mesh, wrap_names: set[str]) -> nn.Module:
    """``fully_shard`` each named block, then the root module.

    Per-block groups keep one all-gather to one block's parameters; the root
    call sweeps the embedders, head and auxiliary projectors into a final
    group. ``fully_shard`` also moves each group's states to the mesh device as
    it goes, so the full model is never resident unsharded.

    This REPLACES every ``nn.Parameter`` with a DTensor-backed one, which is
    why it must run before the optimizer is built.
    """
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    param_dtype, reduce_dtype = mixed_precision_policy(dist_cfg)
    kwargs = {
        "mesh": mesh,
        "mp_policy": MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype),
        "reshard_after_forward": dist_cfg.reshard_after_forward,
    }
    blocks = [module for module in model.modules() if type(module).__name__ in wrap_names]
    if not blocks:
        raise ValueError(
            f"none of {sorted(wrap_names)} found in {type(model).__name__}: fsdp2 would put "
            "the whole model in a single all-gather group"
        )
    for block in blocks:
        fully_shard(block, **kwargs)
    # the root group stays gathered through backward. TrainModel returns a
    # TrainOutput dataclass, which FSDP2 cannot traverse to register its
    # pre-backward hook, so a resharded root is never re-gathered and backward
    # dies reading freed storage. The group is only the embedders, head and
    # projectors, so holding it costs little.
    fully_shard(model, **{**kwargs, "reshard_after_forward": False})
    model.no_sync = MethodType(_fsdp2_no_sync, model)
    return model


def assert_optimizer_sees_shards(optimizer, dist_cfg) -> None:
    """Refuse an optimizer built before ``shard_model``.

    ``fully_shard`` swaps every ``nn.Parameter`` for a DTensor-backed one, so an
    optimizer constructed first holds the orphaned pre-shard tensors. Nothing
    raises: the step runs, the loss moves, peak memory silently carries a full
    unsharded optimizer state, and the parameters the model actually reads are
    never updated. Checked once at startup, so the cost is a single pass.
    """
    if dist_cfg.strategy != "fsdp2":
        return
    from torch.distributed.tensor import DTensor

    params = [p for group in optimizer.param_groups for p in group["params"]]
    orphans = sum(1 for p in params if not isinstance(p, DTensor))
    if orphans:
        raise RuntimeError(
            f"train.dist.strategy=fsdp2: {orphans} of {len(params)} optimizer parameters are not "
            "DTensors, so they are pre-shard tensors the model no longer uses. Build the optimizer "
            "after parallel.shard_model()."
        )


def full_grad_norm(total_norm: torch.Tensor) -> torch.Tensor:
    """Materialize a possibly-sharded gradient norm.

    ``clip_grad_norm_`` over DTensor gradients returns a DTensor whose local
    value is a partial norm, so every scalar read of it — ``isfinite``,
    ``.item()``, logging — has to reduce first.
    """
    if hasattr(total_norm, "full_tensor"):
        return total_norm.full_tensor()
    return total_norm
