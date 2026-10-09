"""Exponential moving average of model weights."""

from contextlib import contextmanager
from copy import deepcopy

import torch
from torch import nn


class EMA:
    """Shadow copy updated as ``p_ema = decay * p_ema + (1 - decay) * p``.

    Parameters are blended each ``update``; buffers are copied verbatim.
    The shadow module never requires grad and stays in eval mode.

    ``foreach=True`` performs the identical update with two multi-tensor
    kernels instead of two kernels per parameter (mathematically exact; the
    tensor pairing is cached on first use). Falls back to the per-parameter
    loop when any shadow/source dtype differs.
    """

    def __init__(self, model: nn.Module, decay: float, foreach: bool = False):
        self.decay = decay
        self.foreach = foreach
        self.module = deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self._param_pairs: tuple[list[torch.Tensor], list[torch.Tensor]] | None = None
        self._buffer_pairs: list[tuple[torch.Tensor, torch.Tensor]] | None = None

    def _build_pairs(self, model: nn.Module) -> None:
        src_params = dict(model.named_parameters())
        ema, src = [], []
        for name, p in self.module.named_parameters():
            ema.append(p)
            src.append(src_params[name])
        if any(e.dtype != s.dtype for e, s in zip(ema, src, strict=True)):
            self.foreach = False  # exact fallback: the loop casts per tensor
        self._param_pairs = (ema, src)
        src_buffers = dict(model.named_buffers())
        self._buffer_pairs = [(b, src_buffers[name]) for name, b in self.module.named_buffers()]

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        if self.foreach:
            if self._param_pairs is None:
                self._build_pairs(model)
            if self.foreach:
                ema, src = self._param_pairs
                torch._foreach_mul_(ema, self.decay)
                torch._foreach_add_(ema, src, alpha=1.0 - self.decay)
                for b, s in self._buffer_pairs:
                    b.copy_(s)
                return
        src_params = dict(model.named_parameters())
        for name, p in self.module.named_parameters():
            p.mul_(self.decay).add_(src_params[name].to(p.dtype), alpha=1.0 - self.decay)
        src_buffers = dict(model.named_buffers())
        for name, b in self.module.named_buffers():
            b.copy_(src_buffers[name])

    def state_dict(self) -> dict:
        return self.module.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        self.module.load_state_dict(state_dict)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        """Write the averaged weights into ``model``."""
        model.load_state_dict(self.module.state_dict())

    def to(self, device) -> "EMA":
        self.module.to(device)
        return self


def _local(tensor: torch.Tensor) -> torch.Tensor:
    """This rank's own storage: a DTensor's shard under fsdp2, else the tensor."""
    return tensor.to_local() if hasattr(tensor, "to_local") else tensor


class ShardEMA:
    """EMA over a sharded model's local parameter shards.

    Under fsdp2 the parameters are DTensors, and the pre/post-forward hooks
    swap them for plain unsharded tensors for the duration of a pass, so this
    class is only correct after ``optimizer.step()``, where the sharded form is
    exposed.

    The update is pointwise on this rank's own storage, so the shard of the
    EMA equals the EMA of the shard and the union over ranks is exactly the
    full-tensor average. The shadow is cloned from the live parameters, which
    makes it a DTensor with identical placements; the blend runs on
    ``to_local()`` views of both sides, so a parameter that is unsharded when
    ``update`` runs fails on the size mismatch instead of averaging a full
    tensor into a shard.

    There is deliberately no ``state_dict``: shard-shaped tensors must never
    reach a checkpoint. Consolidation goes through :meth:`swapped` — write
    the shadow into the live shards, let the caller run a collective
    full-state-dict gather, then restore. Seeding at resume runs the same trick
    in reverse: the trainer loads the checkpoint's EMA weights into the model
    first, constructs this class (the snapshot IS the average), then restores
    the live weights.
    """

    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self._model = (model,)  # tuple hides the live model from any traversal
        self.shadow_params = [p.detach().clone() for p in model.parameters()]
        self.shadow_buffers = [b.detach().clone() for b in model.buffers()]

    @torch.no_grad()
    def update(self) -> None:
        model = self._model[0]
        src = [_local(p.detach()) for p in model.parameters()]
        shadow = [_local(s) for s in self.shadow_params]
        torch._foreach_mul_(shadow, self.decay)
        torch._foreach_add_(shadow, src, alpha=1.0 - self.decay)
        for buffer, live in zip(self.shadow_buffers, model.buffers(), strict=True):
            _local(buffer).copy_(_local(live))

    @contextmanager
    def swapped(self):
        """Write the averaged shards into the live parameters, restore on exit.

        The copy targets each parameter's local storage (the DTensor's shard),
        so a collective gather inside this context returns the EMA weights. NOT
        reentrant; never step inside.
        """
        model = self._model[0]
        backup = [_local(p.detach()).clone() for p in model.parameters()]
        with torch.no_grad():
            for p, shadow in zip(model.parameters(), self.shadow_params, strict=True):
                _local(p.data).copy_(_local(shadow))
        try:
            yield
        finally:
            with torch.no_grad():
                for p, saved in zip(model.parameters(), backup, strict=True):
                    _local(p.data).copy_(saved)
