"""Warmup learning-rate schedules built on LambdaLR."""

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from iris3b.config import OptimizerConfig


def constant_with_warmup(optimizer: Optimizer, warmup_steps: int) -> LambdaLR:
    """Linear 0 -> 1 ramp over ``warmup_steps``, then a constant factor of 1."""

    def factor(step: int) -> float:
        if step < warmup_steps:
            return step / max(1.0, warmup_steps)
        return 1.0

    return LambdaLR(optimizer, factor)


def cosine_with_warmup(optimizer: Optimizer, warmup_steps: int, total_steps: int) -> LambdaLR:
    """Linear warmup followed by a half-cosine decay to 0 at ``total_steps``."""

    def factor(step: int) -> float:
        if step < warmup_steps:
            return step / max(1.0, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, factor)


def build_lr_scheduler(
    cfg: OptimizerConfig, optimizer: Optimizer, world_size: int, total_steps: int
) -> LambdaLR:
    """Build the configured schedule in per-process scheduler steps.

    A scheduler prepared by accelerate advances once per process for every
    optimizer step, so warmup (and the cosine horizon) stretch by
    ``world_size`` when ``cfg.scale_warmup_by_world`` is set.
    """
    mult = world_size if cfg.scale_warmup_by_world else 1
    warmup = cfg.warmup_steps * mult
    if cfg.schedule == "constant":
        return constant_with_warmup(optimizer, warmup)
    if cfg.schedule == "cosine":
        return cosine_with_warmup(optimizer, warmup, total_steps * mult)
    raise ValueError(f"unknown lr schedule '{cfg.schedule}'")
