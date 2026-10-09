"""Dataloader assembly producing the training batch contract."""

import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from iris3b.config import DataConfig
from iris3b.data import datasets as _datasets  # noqa: F401 - registers dataset types
from iris3b.data.samplers import RangedSampler, ShapeBatchSampler, key_cache_path
from iris3b.registry import DATASETS


def collate_batch(items: list[dict]) -> dict:
    """Stack per-sample dicts into {image, caption, index, img_hw, aspect_ratio}.

    ``index`` is the absolute dataset index. It is what makes the frozen
    validation grid invariant to batching: (timestep, noise seed) key on the
    sample, not on its position in the stream, so grouping by shape, changing
    the batch size, or a future packed policy cannot silently re-label it.
    """
    return {
        "image": torch.stack([it["image"] for it in items]).float(),
        "caption": [it["caption"] for it in items],
        "index": torch.tensor([int(it["index"]) for it in items], dtype=torch.int64),
        "img_hw": torch.stack([torch.as_tensor(it["img_hw"], dtype=torch.int64) for it in items]),
        "aspect_ratio": torch.tensor([float(it["aspect_ratio"]) for it in items], dtype=torch.float32),
    }


class _WorkerSeeder:
    """Deterministic per-worker seeding of the caption-sampling rngs."""

    def __init__(self, base: int):
        self.base = base

    def __call__(self, worker_id: int) -> None:
        s = (self.base + worker_id) % 2**32
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)


def shape_policy_id(data_cfg: DataConfig, patch_size: int) -> str:
    """Identity of everything that changes a sample's shape key.

    Part of the shape-plan cache path, so it must be derived in exactly one
    place: a drifting id silently sends the pre-staged plan to a file no rank
    ever reads.
    """
    return (
        f"{data_cfg.resolved_shape_policy()}:{data_cfg.image_size}:{data_cfg.aspect_ratio_bucket}"
        f":{data_cfg.shape_align}:{data_cfg.shape_max_ratio}:{patch_size}"
    )


def build_dataloader(
    data_cfg: DataConfig,
    batch_size: int,
    rank: int,
    world_size: int,
    seed: int,
    patch_size: int = 16,
) -> tuple[DataLoader, RangedSampler | ShapeBatchSampler]:
    """Build the per-rank loader; the returned sampler exposes set_epoch/set_start.

    A uniform shape policy batches the rank's sequential chunk directly. Any
    shape-varying policy routes it through shape-homogeneous batching
    (drop_last), because the dense ``[B, C, H, W]`` collate below admits exactly
    one shape per batch.
    """
    dataset = DATASETS.build(data_cfg.type, data_cfg, patch_size)
    dataset.seed = seed  # keys positional caption sampling; trainer sets .epoch
    sampler = RangedSampler(len(dataset), rank=rank, world_size=world_size)
    common = {
        "num_workers": data_cfg.num_workers,
        "pin_memory": True,
        "collate_fn": collate_batch,
        "worker_init_fn": _WorkerSeeder(seed * 100_003 + rank * 1_009),
    }
    if dataset.policy.uniform:
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, **common), sampler
    policy_id = shape_policy_id(data_cfg, patch_size)
    cache = key_cache_path(data_cfg.data_dirs, world_size, rank, policy_id) if data_cfg.data_dirs else None
    batch_sampler = ShapeBatchSampler(sampler, dataset, batch_size, drop_last=True, cache_path=cache)
    return DataLoader(dataset, batch_sampler=batch_sampler, **common), batch_sampler
