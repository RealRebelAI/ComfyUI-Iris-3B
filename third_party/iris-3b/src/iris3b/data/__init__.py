"""Data pipeline: indexed tar shards, shape policies, datasets, samplers, loaders."""

from iris3b.data.buckets import SHARED_21_512, SHARED_21_1024, TRAIN_BUCKETS, closest_ratio
from iris3b.data.builder import build_dataloader, collate_batch
from iris3b.data.samplers import RangedSampler, ShapeBatchSampler
from iris3b.data.shapes import (
    AreaPolicy,
    BucketPolicy,
    FixedSquarePolicy,
    Shape,
    ShapePolicy,
    build_shape_policy,
)
from iris3b.data.wids import IndexedTarDataset

__all__ = [
    "SHARED_21_512",
    "SHARED_21_1024",
    "TRAIN_BUCKETS",
    "AreaPolicy",
    "BucketPolicy",
    "FixedSquarePolicy",
    "IndexedTarDataset",
    "RangedSampler",
    "Shape",
    "ShapeBatchSampler",
    "ShapePolicy",
    "build_dataloader",
    "build_shape_policy",
    "closest_ratio",
    "collate_batch",
]
