"""Shape policies: the single seam that decides what (H, W) a sample trains at.

A policy answers two questions for one sample, from its native size alone:

- ``target`` — the (height, width) the sample is resized/cropped to;
- ``key`` — the grouping token used to assemble shape-homogeneous batches.

Everything downstream (dataset transform, batch sampler, collate, trainer)
consumes only that pair, so the batching strategy is a configuration choice
rather than an architectural commitment:

- ``fixed`` — one square size, one key.
- ``bucket`` — snap to the nearest entry of a hand-written aspect-ratio table.
- ``area`` — no table: preserve the native ratio, round both sides to a
  multiple of ``align`` under a token budget, and let the realized shape set be
  whatever the corpus contains.
"""

from dataclasses import dataclass
from math import sqrt

from iris3b.data.buckets import TRAIN_BUCKETS, closest_ratio


@dataclass(frozen=True)
class Shape:
    """A concrete training shape in pixels."""

    height: int
    width: int

    def __post_init__(self) -> None:
        if self.height <= 0 or self.width <= 0:
            raise ValueError(f"shape must be positive, got {self.height}x{self.width}")

    def grid(self, patch: int) -> tuple[int, int]:
        if self.height % patch or self.width % patch:
            raise ValueError(f"{self.height}x{self.width} is not divisible by patch {patch}")
        return self.height // patch, self.width // patch

    def tokens(self, patch: int) -> int:
        rows, cols = self.grid(patch)
        return rows * cols

    @property
    def key(self) -> str:
        return f"{self.height}x{self.width}"


class ShapePolicy:
    """Maps a sample's native size to its training shape and grouping key."""

    #: True when every sample resolves to the same shape, so the batch sampler
    #: can skip grouping entirely.
    uniform: bool = False

    def target(self, height: int, width: int) -> Shape:
        raise NotImplementedError

    def key(self, height: int, width: int) -> str:
        return self.target(height, width).key

    def shapes(self) -> tuple[Shape, ...] | None:
        """Enumerate realizable shapes, or None when the set is open."""
        return None


class FixedSquarePolicy(ShapePolicy):
    """One square size for every sample; shortest-side resize then center crop."""

    uniform = True

    def __init__(self, size: int):
        self._shape = Shape(size, size)

    def target(self, height: int, width: int) -> Shape:
        return self._shape

    def key(self, height: int, width: int) -> str:
        return self._shape.key

    def shapes(self) -> tuple[Shape, ...]:
        return (self._shape,)


class BucketPolicy(ShapePolicy):
    """Snap to the nearest aspect-ratio bucket of a fixed table."""

    def __init__(self, table_name: str):
        try:
            table = TRAIN_BUCKETS[table_name]
        except KeyError:
            known = ", ".join(sorted(TRAIN_BUCKETS))
            raise ValueError(f"unknown bucket table '{table_name}' (known: {known})") from None
        self.table_name = table_name
        self.table = table

    def target(self, height: int, width: int) -> Shape:
        _, (bucket_h, bucket_w) = closest_ratio(float(height), float(width), self.table)
        return Shape(int(bucket_h), int(bucket_w))

    def shapes(self) -> tuple[Shape, ...]:
        return tuple(Shape(int(h), int(w)) for h, w in self.table.values())


class AreaPolicy(ShapePolicy):
    """Table-free: keep the native ratio at a target token count.

    Both sides are rounded to a multiple of ``align`` (which must itself be a
    multiple of the patch size, so every realized shape is patchable), and the
    ratio is clamped to ``max_ratio`` so a panorama cannot degenerate into a
    one-patch-tall strip. The realized shape set is not enumerated in advance:
    it is whatever the corpus produces, which is what distinguishes this from a
    bucket table with extra steps.
    """

    def __init__(self, tokens: int, patch: int, align: int = 32, max_ratio: float = 4.0):
        if tokens <= 0:
            raise ValueError(f"target token count must be positive, got {tokens}")
        if align <= 0 or align % patch:
            raise ValueError(f"align {align} must be a positive multiple of patch {patch}")
        if max_ratio < 1.0:
            raise ValueError(f"max_ratio must be >= 1, got {max_ratio}")
        self.tokens = tokens
        self.patch = patch
        self.align = align
        self.max_ratio = max_ratio
        self._area = tokens * patch * patch

    def target(self, height: int, width: int) -> Shape:
        ratio = height / width
        ratio = min(max(ratio, 1.0 / self.max_ratio), self.max_ratio)
        # h * w = area and h / w = ratio
        raw_w = sqrt(self._area / ratio)
        raw_h = raw_w * ratio
        return Shape(self._round(raw_h), self._round(raw_w))

    def _round(self, value: float) -> int:
        return max(self.align, int(round(value / self.align)) * self.align)


def build_shape_policy(
    policy: str, image_size: int, patch: int, bucket_table: str, area_align: int, max_ratio: float
) -> ShapePolicy:
    """Construct the policy named by ``data.shape_policy``."""
    if policy == "fixed":
        return FixedSquarePolicy(image_size)
    if policy == "bucket":
        return BucketPolicy(bucket_table)
    if policy == "area":
        square = Shape(image_size, image_size)
        return AreaPolicy(square.tokens(patch), patch, align=area_align, max_ratio=max_ratio)
    raise ValueError(f"unknown data.shape_policy '{policy}' (fixed | bucket | area)")
