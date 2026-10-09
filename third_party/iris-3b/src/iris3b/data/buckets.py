"""Aspect-ratio bucket tables and nearest-bucket lookup.

Keys are the bucket height/width ratio as a string; values are `[height, width]`
in pixels. The tables drive multi-aspect batching.
"""

SHARED_21_512: dict[str, list[float]] = {
    str(height / width): [float(height), float(width)]
    for height, width in (
        (256, 1024),
        (288, 928),
        (288, 896),
        (320, 832),
        (320, 800),
        (352, 736),
        (384, 672),
        (416, 640),
        (448, 576),
        (480, 544),
        (512, 512),
        (544, 480),
        (576, 448),
        (640, 416),
        (672, 384),
        (736, 352),
        (800, 320),
        (832, 320),
        (896, 288),
        (928, 288),
        (1024, 256),
    )
}

SHARED_21_1024: dict[str, list[float]] = {
    ratio: [height * 2, width * 2] for ratio, (height, width) in SHARED_21_512.items()
}

TRAIN_BUCKETS: dict[str, dict[str, list[float]]] = {
    "shared21-512": SHARED_21_512,
    "shared21-1024": SHARED_21_1024,
}


def closest_ratio(height: float, width: float, table: dict[str, list[float]]) -> tuple[str, list[int]]:
    """Return the (ratio_key, [bucket_h, bucket_w]) entry nearest to height/width."""
    ratio = height / width
    key = min(table, key=lambda k: abs(float(k) - ratio))
    bh, bw = table[key]
    return key, [int(bh), int(bw)]
