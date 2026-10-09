"""Indexed random access over WebDataset-style tar shards.

A dataset directory holds ``wids-meta.json`` describing its shards::

    {"name": ..., "base_path": ..., "shardlist": [{"url": "shard-000.tar", "nsamples": 123}, ...]}

Relative shard URLs resolve against ``base_path`` when given, else against the
directory containing the descriptor. Samples are the members of a shard that
share a basename key (everything before the first dot of the final path
component); within a shard they are ordered by sorted key.
"""

import io
import json
import os
import tarfile
from bisect import bisect_right
from collections import OrderedDict
from collections.abc import Sequence
from itertools import accumulate
from typing import Any

_IMAGE_SUFFIXES = {"png", "jpg", "jpeg", "webp"}


class MissingShardError(RuntimeError):
    """A shard declared in ``wids-meta.json`` is not on disk.

    Distinct from an undecodable sample: substituting the next index would
    silently change which samples the run trains on, so it is never retried.
    """


def split_key(name: str) -> tuple[str, str] | None:
    """Split a member path into (sample key, extension incl. leading dot)."""
    dirpart, _, last = name.rpartition("/")
    stem, dot, ext = last.partition(".")
    if not dot or not stem:
        return None
    key = f"{dirpart}/{stem}" if dirpart else stem
    return key, f".{ext}"


def _decode(ext: str, data: bytes) -> Any:
    """Decode raw member bytes according to the final extension suffix."""
    suffix = ext.rsplit(".", 1)[-1]
    if suffix in _IMAGE_SUFFIXES:
        from PIL import Image

        return Image.open(io.BytesIO(data))
    if suffix == "json":
        return json.loads(data)
    if suffix in ("txt", "text"):
        return data.decode("utf-8")
    return data


class _ShardReader:
    """Random access to the key-grouped samples of one open tar file."""

    def __init__(self, path: str):
        self.path = path
        self.tar = tarfile.open(path)  # noqa: SIM115 - held open for random access, closed by LRU eviction
        groups: dict[str, list[tuple[str, tarfile.TarInfo]]] = {}
        for member in self.tar.getmembers():
            if not member.isfile():
                continue
            split = split_key(member.name)
            if split is None:
                continue
            key, ext = split
            groups.setdefault(key, []).append((ext, member))
        self.keys = sorted(groups)
        self.groups = groups

    def __len__(self) -> int:
        return len(self.keys)

    def sample(self, index: int, exts: set[str] | None = None) -> dict[str, Any]:
        """Decode the sample at ``index``, optionally restricted to given extensions."""
        key = self.keys[index]
        out: dict[str, Any] = {"__key__": key}
        for ext, member in self.groups[key]:
            if exts is not None and ext not in exts:
                continue
            stream = self.tar.extractfile(member)
            if stream is None:
                continue
            out[ext] = _decode(ext, stream.read())
        return out

    def close(self) -> None:
        self.tar.close()


class IndexedTarDataset:
    """Concatenated tar shards with global-index random access."""

    def __init__(self, dirs: Sequence[str], lru_size: int = 8):
        self.lru_size = lru_size
        self.shards: list[str] = []
        lengths: list[int] = []
        for d in dirs:
            d = os.path.expanduser(d)
            descriptor_path = os.path.join(d, "wids-meta.json")
            with open(descriptor_path) as f:
                meta = json.load(f)
            if not isinstance(meta, dict):
                raise ValueError(f"WIDS descriptor must be a mapping: {descriptor_path}")
            base = os.path.expanduser(meta.get("base_path", d))
            for entry in meta["shardlist"]:
                url = os.path.expanduser(entry["url"])
                if not os.path.isabs(url):
                    url = os.path.abspath(os.path.join(base, url))
                self.shards.append(url)
                lengths.append(int(entry["nsamples"]))
        self.cum_lengths = list(accumulate(lengths))
        self.total = self.cum_lengths[-1] if self.cum_lengths else 0
        self._open: OrderedDict[str, _ShardReader] = OrderedDict()

    def __len__(self) -> int:
        return self.total

    def __getstate__(self) -> dict:
        """Open tar handles never cross process boundaries."""
        state = self.__dict__.copy()
        state["_open"] = OrderedDict()
        return state

    def _reader(self, path: str) -> _ShardReader:
        reader = self._open.get(path)
        if reader is None:
            if not os.path.exists(path):
                raise MissingShardError(f"declared shard is absent: {path}")
            reader = _ShardReader(path)
            self._open[path] = reader
            while len(self._open) > self.lru_size:
                _, evicted = self._open.popitem(last=False)
                evicted.close()
        else:
            self._open.move_to_end(path)
        return reader

    def _locate(self, index: int) -> tuple[str, int]:
        """Map a global index to (shard path, index within shard)."""
        if not 0 <= index < self.total:
            raise IndexError(f"index {index} out of range for {self.total} samples")
        shard_idx = bisect_right(self.cum_lengths, index)
        inner = index - (self.cum_lengths[shard_idx - 1] if shard_idx else 0)
        return self.shards[shard_idx], inner

    def info(self, index: int) -> dict:
        """Parsed ``.json`` metadata of one sample; the image stays undecoded."""
        path, inner = self._locate(index)
        return self._reader(path).sample(inner, exts={".json"})[".json"]

    def __getitem__(self, index: int) -> dict[str, Any]:
        path, inner = self._locate(index)
        sample = self._reader(path).sample(inner)
        sample["__index__"] = index
        sample["__shard__"] = path
        sample["__shardindex__"] = inner
        return sample

    def close(self) -> None:
        while self._open:
            _, reader = self._open.popitem()
            reader.close()
