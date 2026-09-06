"""Compact, spawn-safe random access for image-path JSONL manifests."""

from __future__ import annotations

import bisect
import json
import mmap
import os
import struct
import sys
import time
from array import array
from collections.abc import Iterator, Sequence
from contextlib import contextmanager


_INDEX_MAGIC = b"STAMOIX1"
_INDEX_VERSION_SUFFIX = ".stamo-jsonl-index-v1"
_INDEX_HEADER = struct.Struct("<8sQQQ")
_OFFSET = struct.Struct("<Q")
_OFFSET_CHUNK_SIZE = 262_144
_PROGRESS_INTERVAL = 1_000_000


def _source_identity(path: str) -> tuple[int, int]:
    stat = os.stat(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image metadata is not a regular file: {path}")
    return int(stat.st_size), int(stat.st_mtime_ns)


def _index_path(metadata_path: str) -> str:
    return f"{metadata_path}{_INDEX_VERSION_SUFFIX}"


def _read_valid_header(metadata_path: str, index_path: str):
    try:
        source_size, source_mtime_ns = _source_identity(metadata_path)
        index_size = os.path.getsize(index_path)
        with open(index_path, "rb") as index_file:
            header = index_file.read(_INDEX_HEADER.size)
            if len(header) != _INDEX_HEADER.size:
                return None
            magic, indexed_size, indexed_mtime_ns, line_count = (
                _INDEX_HEADER.unpack(header)
            )
            expected_size = _INDEX_HEADER.size + (int(line_count) + 1) * 8
            if (
                magic != _INDEX_MAGIC
                or int(indexed_size) != source_size
                or int(indexed_mtime_ns) != source_mtime_ns
                or index_size != expected_size
                or int(line_count) <= 0
            ):
                return None
            first_offset = _OFFSET.unpack(index_file.read(8))[0]
            index_file.seek(-8, os.SEEK_END)
            final_offset = _OFFSET.unpack(index_file.read(8))[0]
            if first_offset != 0 or final_offset != source_size:
                return None
            return source_size, source_mtime_ns, int(line_count)
    except (FileNotFoundError, OSError, struct.error):
        return None


@contextmanager
def _exclusive_index_lock(lock_path: str):
    """Serialize sidecar creation across the eight rank processes."""
    lock_file = open(lock_path, "a+b")
    try:
        if os.name == "posix":
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        elif os.name == "nt":
            import msvcrt

            if os.path.getsize(lock_path) == 0:
                lock_file.write(b"\0")
                lock_file.flush()
            lock_file.seek(0)
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
        yield
    finally:
        try:
            if os.name == "posix":
                import fcntl

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            elif os.name == "nt":
                import msvcrt

                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
        finally:
            lock_file.close()


def _write_offset_chunk(stream, values) -> None:
    if not values:
        return
    offsets = array("Q", values)
    if offsets.itemsize != 8:
        raise RuntimeError("This platform does not provide 64-bit array('Q').")
    if sys.byteorder != "little":
        offsets.byteswap()
    offsets.tofile(stream)


def _validate_image_record(raw_line: bytes, metadata_path: str, line_number: int):
    if not raw_line.strip():
        raise ValueError(f"{metadata_path}:{line_number} is empty.")
    try:
        item = json.loads(raw_line)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"{metadata_path}:{line_number} is not valid UTF-8 JSON."
        ) from exc
    if not isinstance(item, dict):
        raise ValueError(
            f"{metadata_path}:{line_number} must contain a JSON object."
        )
    image_path = item.get("image")
    if not isinstance(image_path, str) or not image_path.strip():
        raise ValueError(
            f"{metadata_path}:{line_number} has no non-empty 'image' path."
        )


def _build_index(metadata_path: str, index_path: str):
    source_size, source_mtime_ns = _source_identity(metadata_path)
    temporary_path = (
        f"{index_path}.tmp.{os.getpid()}.{time.time_ns()}"
    )
    print(
        "STAMO_METADATA_INDEX_BUILD_BEGIN "
        f"source={metadata_path} bytes={source_size}",
        flush=True,
    )
    started = time.perf_counter()
    line_count = 0
    position = 0
    offsets = [0]
    try:
        with open(metadata_path, "rb", buffering=8 * 1024 * 1024) as source:
            with open(temporary_path, "w+b", buffering=8 * 1024 * 1024) as target:
                target.write(b"\0" * _INDEX_HEADER.size)
                for line_count, raw_line in enumerate(source, start=1):
                    _validate_image_record(raw_line, metadata_path, line_count)
                    position += len(raw_line)
                    offsets.append(position)
                    if len(offsets) >= _OFFSET_CHUNK_SIZE:
                        _write_offset_chunk(target, offsets)
                        offsets.clear()
                    if line_count % _PROGRESS_INTERVAL == 0:
                        print(
                            "STAMO_METADATA_INDEX_BUILD_PROGRESS "
                            f"source={metadata_path} lines={line_count} "
                            f"bytes={position}",
                            flush=True,
                        )
                _write_offset_chunk(target, offsets)
                if line_count <= 0:
                    raise ValueError(
                        f"No images were found in metadata file {metadata_path!r}."
                    )
                if position != source_size:
                    raise RuntimeError(
                        "Metadata size changed while indexing: "
                        f"read={position}, expected={source_size}."
                    )
                if _source_identity(metadata_path) != (
                    source_size,
                    source_mtime_ns,
                ):
                    raise RuntimeError(
                        f"Metadata changed while indexing: {metadata_path}"
                    )
                target.seek(0)
                target.write(
                    _INDEX_HEADER.pack(
                        _INDEX_MAGIC,
                        source_size,
                        source_mtime_ns,
                        line_count,
                    )
                )
                target.flush()
                os.fsync(target.fileno())
        os.replace(temporary_path, index_path)
    except BaseException:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
        raise

    elapsed = time.perf_counter() - started
    print(
        "STAMO_METADATA_INDEX_BUILD_PASS "
        f"source={metadata_path} lines={line_count} "
        f"index_bytes={os.path.getsize(index_path)} "
        f"elapsed_seconds={elapsed:.3f}",
        flush=True,
    )
    return source_size, source_mtime_ns, line_count


def ensure_jsonl_image_index(metadata_path) -> tuple[str, int]:
    """Return a valid offset sidecar, building it exactly once if needed."""
    metadata_path = os.path.abspath(os.path.expanduser(str(metadata_path)))
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(f"Image metadata does not exist: {metadata_path}")
    index_path = _index_path(metadata_path)
    header = _read_valid_header(metadata_path, index_path)
    if header is None:
        lock_path = f"{index_path}.lock"
        with _exclusive_index_lock(lock_path):
            header = _read_valid_header(metadata_path, index_path)
            if header is None:
                header = _build_index(metadata_path, index_path)
    return index_path, int(header[2])


class JsonlImagePathIndex(Sequence):
    """A tiny pickleable handle to a read-only JSONL and uint64 offset map."""

    def __init__(self, metadata_path):
        self.metadata_path = os.path.abspath(
            os.path.expanduser(str(metadata_path))
        )
        self.metadata_dir = os.path.dirname(self.metadata_path)
        self.index_path, self.length = ensure_jsonl_image_index(
            self.metadata_path
        )
        self._source_file = None
        self._source_map = None
        self._index_file = None
        self._index_map = None

    def __len__(self):
        return self.length

    def _ensure_open(self):
        if self._source_map is not None and self._index_map is not None:
            return
        header = _read_valid_header(self.metadata_path, self.index_path)
        if header is None or int(header[2]) != self.length:
            raise RuntimeError(
                "Image metadata index became stale after dataset creation: "
                f"{self.index_path}"
            )
        self._source_file = open(self.metadata_path, "rb")
        self._index_file = open(self.index_path, "rb")
        try:
            self._source_map = mmap.mmap(
                self._source_file.fileno(),
                length=0,
                access=mmap.ACCESS_READ,
            )
            self._index_map = mmap.mmap(
                self._index_file.fileno(),
                length=0,
                access=mmap.ACCESS_READ,
            )
        except BaseException:
            self.close()
            raise

    def __getitem__(self, index):
        index = int(index)
        if index < 0:
            index += self.length
        if index < 0 or index >= self.length:
            raise IndexError(index)
        self._ensure_open()
        offset_position = _INDEX_HEADER.size + index * 8
        start = _OFFSET.unpack_from(self._index_map, offset_position)[0]
        end = _OFFSET.unpack_from(self._index_map, offset_position + 8)[0]
        raw_line = self._source_map[start:end]
        try:
            item = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Indexed metadata line {index + 1} became invalid in "
                f"{self.metadata_path}."
            ) from exc
        image_path = item.get("image") if isinstance(item, dict) else None
        if not isinstance(image_path, str) or not image_path.strip():
            raise RuntimeError(
                f"Indexed metadata line {index + 1} has no image path in "
                f"{self.metadata_path}."
            )
        image_path = os.path.expanduser(image_path.strip())
        if not os.path.isabs(image_path):
            image_path = os.path.join(self.metadata_dir, image_path)
        return os.path.abspath(image_path)

    def close(self):
        for name in ("_index_map", "_source_map"):
            handle = getattr(self, name, None)
            if handle is not None:
                handle.close()
                setattr(self, name, None)
        for name in ("_index_file", "_source_file"):
            handle = getattr(self, name, None)
            if handle is not None:
                handle.close()
                setattr(self, name, None)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.update(
            {
                "_source_file": None,
                "_source_map": None,
                "_index_file": None,
                "_index_map": None,
            }
        )
        return state

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class JsonlImagePathCollection(Sequence):
    """Concatenate mmap-backed manifests without materializing their paths."""

    def __init__(self):
        self.sources = []
        self.cumulative_sizes = []

    def add(self, metadata_path):
        source = JsonlImagePathIndex(metadata_path)
        self.sources.append(source)
        previous = self.cumulative_sizes[-1] if self.cumulative_sizes else 0
        self.cumulative_sizes.append(previous + len(source))

    def __len__(self):
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0

    def __getitem__(self, index):
        length = len(self)
        index = int(index)
        if index < 0:
            index += length
        if index < 0 or index >= length:
            raise IndexError(index)
        source_index = bisect.bisect_right(self.cumulative_sizes, index)
        previous = (
            self.cumulative_sizes[source_index - 1]
            if source_index > 0
            else 0
        )
        return self.sources[source_index][index - previous]

    def __iter__(self) -> Iterator[str]:
        for source in self.sources:
            for index in range(len(source)):
                yield source[index]

    def close(self):
        for source in self.sources:
            source.close()