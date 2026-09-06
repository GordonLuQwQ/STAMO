"""Portable sampled fingerprints for large model files and checkpoint trees.

The fingerprints deliberately hash only fixed-size windows from each file.
They are intended to identify a configured model checkpoint without reading
multi-gigabyte weight shards in full; they are not a replacement for a full
cryptographic checksum when every byte must be authenticated.
"""

from __future__ import annotations

import hashlib
import operator
import os
import unicodedata
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Iterable


_FILE_DOMAIN = b"stamo.sampled-file-fingerprint.v1"
_TREE_DOMAIN = b"stamo.sampled-tree-fingerprint.v1"


def _sample_size(value: int) -> int:
    if isinstance(value, bool):
        raise TypeError("sample_bytes must be a positive integer, not bool.")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError("sample_bytes must be a positive integer.") from exc
    if result <= 0:
        raise ValueError("sample_bytes must be positive.")
    return int(result)


def _normalized_relative_path(value: os.PathLike[str] | str) -> str:
    raw = os.fsdecode(os.fspath(value)).replace("\\", "/")
    if "\x00" in raw:
        raise ValueError("A fingerprint path cannot contain a NUL byte.")
    if PurePosixPath(raw).is_absolute() or PureWindowsPath(raw).drive:
        raise ValueError(f"Fingerprint paths must be relative, got {raw!r}.")

    relative = PurePosixPath(raw)
    if not relative.parts or any(part == ".." for part in relative.parts):
        raise ValueError(
            f"Fingerprint paths must stay below their root, got {raw!r}."
        )
    return unicodedata.normalize("NFC", relative.as_posix())


def _frame(digest, label: bytes, payload: bytes) -> None:
    """Add an unambiguous length-prefixed field to a running digest."""

    digest.update(len(label).to_bytes(2, "big"))
    digest.update(label)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def _file_record(
    digest,
    path: Path,
    logical_path: str,
    sample_bytes: int,
) -> None:
    try:
        before = path.stat()
    except FileNotFoundError:
        raise FileNotFoundError(f"Fingerprint input file does not exist: {path}")
    if not path.is_file():
        raise IsADirectoryError(f"Fingerprint input is not a regular file: {path}")

    file_size = int(before.st_size)
    window_size = min(sample_bytes, file_size)
    offsets = (
        ("head", 0),
        ("middle", max(0, (file_size - window_size) // 2)),
        ("tail", max(0, file_size - window_size)),
    )

    _frame(digest, b"path", logical_path.encode("utf-8"))
    _frame(digest, b"size", file_size.to_bytes(16, "big"))
    with path.open("rb") as handle:
        for region, offset in offsets:
            handle.seek(offset)
            sample = handle.read(window_size)
            if len(sample) != window_size:
                raise OSError(
                    "Fingerprint input changed or became unreadable while sampling: "
                    f"{path} at offset {offset}."
                )
            _frame(digest, b"region", region.encode("ascii"))
            _frame(digest, b"offset", offset.to_bytes(16, "big"))
            _frame(digest, b"sample", sample)

    after = path.stat()
    if (
        int(after.st_size) != file_size
        or int(after.st_mtime_ns) != int(before.st_mtime_ns)
    ):
        raise OSError(f"Fingerprint input changed while it was sampled: {path}")


def sampled_file_fingerprint(
    path: os.PathLike[str] | str,
    sample_bytes: int = 65_536,
) -> str:
    """Return a SHA-256 fingerprint of one file's name, size, and samples.

    The logical path is the normalized file name rather than its absolute
    location, so moving an unchanged checkpoint directory does not alter the
    result. Renaming the file does alter it.
    """

    sample_bytes = _sample_size(sample_bytes)
    input_path = Path(path)
    logical_path = _normalized_relative_path(input_path.name)
    digest = hashlib.sha256()
    _frame(digest, b"domain", _FILE_DOMAIN)
    _file_record(digest, input_path, logical_path, sample_bytes)
    return digest.hexdigest()


def sampled_tree_fingerprint(
    root: os.PathLike[str] | str,
    relative_paths: Iterable[os.PathLike[str] | str] | None = None,
    sample_bytes: int = 65_536,
) -> str:
    """Fingerprint selected files below ``root`` in stable relative-path order.

    Passing ``relative_paths`` is recommended for model checkpoints because it
    makes missing required files an error and excludes unrelated cache files.
    When it is ``None``, all regular files below ``root`` are traversed
    recursively. Both modes hash normalized POSIX-style root-relative paths.
    """

    sample_bytes = _sample_size(sample_bytes)
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Fingerprint root does not exist: {root_path}")
    if not root_path.is_dir():
        raise NotADirectoryError(f"Fingerprint root is not a directory: {root_path}")

    if relative_paths is None:
        requested = [
            path.relative_to(root_path)
            for path in root_path.rglob("*")
            if path.is_file()
        ]
    elif isinstance(relative_paths, (str, os.PathLike)):
        requested = [relative_paths]
    else:
        requested = list(relative_paths)

    entries: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for requested_path in requested:
        logical_path = _normalized_relative_path(requested_path)
        if logical_path in seen:
            raise ValueError(
                f"Duplicate normalized fingerprint path: {logical_path!r}."
            )
        seen.add(logical_path)
        physical_path = root_path.joinpath(*PurePosixPath(logical_path).parts)
        if not physical_path.exists():
            raise FileNotFoundError(
                "Required fingerprint input file does not exist: "
                f"{physical_path}"
            )
        if not physical_path.is_file():
            raise IsADirectoryError(
                f"Fingerprint input is not a regular file: {physical_path}"
            )
        entries.append((logical_path, physical_path))

    entries.sort(key=lambda item: item[0])
    digest = hashlib.sha256()
    _frame(digest, b"domain", _TREE_DOMAIN)
    _frame(digest, b"file-count", len(entries).to_bytes(8, "big"))
    for logical_path, physical_path in entries:
        _file_record(
            digest,
            physical_path,
            logical_path,
            sample_bytes,
        )
    return digest.hexdigest()


__all__ = ["sampled_file_fingerprint", "sampled_tree_fingerprint"]
