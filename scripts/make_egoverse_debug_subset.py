#!/usr/bin/env python3
"""Create a deterministic, lightweight EgoVerse training subset.

The output JSONL references the original JPEG files; images are not copied.
Reservoir sampling gives an exact uniform sample without loading the source
manifest into memory.  This is intended for code/debug overfitting runs, not
as a replacement for the production training distribution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path


DEFAULT_SOURCES = (
    Path(
        "jsons_egoverse_human_egocentric_all_tasks_1m/"
        "train_egoverse_human_egocentric_all_tasks_1m.jsonl"
    ),
    Path(
        "jsons_egoverse_human_egocentric_all_tasks_12k/"
        "train_egoverse_human_egocentric_all_tasks_12k.jsonl"
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-jsonl",
        type=Path,
        default=None,
        help="Source training JSONL. By default, use the local 1M source, then the 12k-per-task source.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("jsons_egoverse_debug_1k"),
    )
    parser.add_argument("--output-stem", default="train_egoverse_debug_1k")
    parser.add_argument("--count", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument(
        "--verify-images",
        action="store_true",
        help="Require every selected image path to exist. Use this on the training machine.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing generated subset.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1_000_000,
        help="Report source rows scanned at this interval; zero disables progress output.",
    )
    return parser.parse_args()


def find_source(explicit: Path | None) -> Path:
    if explicit is not None:
        source = explicit.expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Source JSONL does not exist: {source}")
        return source
    for candidate in DEFAULT_SOURCES:
        source = candidate.resolve()
        if source.is_file():
            return source
    raise FileNotFoundError(
        "No default EgoVerse training JSONL was found. Pass --source-jsonl explicitly."
    )


def task_from_image_path(image_path: str) -> str:
    parts = [part for part in image_path.replace("\\", "/").split("/") if part]
    split_indices = [index for index, part in enumerate(parts) if part.lower() == "train"]
    if split_indices and split_indices[-1] + 1 < len(parts):
        return parts[split_indices[-1] + 1]
    return "unknown"


def reservoir_sample(source: Path, count: int, seed: int, progress_every: int):
    rng = random.Random(seed)
    reservoir: list[tuple[int, bytes]] = []
    source_hash = hashlib.sha256()
    row_count = 0

    with source.open("rb") as handle:
        for physical_line, raw_line in enumerate(handle, start=1):
            source_hash.update(raw_line)
            if not raw_line.strip():
                continue
            row_count += 1
            record = (physical_line, raw_line.rstrip(b"\r\n"))
            if len(reservoir) < count:
                reservoir.append(record)
            else:
                replacement = rng.randrange(row_count)
                if replacement < count:
                    reservoir[replacement] = record
            if progress_every > 0 and row_count % progress_every == 0:
                print(f"SUBSET_SCAN rows={row_count}", flush=True)

    if row_count < count:
        raise ValueError(
            f"Requested {count} rows, but source contains only {row_count} non-empty rows."
        )
    reservoir.sort(key=lambda item: item[0])
    return reservoir, row_count, source_hash.hexdigest()


def validate_records(records, source: Path, verify_images: bool):
    validated = []
    image_paths = []
    task_counts: Counter[str] = Counter()
    for physical_line, raw_line in records:
        try:
            payload = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid JSON at {source}:{physical_line}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Expected a JSON object at {source}:{physical_line}")
        image_path = payload.get("image")
        if not isinstance(image_path, str) or not image_path.strip():
            raise ValueError(f"Missing image path at {source}:{physical_line}")
        image_path = image_path.strip()
        resolved = Path(image_path).expanduser()
        if not resolved.is_absolute():
            resolved = source.parent / resolved
        if verify_images and not resolved.is_file():
            raise FileNotFoundError(
                f"Selected image does not exist ({source}:{physical_line}): {resolved}"
            )
        task_counts[task_from_image_path(image_path)] += 1
        image_paths.append(image_path)
        validated.append((physical_line, payload))
    return validated, image_paths, task_counts


def write_atomic(path: Path, payload: bytes, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}; pass --overwrite if intentional.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    args = parse_args()
    if args.count <= 0:
        raise ValueError("--count must be positive.")
    if args.progress_every < 0:
        raise ValueError("--progress-every must be non-negative.")

    source = find_source(args.source_jsonl)
    output_dir = args.output_dir.expanduser().resolve()
    jsonl_path = output_dir / f"{args.output_stem}.jsonl"
    wrapper_path = output_dir / f"{args.output_stem}.json"
    summary_path = output_dir / f"{args.output_stem}_summary.json"

    destinations = (jsonl_path, wrapper_path, summary_path)
    existing = [str(path) for path in destinations if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing outputs: " + ", ".join(existing)
        )

    print(f"SOURCE={source}", flush=True)
    sampled, source_rows, source_sha256 = reservoir_sample(
        source,
        count=args.count,
        seed=args.seed,
        progress_every=args.progress_every,
    )
    records, image_paths, task_counts = validate_records(
        sampled,
        source,
        verify_images=args.verify_images,
    )

    encoded_rows = [
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        + b"\n"
        for _, payload in records
    ]
    jsonl_bytes = b"".join(encoded_rows)
    wrapper = {"datasets": [jsonl_path.name], "ratios": [1.0]}
    summary = {
        "algorithm": "uniform_reservoir_v1",
        "seed": args.seed,
        "source_jsonl": str(source),
        "source_rows": source_rows,
        "source_sha256": source_sha256,
        "selected_rows": len(records),
        "selected_unique_image_paths": len(set(image_paths)),
        "selected_source_line_numbers": [line_number for line_number, _ in records],
        "selected_task_count": len(task_counts),
        "selected_task_counts": dict(sorted(task_counts.items())),
        "selected_sha256": hashlib.sha256(jsonl_bytes).hexdigest(),
        "images_copied": False,
        "images_verified": bool(args.verify_images),
    }

    write_atomic(jsonl_path, jsonl_bytes, args.overwrite)
    write_atomic(
        wrapper_path,
        (json.dumps(wrapper, ensure_ascii=False, indent=4) + "\n").encode("utf-8"),
        args.overwrite,
    )
    write_atomic(
        summary_path,
        (json.dumps(summary, ensure_ascii=False, indent=4) + "\n").encode("utf-8"),
        args.overwrite,
    )

    print(
        f"EGOVERSE_DEBUG_SUBSET_PASS rows={len(records)} "
        f"unique_images={len(set(image_paths))} tasks={len(task_counts)}",
        flush=True,
    )
    print(f"JSONL={jsonl_path}")
    print(f"JSON={wrapper_path}")
    print(f"SUMMARY={summary_path}")


if __name__ == "__main__":
    main()
