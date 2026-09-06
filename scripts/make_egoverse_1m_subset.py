#!/usr/bin/env python3
"""Create an exact, task-stratified 1M subset of the EgoVerse train JSONL.

The source manifest is scanned twice and is never loaded into memory.  The
first pass counts rows per task.  The second pass uses a deterministic affine
permutation inside every task to select its proportional quota.  This keeps
the output exact, reproducible, and representative while guaranteeing that
every source task is retained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path


ALGORITHM_VERSION = "task_proportional_affine_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-jsonl",
        type=Path,
        default=Path(
            "jsons_egoverse_human_egocentric_all_tasks_12k/"
            "train_egoverse_human_egocentric_all_tasks_12k.jsonl"
        ),
        help="Existing full EgoVerse training JSONL.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("jsons_egoverse_human_egocentric_all_tasks_1m"),
    )
    parser.add_argument(
        "--output-stem",
        default="train_egoverse_human_egocentric_all_tasks_1m",
    )
    parser.add_argument("--count", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument(
        "--source-config",
        type=Path,
        default=Path("configs/flux.yaml"),
        help="Flux YAML to copy while changing only run/data identity fields.",
    )
    parser.add_argument(
        "--output-config",
        type=Path,
        default=Path("configs/flux_egoverse_1m.yaml"),
    )
    parser.add_argument(
        "--task-name",
        default="egoverse_flux2_klein4b_qformer_1m",
    )
    parser.add_argument(
        "--skip-config",
        action="store_true",
        help="Generate only JSONL/JSON/summary and do not create a YAML copy.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing generated outputs.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1_000_000,
        help="Print progress after this many source rows; zero disables it.",
    )
    return parser.parse_args()


def parse_row(line: str, *, path: Path, line_number: int) -> tuple[dict, str]:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Invalid JSON at {path}:{line_number}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"Expected a JSON object at {path}:{line_number}."
        )
    image = payload.get("image")
    if not isinstance(image, str) or not image.strip():
        raise RuntimeError(
            f"Missing non-empty image path at {path}:{line_number}."
        )
    return payload, image.strip()


def task_from_image_path(image_path: str) -> str:
    parts = [part for part in image_path.replace("\\", "/").split("/") if part]
    split_indices = [
        index for index, part in enumerate(parts) if part.lower() == "train"
    ]
    if not split_indices:
        raise RuntimeError(
            "Cannot infer task: expected .../train/<task>/<episode>/<frame>, "
            f"got {image_path!r}."
        )
    split_index = split_indices[-1]
    if split_index + 1 >= len(parts):
        raise RuntimeError(f"Missing task after train/ in {image_path!r}.")
    return parts[split_index + 1]


def scan_source(
    source: Path,
    *,
    progress_every: int,
) -> tuple[Counter[str], int, str]:
    counts: Counter[str] = Counter()
    source_hash = hashlib.sha256()
    total = 0
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            _, image = parse_row(line, path=source, line_number=line_number)
            counts[task_from_image_path(image)] += 1
            source_hash.update(line.encode("utf-8"))
            total += 1
            if progress_every > 0 and total % progress_every == 0:
                print(f"COUNT_PASS rows={total} tasks={len(counts)}", flush=True)
    return counts, total, source_hash.hexdigest()


def proportional_quotas(
    counts: Counter[str],
    target_count: int,
) -> dict[str, int]:
    if not counts:
        raise ValueError("The source manifest has no rows.")
    if target_count < len(counts):
        raise ValueError(
            f"--count={target_count} cannot retain all {len(counts)} tasks."
        )
    total = sum(counts.values())
    if target_count > total:
        raise ValueError(
            f"--count={target_count} exceeds source rows={total}."
        )

    # Reserve one row for every task, then apportion the remaining capacity
    # with Hamilton's largest-remainder method.
    remaining = target_count - len(counts)
    capacities = {task: count - 1 for task, count in counts.items()}
    capacity_total = sum(capacities.values())
    if capacity_total == 0:
        return {task: 1 for task in counts}

    ideals = {
        task: remaining * capacity / capacity_total
        for task, capacity in capacities.items()
    }
    extras = {task: int(math.floor(value)) for task, value in ideals.items()}
    left = remaining - sum(extras.values())
    remainder_order = sorted(
        counts,
        key=lambda task: (-(ideals[task] - extras[task]), task),
    )
    for task in remainder_order:
        if left == 0:
            break
        if extras[task] < capacities[task]:
            extras[task] += 1
            left -= 1
    if left != 0:
        raise RuntimeError(f"Failed to apportion {left} subset rows.")

    quotas = {task: 1 + extras[task] for task in counts}
    if sum(quotas.values()) != target_count:
        raise RuntimeError("Internal quota total mismatch.")
    if any(quotas[task] > counts[task] for task in counts):
        raise RuntimeError("A task quota exceeds its source row count.")
    return quotas


def affine_parameters(task: str, count: int, seed: int) -> tuple[int, int]:
    if count <= 1:
        return 0, 0
    digest = hashlib.sha256(f"{seed}:{task}".encode("utf-8")).digest()
    multiplier = int.from_bytes(digest[:8], "big") % count
    if multiplier == 0:
        multiplier = 1
    while math.gcd(multiplier, count) != 1:
        multiplier += 1
        if multiplier >= count:
            multiplier = 1
    offset = int.from_bytes(digest[8:16], "big") % count
    return multiplier, offset


def write_subset(
    source: Path,
    temporary_output: Path,
    *,
    counts: Counter[str],
    quotas: dict[str, int],
    seed: int,
    progress_every: int,
) -> tuple[Counter[str], int, str]:
    affine = {
        task: affine_parameters(task, count, seed)
        for task, count in counts.items()
    }
    seen: Counter[str] = Counter()
    selected: Counter[str] = Counter()
    output_hash = hashlib.sha256()
    processed = 0
    written = 0

    with source.open("r", encoding="utf-8") as source_file, temporary_output.open(
        "w", encoding="utf-8", newline="\n"
    ) as output_file:
        for line_number, line in enumerate(source_file, start=1):
            if not line.strip():
                continue
            payload, image = parse_row(line, path=source, line_number=line_number)
            task = task_from_image_path(image)
            ordinal = seen[task]
            seen[task] += 1
            multiplier, offset = affine[task]
            permuted_ordinal = (
                0
                if counts[task] == 1
                else (multiplier * ordinal + offset) % counts[task]
            )
            if permuted_ordinal < quotas[task]:
                encoded = (
                    json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                ).encode("utf-8")
                output_file.write(encoded.decode("utf-8"))
                output_hash.update(encoded)
                selected[task] += 1
                written += 1
            processed += 1
            if progress_every > 0 and processed % progress_every == 0:
                print(
                    f"WRITE_PASS rows={processed} selected={written}",
                    flush=True,
                )

        output_file.flush()
        os.fsync(output_file.fileno())

    if seen != counts:
        raise RuntimeError("Source row counts changed between the two passes.")
    if dict(selected) != quotas:
        mismatches = {
            task: {"expected": quotas[task], "actual": selected[task]}
            for task in quotas
            if selected[task] != quotas[task]
        }
        raise RuntimeError(f"Selected task quotas differ: {mismatches}.")
    return selected, written, output_hash.hexdigest()


def write_text_atomic(path: Path, text: str, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}; pass --overwrite.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_config_text(source: Path, *, wrapper: Path, task_name: str) -> str:
    lines = source.read_text(encoding="utf-8").splitlines(keepends=True)
    replacements = {
        "resume": "false",
        "resume_path": json.dumps(""),
        "task_name": json.dumps(task_name),
        "train_json_path": json.dumps(str(wrapper.resolve())),
    }
    matched = Counter()
    output = []
    for line in lines:
        stripped = line.lstrip()
        indent = line[: len(line) - len(stripped)]
        key = stripped.split(":", 1)[0] if ":" in stripped else ""
        if key in replacements:
            output.append(f"{indent}{key}: {replacements[key]}\n")
            matched[key] += 1
        else:
            output.append(line)
    unexpected = {
        key: matched[key]
        for key in replacements
        if matched[key] != 1
    }
    if unexpected:
        raise RuntimeError(
            f"Could not safely rewrite config fields in {source}: {unexpected}."
        )
    return "".join(output)


def main() -> None:
    args = parse_args()
    if args.count <= 0:
        raise ValueError("--count must be positive.")
    if args.progress_every < 0:
        raise ValueError("--progress-every must be non-negative.")
    source = args.source_jsonl.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Source JSONL does not exist: {source}")

    output_dir = args.output_dir.resolve()
    jsonl_path = output_dir / f"{args.output_stem}.jsonl"
    wrapper_path = output_dir / f"{args.output_stem}.json"
    summary_path = output_dir / f"{args.output_stem}_summary.json"
    destinations = [jsonl_path, wrapper_path, summary_path]
    if not args.skip_config:
        destinations.append(args.output_config.resolve())
    existing = [path for path in destinations if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing outputs: "
            + ", ".join(str(path) for path in existing)
            + ". Pass --overwrite only after verifying the targets."
        )

    print(f"SOURCE={source}")
    counts, source_rows, source_sha256 = scan_source(
        source,
        progress_every=args.progress_every,
    )
    quotas = proportional_quotas(counts, args.count)
    print(
        f"SOURCE_PASS rows={source_rows} tasks={len(counts)} "
        f"sha256={source_sha256}"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_jsonl = jsonl_path.with_name(f".{jsonl_path.name}.tmp-{os.getpid()}")
    try:
        selected, selected_rows, output_sha256 = write_subset(
            source,
            temporary_jsonl,
            counts=counts,
            quotas=quotas,
            seed=args.seed,
            progress_every=args.progress_every,
        )
        if selected_rows != args.count:
            raise RuntimeError(
                f"Expected {args.count} rows, wrote {selected_rows}."
            )
        temporary_jsonl.replace(jsonl_path)
    finally:
        if temporary_jsonl.exists():
            temporary_jsonl.unlink()

    wrapper = {"datasets": [str(jsonl_path.resolve())], "ratios": [1.0]}
    summary = {
        "algorithm": ALGORITHM_VERSION,
        "seed": args.seed,
        "source_jsonl": str(source),
        "source_rows": source_rows,
        "source_sha256": source_sha256,
        "selected_jsonl": str(jsonl_path.resolve()),
        "selected_rows": selected_rows,
        "selected_sha256": output_sha256,
        "task_count": len(counts),
        "source_task_counts": dict(sorted(counts.items())),
        "selected_task_counts": dict(sorted(selected.items())),
    }
    write_text_atomic(
        wrapper_path,
        json.dumps(wrapper, ensure_ascii=False, indent=4) + "\n",
        overwrite=args.overwrite,
    )
    write_text_atomic(
        summary_path,
        json.dumps(summary, ensure_ascii=False, indent=4) + "\n",
        overwrite=args.overwrite,
    )

    if not args.skip_config:
        source_config = args.source_config.resolve()
        if not source_config.is_file():
            raise FileNotFoundError(f"Source config does not exist: {source_config}")
        config_text = build_config_text(
            source_config,
            wrapper=wrapper_path,
            task_name=args.task_name,
        )
        write_text_atomic(
            args.output_config.resolve(),
            config_text,
            overwrite=args.overwrite,
        )

    print(f"EGOVERSE_1M_SUBSET_PASS rows={selected_rows} tasks={len(selected)}")
    print(f"JSONL={jsonl_path}")
    print(f"JSON={wrapper_path}")
    print(f"SUMMARY={summary_path}")
    if not args.skip_config:
        print(f"CONFIG={args.output_config.resolve()}")


if __name__ == "__main__":
    main()
