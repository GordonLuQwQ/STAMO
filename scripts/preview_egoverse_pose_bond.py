#!/usr/bin/env python3
"""Save source / solid joint dots / solid-dot overlay for one sidecar row."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


LEFT_COLOR = (0, 255, 96)
RIGHT_COLOR = (255, 24, 96)


def resolve_jsonl(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.suffix.lower() == ".jsonl":
        return path
    with path.open("r", encoding="utf-8") as stream:
        wrapper = json.load(stream)
    datasets = wrapper.get("datasets", [])
    if len(datasets) != 1:
        raise ValueError(f"Expected one dataset in {path}, got {len(datasets)}")
    result = Path(datasets[0]).expanduser()
    return (result if result.is_absolute() else path.parent / result).resolve()


def record_at(jsonl_path: Path, row: int):
    current = -1
    with jsonl_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            current += 1
            if current == row:
                return json.loads(line)
    raise IndexError(f"Row {row} is outside {jsonl_path} (rows={current + 1})")


def draw_solid_joints(
    image: Image.Image,
    pose_uvz: np.ndarray,
    radius: int,
) -> None:
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for hand_index, color in enumerate((LEFT_COLOR, RIGHT_COLOR)):
        for u_norm, v_norm, z_camera in pose_uvz[hand_index]:
            if not np.isfinite((u_norm, v_norm, z_camera)).all() or z_camera <= 0:
                continue
            x = float(u_norm) * width - 0.5
            y = float(v_norm) * height - 0.5
            if not (0.0 <= x < width and 0.0 <= y < height):
                continue
            draw.ellipse(
                (
                    int(round(x)) - radius,
                    int(round(y)) - radius,
                    int(round(x)) + radius,
                    int(round(y)) + radius,
                ),
                fill=color,
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--sidecar", required=True)
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--radius", type=int, default=2)
    args = parser.parse_args()

    if args.size <= 0:
        raise ValueError("--size must be positive")
    if args.radius < 0:
        raise ValueError("--radius cannot be negative")

    jsonl_path = resolve_jsonl(Path(args.manifest))
    record = record_at(jsonl_path, args.row)
    image_path = Path(record["image"]).expanduser()
    if not image_path.is_absolute():
        image_path = jsonl_path.parent / image_path
    pose = np.asarray(
        np.load(args.sidecar, mmap_mode="r")[args.row],
        dtype=np.float32,
    )
    if pose.shape != (2, 21, 3):
        raise ValueError(f"Expected sidecar row [2,21,3], got {pose.shape}")

    with Image.open(image_path) as source:
        resampling_enum = getattr(Image, "Resampling", Image)
        image = source.convert("RGB").resize(
            (args.size, args.size),
            resample=resampling_enum.BICUBIC,
        )
    point_map = Image.new("RGB", image.size, color=(0, 0, 0))
    overlay = image.copy()
    draw_solid_joints(point_map, pose, args.radius)
    draw_solid_joints(overlay, pose, args.radius)

    strip = Image.new("RGB", (args.size * 3, args.size))
    strip.paste(image, (0, 0))
    strip.paste(point_map, (args.size, 0))
    strip.paste(overlay, (args.size * 2, 0))
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    strip.save(output)
    print(f"POSE_BOND_PREVIEW image={image_path} row={args.row} output={output}")


if __name__ == "__main__":
    main()
