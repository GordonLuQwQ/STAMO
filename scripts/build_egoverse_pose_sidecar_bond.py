#!/usr/bin/env python3
"""Build a line-aligned EgoVerse hand-pose sidecar for ``*_bond`` training.

Run this once in the separate Zarr-compatible environment.  Training then
uses only a read-only numpy memmap and never imports Zarr in DataLoader workers.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np


def resolve_jsonl(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.suffix.lower() == ".jsonl":
        return path
    with path.open("r", encoding="utf-8") as stream:
        wrapper = json.load(stream)
    datasets = wrapper.get("datasets", [])
    if len(datasets) != 1:
        raise ValueError(
            f"Expected exactly one dataset in {path}, got {len(datasets)}"
        )
    dataset = Path(datasets[0]).expanduser()
    if not dataset.is_absolute():
        dataset = path.parent / dataset
    return dataset.resolve()


def install_zarr_v3_compatibility_patch() -> None:
    """Accept metadata fields emitted by newer Zarr v3 writers.

    The EgoVerse arrays use no storage transformers.  Some Python 3.10 Zarr
    builds reject the empty metadata field even though the underlying sharded
    array is otherwise readable.
    """

    try:
        from zarr.core.metadata.v3 import ArrayV3Metadata
    except (ImportError, ModuleNotFoundError):
        return
    if getattr(ArrayV3Metadata, "_stamo_bond_compat", False):
        return

    original = ArrayV3Metadata.from_dict

    def compatible_from_dict(cls, data):
        metadata = dict(data)
        transformers = metadata.pop("storage_transformers", None)
        if transformers not in (None, []):
            raise RuntimeError(
                f"Unsupported non-empty storage_transformers: {transformers!r}"
            )
        metadata.pop("dimension_names", None)
        return original(metadata)

    ArrayV3Metadata.from_dict = classmethod(compatible_from_dict)
    ArrayV3Metadata._stamo_bond_compat = True


class EpisodeCache:
    def __init__(self, raw_root: Path, capacity: int = 2) -> None:
        self.raw_root = raw_root
        self.capacity = max(1, int(capacity))
        self.cache = OrderedDict()

    def get(self, episode_id: str):
        if episode_id in self.cache:
            value = self.cache.pop(episode_id)
            self.cache[episode_id] = value
            return value
        value = self._load(episode_id)
        self.cache[episode_id] = value
        while len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
        return value

    def _load(self, episode_id: str):
        import zarr

        episode = self.raw_root / episode_id
        metadata_path = episode / "zarr.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Missing episode metadata: {metadata_path}")
        with metadata_path.open("r", encoding="utf-8") as stream:
            metadata = json.load(stream)
        attrs = metadata.get("attributes", {})
        intrinsics = np.asarray(
            attrs.get("intrinsics", {}).get("front_1"),
            dtype=np.float64,
        )
        if intrinsics.shape != (3, 4) or not np.isfinite(intrinsics).all():
            raise ValueError(f"Invalid front_1 intrinsics in {metadata_path}")
        image_shape = (
            attrs.get("features", {})
            .get("images.front_1", {})
            .get("shape")
        )
        if not isinstance(image_shape, list) or len(image_shape) < 2:
            raise ValueError(f"Missing images.front_1 shape in {metadata_path}")
        height, width = int(image_shape[0]), int(image_shape[1])
        source_start = int(attrs.get("source_frame_start", 0))

        group = zarr.open_group(str(episode), mode="r")
        # Full-array reads intentionally avoid the partial-decode issue seen in
        # older Python 3.10 Zarr builds with sharding_indexed + zstd arrays.
        left = np.asarray(group["left.obs_keypoints"][:], dtype=np.float64)
        right = np.asarray(group["right.obs_keypoints"][:], dtype=np.float64)
        head_pose = np.asarray(group["obs_head_pose"][:], dtype=np.float64)
        if left.ndim != 2 or left.shape[1] != 63 or right.shape != left.shape:
            raise ValueError(
                f"Unexpected hand-keypoint arrays for episode {episode_id}: "
                f"left={left.shape}, right={right.shape}"
            )
        if head_pose.ndim != 2 or head_pose.shape[1] != 7:
            raise ValueError(
                f"Unexpected head-pose array for episode {episode_id}: "
                f"head={head_pose.shape}"
            )
        if head_pose.shape[0] != left.shape[0]:
            raise ValueError(
                f"Frame-count mismatch for episode {episode_id}: "
                f"keypoints={left.shape[0]}, head_pose={head_pose.shape[0]}"
            )
        return {
            "points": np.stack(
                (left.reshape(-1, 21, 3), right.reshape(-1, 21, 3)),
                axis=1,
            ),
            "head_pose": head_pose,
            "intrinsics": intrinsics,
            "height": height,
            "width": width,
            "source_start": source_start,
        }


def quaternion_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Convert EgoVerse's ``[w,x,y,z]`` quaternion to a 3x3 rotation."""

    quaternion = np.asarray(quaternion, dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm <= np.finfo(np.float64).eps:
        raise ValueError(f"Invalid head quaternion: {quaternion!r}")
    w, x, y, z = quaternion / norm
    return np.asarray(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )


def project_pose(
    episode,
    source_frame: int,
    depth_min: float = 0.2,
) -> np.ndarray:
    local_frame = int(source_frame)
    points = episode["points"]
    if local_frame < 0 or local_frame >= points.shape[0]:
        raise IndexError(
            f"Frame {source_frame} maps to row {local_frame}, outside "
            f"[0,{points.shape[0]})"
        )
    # EgoVerse stores MANO keypoints in the SLAM/world frame.  obs_head_pose is
    # T_world_head=[xyz, quaternion_wxyz], and the egocentric head frame is the
    # front RGB camera frame.  For row-vector points this is exactly
    # p_camera = (p_world - t_world_head) @ R_world_head.
    xyz_world = np.asarray(points[local_frame], dtype=np.float64)
    head_pose = np.asarray(episode["head_pose"][local_frame], dtype=np.float64)
    if not np.isfinite(head_pose).all():
        raise ValueError(f"Non-finite head pose at local frame {local_frame}")
    rotation_world_head = quaternion_wxyz_to_matrix(head_pose[3:7])
    finite_world = np.isfinite(xyz_world).all(axis=-1)
    xyz_world_safe = np.where(finite_world[..., None], xyz_world, 0.0)
    xyz = (xyz_world_safe - head_pose[:3]) @ rotation_world_head
    homogeneous = np.concatenate(
        (xyz, np.ones((*xyz.shape[:-1], 1), dtype=np.float64)),
        axis=-1,
    )
    projected = homogeneous @ episode["intrinsics"].T
    z_camera = projected[..., 2]
    valid = (
        finite_world
        & np.isfinite(projected).all(axis=-1)
        & (z_camera > float(depth_min))
    )

    # [-1,-1,0] is an out-of-canvas sentinel, not an extra validity channel.
    # It prevents missing/behind-camera joints from becoming false top-left
    # Gaussians while preserving the required [2,21,3] sidecar contract.
    result = np.zeros((2, 21, 3), dtype=np.float32)
    result[..., :2] = -1.0
    u = np.zeros_like(z_camera)
    v = np.zeros_like(z_camera)
    np.divide(projected[..., 0], z_camera, out=u, where=valid)
    np.divide(projected[..., 1], z_camera, out=v, where=valid)
    # Pixel-centre normalization that exactly follows PIL/torchvision resize:
    # source pixel u maps to target pixel (u+0.5)*W_target/W_source - 0.5.
    # The model recovers that coordinate as u_norm*W_target - 0.5.
    u_norm = (u + 0.5) / max(1, int(episode["width"]))
    v_norm = (v + 0.5) / max(1, int(episode["height"]))

    result[..., 0][valid] = u_norm[valid]
    result[..., 1][valid] = v_norm[valid]
    result[..., 2][valid] = z_camera[valid]
    return result


def parse_image_record(raw_line: str, jsonl_path: Path, line_number: int) -> Path:
    try:
        record = json.loads(raw_line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON at {jsonl_path}:{line_number}") from exc
    image = record.get("image") if isinstance(record, dict) else None
    if not isinstance(image, str) or not image:
        raise ValueError(f"Missing image path at {jsonl_path}:{line_number}")
    image_path = Path(image).expanduser()
    if not image_path.is_absolute():
        image_path = jsonl_path.parent / image_path
    return image_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        required=True,
        help="Dataset wrapper .json or its single source .jsonl",
    )
    parser.add_argument(
        "--raw-root",
        default="/workspace/datasets/EgoVerse_12_tasks",
    )
    parser.add_argument("--output", required=True, help="Output .npy path")
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Write an all-zero row instead of stopping on missing raw pose",
    )
    parser.add_argument("--cache-episodes", type=int, default=2)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument(
        "--depth-min",
        type=float,
        default=0.2,
        help=(
            "Ignore joints at or behind this direct camera-space depth in "
            "metres; no inverse depth is used"
        ),
    )
    args = parser.parse_args()
    if args.depth_min <= 0:
        raise ValueError("--depth-min must be positive")

    install_zarr_v3_compatibility_patch()
    jsonl_path = resolve_jsonl(Path(args.manifest))
    raw_root = Path(args.raw_root).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if output_path.suffix.lower() != ".npy":
        raise ValueError("--output must end in .npy")
    if not jsonl_path.is_file():
        raise FileNotFoundError(jsonl_path)
    if not raw_root.is_dir():
        raise NotADirectoryError(raw_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing sidecar: {output_path}"
        )

    with jsonl_path.open("r", encoding="utf-8") as stream:
        row_count = sum(1 for line in stream if line.strip())
    if row_count <= 0:
        raise ValueError(f"Manifest is empty: {jsonl_path}")

    temporary_path = output_path.with_name(
        f".{output_path.stem}.tmp.{os.getpid()}.npy"
    )
    sidecar = np.lib.format.open_memmap(
        temporary_path,
        mode="w+",
        dtype=np.float16,
        shape=(row_count, 2, 21, 3),
    )
    cache = EpisodeCache(raw_root, args.cache_episodes)
    missing = []
    missing_count = 0
    started = time.perf_counter()
    output_row = 0
    try:
        with jsonl_path.open("r", encoding="utf-8") as stream:
            for line_number, raw_line in enumerate(stream, start=1):
                if not raw_line.strip():
                    continue
                image_path = parse_image_record(raw_line, jsonl_path, line_number)
                episode_id = image_path.parent.name
                try:
                    source_frame = int(image_path.stem)
                except ValueError as exc:
                    raise ValueError(
                        f"Image filename is not a numeric frame: {image_path}"
                    ) from exc
                try:
                    pose = project_pose(
                        cache.get(episode_id),
                        source_frame,
                        depth_min=args.depth_min,
                    )
                except (FileNotFoundError, KeyError, ValueError, IndexError) as exc:
                    if not args.allow_missing:
                        raise RuntimeError(
                            f"Cannot build pose for {image_path}"
                        ) from exc
                    pose = np.zeros((2, 21, 3), dtype=np.float32)
                    pose[..., :2] = -1.0
                    missing_count += 1
                    if len(missing) < 100:
                        missing.append(
                            {"image": str(image_path), "error": repr(exc)}
                        )
                sidecar[output_row] = pose.astype(np.float16)
                output_row += 1
                if output_row % max(1, args.progress_every) == 0:
                    elapsed = time.perf_counter() - started
                    print(
                        f"POSE_BOND_PROGRESS rows={output_row}/{row_count} "
                        f"elapsed_seconds={elapsed:.1f}",
                        flush=True,
                    )
        if output_row != row_count:
            raise RuntimeError(
                f"Wrote {output_row} rows but counted {row_count} manifest rows"
            )
        sidecar.flush()
        del sidecar
        os.replace(temporary_path, output_path)
    except BaseException:
        try:
            del sidecar
        except UnboundLocalError:
            pass
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise

    manifest_stat = jsonl_path.stat()
    summary = {
        # Keep the established reader contract: shape, dtype, and pixel-centre
        # normalization are unchanged.  The added fields below record the
        # corrected source/projection coordinate frames.
        "format": "stamo-egoverse-pose-bond-v5-pixel-centre",
        "manifest": str(jsonl_path),
        "manifest_size_bytes": int(manifest_stat.st_size),
        "manifest_mtime_ns": int(manifest_stat.st_mtime_ns),
        "manifest_rows": row_count,
        "raw_root": str(raw_root),
        "output": str(output_path),
        "shape": [row_count, 2, 21, 3],
        "dtype": "float16",
        "coordinates": [
            "u_resize_pixel_centre_normalized",
            "v_resize_pixel_centre_normalized",
            "z_camera_metres",
        ],
        "source_coordinate_frame": "slam_world",
        "projection_coordinate_frame": "head_front_rgb_camera",
        "invalid_joint_encoding": [-1.0, -1.0, 0.0],
        "depth_min_metres": float(args.depth_min),
        "missing_count": missing_count,
        "missing_first_100": missing,
        "elapsed_seconds": time.perf_counter() - started,
    }
    summary_path = output_path.with_suffix(".summary_bond.json")
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(f"POSE_BOND_COMPLETE output={output_path} rows={row_count}")
    print(f"POSE_BOND_SUMMARY path={summary_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise
