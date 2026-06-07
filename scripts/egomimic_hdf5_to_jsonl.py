"""
Convert EgoMimic HDF5 datasets to STAMO-compatible JSONL/JSON format.

Usage:
    python scripts/egomimic_hdf5_to_jsonl.py \
        --hdf5_dir /workspace/datasets/EgoMimic \
        --output_dir /workspace/datasets/EgoMimic/images \
        --jsonl_dir /workspace/STAMO/jsons \
        --dataset_name egomimic \
        --eval_ratio 0.1 \
        --camera_keys agentview_image eye_in_hand_image

The script will:
  1. Open each .hdf5 file in hdf5_dir
  2. Extract image frames from each demo's observations
  3. Save frames as JPEG to output_dir/<dataset>/<demo>/<cam>/<frame>.jpg
  4. Write JSONL files (one per hdf5 file) listing absolute image paths
  5. Write train/eval JSON config files STAMO expects
"""

import argparse
import json
import os
import random

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm


random.seed(33)


def get_image_keys(obs_group):
    """Heuristically find camera/image keys inside an obs group."""
    candidates = []
    for key in obs_group.keys():
        val = obs_group[key]
        if not hasattr(val, "shape"):
            continue
        # Image tensors are (T, H, W, 3) or (T, H, W, C)
        if len(val.shape) == 4 and val.shape[-1] in (3, 4):
            candidates.append(key)
        # Also accept (T, C, H, W) with C in {3,4}
        elif len(val.shape) == 4 and val.shape[1] in (3, 4):
            candidates.append(key)
    return candidates


def save_frame(arr, path):
    """Save a single HxWxC uint8 numpy array as JPEG."""
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    # CHW -> HWC if needed
    if arr.ndim == 3 and arr.shape[0] in (3, 4):
        arr = arr.transpose(1, 2, 0)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    img = Image.fromarray(arr, mode="RGB")
    img.save(path, format="JPEG", quality=95)


def process_hdf5(hdf5_path, output_root, camera_keys=None):
    """
    Extract all frames from an HDF5 file.

    Returns list of absolute image paths.
    """
    image_paths = []
    dataset_name = os.path.splitext(os.path.basename(hdf5_path))[0]
    dataset_out = os.path.join(output_root, dataset_name)

    with h5py.File(hdf5_path, "r") as f:
        if "data" not in f:
            raise KeyError(f"No 'data' group found in {hdf5_path}. Keys: {list(f.keys())}")

        data_group = f["data"]
        demo_keys = sorted(data_group.keys())
        print(f"\n[{dataset_name}] {len(demo_keys)} demos found.")

        for demo_key in tqdm(demo_keys, desc=dataset_name):
            demo = data_group[demo_key]

            if "obs" not in demo:
                print(f"  Skipping {demo_key}: no 'obs' group.")
                continue

            obs = demo["obs"]

            # Determine which camera keys to use
            if camera_keys:
                keys_to_use = [k for k in camera_keys if k in obs]
                if not keys_to_use:
                    keys_to_use = get_image_keys(obs)
            else:
                keys_to_use = get_image_keys(obs)

            if not keys_to_use:
                print(f"  Skipping {demo_key}: no image observations found. obs keys: {list(obs.keys())}")
                continue

            for cam_key in keys_to_use:
                frames = obs[cam_key][:]  # (T, H, W, C) or (T, C, H, W)
                T = frames.shape[0]
                out_dir = os.path.join(dataset_out, demo_key, cam_key)
                os.makedirs(out_dir, exist_ok=True)

                for t in range(T):
                    out_path = os.path.join(out_dir, f"{t:06d}.jpg")
                    if not os.path.exists(out_path):
                        save_frame(frames[t], out_path)
                    image_paths.append(os.path.abspath(out_path))

    print(f"[{dataset_name}] Extracted {len(image_paths)} frames.")
    return image_paths


def write_jsonl(paths, jsonl_path):
    os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for p in paths:
            f.write(json.dumps({"image": p}) + "\n")
    print(f"Wrote {len(paths)} entries -> {jsonl_path}")


def write_json_config(jsonl_basenames, ratios, json_path):
    config = {"datasets": jsonl_basenames, "ratios": ratios}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
    print(f"Wrote config -> {json_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5_dir", default="/workspace/datasets/EgoMimic",
                        help="Directory containing .hdf5 files")
    parser.add_argument("--output_dir", default="/workspace/datasets/EgoMimic/images",
                        help="Where to save extracted image frames")
    parser.add_argument("--jsonl_dir", default="/workspace/STAMO/jsons",
                        help="Where to write JSONL and JSON config files")
    parser.add_argument("--dataset_name", default="egomimic",
                        help="Name prefix for output files (e.g. 'egomimic')")
    parser.add_argument("--eval_ratio", type=float, default=0.1,
                        help="Fraction of frames per dataset to use as eval (default 0.1)")
    parser.add_argument("--camera_keys", nargs="*", default=None,
                        help="Camera observation keys to extract (default: auto-detect all image keys)")
    args = parser.parse_args()

    hdf5_files = sorted([
        os.path.join(args.hdf5_dir, f)
        for f in os.listdir(args.hdf5_dir)
        if f.endswith(".hdf5")
    ])

    if not hdf5_files:
        raise FileNotFoundError(f"No .hdf5 files found in {args.hdf5_dir}")

    print(f"Found {len(hdf5_files)} HDF5 files: {[os.path.basename(p) for p in hdf5_files]}")

    os.makedirs(args.jsonl_dir, exist_ok=True)

    train_jsonl_names = []
    eval_all_paths = []

    for i, hdf5_path in enumerate(hdf5_files):
        all_paths = process_hdf5(hdf5_path, args.output_dir, args.camera_keys)

        # Shuffle and split
        random.shuffle(all_paths)
        n_eval = max(3, int(len(all_paths) * args.eval_ratio))
        eval_paths = all_paths[:n_eval]
        train_paths = all_paths[n_eval:]

        stem = os.path.splitext(os.path.basename(hdf5_path))[0]
        train_jsonl_name = f"train_{args.dataset_name}_{stem}.jsonl"
        train_jsonl_path = os.path.join(args.jsonl_dir, train_jsonl_name)

        write_jsonl(sorted(train_paths), train_jsonl_path)
        train_jsonl_names.append(train_jsonl_name)
        eval_all_paths.extend(eval_paths)

    # Write eval JSONL (combined from all datasets)
    eval_jsonl_name = f"eval_{args.dataset_name}.jsonl"
    eval_jsonl_path = os.path.join(args.jsonl_dir, eval_jsonl_name)
    write_jsonl(sorted(eval_all_paths), eval_jsonl_path)

    # Write train JSON config (equal ratios across datasets)
    n = len(train_jsonl_names)
    ratios = [round(1.0 / n, 6)] * n
    ratios[-1] = round(1.0 - sum(ratios[:-1]), 6)  # fix rounding

    train_json_path = os.path.join(args.jsonl_dir, f"train_{args.dataset_name}.json")
    write_json_config(train_jsonl_names, ratios, train_json_path)

    # Write eval JSON config
    eval_json_path = os.path.join(args.jsonl_dir, f"eval_{args.dataset_name}.json")
    write_json_config([eval_jsonl_name], [1.0], eval_json_path)

    print("\nDone! Summary:")
    print(f"  Train JSON : {train_json_path}")
    print(f"  Eval JSON  : {eval_json_path}")
    print(f"  Train parts: {train_jsonl_names}")
    print(f"  Eval JSONL : {eval_jsonl_path}")


if __name__ == "__main__":
    main()
