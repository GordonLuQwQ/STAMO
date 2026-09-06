#!/usr/bin/env python3
"""Evaluate a portable hand-conditioned StaMo checkpoint on a fixed subset."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path


_DISTRIBUTED_ENV_KEYS = (
    "RANK",
    "WORLD_SIZE",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
    "GROUP_RANK",
    "ROLE_RANK",
    "ROLE_WORLD_SIZE",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run reconstruction evaluation only on a deterministic subset of "
            "an EgoVerse JSON/pose-sidecar pair."
        )
    )
    parser.add_argument("--config-path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--eval-json", type=Path, default=None)
    parser.add_argument("--eval-sidecar", type=Path, default=None)
    parser.add_argument(
        "--null-pose",
        action="store_true",
        help=(
            "Use an all-invalid hand pose so the spatial hand map is zero. "
            "This is intended for robot RGB datasets without 42 hand joints."
        ),
    )
    parser.add_argument("--num-images", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=15)
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/workspace/STAMO/logs"),
    )
    parser.add_argument("--task-name", default=None)
    return parser.parse_args()


def checkpoint_step(checkpoint: Path) -> int:
    match = re.search(r"(\d+)$", checkpoint.name)
    return int(match.group(1)) if match else 0


def require_file(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def require_portable_checkpoint(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {path}")
    require_file(path / "RenderNet.pth", "RenderNet checkpoint")
    require_file(path / "Projector.pth", "Projector checkpoint")
    return path


def resolve_single_jsonl(wrapper_path: Path) -> Path:
    wrapper = json.loads(wrapper_path.read_text(encoding="utf-8"))
    datasets = list(wrapper.get("datasets", []))
    if len(datasets) != 1:
        raise ValueError(
            "Evaluation currently requires exactly one JSONL, got "
            f"{len(datasets)} in {wrapper_path}."
        )
    jsonl = Path(str(datasets[0])).expanduser()
    if not jsonl.is_absolute():
        jsonl = wrapper_path.parent / jsonl
    return require_file(jsonl, "Eval JSONL")


def build_null_pose_sidecar(
    wrapper_path: Path,
    destination: Path,
) -> tuple[Path, int]:
    """Create line-aligned invalid UVZ rows that produce a zero hand map."""
    import numpy as np

    jsonl = resolve_single_jsonl(wrapper_path)
    with jsonl.open("r", encoding="utf-8") as stream:
        row_count = sum(1 for line in stream if line.strip())
    if row_count <= 0:
        raise ValueError(f"Eval JSONL contains no rows: {jsonl}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    pose = np.lib.format.open_memmap(
        destination,
        mode="w+",
        dtype=np.float16,
        shape=(row_count, 2, 21, 3),
    )
    pose[..., 0] = -1.0
    pose[..., 1] = -1.0
    pose[..., 2] = 0.0
    pose.flush()
    del pose
    return destination, row_count


def save_grouped_comparisons(
    image_dir: Path,
    num_images: int,
    group_size: int = 5,
) -> Path:
    """Save GT-over-pred comparison sheets with ``group_size`` columns."""
    from PIL import Image

    if group_size <= 0:
        raise ValueError("group_size must be positive")
    output_dir = image_dir / f"groups_of_{group_size}"
    output_dir.mkdir(parents=True, exist_ok=True)

    sheet_count = 0
    for start in range(0, num_images, group_size):
        indices = list(range(start, min(start + group_size, num_images)))
        gt_images = []
        pred_images = []
        for index in indices:
            gt_path = image_dir / f"{index}_gt.jpeg"
            pred_path = image_dir / f"{index}_pred.jpeg"
            if not gt_path.is_file() or not pred_path.is_file():
                raise FileNotFoundError(
                    "Missing per-sample eval image: "
                    f"gt={gt_path}, pred={pred_path}"
                )
            with Image.open(gt_path) as image:
                gt_images.append(image.convert("RGB").copy())
            with Image.open(pred_path) as image:
                pred_images.append(image.convert("RGB").copy())

        cell_width, cell_height = gt_images[0].size
        for image in gt_images + pred_images:
            if image.size != (cell_width, cell_height):
                raise ValueError(
                    "All eval images must have the same size, got "
                    f"{image.size} and {(cell_width, cell_height)}"
                )

        sheet = Image.new(
            "RGB",
            (cell_width * len(indices), cell_height * 2),
            color=(255, 255, 255),
        )
        for column, image in enumerate(gt_images):
            sheet.paste(image, (column * cell_width, 0))
        for column, image in enumerate(pred_images):
            sheet.paste(image, (column * cell_width, cell_height))

        end = indices[-1]
        sheet.save(
            output_dir
            / f"group_{sheet_count:02d}_rows_{start:03d}_{end:03d}.jpeg",
            quality=95,
        )
        sheet_count += 1

    print(
        f"EVAL_GROUPS path={output_dir} sheets={sheet_count} "
        f"columns={group_size} layout=gt_top_pred_bottom",
        flush=True,
    )
    return output_dir


def main() -> None:
    args = parse_args()
    if args.num_images <= 0:
        raise ValueError("--num-images must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    # This entry point is deliberately single-device. It does not initialize
    # DeepSpeed, create an optimizer, run backward, or save a checkpoint.
    os.environ.setdefault("DS_ACCELERATOR", "musa")
    # Match the production launcher. Internal qformer diagnostics are very
    # expensive and the trained projector is normalized before FLUX consumes it.
    os.environ.setdefault("CHECK_TENSOR", "0")
    os.environ["MUSA_VISIBLE_DEVICES"] = str(args.device)
    os.environ["MTHREADS_VISIBLE_DEVICES"] = str(args.device)
    os.environ.pop("ACCELERATE_USE_DEEPSPEED", None)
    for key in _DISTRIBUTED_ENV_KEYS:
        os.environ.pop(key, None)

    repo = Path(__file__).resolve().parents[1]
    os.chdir(repo)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader, Subset

    try:
        import torch_musa  # noqa: F401
    except ImportError:
        pass

    musa = getattr(torch, "musa", None)
    if musa is not None and musa.is_available():
        musa.set_device(0)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if musa is not None and musa.is_available():
        musa.manual_seed_all(args.seed)

    config_path = require_file(args.config_path, "Config")
    checkpoint = require_portable_checkpoint(args.checkpoint)
    config = OmegaConf.load(config_path)
    OmegaConf.set_struct(config, False)

    eval_json = require_file(
        args.eval_json
        if args.eval_json is not None
        else Path(str(config.data.eval_json_path)),
        "Eval JSON",
    )

    step = checkpoint_step(checkpoint)
    task_name = args.task_name or f"checkpoint_{step}_{args.num_images}_images"
    output_root = args.output_root.expanduser().resolve()
    result_dir = output_root / task_name
    result_dir.mkdir(parents=True, exist_ok=True)

    if args.null_pose:
        if args.eval_sidecar is not None:
            raise ValueError(
                "Do not pass --eval-sidecar together with --null-pose."
            )
        eval_sidecar, null_pose_rows = build_null_pose_sidecar(
            wrapper_path=eval_json,
            destination=result_dir / "null_pose_uvz.npy",
        )
        verify_pose_manifest = False
        print(
            f"EVAL_POSE_MODE null rows={null_pose_rows} "
            f"sidecar={eval_sidecar}",
            flush=True,
        )
    else:
        eval_sidecar = require_file(
            args.eval_sidecar
            if args.eval_sidecar is not None
            else Path(str(config.pose_condition.eval_sidecar)),
            "Eval pose sidecar",
        )
        verify_pose_manifest = bool(
            config.pose_condition.get("verify_manifest", True)
        )
        print(
            f"EVAL_POSE_MODE sidecar path={eval_sidecar}",
            flush=True,
        )

    config.do_train = False
    config.resume = True
    config.resume_path = str(checkpoint)
    config.fabric = False
    config.deepspeed = False
    config.dist = False
    config.world_size = 1
    config.local_rank = 0
    config.task_name = task_name
    config.log_dir = str(output_root)
    config.train.ckpt_save_dir = str(output_root / "metadata")
    config.train.local_batch_size = int(args.batch_size)
    config.train.max_eval_images_to_save = int(args.num_images)
    config.data.eval_json_path = str(eval_json)
    config.data.eval_mode = "test_mode"
    config.data.eval_num_workers = 0
    config.data.eval_persistent_workers = False
    config.pose_condition.eval_sidecar = str(eval_sidecar)

    from stamo.renderer.model.renderer import RenderNet
    from stamo.renderer.trainer import Trainer
    from stamo.renderer.utils.data import (
        collate_fn,
        load_multi_datasets_form_json,
    )

    pose_config = config.pose_condition
    full_loader = load_multi_datasets_form_json(
        str(eval_json),
        flip_p=0.0,
        img_size=int(config.data.img_size),
        local_batch_size=int(args.batch_size),
        pose_sidecar_path=str(eval_sidecar),
        num_workers=0,
        is_infinite=False,
        shuffle=False,
        drop_last=False,
        make_single_dataset=True,
        max_read_attempts=int(config.data.get("max_image_read_attempts", 8)),
        seed=int(args.seed),
        loader_timeout_seconds=0,
        persistent_workers=False,
        worker_start_method=None,
        prefetch_factor=1,
        eval_mode="test_mode",
        pose_flip_swap_hands=bool(pose_config.get("flip_swap_hands", True)),
        pose_verify_manifest=verify_pose_manifest,
    )
    full_dataset = full_loader.dataset
    if args.num_images > len(full_dataset):
        raise ValueError(
            f"Requested {args.num_images} images, but eval has only "
            f"{len(full_dataset)} rows."
        )

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    selected_indices = torch.randperm(
        len(full_dataset),
        generator=generator,
    )[: args.num_images].tolist()
    selected_dataset = Subset(full_dataset, selected_indices)
    eval_loader = DataLoader(
        selected_dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        drop_last=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    selected_rows = [
        {
            "subset_row": subset_row,
            "source_row": source_row,
            "image": str(full_dataset.metadata[source_row]),
        }
        for subset_row, source_row in enumerate(selected_indices)
    ]
    (result_dir / "selected_rows.json").write_text(
        json.dumps(selected_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        f"EVAL_SUBSET source_rows={len(full_dataset)} "
        f"selected_rows={len(selected_dataset)} seed={args.seed}",
        flush=True,
    )
    print(f"EVAL_CHECKPOINT path={checkpoint}", flush=True)

    model = RenderNet(config)
    model.requires_grad_(False)
    null_pose_hook = None
    if args.null_pose:
        # Keep the checkpoint and model sources untouched, but explicitly
        # ablate the spatial hand condition for this evaluation process.
        # The returned tensor keeps the learned encoder's exact shape/dtype:
        # [B, hand_output_channels, 14, 14].
        null_pose_hook = model.projector.pose_encoder.register_forward_hook(
            lambda _module, _inputs, hand_map: torch.zeros_like(hand_map)
        )
        print(
            "EVAL_NULL_HAND_MAP exact_zero=true "
            f"channels={model.hand_condition_channels}",
            flush=True,
        )
    trainer = Trainer(config, model)
    trainer.prepare_dist_model()
    trainer.move_model_to_device()

    try:
        metrics, elapsed = trainer.eval_fn(eval_loader, use_tqdm=True)
        image_dir = result_dir / "images" / str(trainer.global_step)
        grouped_dir = save_grouped_comparisons(
            image_dir=image_dir,
            num_images=args.num_images,
            group_size=5,
        )
        print(f"EVAL_COMPLETE step={trainer.global_step} time={elapsed}")
        print(metrics.avg)
        print(f"EVAL_IMAGES {image_dir}")
        print(f"EVAL_GROUPED_IMAGES {grouped_dir}")
    finally:
        if null_pose_hook is not None:
            null_pose_hook.remove()
        trainer._close_tensorboard_writer()


if __name__ == "__main__":
    main()
