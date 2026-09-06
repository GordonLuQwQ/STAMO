#!/usr/bin/env python3
"""Create an isolated, non-resuming MUSA performance/admission config.

The source YAML is never edited.  Every profile starts from the same serialized
communication baseline, then changes one optimization family so timings remain
comparable and a failed experiment cannot affect production checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
from datetime import datetime, timezone

from omegaconf import OmegaConf


PROFILE_OVERRIDES = {
    "zero3_baseline": {},
    "zero3_frozen_leaf": {
        "train.deepspeed_zero3_leaf_frozen_modules": True,
    },
    "zero3_condition_leaf": {
        "train.deepspeed_zero3_leaf_condition_modules": True,
        "train.deepspeed_zero3_expected_qformer_blocks": 4,
    },
    "zero3_safe_combined_leaf": {
        "train.deepspeed_zero3_leaf_frozen_modules": True,
        "train.deepspeed_zero3_leaf_condition_modules": True,
        "train.deepspeed_zero3_expected_qformer_blocks": 4,
    },
    "zero3_double_ff_leaf": {
        "train.deepspeed_zero3_leaf_flux_blocks": "double_ff",
        "train.deepspeed_zero3_expected_double_blocks": 19,
        # Each FLUX double-block FeedForward is about 75.5M parameters. This
        # profile therefore requires an 80M MCCL preflight. Attention is
        # deliberately excluded because Diffusers invokes it with kwargs,
        # which DeepSpeed 0.17.2's trainable-leaf hook does not inspect.
        "train.deepspeed_zero3_leaf_max_numel": 80_000_000,
    },
    "zero3_combined_leaf": {
        "train.deepspeed_zero3_leaf_frozen_modules": True,
        "train.deepspeed_zero3_leaf_condition_modules": True,
        "train.deepspeed_zero3_expected_qformer_blocks": 4,
        "train.deepspeed_zero3_leaf_flux_blocks": "double_ff",
        "train.deepspeed_zero3_expected_double_blocks": 19,
        "train.deepspeed_zero3_leaf_max_numel": 80_000_000,
    },
    "zero3_bucket75": {
        "train.deepspeed_reduce_bucket_size": 75_000_000,
    },
    "zero3_persist_small": {
        # DeepSpeed documents the per-parameter threshold as a way to reduce
        # latency-bound messages. Keep the aggregate unpartitioned parameter
        # budget explicit so this experiment cannot consume unbounded memory.
        # This remains an isolated candidate: DeepSpeed gathers persistent
        # parameters after each optimizer step, so it needs longer MUSA tests
        # than the bounded leaf profiles before production admission.
        "train.deepspeed_stage3_param_persistence_threshold": 1_000_000,
        "train.deepspeed_stage3_model_persistence_threshold": 200_000_000,
    },
    "zero3_reuse1500": {
        "train.deepspeed_stage3_max_live_parameters": 1_500_000_000,
        "train.deepspeed_stage3_max_reuse_distance": 1_500_000_000,
    },
    # This profile is expected to be rejected on the current 80 GiB cards. It
    # verifies that every rank reaches the same fail-fast admission decision;
    # it is not a performance run.
    "zero2_admission": {
        "train.deepspeed_zero_stage": 2,
        "train.deepspeed_allow_zero2_full_transformer": True,
    },
    "musa_fused_adamw": {
        "train.optimizer_type": "musa_fused_adamw",
    },
}


SERIAL_BASELINE = {
    "train.mixed_precision": "bf16",
    "train.local_batch_size": 1,
    "train.gradient_accumulate_steps": 1,
    "train.optimizer_type": "adamw",
    "train.deepspeed_zero_stage": 3,
    "train.deepspeed_allow_zero2_full_transformer": False,
    "train.deepspeed_zero2_min_device_headroom_gib": 36.0,
    "train.deepspeed_zero2_host_buffer_factor": 1.5,
    "train.deepspeed_overlap_comm": False,
    "train.deepspeed_stage3_use_all_reduce_for_fetch_params": False,
    "train.deepspeed_reduce_bucket_size": 50_000_000,
    "train.deepspeed_allgather_bucket_size": 50_000_000,
    "train.deepspeed_stage3_prefetch_bucket_size": 0,
    "train.deepspeed_stage3_param_persistence_threshold": 100_000,
    "train.deepspeed_stage3_model_persistence_threshold": 100_000_000,
    "train.deepspeed_stage3_max_live_parameters": 1_000_000_000,
    "train.deepspeed_stage3_max_reuse_distance": 1_000_000_000,
    "train.deepspeed_offload_optimizer_device": "none",
    "train.deepspeed_offload_param_device": "none",
    "train.deepspeed_zero3_leaf_frozen_modules": False,
    "train.deepspeed_zero3_leaf_condition_modules": False,
    "train.deepspeed_zero3_leaf_min_numel": 1_000_000,
    "train.deepspeed_zero3_leaf_max_numel": 50_000_000,
    "train.deepspeed_zero3_leaf_flux_blocks": "none",
    "train.deepspeed_zero3_expected_double_blocks": 0,
    "train.deepspeed_zero3_expected_qformer_blocks": 0,
    "render_net.flux.train_transformer": True,
    "render_net.flux.torch_dtype": "bfloat16",
    "render_net.flux.gradient_checkpointing": False,
}


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")


def build_config(args: argparse.Namespace):
    base_path = pathlib.Path(args.base_config).expanduser().resolve()
    output_path = pathlib.Path(args.output).expanduser().resolve()
    if base_path == output_path:
        raise ValueError("Refusing to overwrite the base/production config.")
    if not base_path.is_file():
        raise FileNotFoundError(f"Base config does not exist: {base_path}")
    if args.steps < 2:
        raise ValueError("--steps must be at least 2 to measure a completed interval.")
    if args.timeout_seconds < 60:
        raise ValueError("--timeout-seconds must be at least 60.")

    config = OmegaConf.load(base_path)
    OmegaConf.set_struct(config, False)
    for path, value in SERIAL_BASELINE.items():
        OmegaConf.update(config, path, value, merge=False)
    for path, value in PROFILE_OVERRIDES[args.profile].items():
        OmegaConf.update(config, path, value, merge=False)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    task_name = _safe_name(
        args.task_name
        or f"musa_perf_{args.profile}_{timestamp}_{os.getpid()}"
    )
    if not task_name:
        raise ValueError("The resolved task name is empty.")
    artifact_root = pathlib.Path(args.artifact_root).expanduser().resolve()
    trace_dir = artifact_root / "phase_trace"

    config.task_name = task_name
    config.log_dir = str((artifact_root / "tensorboard").resolve())
    config.resume = False
    config.resume_path = ""
    config.data.num_workers = 0
    config.data.persistent_workers = False
    config.train.num_iters = int(args.steps)
    config.train.enable_eval = False
    config.train.run_final_eval = False
    config.train.eval_step = 0
    config.train.enable_checkpointing = False
    config.train.save_step = 0
    config.train.ckpt_save_dir = str((artifact_root / "forbidden_checkpoints").resolve())
    config.train.save_renderer_export = False
    # Match production's synchronization cadence. The phase files still prove
    # progress on every rank, but rank 0 does not call loss.cpu().item() every
    # step and distort the comparison.
    config.train.log_interval = min(20, int(args.steps))
    config.train.distributed_timeout_seconds = int(args.timeout_seconds)
    config.train.phase_trace = True
    config.train.phase_trace_steps = int(args.steps) + 1
    config.train.phase_trace_dir = str(trace_dir)
    config.train.phase_trace_synchronize = False
    config.train.hang_dump_after_seconds = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=config, f=str(output_path), resolve=False)

    reloaded = OmegaConf.load(output_path)
    required = dict(SERIAL_BASELINE)
    required.update(PROFILE_OVERRIDES[args.profile])
    for path, expected in required.items():
        actual = OmegaConf.select(reloaded, path)
        if actual != expected:
            raise RuntimeError(
                f"Generated config mismatch at {path}: expected {expected!r}, got {actual!r}."
            )
    if bool(reloaded.resume):
        raise RuntimeError("Generated performance config unexpectedly enables resume.")
    if bool(reloaded.train.enable_eval) or bool(reloaded.train.run_final_eval):
        raise RuntimeError("Generated performance config unexpectedly enables evaluation.")
    if bool(reloaded.train.enable_checkpointing):
        raise RuntimeError("Generated performance config unexpectedly enables checkpointing.")

    manifest = {
        "profile": args.profile,
        "task_name": task_name,
        "steps": int(args.steps),
        "base_config": str(base_path),
        "base_sha256": _sha256(base_path),
        "output_config": str(output_path),
        "output_sha256": _sha256(output_path),
        "artifact_root": str(artifact_root),
        "serial_baseline": SERIAL_BASELINE,
        "overrides": PROFILE_OVERRIDES[args.profile],
    }
    manifest_path = artifact_root / "config_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", default="configs/flux.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--profile", choices=sorted(PROFILE_OVERRIDES), required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--task-name", default="")
    return parser.parse_args()


def main() -> None:
    manifest = build_config(parse_args())
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print("PASS: isolated performance config generated; base config was not edited")


if __name__ == "__main__":
    main()
