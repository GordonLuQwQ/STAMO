#!/usr/bin/env python3
"""Create an isolated FLUX.2 Klein ZeRO-2 smoke configuration."""

from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--distributed-timeout-seconds", type=int, required=True)
    parser.add_argument("--enable-checkpointing", action="store_true")
    parser.add_argument("--resume-path", default="")
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    if cli.steps <= 0:
        raise ValueError("--steps must be positive")
    if cli.num_workers <= 0:
        raise ValueError("--num-workers must be positive")
    if cli.distributed_timeout_seconds < 60:
        raise ValueError("--distributed-timeout-seconds must be at least 60")

    base_path = Path(cli.base).resolve()
    output_path = Path(cli.output).resolve()
    work_dir = Path(cli.work_dir).resolve()
    config = OmegaConf.load(base_path)

    if str(config.render_net.flux.training_mode).strip().lower() != "lora":
        raise ValueError("The FLUX.2 smoke verifier requires training_mode=lora")
    if int(config.train.deepspeed_zero_stage) != 2:
        raise ValueError("The FLUX.2 smoke verifier requires DeepSpeed ZeRO-2")

    config.task_name = cli.task_name
    config.do_train = True
    config.fabric = False
    config.resume = bool(cli.resume_path)
    config.resume_path = str(Path(cli.resume_path).resolve()) if cli.resume_path else ""
    config.log_dir = str(work_dir / "logs")

    config.data.num_workers = cli.num_workers
    config.data.loader_timeout_seconds = 120
    config.data.persistent_workers = True
    config.data.worker_start_method = "spawn"
    config.data.read_trace_dir = str(work_dir / "data_trace")
    config.data.read_trace_samples = max(4, cli.steps + 2)

    config.train.epochs = 1
    config.train.num_iters = cli.steps
    config.train.enable_eval = False
    config.train.run_final_eval = False
    config.train.eval_step = 0
    config.train.enable_checkpointing = bool(cli.enable_checkpointing)
    # A tag at step 1 lets the verifier restore that immutable intermediate
    # state and prove that optimizer/scheduler training continues to step N.
    config.train.save_step = 1 if cli.enable_checkpointing else 0
    config.train.ckpt_save_dir = str(work_dir / "ckpts")
    config.train.log_interval = 1
    config.train.distributed_timeout_seconds = cli.distributed_timeout_seconds
    config.train.deepspeed_zero_stage = 2
    config.train.deepspeed_overlap_comm = False
    config.train.deepspeed_offload_optimizer_device = "none"
    config.train.deepspeed_offload_param_device = "none"
    config.train.deepspeed_skip_model_broadcast = True
    config.train.save_renderer_export = bool(cli.enable_checkpointing)
    config.train.phase_trace = True
    config.train.phase_trace_steps = cli.steps
    config.train.phase_trace_dir = str(work_dir / "phase_trace")
    config.train.hang_dump_after_seconds = 0
    config.train.phase_trace_synchronize = False
    config.train.verify_step_numerics = True
    config.train.verify_parameter_updates = True
    # Keep generation in the full checkpoint verifier bounded to two denoise
    # steps; the production YAML remains unchanged at its requested quality.
    config.render_net.num_inference_steps = 2

    output_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, output_path)
    print(output_path)


if __name__ == "__main__":
    main()
