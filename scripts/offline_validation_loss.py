#!/usr/bin/env python3
"""
Offline validation-loss evaluation for StaMo checkpoints.

Goals
-----
1. Never modify trainer.py, data.py, the original YAML, or checkpoints.
2. Never call backward(), optimizer.step(), or save_checkpoint().
3. Evaluate each checkpoint in a fresh process so MUSA memory is released.
4. Use one MUSA device by default and no custom MCCL metric collectives.
5. Reuse train_renderer.py so model/criterion construction matches training.
6. Repeat with deterministic seeds and report mean/std.
7. Save JSON, CSV, and TensorBoard scalars.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import runpy
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

RESULT_PREFIX = "__STAMO_OFFLINE_VAL_RESULT__="


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute offline StaMo validation loss from checkpoints."
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path("/workspace/STAMO"),
        help="StaMo repository root.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("/workspace/STAMO/configs/egoverse.yaml"),
        help="Base YAML config. It is read only and never modified.",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path(
            "/workspace/datasets/stamo_egoverse_output/ckpts/"
            "egoverse_human_2token_1024"
        ),
        help="Directory containing numeric checkpoint directories.",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        default=["latest"],
        help=(
            "Checkpoint steps, or 'latest', or 'all'. "
            "Example: --steps 250000 300000 310000"
        ),
    )
    parser.add_argument(
        "--checkpoints",
        nargs="*",
        type=Path,
        default=[],
        help="Additional explicit checkpoint directories.",
    )
    parser.add_argument(
        "--eval-json",
        type=Path,
        default=None,
        help="Optional eval JSON override. Defaults to the YAML value.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Deterministic noise/timestep repeats per checkpoint.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=33,
        help="Base seed. Repeat r uses seed + r.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Single-device validation batch size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers. Zero is safest.",
    )
    parser.add_argument(
        "--device",
        default="0",
        help="MUSA device exposed to the worker.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/workspace/datasets/stamo_egoverse_output/"
            "offline_validation_loss"
        ),
        help="Output directory for JSON, CSV, logs, and TensorBoard.",
    )
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Do not write TensorBoard scalars.",
    )

    # Internal worker options.
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_checkpoint", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--_worker-config", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def checkpoint_step(path: Path) -> int:
    matches = re.findall(r"\d+", path.name)
    if not matches:
        raise ValueError(f"Cannot infer checkpoint step from: {path}")
    return int(matches[-1])


def numeric_checkpoint_dirs(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Checkpoint root does not exist: {root}")
    paths = [
        path
        for path in root.iterdir()
        if path.is_dir() and path.name.isdigit()
    ]
    return sorted(paths, key=checkpoint_step)


def validate_checkpoint(path: Path) -> Path:
    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {path}")

    latest_file = path / "latest"
    if not latest_file.is_file():
        raise FileNotFoundError(
            f"DeepSpeed checkpoint is missing 'latest': {path}"
        )

    tag = latest_file.read_text(encoding="utf-8").strip()
    if not tag:
        raise RuntimeError(f"Empty latest file: {latest_file}")

    tagged_dir = path / tag
    if not tagged_dir.is_dir():
        raise FileNotFoundError(
            f"Checkpoint tag directory does not exist: {tagged_dir}"
        )
    return path


def resolve_checkpoints(args: argparse.Namespace) -> list[Path]:
    numeric = numeric_checkpoint_dirs(args.checkpoint_root)
    selected: list[Path] = []

    for token in args.steps:
        token_lower = token.lower()
        if token_lower == "latest":
            if not numeric:
                raise RuntimeError(
                    f"No numeric checkpoints under {args.checkpoint_root}"
                )
            selected.append(numeric[-1])
        elif token_lower == "all":
            selected.extend(numeric)
        elif token.isdigit():
            selected.append(args.checkpoint_root / token)
        else:
            selected.append(Path(token))

    selected.extend(args.checkpoints)

    unique: dict[str, Path] = {}
    for path in selected:
        checked = validate_checkpoint(path)
        unique[str(checked)] = checked

    return sorted(unique.values(), key=checkpoint_step)


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required. Install it with: pip install pyyaml"
        ) from exc

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise RuntimeError(f"YAML root must be a mapping: {path}")
    return data


def write_worker_config(
    base_config: Path,
    checkpoint: Path,
    output_dir: Path,
    eval_json: Path | None,
    batch_size: int,
    num_workers: int,
) -> Path:
    import yaml

    config = load_yaml(base_config)
    step = checkpoint_step(checkpoint)

    # Keep the normal training-construction path. The worker replaces only
    # Trainer.train_eval_by_iter before train_renderer.py executes.
    config["do_train"] = True
    config["resume"] = True
    config["resume_path"] = str(checkpoint)
    config["fabric"] = False
    config["log_dir"] = str(output_dir / "construction_logs")
    config["task_name"] = f"offline_validation_loss_{step}"

    data = config.setdefault("data", {})
    data["num_workers"] = int(num_workers)
    if eval_json is not None:
        data["eval_json_path"] = str(eval_json.resolve())

    train = config.setdefault("train", {})
    train["local_batch_size"] = int(batch_size)
    train["reduce_metrics_across_ranks"] = False
    train["deepspeed_skip_model_broadcast"] = False

    temp_dir = output_dir / "temporary_configs"
    temp_dir.mkdir(parents=True, exist_ok=True)
    destination = temp_dir / f"offline_val_{step}.yaml"
    with destination.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False, allow_unicode=True)
    return destination


class _DummyDataset:
    def __len__(self) -> int:
        return 1


class _DummyTrainLoader:
    """Avoid parsing the 1.76M-image train index during offline evaluation."""

    dataset = _DummyDataset()

    def __len__(self) -> int:
        return 1

    def __iter__(self):
        return iter(())


def _argument_value(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    name: str,
    position: int,
    default: Any,
) -> Any:
    if name in kwargs:
        return kwargs[name]
    if len(args) > position:
        return args[position]
    return default


def patch_training_data_loader() -> None:
    """Return a tiny placeholder only for the infinite training loader."""
    import stamo.renderer.utils.data as data_module

    original_multi = data_module.load_multi_datasets_form_json
    original_unsampled = data_module.load_unsampler_datasets_from_json

    def patched_unsampled(*args, **kwargs):
        is_infinite = bool(
            _argument_value(args, kwargs, "is_infinite", 5, True)
        )
        if is_infinite:
            return _DummyTrainLoader()
        return original_unsampled(*args, **kwargs)

    def patched_multi(*args, **kwargs):
        is_infinite = bool(
            _argument_value(args, kwargs, "is_infinite", 5, True)
        )
        if is_infinite:
            return _DummyTrainLoader()
        return original_multi(*args, **kwargs)

    data_module.load_unsampler_datasets_from_json = patched_unsampled
    data_module.load_multi_datasets_form_json = patched_multi


def seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    musa = getattr(torch, "musa", None)
    if musa is not None:
        manual_seed_all = getattr(musa, "manual_seed_all", None)
        if callable(manual_seed_all):
            manual_seed_all(seed)


def infer_step_from_load(
    loaded_path: Any,
    load_path: Path,
    client_state: Any,
) -> int:
    if isinstance(client_state, dict):
        value = client_state.get("global_step")
        if value is not None:
            return int(value)

    for candidate in (loaded_path, load_path):
        name = Path(str(candidate).rstrip("/")).name
        match = re.search(r"(\d+)$", name)
        if match:
            return int(match.group(1))
    return 0


def patch_trainer_for_offline_eval(
    repeats: int,
    base_seed: int,
    expected_checkpoint: Path,
) -> None:
    import torch
    from tqdm import tqdm
    from stamo.renderer.trainer import Trainer

    def load_module_only(self, load_path) -> None:
        """Load model/client state, never optimizer or scheduler state."""
        if not self.use_deepspeed:
            result = self.model.load_checkpoint(load_path)
            self.global_step = int(result)
            return

        kwargs = {
            "load_optimizer_states": False,
            "load_lr_scheduler_states": False,
            "load_module_only": True,
        }
        try:
            result = self.model.load_checkpoint(load_path, **kwargs)
        except TypeError:
            # Compatibility with older/custom DeepSpeed builds.
            kwargs.pop("load_module_only")
            result = self.model.load_checkpoint(load_path, **kwargs)

        loaded_path, client_state = result
        if loaded_path is None:
            raise RuntimeError(
                f"Failed to load DeepSpeed checkpoint: {load_path}"
            )
        self.global_step = infer_step_from_load(
            loaded_path=loaded_path,
            load_path=Path(load_path),
            client_state=client_state,
        )
        print(
            f"[offline-val] Loaded module from {loaded_path}; "
            f"global_step={self.global_step}",
            flush=True,
        )

    def offline_validation(
        self,
        train_loader,
        eval_loader=None,
        use_tqdm=True,
    ) -> None:
        if eval_loader is None:
            raise RuntimeError("eval_loader is None")

        # RenderNet.forward() routes by RenderNet.training:
        #   True  -> train_step(..., criterion=...)
        #   False -> eval_step(...), which does not accept criterion.
        #
        # Put the full network in eval mode first, then flip only the top-level
        # RenderNet.training flag. This selects the loss-producing train_step
        # branch while keeping child dropout/normalization modules in eval mode.
        # torch.no_grad() below additionally guarantees that no gradient graph,
        # backward pass, or optimizer update is created.
        self.model.eval()
        render_module = (
            self.model.module
            if hasattr(self.model, "module")
            else self.model
        )
        render_module.training = True
        repeat_results: list[dict[str, Any]] = []

        try:
            with torch.no_grad():
                for repeat_index in range(repeats):
                    repeat_seed = base_seed + repeat_index
                    seed_everything(repeat_seed)

                    total_weighted_loss = torch.zeros(
                        (),
                        device=self.device,
                        dtype=torch.float32,
                    )
                    total_samples = 0
                    batch_count = 0

                    iterator = tqdm(
                        eval_loader,
                        total=len(eval_loader),
                        desc=(
                            f"checkpoint {self.global_step} "
                            f"repeat {repeat_index + 1}/{repeats}"
                        ),
                        dynamic_ncols=True,
                        disable=not use_tqdm,
                    )

                    for inputs in iterator:
                        inputs = self.prepare_batch(inputs)
                        inputs["global_step"] = int(self.global_step)
                        inputs["epoch"] = int(
                            self.global_step // self.iter_per_ep
                            if self.iter_per_ep
                            else 0
                        )

                        outputs = self.forward_step(
                            inputs,
                            criterion=self.criterion,
                        )
                        if "loss" not in outputs:
                            raise RuntimeError(
                                "StaMo validation forward did not return "
                                f"'loss'. Output keys: {list(outputs.keys())}"
                            )

                        loss = outputs["loss"]
                        if not torch.is_tensor(loss):
                            loss = torch.as_tensor(
                                loss,
                                device=self.device,
                                dtype=torch.float32,
                            )
                        loss = loss.detach().float().mean()

                        batch_size = int(inputs["images"].shape[0])
                        total_weighted_loss.add_(loss * batch_size)
                        total_samples += batch_size
                        batch_count += 1

                    if total_samples <= 0:
                        raise RuntimeError(
                            "Validation DataLoader produced zero samples."
                        )

                    mean_loss = float(
                        (total_weighted_loss / total_samples)
                        .detach()
                        .cpu()
                        .item()
                    )
                    repeat_results.append(
                        {
                            "repeat": repeat_index,
                            "seed": repeat_seed,
                            "loss": mean_loss,
                            "samples": total_samples,
                            "batches": batch_count,
                        }
                    )
                    print(
                        f"[offline-val] step={self.global_step} "
                        f"repeat={repeat_index + 1}/{repeats} "
                        f"seed={repeat_seed} samples={total_samples} "
                        f"loss={mean_loss:.8f}",
                        flush=True,
                    )

            losses = [item["loss"] for item in repeat_results]
            payload = {
                "checkpoint": str(expected_checkpoint.resolve()),
                "step": int(self.global_step),
                "repeats": int(repeats),
                "base_seed": int(base_seed),
                "samples_per_repeat": int(
                    repeat_results[0]["samples"]
                ),
                "losses": losses,
                "mean_validation_loss": float(statistics.fmean(losses)),
                "std_validation_loss": float(
                    statistics.pstdev(losses)
                    if len(losses) > 1
                    else 0.0
                ),
                "repeat_results": repeat_results,
            }
            print(
                RESULT_PREFIX
                + json.dumps(payload, ensure_ascii=False),
                flush=True,
            )
        finally:
            # Return to inference mode before process cleanup.
            self.model.eval()

    Trainer.load_checkpoint = load_module_only
    Trainer.train_eval_by_iter = offline_validation


def worker_main(args: argparse.Namespace) -> int:
    if args._checkpoint is None or args._worker_config is None:
        raise ValueError("Worker checkpoint/config arguments are required.")

    # Import torch_musa before any possible DeepSpeed import. New torch_musa
    # releases usually auto-initialize with torch, but the explicit import is
    # harmless and supports older/custom environments.
    try:
        import torch_musa  # noqa: F401
    except ImportError:
        pass

    repo = args.repo.resolve()
    os.chdir(repo)
    sys.path.insert(0, str(repo))

    patch_training_data_loader()
    patch_trainer_for_offline_eval(
        repeats=args.repeats,
        base_seed=args.seed,
        expected_checkpoint=args._checkpoint,
    )

    original_argv = sys.argv[:]
    sys.argv = [
        str(repo / "train_renderer.py"),
        "--config_path",
        str(args._worker_config.resolve()),
        "--deepspeed",
    ]

    try:
        runpy.run_path(
            str(repo / "train_renderer.py"),
            run_name="__main__",
        )
    finally:
        sys.argv = original_argv
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
        except Exception as exc:
            print(
                f"[offline-val] process-group cleanup warning: {exc!r}",
                file=sys.stderr,
                flush=True,
            )
    return 0


def free_tcp_port() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return str(sock.getsockname()[1])


def run_worker(
    args: argparse.Namespace,
    checkpoint: Path,
    worker_config: Path,
    log_path: Path,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_worker",
        "--repo",
        str(args.repo.resolve()),
        "--config",
        str(args.config.resolve()),
        "--checkpoint-root",
        str(args.checkpoint_root.resolve()),
        "--_checkpoint",
        str(checkpoint.resolve()),
        "--_worker-config",
        str(worker_config.resolve()),
        "--repeats",
        str(args.repeats),
        "--seed",
        str(args.seed),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--device",
        str(args.device),
        "--output-dir",
        str(args.output_dir.resolve()),
    ]

    env = os.environ.copy()

    # Force the Moore Threads backend before DeepSpeed is imported.
    # The normal StaMo launch script provides MUSA-specific setup, while this
    # offline worker starts a fresh Python process and must set it explicitly.
    env.setdefault("DS_ACCELERATOR", "musa")
    env["MTHREADS_VISIBLE_DEVICES"] = str(args.device)
    # Retain this alias for compatibility with existing local MUSA stacks.
    env["MUSA_VISIBLE_DEVICES"] = str(args.device)

    env["RANK"] = "0"
    env["LOCAL_RANK"] = "0"
    env["WORLD_SIZE"] = "1"
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = free_tcp_port()
    env.setdefault("PYTHONUNBUFFERED", "1")

    result_payload: dict[str, Any] | None = None
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=str(args.repo.resolve()),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
            log_file.flush()
            if line.startswith(RESULT_PREFIX):
                result_payload = json.loads(
                    line[len(RESULT_PREFIX):]
                )

        return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(
            f"Offline validation worker failed for {checkpoint} "
            f"with return code {return_code}. Log: {log_path}"
        )
    if result_payload is None:
        raise RuntimeError(
            f"Worker exited without a result payload. Log: {log_path}"
        )
    return result_payload


def write_outputs(
    output_dir: Path,
    results: list[dict[str, Any]],
    write_tensorboard: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "validation_loss_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "created_at_unix": time.time(),
                "results": results,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    csv_path = output_dir / "validation_loss_summary.csv"
    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "step",
                "checkpoint",
                "samples_per_repeat",
                "repeats",
                "base_seed",
                "mean_validation_loss",
                "std_validation_loss",
                "losses",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "step": result["step"],
                    "checkpoint": result["checkpoint"],
                    "samples_per_repeat":
                        result["samples_per_repeat"],
                    "repeats": result["repeats"],
                    "base_seed": result["base_seed"],
                    "mean_validation_loss":
                        result["mean_validation_loss"],
                    "std_validation_loss":
                        result["std_validation_loss"],
                    "losses": json.dumps(result["losses"]),
                }
            )

    if write_tensorboard:
        from torch.utils.tensorboard import SummaryWriter

        tb_dir = output_dir / "tensorboard"
        writer = SummaryWriter(log_dir=str(tb_dir))
        try:
            for result in results:
                step = int(result["step"])
                writer.add_scalar(
                    "validation/offline_loss_mean",
                    result["mean_validation_loss"],
                    step,
                )
                writer.add_scalar(
                    "validation/offline_loss_std",
                    result["std_validation_loss"],
                    step,
                )
                for repeat_index, loss in enumerate(result["losses"]):
                    writer.add_scalar(
                        f"validation/offline_loss_repeat_{repeat_index}",
                        loss,
                        step,
                    )
        finally:
            writer.close()

    print(f"\nSaved JSON: {summary_path}")
    print(f"Saved CSV:  {csv_path}")
    if write_tensorboard:
        print(f"TensorBoard: {output_dir / 'tensorboard'}")


def parent_main(args: argparse.Namespace) -> int:
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    if not args.repo.is_dir():
        raise FileNotFoundError(f"Repo does not exist: {args.repo}")
    if not args.config.is_file():
        raise FileNotFoundError(f"Config does not exist: {args.config}")
    if not (args.repo / "train_renderer.py").is_file():
        raise FileNotFoundError(
            f"train_renderer.py not found under {args.repo}"
        )

    checkpoints = resolve_checkpoints(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Offline validation checkpoints:")
    for checkpoint in checkpoints:
        print(f"  step {checkpoint_step(checkpoint)}: {checkpoint}")
    print(
        f"Device={args.device}, batch_size={args.batch_size}, "
        f"repeats={args.repeats}, seeds="
        f"{args.seed}..{args.seed + args.repeats - 1}"
    )

    results: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        step = checkpoint_step(checkpoint)
        worker_config = write_worker_config(
            base_config=args.config,
            checkpoint=checkpoint,
            output_dir=args.output_dir,
            eval_json=args.eval_json,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        log_path = args.output_dir / "logs" / f"step_{step}.log"
        result = run_worker(
            args=args,
            checkpoint=checkpoint,
            worker_config=worker_config,
            log_path=log_path,
        )
        results.append(result)
        write_outputs(
            output_dir=args.output_dir,
            results=results,
            write_tensorboard=not args.no_tensorboard,
        )

    print("\nFinal offline validation-loss results:")
    for result in results:
        print(
            f"  step={result['step']} "
            f"loss={result['mean_validation_loss']:.8f} "
            f"std={result['std_validation_loss']:.8f} "
            f"samples={result['samples_per_repeat']}"
        )
    return 0


def main() -> int:
    args = parse_args()
    if args._worker:
        return worker_main(args)
    return parent_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
