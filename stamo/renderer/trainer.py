import faulthandler
import inspect
import json
import os
import random
import re
import tempfile
import time
from contextlib import nullcontext
from functools import wraps
from typing import Any, Dict

import numpy as np
import torch

try:
    import torch_musa  # noqa: F401
except ImportError:
    torch_musa = None

import torch.distributed as DIST
import torchvision.transforms as T
from lightning.fabric import Fabric
from lightning.fabric.loggers import TensorBoardLogger
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm

from stamo.renderer.model.renderer import RenderNet
from stamo.renderer.utils.data import complex_to_device, move_to_cuda
from stamo.renderer.utils.device import get_accelerator_device
from stamo.renderer.utils.files import ensure_directory, ensure_dirname
from stamo.renderer.utils.metrics import Meter, Timer, get_parameters
from stamo.renderer.utils.overwatch import initialize_overwatch


overwatch = initialize_overwatch(__name__)


def _get_runtime_device(local_rank: int):
    """Prefer MUSA when torch_musa is available, otherwise use the shared helper."""
    musa = getattr(torch, "musa", None)
    if musa is not None and musa.is_available():
        device_count = int(musa.device_count())
        if local_rank < 0 or local_rank >= device_count:
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} but only {device_count} MUSA device(s) are visible. "
                "Check MUSA_VISIBLE_DEVICES and the DeepSpeed launch command."
            )
        musa.set_device(local_rank)
        return torch.device("musa", local_rank)

    return get_accelerator_device(local_rank)


def _as_bool(value, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "on", "1"}:
            return True
        if normalized in {"false", "no", "off", "0"}:
            return False
    raise ValueError(f"{name} must be a boolean, got {value!r}.")


def _ensure_deepspeed_all_reduce_wait_compatibility() -> bool:
    """Backport DeepSpeed #7414 for MUSA builds based on DeepSpeed 0.17.2.

    DeepSpeed's ZeRO-3 coordinator calls parameter-fetch handles with the
    ``handle_dependency`` keyword.  In 0.17.2, AllReduceCoalescedHandle.wait
    omitted ``**kwargs`` even though the all-gather handles accepted it.  The
    upstream 0.17.3 fix only adds ``**kwargs`` and intentionally leaves the
    original wait implementation unchanged.

    Returns True only when this process applied the compatibility wrapper.
    """
    from deepspeed.runtime.zero.partition_parameters import (
        AllReduceCoalescedHandle,
    )

    original_wait = AllReduceCoalescedHandle.wait
    compatibility_marker = "_stamo_all_reduce_wait_keyword_compatibility"
    if getattr(original_wait, compatibility_marker, False):
        return False

    # DeepSpeed's instrument_w_nvtx decorator in 0.17.2 does not use
    # functools.wraps.  Its outer wrapper therefore always appears to accept
    # (*args, **kwargs), even when the wrapped wait implementation accepts only
    # (self).  Inspect the callable captured by that wrapper when available.
    wait_implementation = original_wait
    for closure_cell in getattr(original_wait, "__closure__", ()) or ():
        try:
            candidate = closure_cell.cell_contents
        except ValueError:
            continue
        candidate_qualname = getattr(candidate, "__qualname__", "")
        if (
            inspect.isfunction(candidate)
            and candidate_qualname.endswith("AllReduceCoalescedHandle.wait")
        ):
            wait_implementation = candidate
            break

    wait_parameters = inspect.signature(wait_implementation).parameters
    accepts_keyword_arguments = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in wait_parameters.values()
    )
    if accepts_keyword_arguments or "handle_dependency" in wait_parameters:
        return False

    # Match the upstream 0.17.3 behavior: accept and ignore handle-specific
    # keyword hints, while retaining the original decorated wait body.
    def wait_with_keyword_compatibility(self, **kwargs) -> None:
        del kwargs
        return original_wait(self)

    wait_with_keyword_compatibility.__name__ = getattr(
        original_wait,
        "__name__",
        "wait",
    )
    wait_with_keyword_compatibility.__qualname__ = getattr(
        original_wait,
        "__qualname__",
        "AllReduceCoalescedHandle.wait",
    )
    setattr(
        wait_with_keyword_compatibility,
        compatibility_marker,
        True,
    )
    AllReduceCoalescedHandle.wait = wait_with_keyword_compatibility
    return True


def _ensure_deepspeed_zero3_ipg_bucket_reset_compatibility(
    zero_optimizer_cls=None,
) -> bool:
    """Backport DeepSpeed #7418 for ZeRO-3 builds based on 0.17.2.

    DeepSpeed 0.17.2 clears the list of reduced parameters but leaves the
    bucket's element counter unchanged.  After the first reduction, that stale
    counter makes nearly every following parameter trigger an independent
    collective.  Upstream 0.17.3 fixes the issue by setting
    ``bucket.elements = 0`` immediately after ``params_in_bucket.clear()``.

    This process-local wrapper restores the same invariant after the original
    reduction method returns successfully.  It deliberately does not reset
    state when the original method raises, so the real failure remains visible.
    Returns True only when this process installed the compatibility wrapper.
    """
    if zero_optimizer_cls is None:
        from deepspeed.runtime.zero.stage3 import (
            DeepSpeedZeroOptimizer_Stage3,
        )

        zero_optimizer_cls = DeepSpeedZeroOptimizer_Stage3

    method_name = (
        "_DeepSpeedZeroOptimizer_Stage3__reduce_and_partition_ipg_grads"
    )
    original_method = getattr(zero_optimizer_cls, method_name, None)
    compatibility_marker = "_stamo_zero3_ipg_bucket_reset_compatibility"
    if original_method is None:
        raise RuntimeError(
            "Unsupported DeepSpeed ZeRO-3 internals: missing " + method_name
        )
    if getattr(original_method, compatibility_marker, False):
        return False

    # Inspect the whole class instead of the decorated method.  DeepSpeed
    # 0.17.2's NVTX decorator does not preserve __wrapped__, while the class
    # source still exposes the exact upstream fix when it has been backported.
    try:
        class_source = inspect.getsource(zero_optimizer_cls)
    except (OSError, TypeError):
        class_source = ""
    upstream_fix_pattern = (
        r"params_in_bucket\.clear\(\)[ \t]*"
        r"(?:\r?\n[ \t]*(?:#.*\r?\n[ \t]*)?)*"
        r"bucket\.elements[ \t]*=[ \t]*0"
    )
    if re.search(upstream_fix_pattern, class_source):
        return False

    @wraps(original_method)
    def reduce_with_bucket_reset(
        self,
        communication_data_type,
        *args,
        **kwargs,
    ):
        bucket = self.ipg_buckets[communication_data_type]
        had_parameters = bool(bucket.params)
        result = original_method(
            self,
            communication_data_type,
            *args,
            **kwargs,
        )
        bucket = self.ipg_buckets[communication_data_type]
        if bucket.params:
            raise RuntimeError(
                "DeepSpeed ZeRO-3 gradient reduction returned without "
                "clearing its IPG parameter bucket "
                f"({len(bucket.params)} parameter(s) remain)."
            )
        bucket.elements = 0
        if had_parameters:
            reset_count = int(
                getattr(self, "_stamo_zero3_ipg_bucket_reset_count", 0)
            )
            self._stamo_zero3_ipg_bucket_reset_count = reset_count + 1
        return result

    setattr(
        reduce_with_bucket_reset,
        compatibility_marker,
        True,
    )
    setattr(
        zero_optimizer_cls,
        method_name,
        reduce_with_bucket_reset,
    )
    return True


def _assert_deepspeed_zero3_ipg_buckets_cleared(optimizer) -> None:
    """Fail fast if DeepSpeed returns from backward with #7418's stale state."""
    buckets = getattr(optimizer, "ipg_buckets", None)
    if buckets is None:
        raise RuntimeError(
            "Cannot validate ZeRO-3 IPG buckets on this DeepSpeed build."
        )

    uncleared_buckets = []
    for communication_dtype, bucket in buckets.items():
        parameter_count = len(bucket.params)
        element_count = int(bucket.elements)
        if parameter_count != 0 or element_count != 0:
            uncleared_buckets.append(
                f"{communication_dtype}: params={parameter_count}, "
                f"elements={element_count}"
            )
    if uncleared_buckets:
        raise RuntimeError(
            "DeepSpeed ZeRO-3 returned from backward without fully clearing "
            "its IPG buckets; upstream PR #7418 is not active or the private "
            "bucket lifecycle changed: "
            + "; ".join(uncleared_buckets)
        )

    method_name = (
        "_DeepSpeedZeroOptimizer_Stage3__reduce_and_partition_ipg_grads"
    )
    reduce_method = getattr(type(optimizer), method_name, None)
    compatibility_marker = "_stamo_zero3_ipg_bucket_reset_compatibility"
    if (
        getattr(reduce_method, compatibility_marker, False)
        and int(
            getattr(optimizer, "_stamo_zero3_ipg_bucket_reset_count", 0)
        )
        <= 0
    ):
        raise RuntimeError(
            "The DeepSpeed ZeRO-3 #7418 compatibility wrapper was installed "
            "but did not process any non-empty gradient bucket during backward."
        )


def _as_nonnegative_int(value, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}.") from exc
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative, got {parsed}.")
    return parsed


def _build_training_contract(optimizer, lr_scheduler, num_iters: int) -> Dict[str, Any]:
    """Capture settings that DeepSpeed restores instead of re-reading from YAML."""
    if optimizer is None or lr_scheduler is None:
        return {}

    base_lrs = [float(value) for value in getattr(lr_scheduler, "base_lrs", [])]
    parameter_groups = []
    for index, group in enumerate(optimizer.param_groups):
        base_lr = base_lrs[index] if index < len(base_lrs) else float(group["lr"])
        betas = group.get("betas")
        parameter_groups.append(
            {
                "base_lr": float(base_lr),
                "weight_decay": float(group.get("weight_decay", 0.0)),
                "betas": (
                    [float(betas[0]), float(betas[1])]
                    if betas is not None
                    else None
                ),
                "eps": (
                    float(group["eps"])
                    if "eps" in group
                    else None
                ),
                "amsgrad": bool(group.get("amsgrad", False)),
            }
        )

    return {
        "num_iters": int(num_iters),
        "optimizer_class": (
            f"{optimizer.__class__.__module__}.{optimizer.__class__.__qualname__}"
        ),
        "optimizer_groups": parameter_groups,
        "scheduler_class": (
            f"{lr_scheduler.__class__.__module__}."
            f"{lr_scheduler.__class__.__qualname__}"
        ),
        "scheduler_max_iter": int(getattr(lr_scheduler, "max_iter", num_iters)),
        "scheduler_warmup_ratio": float(
            getattr(lr_scheduler, "warmup_ratio", 0.0)
        ),
        "scheduler_min_lr": float(getattr(lr_scheduler, "min_lr", 0.0)),
    }


def _preserve_model_mode(method):
    """Restore the Engine mode even if validation or image saving raises."""

    @wraps(method)
    def wrapped(self, *args, **kwargs):
        was_training = self.model.training
        try:
            return method(self, *args, **kwargs)
        finally:
            self.model.train(was_training)

    return wrapped


class _PhaseTracer:
    """Per-process JSONL trace that never introduces distributed collectives."""

    def __init__(
        self,
        enabled: bool,
        directory: str,
        task_name: str,
        rank: int,
        step_limit: int,
        hang_dump_after_seconds: int,
    ) -> None:
        self.enabled = bool(enabled)
        self.rank = int(rank)
        self.step_limit = max(0, int(step_limit))
        self.file = None
        self.path = None
        self.stack_file = None
        self.stack_path = None
        self._faulthandler_scheduled = False
        if not self.enabled:
            return

        directory = os.path.abspath(os.path.expanduser(str(directory)))
        os.makedirs(directory, exist_ok=True)
        safe_task = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(task_name))
        file_stem = f"{safe_task}_rank{self.rank:02d}_pid{os.getpid()}"
        self.path = os.path.join(directory, file_stem + ".jsonl")
        self.stack_path = os.path.join(directory, file_stem + ".stacks.log")
        self.file = open(self.path, "a", encoding="utf-8", buffering=1)
        self.stack_file = open(
            self.stack_path,
            "a",
            encoding="utf-8",
            buffering=1,
        )
        faulthandler.enable(file=self.stack_file, all_threads=True)
        hang_dump_after_seconds = int(hang_dump_after_seconds)
        if hang_dump_after_seconds > 0:
            faulthandler.dump_traceback_later(
                hang_dump_after_seconds,
                repeat=True,
                file=self.stack_file,
            )
            self._faulthandler_scheduled = True
        self.emit(
            "TRACE_OPEN",
            step=-1,
            trace_path=self.path,
            stack_path=self.stack_path,
        )
        print(
            f"[rank {self.rank}] phase trace: {self.path}; "
            f"stacks: {self.stack_path}",
            flush=True,
        )

    def emit(self, phase: str, step: int, **details) -> None:
        if not self.enabled or self.file is None:
            return
        step = int(step)
        if step >= 0 and self.step_limit > 0 and step >= self.step_limit:
            return
        record = {
            "wall_time_ns": time.time_ns(),
            "monotonic_ns": time.monotonic_ns(),
            "pid": os.getpid(),
            "rank": self.rank,
            "step": step,
            "phase": str(phase),
        }
        record.update(details)
        try:
            self.file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            self.file.flush()
        except OSError as exc:
            print(
                f"[rank {self.rank}] phase trace write failed: {exc!r}",
                flush=True,
            )

    def close(self) -> None:
        if self.file is None:
            return
        try:
            self.emit("TRACE_CLOSE", step=-1)
            if self._faulthandler_scheduled:
                faulthandler.cancel_dump_traceback_later()
            faulthandler.disable()
        finally:
            try:
                self.file.close()
            finally:
                self.file = None
                if self.stack_file is not None:
                    self.stack_file.close()
                    self.stack_file = None


class Trainer:
    def __init__(self, args, model: RenderNet, criterion=None, optimizer=None, lr_scheduler=None) -> None:
        self.model: RenderNet = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.rank = DIST.get_rank() if DIST.is_initialized() else 0
        self.world_size = DIST.get_world_size() if DIST.is_initialized() else 1
        expected_rank = int(os.environ.get("RANK", self.rank))
        expected_world_size = int(os.environ.get("WORLD_SIZE", self.world_size))
        if self.rank != expected_rank or self.world_size != expected_world_size:
            raise RuntimeError(
                "Trainer distributed state disagrees with the launcher: "
                f"group={self.rank}/{self.world_size}, "
                f"environment={expected_rank}/{expected_world_size}."
            )
        runtime_rank = self.local_rank if self.world_size > 1 else 0
        self.device = _get_runtime_device(runtime_rank)

        self.epoch = -1
        self.global_step = -1
        self.consumed_local_samples = 0

        self.eval_before_train = False

        self.use_deepspeed = bool(getattr(args, "deepspeed", False))
        self.use_fabric = bool(getattr(args, "fabric", False)) and not self.use_deepspeed

        mixed_precision = str(args.train.get("mixed_precision", "no")).strip().lower()
        if mixed_precision in {"none", "false", "fp32", "float32"}:
            mixed_precision = "no"
        elif mixed_precision in {"float16", "half"}:
            mixed_precision = "fp16"
        elif mixed_precision in {"bfloat16"}:
            mixed_precision = "bf16"
        if mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError(
                "train.mixed_precision must be one of no/fp16/bf16, "
                f"got {mixed_precision!r}."
            )
        self.mixed_precision = mixed_precision

        # One precision setting is the source of truth for both model execution
        # and the generated DeepSpeed config.
        self.deepspeed_fp16 = self.mixed_precision == "fp16"
        self.deepspeed_bf16 = self.mixed_precision == "bf16"

        self.train_transformer = bool(getattr(model, "train_transformer", False))
        if self.train_transformer and "deepspeed_zero_stage" not in args.train:
            raise ValueError(
                "Full FLUX training requires an explicit "
                "train.deepspeed_zero_stage=3."
            )
        zero_stage_value = args.train.get(
            "deepspeed_zero_stage",
            3 if self.train_transformer else 0,
        )
        if str(zero_stage_value).strip().lower() == "auto":
            raise ValueError(
                "train.deepspeed_zero_stage must be explicit; use 3 for the "
                "full-parameter FLUX configuration."
            )
        self.deepspeed_zero_stage = int(zero_stage_value)
        if self.deepspeed_zero_stage not in {0, 1, 2, 3}:
            raise ValueError(
                "train.deepspeed_zero_stage must be 0, 1, 2, or 3; "
                f"got {zero_stage_value!r}."
            )
        self.save_renderer_export = _as_bool(
            args.train.get(
                "save_renderer_export",
                not self.train_transformer,
            ),
            "train.save_renderer_export",
        )
        self.run_final_eval = _as_bool(
            args.train.get("run_final_eval", True),
            "train.run_final_eval",
        )
        self.enable_checkpointing = _as_bool(
            args.train.get("enable_checkpointing", True),
            "train.enable_checkpointing",
        )
        self.deepspeed_exclude_frozen_parameters = bool(
            args.train.get("deepspeed_exclude_frozen_parameters", True)
        )
        self.gradient_clipping = float(args.train.get("gradient_clipping", 1.0))
        if self.gradient_clipping < 0:
            raise ValueError("train.gradient_clipping must be non-negative.")
        self.deepspeed_overlap_comm = _as_bool(
            args.train.get("deepspeed_overlap_comm", False),
            "train.deepspeed_overlap_comm",
        )
        self.deepspeed_stage3_use_all_reduce_for_fetch_params = _as_bool(
            args.train.get(
                "deepspeed_stage3_use_all_reduce_for_fetch_params",
                False,
            ),
            "train.deepspeed_stage3_use_all_reduce_for_fetch_params",
        )
        self.deepspeed_reduce_bucket_size = _as_nonnegative_int(
            args.train.get("deepspeed_reduce_bucket_size", 50_000_000),
            "train.deepspeed_reduce_bucket_size",
        )
        self.deepspeed_allgather_bucket_size = _as_nonnegative_int(
            args.train.get("deepspeed_allgather_bucket_size", 50_000_000),
            "train.deepspeed_allgather_bucket_size",
        )
        self.deepspeed_stage3_prefetch_bucket_size = _as_nonnegative_int(
            args.train.get("deepspeed_stage3_prefetch_bucket_size", 50_000_000),
            "train.deepspeed_stage3_prefetch_bucket_size",
        )
        self.deepspeed_stage3_param_persistence_threshold = _as_nonnegative_int(
            args.train.get("deepspeed_stage3_param_persistence_threshold", 100_000),
            "train.deepspeed_stage3_param_persistence_threshold",
        )
        self.deepspeed_stage3_max_live_parameters = _as_nonnegative_int(
            args.train.get("deepspeed_stage3_max_live_parameters", 1_000_000_000),
            "train.deepspeed_stage3_max_live_parameters",
        )
        self.deepspeed_stage3_max_reuse_distance = _as_nonnegative_int(
            args.train.get("deepspeed_stage3_max_reuse_distance", 1_000_000_000),
            "train.deepspeed_stage3_max_reuse_distance",
        )
        self.deepspeed_offload_optimizer_device = str(
            args.train.get("deepspeed_offload_optimizer_device", "none")
        ).lower()
        self.deepspeed_offload_param_device = str(
            args.train.get("deepspeed_offload_param_device", "none")
        ).lower()
        valid_offload_devices = {"none", "cpu"}
        if self.deepspeed_offload_optimizer_device not in valid_offload_devices:
            raise ValueError(
                "deepspeed_offload_optimizer_device must be none or cpu."
            )
        if self.deepspeed_offload_param_device not in valid_offload_devices:
            raise ValueError("deepspeed_offload_param_device must be none or cpu.")

        if self.train_transformer and self.use_deepspeed:
            if not self.deepspeed_bf16:
                raise ValueError(
                    "Full FLUX transformer training requires DeepSpeed BF16. "
                    "Set train.mixed_precision=bf16."
                )
            if self.deepspeed_zero_stage != 3:
                raise ValueError(
                    "Full FLUX transformer training requires ZeRO stage 3 in "
                    "this 12B, 8-card profile."
                )
        elif self.train_transformer:
            raise ValueError(
                "Full FLUX transformer training on MUSA requires "
                "DeepSpeed. Launch with --deepspeed."
            )
        # Limit preview images written during each validation.
        self.max_eval_images_to_save = max(
            0,
            int(args.train.get("max_eval_images_to_save", 32)),
        )
        # Log less frequently to avoid a MUSA->CPU synchronization on every
        # micro-step. This changes logging frequency only, not optimization.
        self.log_interval = max(
            1,
            int(args.train.get("log_interval", 20)),
        )
        # Save validation previews from every rank into separate directories.
        # No image gather is used, so this adds no extra MCCL traffic.
        self.save_eval_images_all_ranks = bool(
            args.train.get("save_eval_images_all_ranks", False)
        )
        # Limit TensorBoard preview size even if many JPEGs are saved per rank.
        self.tensorboard_eval_images = max(
            1,
            int(args.train.get("tensorboard_eval_images", 5)),
        )
        self.deepspeed_steps_per_print = max(
            1,
            int(args.train.get("deepspeed_steps_per_print", 1000)),
        )
        self.eval_jpeg_quality = min(
            100,
            max(1, int(args.train.get("eval_jpeg_quality", 90))),
        )
        self.save_eval_gt = bool(
            args.train.get("save_eval_gt", True)
        )

        self.resume = args.resume
        self.resume_path = args.resume_path
        self.do_train = args.do_train

        self.num_iters = int(args.train.num_iters)
        self.epochs = int(args.train.epochs)
        self.eval_step = int(args.train.eval_step)
        self.save_step = int(args.train.save_step)
        self.local_batch_size = int(args.train.local_batch_size)
        self.gradient_accumulate_steps = int(
            args.train.gradient_accumulate_steps
        )
        self.iter_per_ep = None

        if self.num_iters <= 0:
            raise ValueError("num_iters must be positive")
        if self.local_batch_size <= 0:
            raise ValueError("local_batch_size must be positive")
        if self.gradient_accumulate_steps <= 0:
            raise ValueError("gradient_accumulate_steps must be positive")

        self.seed = args.seed
        self.task_name = args.task_name
        self.img_size = args.data.img_size
        self.training_contract = _build_training_contract(
            self.optimizer,
            self.lr_scheduler,
            self.num_iters,
        )
        self.training_contract.update(
            {
                "seed": int(self.seed),
                "image_size": int(self.img_size),
                "flip_probability": float(args.data.flip_p),
                "weighting_scheme": str(
                    getattr(model, "weighting_scheme", "none")
                ),
                "guidance_scale": float(
                    getattr(model, "guidance_scale", 0.0)
                ),
                "condition_tokens": int(getattr(model, "num_token", 0)),
                "token_dropout": bool(getattr(model, "token_dropout", False)),
                "train_transformer": bool(self.train_transformer),
            }
        )
        self.log_dir = os.path.join(args.log_dir, args.task_name)
        self.ckpt_save_dir = os.path.join(args.train.ckpt_save_dir, args.task_name)
        phase_trace_enabled = _as_bool(
            args.train.get("phase_trace", False),
            "train.phase_trace",
        )
        self.phase_trace_synchronize = _as_bool(
            args.train.get("phase_trace_synchronize", False),
            "train.phase_trace_synchronize",
        )
        default_phase_dir = os.path.join(
            tempfile.gettempdir(),
            "stamo_phase_trace",
        )
        self.phase_tracer = _PhaseTracer(
            enabled=phase_trace_enabled,
            directory=str(args.train.get("phase_trace_dir", default_phase_dir)),
            task_name=self.task_name,
            rank=self.rank,
            step_limit=int(args.train.get("phase_trace_steps", 10)),
            hang_dump_after_seconds=int(
                args.train.get("hang_dump_after_seconds", 120)
            ),
        )

        if self.rank == 0 and args.do_train:
            ensure_directory(self.log_dir)
            self.writer = SummaryWriter(
                log_dir=self.log_dir,
                comment="STAMO FLUX Renderer",
            )

        overwatch.info(
            "MUSA trainer configuration: "
            f"device={self.device}, mixed_precision={self.mixed_precision}, "
            f"deepspeed={self.use_deepspeed}, zero_stage={self.deepspeed_zero_stage}, "
            f"train_transformer={self.train_transformer}, "
            f"gradient_accumulation={self.gradient_accumulate_steps}."
        )

    def _trace_phase(self, phase: str, **details) -> None:
        engine_global_steps = getattr(self.model, "global_steps", None)
        engine_micro_steps = getattr(self.model, "micro_steps", None)
        self.phase_tracer.emit(
            phase,
            step=max(int(self.global_step), 0),
            engine_global_steps=engine_global_steps,
            engine_micro_steps=engine_micro_steps,
            **details,
        )

    def _diagnostic_synchronize(self) -> None:
        if not self.phase_trace_synchronize:
            return
        musa = getattr(torch, "musa", None)
        if self.device.type == "musa" and musa is not None:
            musa.synchronize()
        elif self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _seed_training_microbatch(self) -> None:
        """Derive each rank's RNG stream from its exact sampler cursor."""
        # Counter-based seeding makes an interrupted/resumed run consume the
        # same flip, dropout, VAE sample, noise, and timestep randomness as an
        # uninterrupted run. Constants are odd 32-bit mixing multipliers.
        seed = (
            int(self.seed)
            + 0x9E3779B1 * int(self.rank)
            + 0x85EBCA77 * int(self.consumed_local_samples)
        ) % (2**32)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        musa = getattr(torch, "musa", None)
        if self.device.type == "musa" and musa is not None:
            musa.manual_seed(seed)

    def close(self) -> None:
        active_progress = getattr(self, "_active_train_pbar", None)
        if active_progress is not None:
            try:
                active_progress.close()
            finally:
                self._active_train_pbar = None
        if hasattr(self, "writer"):
            try:
                self.writer.flush()
                self.writer.close()
            except Exception as exc:
                print(
                    f"[rank {self.rank}] TensorBoard writer close failed: {exc!r}",
                    flush=True,
                )
        self.phase_tracer.close()

    def unwrap_model(self) -> RenderNet:
        """Return the underlying RenderNet without bypassing Engine forward/backward."""
        model = self.model
        visited = set()
        while hasattr(model, "module") and id(model) not in visited:
            visited.add(id(model))
            model = model.module
        return model

    def _set_model_progress_bar(self, disable: bool) -> None:
        self.unwrap_model()._progress_bar_config = {
            "disable": bool(disable),
            "leave": False,
        }

    def move_model_to_cuda(self) -> None:
        self.model.to(self.device)
        if self.optimizer is not None:
            if isinstance(self.optimizer, list):
                for i in range(len(self.optimizer)):
                    self.optimizer[i].load_state_dict(
                        complex_to_device(self.optimizer[i].state_dict(), device=self.device)
                    )
            else:
                self.optimizer.load_state_dict(complex_to_device(self.optimizer.state_dict(), device=self.device))

    def validate_training_model(self) -> None:
        model = self.unwrap_model()
        if hasattr(model, "DiT"):
            first_transformer_parameter = next(model.DiT.parameters())
            model.dtype = first_transformer_parameter.dtype
            model.device = getattr(
                self.model,
                "device",
                first_transformer_parameter.device,
            )
        trainable = [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        if not trainable:
            raise RuntimeError("RenderNet has no trainable parameters after setup.")

        if self.use_deepspeed and self.deepspeed_bf16:
            wrong_dtype = [
                name
                for name, parameter in trainable
                if parameter.is_floating_point()
                and parameter.dtype != torch.bfloat16
            ]
            if wrong_dtype:
                raise RuntimeError(
                    "DeepSpeed BF16 was enabled but trainable parameters were not "
                    f"converted to BF16 (examples: {wrong_dtype[:5]})."
                )

        if hasattr(model, "DiT"):
            transformer_trainable = any(
                parameter.requires_grad for parameter in model.DiT.parameters()
            )
            if transformer_trainable != self.train_transformer:
                raise RuntimeError(
                    "FLUX transformer trainability changed during distributed setup: "
                    f"expected={self.train_transformer}, actual={transformer_trainable}."
                )

    def prepare_dist_model(self) -> None:
        if self.use_deepspeed:
            import deepspeed

            if self.deepspeed_zero_stage == 3:
                bucket_reset_applied = (
                    _ensure_deepspeed_zero3_ipg_bucket_reset_compatibility()
                )
                if self.rank == 0:
                    if bucket_reset_applied:
                        overwatch.warning(
                            "Applied the DeepSpeed ZeRO-3 IPG bucket reset "
                            "fix from upstream PR #7418."
                        )
                    else:
                        overwatch.info(
                            "DeepSpeed ZeRO-3 IPG bucket reset fix #7418 "
                            "is already present."
                        )

            if (
                self.deepspeed_zero_stage == 3
                and self.deepspeed_stage3_use_all_reduce_for_fetch_params
            ):
                compatibility_applied = (
                    _ensure_deepspeed_all_reduce_wait_compatibility()
                )
                if compatibility_applied and self.rank == 0:
                    overwatch.warning(
                        "Applied the DeepSpeed 0.17.2 "
                        "AllReduceCoalescedHandle.wait compatibility fix "
                        "from upstream PR #7414."
                    )

            if self.device.type == "musa":
                from deepspeed.accelerator import get_accelerator

                deepspeed_device = str(get_accelerator().device_name()).lower()
                if not deepspeed_device.startswith("musa"):
                    raise RuntimeError(
                        "DeepSpeed did not select its MUSA accelerator "
                        f"(reported {deepspeed_device!r}). Use the MUSA-enabled "
                        "DeepSpeed build and export DS_ACCELERATOR=musa."
                    )

            ds_config = {
                "train_micro_batch_size_per_gpu": self.local_batch_size,
                "gradient_accumulation_steps": self.gradient_accumulate_steps,
                "train_batch_size": (
                    self.local_batch_size
                    * self.world_size
                    * self.gradient_accumulate_steps
                ),
                "steps_per_print": self.deepspeed_steps_per_print,
                "fp16": {"enabled": self.deepspeed_fp16},
                "bf16": {"enabled": self.deepspeed_bf16},
                "gradient_clipping": self.gradient_clipping,
            }
            zero_config = None
            if self.deepspeed_zero_stage > 0:
                zero_config = {
                    "stage": self.deepspeed_zero_stage,
                    "allgather_partitions": True,
                    "reduce_scatter": self.deepspeed_zero_stage >= 2,
                    "overlap_comm": self.deepspeed_overlap_comm,
                    "contiguous_gradients": True,
                    "reduce_bucket_size": self.deepspeed_reduce_bucket_size,
                    "allgather_bucket_size": self.deepspeed_allgather_bucket_size,
                }
                if self.deepspeed_zero_stage == 3:
                    zero_config.update(
                        {
                            "stage3_prefetch_bucket_size": (
                                self.deepspeed_stage3_prefetch_bucket_size
                            ),
                            "stage3_param_persistence_threshold": (
                                self.deepspeed_stage3_param_persistence_threshold
                            ),
                            "stage3_max_live_parameters": (
                                self.deepspeed_stage3_max_live_parameters
                            ),
                            "stage3_max_reuse_distance": (
                                self.deepspeed_stage3_max_reuse_distance
                            ),
                            "stage3_use_all_reduce_for_fetch_params": (
                                self.deepspeed_stage3_use_all_reduce_for_fetch_params
                            ),
                        }
                    )
            if self.deepspeed_offload_optimizer_device == "cpu":
                if zero_config is None:
                    raise ValueError(
                        "Optimizer offload requires DeepSpeed ZeRO stage 1, 2, or 3."
                    )
                zero_config["offload_optimizer"] = {
                    "device": "cpu",
                    "pin_memory": True,
                }
            if self.deepspeed_offload_param_device == "cpu":
                if self.deepspeed_zero_stage != 3:
                    raise ValueError(
                        "Parameter offload requires DeepSpeed ZeRO stage 3."
                    )
                zero_config["offload_param"] = {
                    "device": "cpu",
                    "pin_memory": True,
                }
            if zero_config is not None:
                ds_config["zero_optimization"] = zero_config
            self._trace_phase(
                "BEFORE_DEEPSPEED_INITIALIZE",
                zero_stage=self.deepspeed_zero_stage,
                overlap_comm=self.deepspeed_overlap_comm,
            )
            self.model, self.optimizer, _, self.lr_scheduler = deepspeed.initialize(
                model=self.model,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                config=ds_config,
            )
            self._trace_phase("AFTER_DEEPSPEED_INITIALIZE")
            if self.device.type == "musa" and DIST.is_initialized():
                distributed_backend = str(DIST.get_backend()).lower()
                if "mccl" not in distributed_backend:
                    raise RuntimeError(
                        "MUSA distributed training requires MCCL, but the active "
                        f"process group backend is {distributed_backend!r}."
                    )
            if self.resume:
                assert os.path.exists(self.resume_path)
                overwatch.warning(f"Resuming from {self.resume_path}")
                self.load_checkpoint(self.resume_path)
            self.validate_training_model()
            if not self.do_train:
                self.model.eval()
            overwatch.info(f"Successfully built models with {get_parameters(self.model)} parameters")
            return

        if self.use_fabric:
            tb = TensorBoardLogger(root_dir=self.log_dir, version=0)
            self.fabric = Fabric(loggers=tb)
            self.model, self.optimizer = self.fabric.setup(
                self.model,
                self.optimizer,
            )

        if self.resume:
            assert os.path.exists(self.resume_path)
            overwatch.warning(f"Resuming from {self.resume_path}")
            self.load_checkpoint(self.resume_path)

        self.validate_training_model()

        if not self.do_train:
            self.model.eval()
        overwatch.info(f"Successfully built models with {get_parameters(self.model)} parameters")

    def forward_step(self, inputs, **kwargs) -> Dict[str, Any]:
        outputs = self.model(inputs, **kwargs)
        return outputs

    def backward_step(self, loss) -> None:
        if self.use_deepspeed:
            self.model.backward(loss)
        elif self.use_fabric:
            self.fabric.backward(loss)
        else:
            loss.backward()

    def prepare_batch(self, batch) -> Dict[str, Any]:
        # Keep raw images in FP32. RenderNet performs the required per-module
        # casts for DINO, the VAE, and the FLUX transformer.
        return move_to_cuda(batch, device=self.device)

    def clip_gradients(self) -> None:
        if self.use_deepspeed or self.gradient_clipping <= 0:
            return
        if self.use_fabric and hasattr(self.fabric, "clip_gradients"):
            self.fabric.clip_gradients(
                self.model,
                self.optimizer,
                max_norm=self.gradient_clipping,
            )
            return
        torch.nn.utils.clip_grad_norm_(
            [
                parameter
                for parameter in self.model.parameters()
                if parameter.requires_grad and parameter.grad is not None
            ],
            max_norm=self.gradient_clipping,
        )

    def step(self, optimizer_idx=-1) -> None:
        if self.use_deepspeed:
            self.model.step()
            return
        if optimizer_idx >= 0 and isinstance(self.optimizer, list):
            optimizer = self.optimizer[optimizer_idx]
        else:
            optimizer = self.optimizer
        optimizer.step()
        try:
            optimizer.zero_grad(set_to_none=True)
        except TypeError:
            optimizer.zero_grad()

    def deepspeed_train_microbatch(self, loss):
        """Run one micro-batch and return (optimizer_updated, engine_global_step)."""
        if not hasattr(self.model, "global_steps"):
            raise RuntimeError(
                "DeepSpeedEngine.global_steps is required for optimizer-step accounting."
            )
        engine_step_before = int(self.model.global_steps)

        self._trace_phase("BEFORE_BACKWARD")
        self.model.backward(loss)
        if self.deepspeed_zero_stage == 3:
            _assert_deepspeed_zero3_ipg_buckets_cleared(self.model.optimizer)
            if self.rank == 0 and not getattr(
                self,
                "_zero3_bucket_validation_logged",
                False,
            ):
                reset_count = int(
                    getattr(
                        self.model.optimizer,
                        "_stamo_zero3_ipg_bucket_reset_count",
                        0,
                    )
                )
                overwatch.info(
                    "Validated ZeRO-3 IPG bucket state after backward "
                    f"(compatibility resets={reset_count})."
                )
                self._zero3_bucket_validation_logged = True
        self._diagnostic_synchronize()
        self._trace_phase("AFTER_BACKWARD")

        self._trace_phase("BEFORE_STEP")
        self.model.step()
        self._diagnostic_synchronize()
        self._trace_phase("AFTER_STEP")

        engine_step_after = int(self.model.global_steps)
        if engine_step_after not in {engine_step_before, engine_step_before + 1}:
            raise RuntimeError(
                "Unexpected DeepSpeed global-step transition: "
                f"{engine_step_before} -> {engine_step_after}."
            )
        return engine_step_after > engine_step_before, engine_step_after

    def reduce_mean(self, v) -> float:
        """Convert a local metric to a Python float without MCCL collectives.

        DeepSpeed still synchronizes gradients. This function is used only
        for logging and deliberately avoids distributed all_reduce calls.
        """
        if not torch.is_tensor(v):
            return float(v)

        value = v.detach().float()
        if value.numel() != 1:
            value = value.mean()
        return float(value.cpu().item())

    def save_checkpoint(self) -> None:
        if self.use_deepspeed:
            engine_step = getattr(self.model, "global_steps", None)
            if engine_step is None or int(engine_step) != int(self.global_step):
                raise RuntimeError(
                    "Refusing to save a checkpoint with inconsistent steps: "
                    f"engine={engine_step}, trainer={self.global_step}."
                )
        save_path = (
            self.ckpt_save_dir
            if self.use_deepspeed
            else os.path.join(self.ckpt_save_dir, str(self.global_step))
        )
        overwatch.warning(f"Saving models to {save_path}")

        if self.use_deepspeed:
            checkpoint_kwargs = {
                "tag": str(self.global_step),
                "client_state": {
                    "global_step": int(self.global_step),
                    "step_unit": "optimizer_step",
                    "gradient_accumulation_steps": int(
                        self.gradient_accumulate_steps
                    ),
                    "local_batch_size": int(self.local_batch_size),
                    "world_size": int(self.world_size),
                    "consumed_local_samples": int(self.consumed_local_samples),
                    "backend": "flux",
                    "checkpoint_format": "stamo_flux_deepspeed_v2",
                    "training_contract": self.training_contract,
                },
            }
            try:
                save_parameters = inspect.signature(
                    self.model.save_checkpoint
                ).parameters
            except (TypeError, ValueError):
                save_parameters = {}
            if "exclude_frozen_parameters" in save_parameters:
                checkpoint_kwargs["exclude_frozen_parameters"] = (
                    self.deepspeed_exclude_frozen_parameters
                )
            self.model.save_checkpoint(save_path, **checkpoint_kwargs)

            # DeepSpeed checkpoints are the authoritative resumable state.
            # Also write the small STAMO-only files used by validate_renderer.py
            # when FLUX is frozen. ZeRO-3 shards parameters, so exporting a
            # normal state_dict from one rank is deliberately not attempted.
            if self.save_renderer_export:
                if self.deepspeed_zero_stage == 3:
                    if self.rank == 0:
                        overwatch.warning(
                            "Skipping renderer export for ZeRO-3. Resume from "
                            "the DeepSpeed checkpoint, or consolidate it with "
                            "the matching DeepSpeed installation first."
                        )
                else:
                    export_path = os.path.join(
                        self.ckpt_save_dir,
                        "renderer_exports",
                        str(self.global_step),
                    )
                    if self.rank == 0:
                        ensure_directory(export_path)
                        self.unwrap_model().save_checkpoint(
                            export_path,
                            self.global_step,
                        )
                        overwatch.info(
                            f"Saved lightweight FLUX renderer export to {export_path}"
                        )
                    if (
                        DIST.is_available()
                        and DIST.is_initialized()
                        and self.world_size > 1
                    ):
                        DIST.barrier()
        elif self.rank == 0:
            ensure_directory(save_path)
            self.unwrap_model().save_checkpoint(save_path, self.global_step)

        if hasattr(self, "writer"):
            self.writer.flush()

    def load_checkpoint(self, load_path) -> None:
        if self.use_deepspeed:
            load_kwargs = {}
            try:
                load_parameters = inspect.signature(
                    self.model.load_checkpoint
                ).parameters
            except (TypeError, ValueError):
                load_parameters = {}
            if "load_module_strict" in load_parameters:
                load_kwargs["load_module_strict"] = not (
                    self.deepspeed_exclude_frozen_parameters
                )

            # Accept either the DeepSpeed checkpoint root (which contains
            # ``latest``) or one explicit tag directory such as ``.../2000``.
            # DeepSpeed itself expects the former plus an optional ``tag``.
            load_dir = os.path.normpath(str(load_path))
            if os.path.isfile(os.path.join(load_dir, "RenderNet.pth")):
                raise ValueError(
                    "resume_path points to a lightweight renderer export. "
                    "DeepSpeed training must resume from ckpts/<task_name> "
                    "or from one of its numeric tag directories."
                )
            explicit_tag = None
            if not os.path.isfile(os.path.join(load_dir, "latest")):
                try:
                    checkpoint_files = os.listdir(load_dir)
                except OSError:
                    checkpoint_files = []
                looks_like_tag_dir = any(
                    filename.endswith(("model_states.pt", "optim_states.pt"))
                    for filename in checkpoint_files
                )
                if looks_like_tag_dir:
                    explicit_tag = os.path.basename(load_dir)
                    load_dir = os.path.dirname(load_dir)
                    if "tag" not in load_parameters:
                        raise RuntimeError(
                            "This DeepSpeed build cannot load an explicit tag "
                            "directory; set resume_path to the checkpoint root."
                        )
                    load_kwargs["tag"] = explicit_tag

            result = self.model.load_checkpoint(load_dir, **load_kwargs)
            loaded_path, client_state = result
            if loaded_path is None:
                raise RuntimeError(
                    "Failed to load DeepSpeed checkpoint: "
                    f"load_dir={load_dir}, tag={explicit_tag!r}."
                )

            if not isinstance(client_state, dict):
                raise RuntimeError(
                    "FLUX checkpoints must contain DeepSpeed client_state."
                )

            required_fields = {
                "backend",
                "checkpoint_format",
                "global_step",
                "step_unit",
                "gradient_accumulation_steps",
                "local_batch_size",
                "world_size",
                "consumed_local_samples",
                "training_contract",
            }
            missing_fields = sorted(required_fields.difference(client_state))
            if missing_fields:
                raise RuntimeError(
                    "FLUX checkpoint client_state is incomplete: "
                    f"missing {missing_fields}."
                )
            if client_state["backend"] != "flux":
                raise ValueError(
                    "Cannot resume a non-FLUX DeepSpeed checkpoint: "
                    f"backend={client_state['backend']!r}."
                )
            if client_state["checkpoint_format"] != "stamo_flux_deepspeed_v2":
                raise ValueError(
                    "Unsupported FLUX checkpoint format: "
                    f"{client_state['checkpoint_format']!r}."
                )
            saved_training_contract = client_state["training_contract"]
            if saved_training_contract != self.training_contract:
                raise ValueError(
                    "The FLUX training configuration changed across resume. "
                    "DeepSpeed restores the old optimizer and LR scheduler, so "
                    "num_iters, learning rates, schedule, seed, image size, and "
                    "flow-conditioning settings must remain identical. "
                    f"checkpoint={saved_training_contract!r}, "
                    f"current={self.training_contract!r}."
                )
            if client_state["step_unit"] != "optimizer_step":
                raise ValueError(
                    "FLUX checkpoint global_step must use optimizer_step units."
                )

            saved_world_size = int(client_state["world_size"])
            saved_gas = int(client_state["gradient_accumulation_steps"])
            saved_batch = int(client_state["local_batch_size"])
            current_world_size = int(self.world_size)
            if saved_world_size != current_world_size:
                raise ValueError(
                    "Changing world size while resuming is unsupported: "
                    f"checkpoint={saved_world_size}, current={current_world_size}."
                )
            if saved_gas != int(self.gradient_accumulate_steps):
                raise ValueError(
                    "Changing gradient_accumulation_steps while resuming is "
                    f"unsupported: checkpoint={saved_gas}, "
                    f"current={self.gradient_accumulate_steps}."
                )
            if saved_batch != int(self.local_batch_size):
                raise ValueError(
                    "Changing local_batch_size while resuming is unsupported: "
                    f"checkpoint={saved_batch}, current={self.local_batch_size}."
                )

            engine_global_step = getattr(self.model, "global_steps", None)
            if engine_global_step is None:
                raise RuntimeError(
                    "DeepSpeedEngine did not restore global_steps."
                )
            engine_global_step = int(engine_global_step)
            client_global_step = int(client_state["global_step"])
            if engine_global_step != client_global_step:
                raise RuntimeError(
                    "DeepSpeed engine/client global step mismatch: "
                    f"engine={engine_global_step}, client={client_global_step}."
                )

            consumed_local_samples = int(
                client_state["consumed_local_samples"]
            )
            expected_consumed = (
                client_global_step * saved_gas * saved_batch
            )
            if consumed_local_samples != expected_consumed:
                raise RuntimeError(
                    "FLUX checkpoint sample cursor is inconsistent with its "
                    f"optimizer step: cursor={consumed_local_samples}, "
                    f"expected={expected_consumed}."
                )
            self.global_step = client_global_step
            self.consumed_local_samples = consumed_local_samples
        else:
            result = self.unwrap_model().load_checkpoint(load_path)
            self.global_step = int(result)
            self.consumed_local_samples = (
                self.global_step
                * self.gradient_accumulate_steps
                * self.local_batch_size
            )

        overwatch.warning(
            f"Resumed at global_step={self.global_step}"
        )

    def setup_model_for_training(self) -> None:
        self._trace_phase("BEFORE_TRAINING_SETUP")
        if self.rank == 0:
            overwatch.warning(f"Existing dirs detected {self.log_dir}")
            ensure_dirname(self.log_dir, override=False)

        self.model.set_trainable_params()
        if not self.use_fabric and not self.use_deepspeed:
            self.move_model_to_cuda()
        self.prepare_dist_model()
        if self.use_deepspeed:
            if not hasattr(self.model, "global_steps"):
                raise RuntimeError("DeepSpeedEngine has no global_steps attribute.")
            engine_step = int(self.model.global_steps)
            if self.global_step < 0:
                self.global_step = engine_step
            elif self.global_step != engine_step:
                raise RuntimeError(
                    "Checkpoint step mismatch after DeepSpeed setup: "
                    f"trainer={self.global_step}, engine={engine_step}."
                )
        self._trace_phase("AFTER_TRAINING_SETUP")

    def train_eval_by_iter(
        self,
        train_loader,
        eval_loader=None,
        use_tqdm=True,
    ) -> None:
        if self.use_fabric:
            train_loader = self.fabric.setup_dataloaders(
                train_loader,
                use_distributed_sampler=False,
            )

        if not self.num_iters:
            overwatch.warning("Skip train & val phase...")
            return

        overwatch.warning("Start train & val phase...")

        val_examples = (
            len(eval_loader.dataset)
            if eval_loader is not None
            else 0
        )
        val_batches = (
            len(eval_loader)
            if eval_loader is not None
            else 0
        )

        overwatch.warning(
            f"Train examples: {len(train_loader.dataset)},\n"
            f"Val examples: {val_examples}, {val_batches}\n"
            f"epochs: {self.epochs}, iters: {self.num_iters},\n"
            f"eval_step: {self.eval_step}, "
            f"save_step: {self.save_step},\n"
            f"log_interval: {self.log_interval},\n"
            f"global_batch_size: "
            f"{self.local_batch_size * self.world_size * self.gradient_accumulate_steps}, "
            f"local_batch_size: {self.local_batch_size}."
        )

        show_progress = bool(
            use_tqdm and self.rank == 0
        )
        train_pbar = tqdm(
            total=self.num_iters,
            disable=not show_progress,
            dynamic_ncols=True,
        )
        self._active_train_pbar = train_pbar
        train_meter = Meter()

        if not isinstance(self.global_step, int):
            raise TypeError(
                f"global_step must be int, got "
                f"{type(self.global_step)!r}: "
                f"{self.global_step!r}"
            )

        if self.global_step > 0:
            train_pbar.update(
                min(self.global_step, self.num_iters)
            )
        else:
            self.global_step = 0

        if (
            self.eval_before_train
            and self.global_step == 0
            and eval_loader is not None
        ):
            eval_meter, eval_time = self.eval_fn(
                eval_loader,
                use_tqdm=use_tqdm,
            )
            if self.rank == 0:
                overwatch.info(
                    f"[Rank {self.rank}] Valid before train. "
                    f"Time: {eval_time}"
                )

        self.model.train()

        log_window_start_time = time.perf_counter()
        log_window_start_step = self.global_step
        micro_step_in_update = 0
        if not self.use_deepspeed and self.optimizer is not None:
            try:
                self.optimizer.zero_grad(set_to_none=True)
            except TypeError:
                self.optimizer.zero_grad()

        while self.global_step < self.num_iters:
            train_iter = iter(train_loader)

            while self.global_step < self.num_iters:
                try:
                    self._trace_phase("BEFORE_DATA")
                    self._seed_training_microbatch()
                    inputs = next(train_iter)
                    image_tensor = inputs.get("images") if isinstance(inputs, dict) else None
                    if not torch.is_tensor(image_tensor) or image_tensor.ndim < 1:
                        raise ValueError(
                            "Every FLUX training batch must contain an images tensor."
                        )
                    microbatch_sample_count = int(image_tensor.shape[0])
                    if microbatch_sample_count != self.local_batch_size:
                        raise RuntimeError(
                            "DeepSpeed requires a fixed train micro-batch size: "
                            f"expected {self.local_batch_size}, got "
                            f"{microbatch_sample_count}. Set drop_last=True."
                        )
                    self._trace_phase(
                        "AFTER_DATA",
                        image_shape=(
                            tuple(image_tensor.shape)
                            if torch.is_tensor(image_tensor)
                            else None
                        ),
                        paths=(
                            list(inputs.get("paths", []))
                            if isinstance(inputs, dict)
                            else []
                        ),
                    )
                except StopIteration:
                    break

                self.epoch = (
                    self.global_step
                    // max(1, int(self.iter_per_ep or 1))
                )

                inputs["epoch"] = self.epoch
                inputs["global_step"] = self.global_step

                if self.use_deepspeed:
                    # DeepSpeed owns gradient accumulation. Every micro-batch
                    # must call backward() and step(), but global_step advances
                    # only when DeepSpeed performs an optimizer update.
                    self._trace_phase("BEFORE_H2D")
                    inputs = self.prepare_batch(inputs)
                    self._diagnostic_synchronize()
                    self._trace_phase("AFTER_H2D")

                    self._trace_phase("BEFORE_FORWARD")
                    outputs = self.forward_step(
                        inputs,
                        criterion=self.criterion,
                    )
                    self._diagnostic_synchronize()
                    self._trace_phase("AFTER_FORWARD")
                    optimizer_updated, engine_step = self.deepspeed_train_microbatch(
                        outputs["loss"]
                    )

                else:
                    micro_step_in_update += 1
                    is_accumulating = (
                        micro_step_in_update
                        % self.gradient_accumulate_steps
                        != 0
                    )

                    sync_context = (
                        self.fabric.no_backward_sync(
                            self.model,
                            enabled=is_accumulating,
                        )
                        if self.use_fabric
                        else nullcontext()
                    )

                    with sync_context:
                        self._trace_phase("BEFORE_H2D")
                        inputs = self.prepare_batch(inputs)
                        self._diagnostic_synchronize()
                        self._trace_phase("AFTER_H2D")

                        self._trace_phase("BEFORE_FORWARD")
                        outputs = self.forward_step(
                            inputs,
                            criterion=self.criterion,
                        )
                        self._diagnostic_synchronize()
                        self._trace_phase("AFTER_FORWARD")

                        self._trace_phase("BEFORE_BACKWARD")
                        self.backward_step(
                            outputs["loss"]
                            / self.gradient_accumulate_steps
                        )
                        self._diagnostic_synchronize()
                        self._trace_phase("AFTER_BACKWARD")

                    if not is_accumulating:
                        self.clip_gradients()
                        self._trace_phase("BEFORE_STEP")
                        self.step()
                        if self.lr_scheduler is not None:
                            self.lr_scheduler.step()
                        self._diagnostic_synchronize()
                        self._trace_phase("AFTER_STEP")
                        micro_step_in_update = 0
                    optimizer_updated = not is_accumulating

                self.consumed_local_samples += microbatch_sample_count

                if not optimizer_updated:
                    continue

                if self.use_deepspeed:
                    expected_step = self.global_step + 1
                    if engine_step != expected_step:
                        raise RuntimeError(
                            "Trainer/DeepSpeed optimizer-step mismatch: "
                            f"trainer expected {expected_step}, engine reported {engine_step}."
                        )
                    completed_step = engine_step
                else:
                    completed_step = self.global_step + 1
                should_log = (
                    completed_step == 1
                    or completed_step % self.log_interval == 0
                )

                metric_and_loss = {}
                if should_log:
                    raw_metric_and_loss = {
                        key: value
                        for key, value in outputs.items()
                        if key.split("_")[0]
                        in {"metric", "loss"}
                    }

                    if self.rank == 0:
                        metric_and_loss = {
                            key: float(
                                value.detach()
                                .float()
                                .mean()
                                .cpu()
                                .item()
                            )
                            if torch.is_tensor(value)
                            else float(value)
                            for key, value
                            in raw_metric_and_loss.items()
                        }

                    if self.rank == 0:
                        train_meter.update(metric_and_loss)

                        elapsed = max(
                            time.perf_counter()
                            - log_window_start_time,
                            1e-9,
                        )
                        window_steps = max(
                            completed_step
                            - log_window_start_step,
                            1,
                        )
                        images_per_second = (
                            window_steps
                            * self.local_batch_size
                            * self.world_size
                            * self.gradient_accumulate_steps
                            / elapsed
                        )

                        if show_progress:
                            train_pbar.set_description(
                                "Metering: "
                                + str(train_meter)
                                + f", {images_per_second:.2f} img/s"
                            )

                        if hasattr(self, "writer"):
                            for key, value in metric_and_loss.items():
                                self.writer.add_scalar(
                                    key,
                                    value,
                                    completed_step,
                                )
                            self.writer.add_scalar(
                                "performance/images_per_second",
                                images_per_second,
                                completed_step,
                            )

                        log_window_start_time = time.perf_counter()
                        log_window_start_step = completed_step

                self._trace_phase("BEFORE_PROGRESS", completed_step=completed_step)
                self.global_step = completed_step
                train_pbar.update(1)
                self._trace_phase("AFTER_PROGRESS")

                if (
                    self.enable_checkpointing
                    and self.save_step > 0
                    and self.global_step
                    % self.save_step
                    == 0
                ):
                    overwatch.warning("Saving model...")
                    self.save_checkpoint()

                if (
                    eval_loader is not None
                    and self.eval_step > 0
                    and self.global_step
                    % self.eval_step
                    == 0
                ):
                    overwatch.warning("Evaluating...")
                    eval_meter, eval_time = self.eval_fn(
                        eval_loader,
                        use_tqdm=use_tqdm,
                    )
                    if self.rank == 0:
                        overwatch.info(
                            f"[Rank {self.rank}] "
                            f"Valid Step: "
                            f"{self.global_step}, "
                            f"Time: {eval_time}"
                        )
                    train_meter = Meter()
                    log_window_start_time = time.perf_counter()
                    log_window_start_step = self.global_step

        train_pbar.close()
        self._active_train_pbar = None

        if (
            self.enable_checkpointing
            and (
                self.save_step <= 0
                or self.global_step
                % self.save_step
                != 0
            )
        ):
            overwatch.warning("Saving model...")
            self.save_checkpoint()

        if (
            self.run_final_eval
            and eval_loader is not None
            and (
                self.eval_step <= 0
                or self.global_step
                % self.eval_step
                != 0
            )
        ):
            overwatch.warning("Evaluating...")
            eval_meter, eval_time = self.eval_fn(
                eval_loader,
                use_tqdm=use_tqdm,
            )
            if self.rank == 0:
                overwatch.info(
                    f"[Rank {self.rank}] Valid Step: "
                    f"{self.global_step}, "
                    f"Time: {eval_time}"
                )

    @_preserve_model_mode
    def eval_fn(self, eval_loader, use_tqdm=True):
        """Run equal-length ZeRO-3 validation on every rank.

        Predictions stay rank-local. One scalar all-reduce verifies that the
        equally padded distributed shards cover the eval set exactly once.
        Rank 0 logs a single grid with ground truth on the first row and the
        corresponding generated images on the second row.
        """
        was_training = self.model.training
        self.model.eval()
        self._set_model_progress_bar(
            disable=not (use_tqdm and self.rank == 0)
        )
        eval_meter = Meter()
        eval_timer = Timer()

        label_previews = []
        pred_previews = []
        local_count = 0
        local_sampler_position = 0
        eval_sampler = getattr(eval_loader, "sampler", None)
        sampler_world_size = int(
            getattr(eval_sampler, "num_replicas", 1)
        )
        sampler_rank = int(getattr(eval_sampler, "rank", 0))
        eval_dataset_size = len(eval_loader.dataset)
        sampler_pads_to_equal_length = (
            sampler_world_size > 1
            and hasattr(eval_sampler, "total_size")
            and not bool(getattr(eval_sampler, "drop_last", False))
        )

        show_progress = bool(
            use_tqdm and self.rank == 0
        )
        iterator = tqdm(
            eval_loader,
            total=len(eval_loader),
            disable=not show_progress,
            dynamic_ncols=True,
        )

        try:
            with torch.no_grad():
                for inputs in iterator:
                    inputs = self.prepare_batch(inputs)
                    outputs = self.forward_step(inputs)

                    label_img = (
                        inputs["images"]
                        .detach()
                        .float()
                        .cpu()
                    )
                    pred_img = self.unwrap_model().inv_vae_transform(
                        outputs["images"]
                    )
                    pred_img = (
                        torch.clamp(pred_img, 0, 1)
                        .detach()
                        .float()
                        .cpu()
                    )

                    batch_count = int(label_img.shape[0])
                    if sampler_pads_to_equal_length:
                        local_positions = torch.arange(
                            local_sampler_position,
                            local_sampler_position + batch_count,
                            dtype=torch.long,
                        )
                        valid_mask = (
                            sampler_rank
                            + local_positions * sampler_world_size
                            < eval_dataset_size
                        )
                    else:
                        valid_mask = torch.ones(
                            batch_count,
                            dtype=torch.bool,
                        )
                    local_sampler_position += batch_count

                    valid_label_img = label_img[valid_mask]
                    valid_pred_img = pred_img[valid_mask]
                    valid_count = int(valid_mask.sum().item())
                    if valid_count > 0:
                        local_count += valid_count

                    remaining_previews = max(
                        self.max_eval_images_to_save
                        - sum(item.shape[0] for item in pred_previews),
                        0,
                    )
                    if remaining_previews > 0:
                        preview_batch = min(remaining_previews, valid_count)
                        if preview_batch > 0:
                            label_previews.append(
                                valid_label_img[:preview_batch]
                            )
                            pred_previews.append(
                                valid_pred_img[:preview_batch]
                            )

            # Every rank executes exactly one fixed-shape reduction after its
            # finite DistributedSampler is exhausted. Metrics are intentionally
            # not computed; this count is retained as a common ZeRO boundary
            # and catches sampler/accounting divergence before training resumes.
            eval_count = torch.tensor(
                [float(local_count)],
                device=self.device,
                dtype=torch.float32,
            )
            if (
                DIST.is_available()
                and DIST.is_initialized()
                and self.world_size > 1
            ):
                DIST.all_reduce(eval_count, op=DIST.ReduceOp.SUM)

            global_count = int(round(float(eval_count.cpu().item())))

            if global_count != eval_dataset_size:
                raise RuntimeError(
                    "Distributed eval sample accounting is inconsistent: "
                    f"counted={global_count}, dataset={eval_dataset_size}."
                )
            if global_count == 0:
                return (
                    eval_meter,
                    eval_timer.elapse(True),
                )

            if pred_previews:
                pred_preview = torch.cat(pred_previews, dim=0)
                label_preview = torch.cat(label_previews, dim=0)
            else:
                empty_preview = torch.empty(
                    (0, 3, self.img_size, self.img_size),
                    dtype=torch.float32,
                )
                pred_preview = empty_preview
                label_preview = empty_preview.clone()

            step_image_path = os.path.join(
                self.log_dir,
                "images",
                str(self.global_step),
            )
            rank_image_path = os.path.join(
                step_image_path,
                f"rank_{self.rank}",
            )

            should_save_rank = (
                self.save_eval_images_all_ranks
                or self.rank == 0
            )
            preview_count = int(pred_preview.shape[0])
            comparison_count = min(
                self.tensorboard_eval_images,
                preview_count,
            )
            comparison_grid = None
            if should_save_rank and comparison_count > 0:
                # Concatenating all GT samples before all predictions and using
                # nrow=comparison_count guarantees exactly two aligned rows:
                # GT on top and the corresponding generated images below.
                comparison_grid = make_grid(
                    torch.cat(
                        [
                            label_preview[:comparison_count],
                            pred_preview[:comparison_count],
                        ],
                        dim=0,
                    ),
                    nrow=comparison_count,
                    padding=2,
                    pad_value=1.0,
                )

            image_save_errors = 0
            preview_directory_ready = should_save_rank
            if should_save_rank:
                try:
                    ensure_directory(rank_image_path)
                except OSError as exc:
                    preview_directory_ready = False
                    overwatch.warning(
                        f"[Rank {self.rank}] Failed to create the eval preview "
                        f"directory {rank_image_path!r}: {exc!r}"
                    )

            if preview_directory_ready:
                toimg = T.ToPILImage()

                if comparison_grid is not None:
                    try:
                        comparison_path = os.path.join(
                            rank_image_path,
                            "gt_top_pred_bottom.jpeg",
                        )
                        toimg(comparison_grid).save(
                            comparison_path,
                            format="JPEG",
                            quality=self.eval_jpeg_quality,
                        )
                    except (OSError, ValueError) as exc:
                        image_save_errors += 1
                        overwatch.warning(
                            f"[Rank {self.rank}] Failed to save the combined "
                            f"GT/prediction preview: {exc!r}"
                        )

                for index in range(preview_count):
                    try:
                        pred_path = os.path.join(
                            rank_image_path,
                            f"{index:04d}_pred.jpeg",
                        )
                        toimg(pred_preview[index]).save(
                            pred_path,
                            format="JPEG",
                            quality=self.eval_jpeg_quality,
                        )

                        if self.save_eval_gt:
                            gt_path = os.path.join(
                                rank_image_path,
                                f"{index:04d}_gt.jpeg",
                            )
                            toimg(label_preview[index]).save(
                                gt_path,
                                format="JPEG",
                                quality=self.eval_jpeg_quality,
                            )
                    except (OSError, ValueError) as exc:
                        # A preview write failure must not destroy a long
                        # training run. Keep evaluating and report the count.
                        image_save_errors += 1
                        if image_save_errors == 1:
                            overwatch.warning(
                                f"[Rank {self.rank}] Failed to save an eval "
                                f"preview: {exc!r}"
                            )

            if self.rank == 0:
                if self.save_eval_images_all_ranks:
                    overwatch.info(
                        "Eval previews are stored under all rank directories: "
                        f"{step_image_path}/rank_0 ... rank_"
                        f"{self.world_size - 1}"
                    )
                else:
                    overwatch.info(
                        f"Eval previews are stored under {rank_image_path}."
                    )

                if hasattr(self, "writer") and comparison_grid is not None:
                    try:
                        self.writer.add_image(
                            "validation/gt_top_pred_bottom",
                            comparison_grid,
                            self.global_step,
                            dataformats="CHW",
                        )
                    except Exception as exc:
                        # TensorBoard is diagnostic output. A writer failure on
                        # rank 0 must not terminate the distributed training run.
                        overwatch.warning(
                            "Failed to write the combined eval preview to "
                            f"TensorBoard: {exc!r}"
                        )

            return (
                eval_meter,
                eval_timer.elapse(True),
            )
        finally:
            self.model.train(was_training)

    @_preserve_model_mode
    def manually_eval(self, images, batch_size=64):
        was_training = self.model.training
        self.model.eval()
        self._set_model_progress_bar(disable=self.rank != 0)

        label_imgs = images
        toimg = T.ToPILImage()
        transforms = T.Compose(
            [T.Resize((self.img_size, self.img_size), interpolation=T.InterpolationMode.BICUBIC), T.ToTensor()]
        )

        image_path = os.path.join(self.log_dir, "images", str(self.global_step))
        ensure_directory(os.path.join(image_path))

        with torch.no_grad():
            for start_idx in range(0, len(images), batch_size):
                end_idx = min(start_idx + batch_size, len(images))
                batch_images = images[start_idx:end_idx]

                tensor_images = torch.stack([transforms(image).to(self.device) for image in batch_images])
                inputs = {"images": tensor_images}

                inputs = self.prepare_batch(inputs)
                outputs = self.forward_step(inputs)

                pred_imgs = self.unwrap_model().inv_vae_transform(outputs["images"])
                pred_imgs = torch.clamp(pred_imgs, 0, 1)

                pred_imgs = [
                    toimg(pred_img.squeeze().detach().float().cpu())
                    for pred_img in pred_imgs
                ]

                for idx, pred_img in enumerate(pred_imgs):
                    pred_img.save(os.path.join(image_path, f"{start_idx + idx}_pred.jpeg"))
                    label_imgs[start_idx + idx].save(os.path.join(image_path, f"{start_idx + idx}_gt.jpeg"))

        self.model.train(was_training)

    @_preserve_model_mode
    def interpolation_eval(
        self,
        image1,
        image2,
        tokens=None,
        num_interpolation=5,
        to_video=False,
        name="interpolation.mp4",
    ):
        """
        对压缩token进行线性插值
        """
        was_training = self.model.training
        self.model.eval()
        self._set_model_progress_bar(disable=self.rank != 0)

        transforms = T.Compose(
            [T.Resize((self.img_size, self.img_size), interpolation=T.InterpolationMode.BICUBIC), T.ToTensor()]
        )

        with torch.no_grad():
            image1 = transforms(image1).to(self.device).unsqueeze(0)
            image2 = transforms(image2).to(self.device).unsqueeze(0)

            inputs1 = self.prepare_batch(image1)
            inputs2 = self.prepare_batch(image2)

            generator = torch.Generator(device=self.device)
            generator.manual_seed(self.seed)

            outputs = self.unwrap_model().interpolation_eval(
                inputs1,
                inputs2,
                generator,
                tokens=tokens,
                num_interpolation=num_interpolation,
            )

        toimg = T.ToPILImage()

        images = []
        for pred_image in outputs:
            pred_image = self.unwrap_model().inv_vae_transform(pred_image)
            pred_image = torch.clamp(pred_image, 0, 1)
            images.append(toimg(pred_image.detach().float().cpu()))

        if to_video:
            import imageio

            video_path = os.path.join(self.log_dir, "images", str(self.global_step))
            ensure_directory(video_path)
            save_path = os.path.join(video_path, name)
            imageio.mimsave(save_path, images, fps=10)
            self.model.train(was_training)
            return

        image_path = os.path.join(self.log_dir, "images", str(self.global_step))
        ensure_directory(image_path)
        for i in range(len(images)):
            images[i].save(os.path.join(image_path, f"interpolation_{i}.jpeg"))

        widths, heights = zip(*(img.size for img in images))
        total_width = sum(widths)
        max_height = max(heights)

        combined_image = Image.new("RGB", (total_width, max_height))
        x_offset = 0
        for img in images:
            combined_image.paste(img, (x_offset, 0))
            x_offset += img.size[0]

        # 保存拼接后的图像
        combined_image.save(os.path.join(image_path, f"combined_step_{self.global_step}.jpeg"))
        self.model.train(was_training)

    @_preserve_model_mode
    def delta_interpolation(self, image, start, end):
        """
        进行delta插值
        """
        was_training = self.model.training
        self.model.eval()
        self._set_model_progress_bar(disable=self.rank != 0)

        toimg = T.ToPILImage()
        transforms = T.Compose(
            [T.Resize((self.img_size, self.img_size), interpolation=T.InterpolationMode.BICUBIC), T.ToTensor()]
        )
        size = image.size

        with torch.no_grad():
            start_inputs = transforms(start).to(self.device).unsqueeze(0)
            end_inputs = transforms(end).to(self.device).unsqueeze(0)
            image_inputs = transforms(image).to(self.device).unsqueeze(0)

            start_inputs = self.prepare_batch(start_inputs)
            end_inputs = self.prepare_batch(end_inputs)
            image_inputs = self.prepare_batch(image_inputs)

            generator = torch.Generator(device=self.device)
            generator.manual_seed(self.seed)

            outputs = self.unwrap_model().delta_interpolation(
                image_inputs,
                start_inputs,
                end_inputs,
                generator,
            )

        pred_image = self.unwrap_model().inv_vae_transform(outputs).squeeze(0)
        pred_image = torch.clamp(pred_image, 0, 1)
        pred_image = toimg(pred_image.detach().float().cpu())

        image_path = os.path.join(self.log_dir, "images", str(self.global_step))
        ensure_directory(os.path.join(image_path))

        pred_image.save(os.path.join(image_path, f"delta_interpolation_{self.global_step}.jpeg"))

        images = [start.resize(size), end.resize(size), image, pred_image.resize(size)]
        widths, heights = zip(*(img.size for img in images))
        total_width = sum(widths)
        max_height = max(heights)

        combined_image = Image.new("RGB", (total_width, max_height))
        x_offset = 0
        for img in images:
            combined_image.paste(img, (x_offset, 0))
            x_offset += img.size[0]

        combined_image.save(os.path.join(image_path, f"delta_interpolation_combined_{self.global_step}.jpeg"))
        self.model.train(was_training)
