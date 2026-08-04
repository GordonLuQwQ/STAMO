import os
import random
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist

try:
    import torch_musa  # noqa: F401
except ImportError:
    torch_musa = None


def _bind_launcher_musa_device() -> None:
    """Bind LOCAL_RANK before any helper can initialize distributed state."""
    musa = getattr(torch, "musa", None)
    local_rank_value = os.environ.get("LOCAL_RANK")
    if local_rank_value is None:
        return
    if musa is None or not musa.is_available():
        raise RuntimeError(
            "The DeepSpeed launcher provided LOCAL_RANK, but torch_musa/MUSA "
            "is unavailable."
        )

    local_rank = int(local_rank_value)
    device_count = int(musa.device_count())
    if local_rank < 0 or local_rank >= device_count:
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} but only {device_count} MUSA devices are visible."
        )
    musa.set_device(local_rank)


_bind_launcher_musa_device()


def _initialize_launcher_mccl() -> None:
    """Let DeepSpeed own MCCL initialization before project imports."""
    expected_world_size = int(os.environ.get("WORLD_SIZE", 1))
    if expected_world_size <= 1 or dist.is_initialized():
        return
    if os.environ.get("DS_ACCELERATOR", "").strip().lower() != "musa":
        raise EnvironmentError(
            "Set DS_ACCELERATOR=musa before launching train_renderer.py."
        )

    import deepspeed

    timeout_seconds = int(
        os.environ.get("STAMO_DISTRIBUTED_TIMEOUT_SECONDS", 600)
    )
    if timeout_seconds < 60:
        raise ValueError(
            "STAMO_DISTRIBUTED_TIMEOUT_SECONDS must be at least 60."
        )
    deepspeed.init_distributed(
        dist_backend="mccl",
        timeout=timedelta(seconds=timeout_seconds),
    )


_initialize_launcher_mccl()


from stamo.renderer.model.renderer import RenderNet  # noqa: E402
from stamo.renderer.trainer import Trainer  # noqa: E402
from stamo.renderer.utils.args import init_args  # noqa: E402
from stamo.renderer.utils.data import (  # noqa: E402
    get_loader_info,
    load_multi_datasets_form_json,
)
from stamo.renderer.utils.optim import (  # noqa: E402
    WarmupLinearConstantLR,
    WarmupLinearLR,
    get_criterion,
    get_optimizer,
)
from stamo.renderer.utils.overwatch import initialize_overwatch  # noqa: E402


torch.multiprocessing.set_sharing_strategy("file_system")

overwatch = initialize_overwatch(__name__)


def validate_deepspeed_musa() -> None:
    import deepspeed
    from deepspeed.accelerator import get_accelerator

    device_name = str(get_accelerator().device_name()).lower()
    if not device_name.startswith("musa"):
        raise RuntimeError(
            "DeepSpeed did not select its MUSA accelerator "
            f"(reported {device_name!r}). Check the MUSA DeepSpeed install and "
            "DS_ACCELERATOR=musa before loading FLUX weights."
        )
    overwatch.info(
        f"DeepSpeed {getattr(deepspeed, '__version__', 'unknown')} "
        f"accelerator={device_name}"
    )


def validate_input_paths(args) -> None:
    flux_root = os.path.abspath(str(args.render_net.flux.local_ckpt))
    missing_flux_parts = [
        part
        for part in ("transformer", "vae", "scheduler")
        if not os.path.isdir(os.path.join(flux_root, part))
    ]
    if missing_flux_parts:
        raise FileNotFoundError(
            f"FLUX checkpoint {flux_root!r} is missing Diffusers subdirectories: "
            f"{missing_flux_parts}."
        )

    vision_checkpoint = str(args.vision_backbone.get("local_ckpt", "")).strip()
    if vision_checkpoint and not os.path.isfile(vision_checkpoint):
        raise FileNotFoundError(
            f"DINO vision checkpoint not found: {vision_checkpoint}"
        )

    required_json_paths = [("train", str(args.data.train_json_path))]
    if bool(args.train.get("enable_eval", True)):
        required_json_paths.append(("eval", str(args.data.eval_json_path)))
    for split, json_path in required_json_paths:
        if not os.path.isfile(json_path):
            raise FileNotFoundError(
                f"{split} dataset JSON not found: {os.path.abspath(json_path)}"
            )

    if bool(args.resume) and not os.path.exists(str(args.resume_path)):
        raise FileNotFoundError(
            f"resume_path does not exist: {os.path.abspath(str(args.resume_path))}"
        )


def resolve_eval_options(args):
    """Validate the YAML evaluation mode before loading FLUX weights."""
    eval_mode = str(
        args.data.get("eval_mode", "test_mode")
    ).strip().lower()
    if eval_mode not in {"test_mode", "fast_mode"}:
        raise ValueError(
            "data.eval_mode must be 'test_mode' or 'fast_mode', "
            f"got {eval_mode!r}."
        )

    fast_eval_num_tasks = int(
        args.data.get("fast_eval_num_tasks", 100)
    )
    if fast_eval_num_tasks <= 0:
        raise ValueError("data.fast_eval_num_tasks must be positive.")

    fast_eval_task_seed = int(
        args.data.get("fast_eval_task_seed", args.seed)
    )
    if fast_eval_task_seed < 0:
        raise ValueError("data.fast_eval_task_seed must be non-negative.")

    num_inference_steps = int(args.render_net.num_inference_steps)
    if num_inference_steps <= 0:
        raise ValueError("render_net.num_inference_steps must be positive.")

    return (
        eval_mode,
        fast_eval_num_tasks,
        fast_eval_task_seed,
        num_inference_steps,
    )


def setup_musa_device(args) -> None:
    musa = getattr(torch, "musa", None)
    if musa is None or not musa.is_available():
        raise RuntimeError(
            "torch_musa/MUSA is unavailable. Run this entrypoint inside the "
            "Moore Threads PyTorch environment."
        )

    launcher_local_rank = int(os.environ.get("LOCAL_RANK", 0))
    local_rank_value = getattr(args, "local_rank", launcher_local_rank)
    local_rank = int(local_rank_value)
    if local_rank != launcher_local_rank:
        raise RuntimeError(
            "Local-rank mismatch between parsed arguments and launcher: "
            f"args={local_rank}, environment={launcher_local_rank}."
        )
    device_count = int(musa.device_count())
    if local_rank < 0 or local_rank >= device_count:
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} but only {device_count} MUSA device(s) are visible. "
            "Check MUSA_VISIBLE_DEVICES and the DeepSpeed launch command."
        )
    musa.set_device(local_rank)


def validate_mccl_process_group(args) -> None:
    """Validate the launcher-owned MCCL group and rank/device mapping."""
    expected_world_size = int(os.environ.get("WORLD_SIZE", 1))
    expected_rank = int(os.environ.get("RANK", 0))
    expected_local_rank = int(os.environ.get("LOCAL_RANK", 0))
    timeout_seconds = int(args.train.get("distributed_timeout_seconds", 600))
    if timeout_seconds < 60:
        raise ValueError("train.distributed_timeout_seconds must be at least 60")
    launcher_timeout = int(
        os.environ.get("STAMO_DISTRIBUTED_TIMEOUT_SECONDS", 600)
    )
    if timeout_seconds != launcher_timeout:
        raise ValueError(
            "The YAML and launcher distributed timeouts must match: "
            f"yaml={timeout_seconds}, launcher={launcher_timeout}."
        )
    if expected_world_size > 1 and not dist.is_initialized():
        raise RuntimeError(
            "DeepSpeed did not initialize the MCCL process group before "
            "project modules were imported."
        )

    if dist.is_initialized():
        backend = str(dist.get_backend()).lower()
        if "mccl" not in backend:
            raise RuntimeError(
                "MUSA distributed training requires MCCL, but the active "
                f"process group backend is {backend!r}."
            )
        if int(dist.get_world_size()) != expected_world_size:
            raise RuntimeError(
                "Distributed world-size mismatch: "
                f"environment={expected_world_size}, group={dist.get_world_size()}."
            )
        if int(dist.get_rank()) != expected_rank:
            raise RuntimeError(
                "Distributed rank mismatch: "
                f"environment={expected_rank}, group={dist.get_rank()}."
            )

    musa = getattr(torch, "musa", None)
    if musa is not None and musa.is_available():
        current_device = int(musa.current_device())
        if current_device != expected_local_rank:
            raise RuntimeError(
                "MUSA device/rank mismatch after MCCL initialization: "
                f"LOCAL_RANK={expected_local_rank}, current_device={current_device}."
            )


def validate_distributed_sampler(dataloader, name: str) -> None:
    if not dist.is_initialized() or dist.get_world_size() <= 1:
        return
    sampler = getattr(dataloader, "sampler", None)
    sampler_world_size = getattr(sampler, "num_replicas", None)
    sampler_rank = getattr(sampler, "rank", None)
    if (
        int(sampler_world_size or -1) != int(dist.get_world_size())
        or int(sampler_rank if sampler_rank is not None else -1) != int(dist.get_rank())
    ):
        raise RuntimeError(
            f"{name} sampler was not constructed for the active MCCL rank: "
            f"sampler rank/world={sampler_rank}/{sampler_world_size}, "
            f"process rank/world={dist.get_rank()}/{dist.get_world_size()}."
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    musa = getattr(torch, "musa", None)
    if musa is not None and musa.is_available():
        if hasattr(musa, "manual_seed"):
            musa.manual_seed(seed)


def position_train_sampler(
    train_dataloader,
    consumed_local_samples: int,
) -> None:
    """Start a resumed run from a new deterministic sampler epoch.

    DeepSpeed restores model/optimizer/scheduler state. The checkpoint also
    records the exact number of samples consumed by each rank.
    """
    consumed_local_samples = max(0, int(consumed_local_samples))
    sampler = getattr(train_dataloader, "sampler", None)
    if hasattr(sampler, "set_start_sample"):
        sampler.set_start_sample(consumed_local_samples)
    elif consumed_local_samples > 0:
        overwatch.warning(
            "The train sampler has no resumable sample cursor; model and optimizer "
            "state will resume correctly, but sample order is not exact."
        )


def get_warmup_ratio(args) -> float:
    return float(args.train.get("warmup_ratio", 0.03))


def build_optimizer_groups(model, args):
    """Use separate FLUX/condition LRs and no decay for bias/norm parameters."""
    base_lr = float(args.train.learning_rate)
    transformer_lr = float(args.train.get("transformer_learning_rate", base_lr))
    weight_decay = float(args.train.decay)

    grouped = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        # Projector and DiTConditionHead are the STAMO condition branch and use
        # train.learning_rate. Only FLUX transformer parameters use the lower
        # optional transformer_learning_rate.
        learning_rate = transformer_lr if name.startswith("DiT.") else base_lr
        normalized_name = name.lower()
        no_decay = (
            parameter.ndim < 2
            or name.endswith(".bias")
            or "norm" in normalized_name
        )
        decay = 0.0 if no_decay else weight_decay
        grouped.setdefault((learning_rate, decay), []).append(parameter)

    if not grouped:
        raise RuntimeError("RenderNet has no trainable parameters.")
    return [
        {"params": parameters, "lr": learning_rate, "weight_decay": decay}
        for (learning_rate, decay), parameters in grouped.items()
    ]


def main(args) -> None:
    if os.environ.get("DS_ACCELERATOR", "").strip().lower() != "musa":
        raise EnvironmentError(
            "Set DS_ACCELERATOR=musa before launching DeepSpeed."
        )
    if not bool(args.deepspeed):
        raise ValueError(
            "train_renderer.py must be launched with DeepSpeed and "
            "the --deepspeed flag."
        )

    setup_musa_device(args)
    validate_deepspeed_musa()
    validate_mccl_process_group(args)
    validate_input_paths(args)
    (
        eval_mode,
        fast_eval_num_tasks,
        fast_eval_task_seed,
        num_inference_steps,
    ) = resolve_eval_options(args)
    set_seed(int(args.seed))

    overwatch.info("Building FLUX RenderNet on MUSA...")
    model = RenderNet(args)
    if not args.do_train:
        raise ValueError("This entrypoint is for training; set do_train=true.")

    model.train()
    model.set_trainable_params()

    train_dataloader = load_multi_datasets_form_json(
        args.data.train_json_path,
        flip_p=args.data.flip_p,
        img_size=args.data.img_size,
        local_batch_size=args.train.local_batch_size,
        num_workers=args.data.num_workers,
        is_infinite=True,
        shuffle=True,
        drop_last=True,
        make_single_dataset=True,
        max_read_attempts=int(args.data.get("max_image_read_attempts", 8)),
        seed=int(args.seed),
        loader_timeout_seconds=float(args.data.get("loader_timeout_seconds", 0)),
        persistent_workers=bool(args.data.get("persistent_workers", False)),
        worker_start_method=args.data.get("worker_start_method", None),
        read_trace_dir=args.data.get("read_trace_dir", None),
        read_trace_samples=int(args.data.get("read_trace_samples", 0)),
    )
    validate_distributed_sampler(train_dataloader, "train")
    enable_eval = bool(args.train.get("enable_eval", True))
    eval_dataloader = None
    if enable_eval:
        eval_dataloader = load_multi_datasets_form_json(
            args.data.eval_json_path,
            flip_p=0,
            img_size=args.data.img_size,
            local_batch_size=args.train.local_batch_size,
            num_workers=args.data.num_workers,
            is_infinite=False,
            shuffle=False,
            drop_last=False,
            make_single_dataset=True,
            max_read_attempts=int(args.data.get("max_image_read_attempts", 8)),
            seed=int(args.seed) + 1000003,
            loader_timeout_seconds=float(args.data.get("loader_timeout_seconds", 0)),
            persistent_workers=bool(args.data.get("persistent_workers", False)),
            worker_start_method=args.data.get("worker_start_method", None),
            read_trace_dir=args.data.get("read_trace_dir", None),
            read_trace_samples=int(args.data.get("read_trace_samples", 0)),
            eval_mode=eval_mode,
            fast_eval_num_tasks=fast_eval_num_tasks,
            fast_eval_task_seed=fast_eval_task_seed,
        )
        validate_distributed_sampler(eval_dataloader, "eval")

    _, images_per_batch, iter_per_ep, epoch_num_iters = get_loader_info(
        train_dataloader,
        args.train.epochs,
        args.train.local_batch_size,
        args.train.gradient_accumulate_steps,
    )
    configured_num_iters = int(args.train.get("num_iters", 0))
    args.train.iter_per_ep = max(1, int(iter_per_ep))
    args.train.num_iters = (
        configured_num_iters if configured_num_iters > 0 else max(1, int(epoch_num_iters))
    )

    optimizer = get_optimizer(
        build_optimizer_groups(model, args),
        opt_type="AdamW",
        lr=float(args.train.learning_rate),
        betas=(0.9, 0.98),
        weight_decay=0.0,
    )
    criterion = get_criterion(loss_type="diffusion", reduction="mean")

    scheduler_kwargs = {
        "max_iter": int(args.train.num_iters),
        "min_lr": float(args.train.get("min_learning_rate", 0.0)),
        "warmup_ratio": get_warmup_ratio(args),
    }
    if bool(args.train.constant_lr):
        scheduler = WarmupLinearConstantLR(optimizer, **scheduler_kwargs)
    else:
        scheduler = WarmupLinearLR(optimizer, **scheduler_kwargs)

    trainer = Trainer(args, model, criterion, optimizer, scheduler)
    try:
        trainer.setup_model_for_training()
        position_train_sampler(
            train_dataloader,
            consumed_local_samples=int(trainer.consumed_local_samples),
        )
        # The Trainer derives rank-local RNG state from the exact consumed
        # sample cursor before every micro-batch, including after resume.
        trainer.iter_per_ep = int(args.train.iter_per_ep)
        trainer.num_iters = int(args.train.num_iters)

        overwatch.info(f"Total effective batch size: {images_per_batch}")
        overwatch.info(f"Total optimizer steps: {trainer.num_iters}")
        overwatch.info(
            f"Starting optimizer step: {max(trainer.global_step, 0) + 1}"
        )
        overwatch.info(f"Optimizer steps per epoch: {trainer.iter_per_ep}")
        if enable_eval:
            overwatch.info(
                f"Evaluation every {args.train.eval_step} optimizer steps: "
                f"mode={eval_mode}, tasks="
                f"{len(getattr(eval_dataloader.dataset, 'eval_selected_tasks', ()))}/"
                f"{getattr(eval_dataloader.dataset, 'eval_available_task_count', 'unknown')}, "
                f"images={len(eval_dataloader.dataset)}, "
                f"inference_steps={num_inference_steps}, "
                f"task_seed="
                f"{fast_eval_task_seed if eval_mode == 'fast_mode' else 'n/a'}"
            )
        else:
            overwatch.info("Evaluation is disabled by train.enable_eval=false")
        overwatch.info(
            f"Checkpoint every {args.train.save_step} optimizer steps"
        )

        trainer.train_eval_by_iter(
            train_loader=train_dataloader,
            eval_loader=eval_dataloader,
        )
    finally:
        trainer.close()


if __name__ == "__main__":
    try:
        main(init_args())
    finally:
        # Never add a barrier here: after a rank failure it would only turn the
        # original exception into another distributed hang.
        if dist.is_available() and dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception as exc:
                print(
                    f"Failed to destroy the distributed process group: {exc!r}",
                    flush=True,
                )
