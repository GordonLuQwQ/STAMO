
# import os
# import random


# _DATALOADER_DISTRIBUTED_ENV_KEYS = (
#     "RANK",
#     "WORLD_SIZE",
#     "LOCAL_RANK",
#     "LOCAL_WORLD_SIZE",
#     "MASTER_ADDR",
#     "MASTER_PORT",
#     "GROUP_RANK",
#     "ROLE_RANK",
#     "ROLE_WORLD_SIZE",
# )


# def _isolate_dataloader_spawn_worker_environment() -> None:
#     """Keep a spawned DataLoader worker CPU-only and non-distributed."""
#     os.environ["STAMO_DATALOADER_SPAWN_CHILD"] = "1"
#     os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
#     os.environ.pop("DS_ACCELERATOR", None)
#     for key in _DATALOADER_DISTRIBUTED_ENV_KEYS:
#         os.environ.pop(key, None)


# if __name__ == "__mp_main__":
#     _isolate_dataloader_spawn_worker_environment()


# import numpy as np
# import torch


# def _install_cpu_worker_musa_storage_predicate() -> None:
#     if os.environ.get("STAMO_DATALOADER_SPAWN_CHILD") != "1":
#         return
#     storage_type = getattr(torch, "UntypedStorage", None)
#     if storage_type is not None and not hasattr(storage_type, "is_musa"):
#         storage_type.is_musa = property(
#             lambda storage: getattr(storage.device, "type", None) == "musa"
#         )


# _install_cpu_worker_musa_storage_predicate()


# if __name__ != "__mp_main__":
#     import torch_musa

#     local_rank = int(os.environ.get("LOCAL_RANK", 0))
#     torch.musa.set_device(local_rank)

#     from stamo.renderer.model.renderer import RenderNet
#     from stamo.renderer.trainer import Trainer
#     from stamo.renderer.utils.args import init_args
#     from stamo.renderer.utils.data import get_loader_info, load_multi_datasets_form_json
#     from stamo.renderer.utils.optim import (
#         WarmupLinearConstantLR,
#         WarmupLinearLR,
#         get_criterion,
#         get_optimizer,
#     )
#     from stamo.renderer.utils.overwatch import initialize_overwatch

#     overwatch = initialize_overwatch(__name__)


# torch.multiprocessing.set_sharing_strategy("file_system")

# def set_seed(seed: int) -> None:
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if hasattr(torch, "musa") and torch.musa.is_available():
#         torch.musa.manual_seed_all(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)


# def setup_cuda_device(args) -> None:
#     musa = getattr(torch, "musa", None)
#     if musa is not None and musa.is_available():
#         local_rank = int(getattr(args, "local_rank", os.environ.get("LOCAL_RANK", 0)))
#         device_count = musa.device_count()
#         if local_rank >= device_count:
#             raise RuntimeError(
#                 f"LOCAL_RANK={local_rank} but only {device_count} MUSA device(s) are visible. "
#                 "Check MUSA_VISIBLE_DEVICES and the launch command."
#             )
#         musa.set_device(local_rank)
#         return

#     if not torch.cuda.is_available():
#         overwatch.warning("MUSA/CUDA is not available; falling back to CPU.")
#         return

#     local_rank = int(getattr(args, "local_rank", os.environ.get("LOCAL_RANK", 0)))
#     device_count = torch.cuda.device_count()
#     if local_rank >= device_count:
#         raise RuntimeError(
#             f"LOCAL_RANK={local_rank} but only {device_count} CUDA device(s) are visible. "
#             "Check CUDA_VISIBLE_DEVICES and the launch command."
#         )
#     torch.cuda.set_device(local_rank)

#     torch.backends.cuda.matmul.allow_tf32 = True
#     torch.backends.cudnn.allow_tf32 = True


# def get_warmup_ratio(args) -> float:
#     if "warmup_ratio" in args.train:
#         return float(args.train.warmup_ratio)
#     return float(getattr(args, "warmup_ratio", 0.00001))


# def main(args):
#     set_seed(args.seed)
#     setup_cuda_device(args)

#     # init models
#     overwatch.info("Building models...")
#     model = RenderNet(args)
#     if args.do_train:
#         overwatch.warning("Do training...")
#         model.train()
#         model.set_trainable_params()

#         train_dataloader = load_multi_datasets_form_json(
#             args.data.train_json_path,
#             flip_p=args.data.flip_p,
#             img_size=args.data.img_size,
#             local_batch_size=args.train.local_batch_size,
#             num_workers=args.data.num_workers,
#             is_infinite=True,
#             shuffle=True,
#             make_single_dataset=True,
#             worker_start_method=args.data.worker_start_method,
#         )

#         eval_dataloader = load_multi_datasets_form_json(
#             args.data.eval_json_path,
#             flip_p=0,
#             img_size=args.data.img_size,
#             local_batch_size=args.train.local_batch_size,
#             num_workers=args.data.num_workers,
#             is_infinite=False,
#             shuffle=False,
#             drop_last=False,
#             make_single_dataset=True,
#             worker_start_method=args.data.worker_start_method,
#             eval_mode=args.data.get("eval_mode", None),
#             fast_eval_num_tasks=int(args.data.get("fast_eval_num_tasks", 8)),
#             fast_eval_task_seed=int(args.data.get("fast_eval_task_seed", args.seed)),
#         )

#         train_info = get_loader_info(
#             train_dataloader,
#             args.train.epochs,
#             args.train.local_batch_size,
#             args.train.gradient_accumulate_steps,
#         )
#         _, images_per_batch, args.train.iter_per_ep, args.train.num_iters = train_info

#         projector_params = [
#             parameter
#             for parameter in model.projector.parameters()
#             if parameter.requires_grad
#         ]
#         transformer_params = [
#             parameter
#             for parameter in model.DiT.parameters()
#             if parameter.requires_grad
#         ]

#         if not projector_params:
#             raise RuntimeError("Q-Former has no trainable parameters.")
#         if not transformer_params:
#             raise RuntimeError("FLUX Transformer has no trainable parameters.")

#         projector_param_ids = {id(parameter) for parameter in projector_params}
#         transformer_param_ids = {id(parameter) for parameter in transformer_params}
#         overlap = projector_param_ids & transformer_param_ids
#         if overlap:
#             raise RuntimeError(
#                 "Q-Former and FLUX Transformer optimizer groups overlap."
#             )

#         assigned_param_ids = projector_param_ids | transformer_param_ids
#         unassigned_trainable = [
#             name
#             for name, parameter in model.named_parameters()
#             if parameter.requires_grad and id(parameter) not in assigned_param_ids
#         ]
#         if unassigned_trainable:
#             raise RuntimeError(
#                 "Trainable parameters are missing from the optimizer groups: "
#                 + ", ".join(unassigned_trainable[:20])
#             )

#         projector_lr = float(args.train.learning_rate)
#         transformer_lr = float(
#             args.train.get("full_transformer_learning_rate", projector_lr)
#         )
#         optimizer_groups = [
#             {"params": projector_params, "lr": projector_lr},
#             {"params": transformer_params, "lr": transformer_lr},
#         ]
#         optimizer = get_optimizer(
#             optimizer_groups,
#             opt_type="AdamW",
#             lr=projector_lr,
#             betas=(0.9, 0.98),
#             weight_decay=args.train.decay,
#         )
#         overwatch.info(
#             "Optimizer parameter groups: "
#             f"Q-Former={sum(parameter.numel() for parameter in projector_params):,} "
#             f"parameters at lr={projector_lr:.9g}; "
#             f"FLUX Transformer={sum(parameter.numel() for parameter in transformer_params):,} "
#             f"parameters at lr={transformer_lr:.9g}; "
#             f"weight_decay={float(args.train.decay):.9g}."
#         )

#         criterion = get_criterion(
#             loss_type="diffusion",
#             reduction="mean",
#         )

#         if args.train.constant_lr:
#             scheduler = WarmupLinearConstantLR(
#                 optimizer,
#                 max_iter=args.train.num_iters + 1,
#                 warmup_ratio=get_warmup_ratio(args),
#             )
#         else:
#             scheduler = WarmupLinearLR(
#                 optimizer,
#                 max_iter=args.train.num_iters + 1,
#                 warmup_ratio=get_warmup_ratio(args),
#             )

#         trainer = Trainer(args, model, criterion, optimizer, scheduler)
#         trainer.setup_model_for_training()
#         trainer.iter_per_ep = args.train.iter_per_ep
#         trainer.num_iters = args.train.num_iters
#         if not isinstance(trainer.global_step, int):
#             trainer.global_step = 30000
#         overwatch.info(f"Total batch size {images_per_batch}")
#         overwatch.info(f"Total training steps {args.train.num_iters}")
#         overwatch.info(f"Starting train iter: {trainer.global_step + 1}")
#         overwatch.info(f"Training steps per epoch (accumulated) {args.train.iter_per_ep}")
#         overwatch.info(f"Training dataloader length {len(train_dataloader)}")
#         overwatch.info(f"Evaluation happens every {args.train.eval_step} steps")
#         overwatch.info(f"Checkpoint saves every {args.train.save_step} steps")

#         trainer.train_eval_by_iter(train_loader=train_dataloader, eval_loader=eval_dataloader)


# if __name__ == "__main__":
#     config = init_args()
#     main(config)


# """Standalone trainer for the StaMo FLUX.2 bond path."""

# import os
# import re
# import time
# from typing import Any, Dict

# import torch
# import torch.distributed as DIST
# import torchvision.transforms as T
# from torchvision.utils import make_grid
# from accelerate import Accelerator
# from omegaconf import OmegaConf
# from PIL import Image
# from torch.utils.tensorboard import SummaryWriter
# from tqdm import tqdm

# from stamo.renderer.model.renderer_bond import RenderNetBond
# from stamo.renderer.utils.data_bond import (
#     check_tensor,
#     fp32_to_bf16,
#     fp32_to_fp16,
#     move_to_cuda,
# )
# from stamo.renderer.utils.device import get_accelerator_device
# from stamo.renderer.utils.files import ensure_directory, ensure_dirname
# from stamo.renderer.utils.metrics import Meter, Timer, calculate_psnr, calculate_ssim, get_parameters
# from stamo.renderer.utils.overwatch import initialize_overwatch


# overwatch = initialize_overwatch(__name__)


# class TrainerBond:
#     def __init__(self, args, model: RenderNetBond, criterion=None, optimizer=None, lr_scheduler=None) -> None:
#         self.model: RenderNetBond = model
#         self.criterion = criterion
#         self.optimizer = optimizer
#         self.lr_scheduler = lr_scheduler
#         self.accelerator = None
#         self.writer = None
#         self._deferred_deepspeed_resume = False

#         self.local_rank = overwatch.local_rank()
#         self.rank = overwatch.rank()
#         self.device = get_accelerator_device(self.local_rank if overwatch.world_size() > 1 else None)

#         self.epoch = -1
#         self.global_step = -1
#         self.eval_before_train = False

#         self.resume = args.resume
#         self.resume_path = args.resume_path
#         self.reset_global_step = bool(args.train.get("reset_global_step", False))
#         self.do_train = args.do_train

#         self.num_iters = args.train.num_iters
#         self.epochs = args.train.epochs
#         self.eval_step = args.train.eval_step
#         self.save_step = args.train.save_step
#         self.local_batch_size = args.train.local_batch_size
#         self.gradient_accumulate_steps = args.train.gradient_accumulate_steps
#         self.iter_per_ep = None

#         self.mixed_precision = str(args.train.get("mixed_precision", "bf16")).lower()
#         if self.mixed_precision in {"none", "no", "false"}:
#             self.mixed_precision = "no"
#         if self.mixed_precision not in {"no", "fp16", "bf16"}:
#             raise ValueError(f"Unsupported mixed_precision={self.mixed_precision!r}")

#         self.log_interval = max(1, int(args.train.get("log_interval", 20)))
#         self.max_eval_images_to_save = max(1, int(args.train.get("max_eval_images_to_save", 32)))
#         self.tensorboard_eval_images = max(1, int(args.train.get("tensorboard_eval_images", 16)))
#         self.eval_grid_pairs = max(1, int(args.train.get("eval_grid_pairs", 6)))
#         self.eval_jpeg_quality = min(100, max(1, int(args.train.get("eval_jpeg_quality", 90))))
#         self.save_training_state = bool(args.train.get("save_training_state", False))

#         self.seed = args.seed
#         self.task_name = args.task_name
#         self.img_size = args.data.img_size
#         self.log_dir = os.path.join(args.log_dir, args.task_name)
#         self.ckpt_save_dir = os.path.join(args.train.ckpt_save_dir, args.task_name)

#         if overwatch.is_rank_zero() and args.do_train:
#             ensure_directory(self.log_dir)
#             ensure_directory(self.ckpt_save_dir)

#         OmegaConf.resolve(args)
#         if overwatch.is_rank_zero():
#             ensure_directory(self.ckpt_save_dir)
#             OmegaConf.save(args, os.path.join(self.ckpt_save_dir, "config.yaml"))

#     def unwrap_model(self) -> RenderNet:
#         if self.accelerator is not None:
#             return self.accelerator.unwrap_model(self.model)
#         return self.model.module if hasattr(self.model, "module") else self.model

#     def move_model_to_cuda(self) -> None:
#         self.move_model_to_device()

#     def move_model_to_device(self) -> None:
#         self.model.to(self.device)

#     def prepare_dist_model(self) -> None:
#         self.accelerator = Accelerator(
#             mixed_precision=self.mixed_precision,
#             gradient_accumulation_steps=self.gradient_accumulate_steps,
#         )
#         self.device = self.accelerator.device

#         if self.lr_scheduler is not None:
#             self.accelerator.register_for_checkpointing(self.lr_scheduler)

#         if self.resume:
#             assert os.path.exists(self.resume_path), self.resume_path
#             if os.path.exists(os.path.join(self.resume_path, "RenderNet.pth")):
#                 overwatch.warning(f"Resuming model weights from {self.resume_path}")
#                 self.load_checkpoint(self.resume_path)
#             else:
#                 self._deferred_deepspeed_resume = True
#                 overwatch.warning(
#                     f"No RenderNet.pth found at {self.resume_path}; "
#                     "will try DeepSpeed checkpoint loading after accelerator.prepare()."
#                 )

#         if not self.do_train:
#             self.model.eval()

#         overwatch.info(f"Successfully built models with {get_parameters(self.model)} parameters")

#     def _ensure_tensorboard_writer(self):
#         if self.accelerator is None or not self.accelerator.is_main_process:
#             return None
#         if self.writer is None:
#             purge_step = None
#             if self.resume and self.global_step >= 0:
#                 purge_step = self.global_step + 1
#             ensure_directory(self.log_dir)
#             self.writer = SummaryWriter(
#                 log_dir=self.log_dir,
#                 purge_step=purge_step,
#                 max_queue=10,
#                 flush_secs=30,
#             )
#         return self.writer

#     def _close_tensorboard_writer(self) -> None:
#         if self.writer is not None:
#             self.writer.flush()
#             self.writer.close()
#             self.writer = None

#     def forward_step(self, inputs, **kwargs) -> Dict[str, Any]:
#         return self.model(inputs, **kwargs)

#     def backward_step(self, loss) -> None:
#         if self.accelerator is not None:
#             self.accelerator.backward(loss)
#         else:
#             loss.backward()

#     def prepare_batch(self, batch) -> Dict[str, Any]:
#         batch = move_to_cuda(batch, device=self.device)
#         if self.mixed_precision == "bf16":
#             batch = fp32_to_bf16(batch)
#         elif self.mixed_precision == "fp16":
#             batch = fp32_to_fp16(batch)
#         return batch

#     def step(self) -> None:
#         if self.accelerator is not None and not self.accelerator.sync_gradients:
#             return

#         self.optimizer.step()
#         if self.lr_scheduler is not None:
#             self.lr_scheduler.step()
#         try:
#             self.optimizer.zero_grad(set_to_none=True)
#         except TypeError:
#             self.optimizer.zero_grad()

#     def reduce_mean(self, value) -> float:
#         if not torch.is_tensor(value):
#             return float(value)

#         tensor = value.detach().float()
#         if tensor.numel() != 1:
#             tensor = tensor.mean()
#         tensor = tensor.reshape(1).to(self.device)

#         if self.accelerator is not None and self.accelerator.num_processes > 1:
#             gathered = self.accelerator.gather(tensor)
#             return gathered.float().mean().item()

#         if DIST.is_available() and DIST.is_initialized() and overwatch.world_size() > 1:
#             DIST.all_reduce(tensor, op=DIST.ReduceOp.SUM)
#             tensor /= overwatch.world_size()

#         return tensor.item()

#     def _shared_eval_batches(self, eval_loader) -> int:
#         n_batches = len(eval_loader)
#         if self.accelerator is None or self.accelerator.num_processes <= 1:
#             return n_batches

#         n_batches_tensor = torch.tensor([n_batches], device=self.device, dtype=torch.long)
#         if DIST.is_available() and DIST.is_initialized():
#             DIST.all_reduce(n_batches_tensor, op=DIST.ReduceOp.MIN)
#             return int(n_batches_tensor.item())

#         gathered = self.accelerator.gather(n_batches_tensor)
#         return int(gathered.min().item())

#     @staticmethod
#     def _cpu_state_dict(state_dict):
#         if torch.is_tensor(state_dict):
#             return state_dict.detach().cpu()
#         if isinstance(state_dict, dict):
#             return {k: Trainer._cpu_state_dict(v) for k, v in state_dict.items()}
#         return state_dict

#     def _save_rendernet_checkpoint(self, state_dict: Dict[str, torch.Tensor], save_path: str) -> None:
#         rendernet_dict = {"model": {}, "global_step": int(self.global_step)}
#         projector_dict = {}
#         exclude_prefixes = ("vae.", "projector.", "vision_backbone.")

#         for key, value in state_dict.items():
#             if key.startswith("module."):
#                 key = key[len("module.") :]
#             if key.startswith("projector."):
#                 projector_dict[key[len("projector.") :]] = value
#             elif not key.startswith(exclude_prefixes):
#                 rendernet_dict["model"][key] = value

#         torch.save(self._cpu_state_dict(rendernet_dict), os.path.join(save_path, "RenderNet.pth"))
#         torch.save(self._cpu_state_dict(projector_dict), os.path.join(save_path, "Projector.pth"))

#     def save_checkpoint(self) -> None:
#         save_path = os.path.join(self.ckpt_save_dir, str(self.global_step))
#         overwatch.warning(f"Saving models to {save_path}")

#         if self.accelerator is None:
#             if overwatch.is_rank_zero():
#                 ensure_directory(save_path)
#                 self.unwrap_model().save_checkpoint(save_path, self.global_step)
#             return

#         self.accelerator.wait_for_everyone()
#         full_state_dict = self.accelerator.get_state_dict(self.model)

#         if self.accelerator.is_main_process:
#             ensure_directory(save_path)
#             self._save_rendernet_checkpoint(full_state_dict, save_path)

#         if self.save_training_state:
#             train_state_dir = os.path.join(save_path, "train_state")
#             self.accelerator.save_state(train_state_dir, safe_serialization=False)

#         self.accelerator.wait_for_everyone()

#     def load_checkpoint(self, load_path) -> None:
#         global_step = self.unwrap_model().load_checkpoint(load_path)
#         if self.reset_global_step:
#             overwatch.warning(f"reset_global_step=True: ignoring checkpoint global_step={global_step}")
#             self.global_step = 0
#         else:
#             self.global_step = int(global_step)

#     def _resume_training_state(self) -> None:
#         if self.accelerator is None:
#             return

#         train_state_dir = os.path.join(self.resume_path, "train_state")
#         if os.path.isdir(train_state_dir):
#             self.accelerator.load_state(train_state_dir)
#             overwatch.warning(f"Resumed optimizer/scheduler/RNG state from {train_state_dir}")
#         elif self.save_training_state:
#             overwatch.warning(f"No training state found at {train_state_dir}; model weights were loaded only.")

#     def _resume_deepspeed_checkpoint(self) -> None:
#         if not hasattr(self.model, "load_checkpoint"):
#             raise RuntimeError(
#                 f"{self.resume_path} does not look like a RenderNet checkpoint, "
#                 "and the prepared model cannot load DeepSpeed checkpoints."
#             )

#         result = self.model.load_checkpoint(self.resume_path)
#         if not isinstance(result, tuple):
#             raise RuntimeError(f"Unexpected DeepSpeed load_checkpoint result: {result!r}")

#         loaded_path, client_state = result
#         if loaded_path is None:
#             raise RuntimeError(f"Failed to load DeepSpeed checkpoint from {self.resume_path}")

#         global_step = None
#         if isinstance(client_state, dict):
#             global_step = client_state.get("global_step")

#         if global_step is None:
#             for candidate in [str(loaded_path).rstrip("/\\"), str(self.resume_path).rstrip("/\\")]:
#                 match = re.search(r"(\d+)$", os.path.basename(candidate))
#                 if match:
#                     global_step = int(match.group(1))
#                     break

#         self.global_step = 0 if self.reset_global_step else int(global_step or 0)
#         overwatch.warning(f"Resumed DeepSpeed checkpoint at global_step={self.global_step}")

#     def setup_model_for_training(self) -> None:
#         if overwatch.is_rank_zero():
#             overwatch.warning(f"Existing dirs detected {self.log_dir}")
#             ensure_dirname(self.log_dir, override=False)

#         self.model.set_trainable_params()
#         self.prepare_dist_model()

#     def train_eval_by_iter(self, train_loader, eval_loader=None, use_tqdm=True) -> None:
#         deepspeed_plugin = getattr(self.accelerator, "deepspeed_plugin", None)
#         if deepspeed_plugin is not None and (
#             deepspeed_plugin.is_auto("train_micro_batch_size_per_gpu")
#             or deepspeed_plugin.get_value("train_micro_batch_size_per_gpu") is None
#         ):
#             deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = self.local_batch_size

#         # Data loaders already use rank-aware samplers; do not let Accelerate shard them again.
#         original_broadcast_model = None
#         if self.device.type == "musa" and overwatch.world_size() > 1:
#             from deepspeed.runtime.engine import DeepSpeedEngine

#             original_broadcast_model = DeepSpeedEngine._broadcast_model
#             DeepSpeedEngine._broadcast_model = lambda engine: None
#         try:
#             self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
#         finally:
#             if original_broadcast_model is not None:
#                 DeepSpeedEngine._broadcast_model = original_broadcast_model

#         if self._deferred_deepspeed_resume:
#             self._resume_deepspeed_checkpoint()
#         elif self.resume:
#             self._resume_training_state()

#         if not self.num_iters:
#             overwatch.warning("Skip train & val phase...")
#             return

#         self._ensure_tensorboard_writer()

#         val_examples = len(eval_loader.dataset) if eval_loader is not None else 0
#         val_batches = len(eval_loader) if eval_loader is not None else 0
#         overwatch.warning(
#             f"Start train & val phase...\n"
#             f"Train examples: {len(train_loader.dataset)},\n"
#             f"Val examples: {val_examples}, {val_batches}\n"
#             f"epochs: {self.epochs}, iters: {self.num_iters},\n"
#             f"eval_step: {self.eval_step}, save_step: {self.save_step}, log_interval: {self.log_interval},\n"
#             f"global_batch_size: {self.local_batch_size * overwatch.world_size() * self.gradient_accumulate_steps}, "
#             f"local_batch_size: {self.local_batch_size}."
#         )

#         show_progress = bool(use_tqdm and self.accelerator.is_main_process)
#         train_pbar = tqdm(total=self.num_iters, disable=not show_progress, dynamic_ncols=True)
#         train_meter = Meter()

#         if self.global_step > 0:
#             train_pbar.update(min(self.global_step, self.num_iters))
#         else:
#             self.global_step = 0

#         if self.eval_before_train and self.global_step == 0 and eval_loader is not None:
#             eval_meter, eval_time = self.eval_fn(eval_loader, use_tqdm=use_tqdm)
#             if self.accelerator.is_main_process:
#                 overwatch.info(f"[Rank {self.rank}] Valid before train. Time: {eval_time}\n{eval_meter.avg}")

#         self.model.train()
#         last_log_time = time.perf_counter()
#         last_log_step = self.global_step

#         while self.global_step < self.num_iters:
#             train_iter = iter(train_loader)
#             while self.global_step < self.num_iters:
#                 try:
#                     inputs = next(train_iter)
#                 except StopIteration:
#                     break

#                 self.epoch = (self.global_step + 1) // max(1, int(self.iter_per_ep or 1))
#                 inputs["epoch"] = self.epoch
#                 inputs["global_step"] = self.global_step

#                 with self.accelerator.accumulate(self.model):
#                     inputs = self.prepare_batch(inputs)
#                     check_tensor(inputs, "inputs(prepare_batch)")
#                     outputs = self.forward_step(inputs, criterion=self.criterion)
#                     check_tensor(outputs["loss"], "loss", check_bound=10, check_std=10)
#                     self.backward_step(outputs["loss"])
#                     self.step()

#                 if not self.accelerator.sync_gradients:
#                     continue

#                 self.global_step += 1
#                 train_pbar.update(1)

#                 should_log = self.global_step == 1 or self.global_step % self.log_interval == 0
#                 if should_log:
#                     metric_and_loss = {
#                         k: self.reduce_mean(v)
#                         for k, v in outputs.items()
#                         if k.split("_")[0] in {"metric", "loss"}
#                     }
#                     train_meter.update(metric_and_loss)

#                     elapsed = max(time.perf_counter() - last_log_time, 1e-9)
#                     step_delta = max(self.global_step - last_log_step, 1)
#                     images_per_second = (
#                         step_delta
#                         * self.local_batch_size
#                         * overwatch.world_size()
#                         * self.gradient_accumulate_steps
#                         / elapsed
#                     )

#                     if self.accelerator.is_main_process:
#                         print(
#                             "STAMO_PERF_WINDOW "
#                             f"step={self.global_step} "
#                             f"window_steps={step_delta} "
#                             f"elapsed_seconds={elapsed:.9f} "
#                             f"global_images_per_second={images_per_second:.9f} "
#                             f"loss={metric_and_loss['loss']:.9g}",
#                             flush=True,
#                         )

#                     if show_progress:
#                         train_pbar.set_description(
#                             "Metering: " + str(train_meter)
#                             + f", {images_per_second:.2f} img/s"
#                         )
#                     if self.accelerator.is_main_process:
#                         writer = self._ensure_tensorboard_writer()
#                         for key, value in metric_and_loss.items():
#                             writer.add_scalar(key, value, self.global_step)
#                         writer.add_scalar(
#                             "performance/images_per_second",
#                             images_per_second,
#                             self.global_step,
#                         )
#                     last_log_time = time.perf_counter()
#                     last_log_step = self.global_step

#                 if self.save_step > 0 and self.global_step % self.save_step == 0:
#                     overwatch.warning("Saving model...")
#                     self.save_checkpoint()

#                 if self.eval_step > 0 and self.global_step % self.eval_step == 0:
#                     overwatch.warning("Evaluating...")
#                     if eval_loader is not None:
#                         eval_meter, eval_time = self.eval_fn(eval_loader, use_tqdm=use_tqdm)
#                         if self.accelerator.is_main_process:
#                             overwatch.info(
#                                 f"[Rank {self.rank}] Valid Step: {self.global_step}, "
#                                 f"Time: {eval_time}\n{eval_meter.avg}"
#                             )
#                     train_meter = Meter()
#                     last_log_time = time.perf_counter()
#                     last_log_step = self.global_step

#             if self.global_step >= self.num_iters:
#                 break

#         train_pbar.close()

#         if self.save_step <= 0 or self.global_step % self.save_step != 0:
#             overwatch.warning("Saving model...")
#             self.save_checkpoint()

#         if eval_loader is not None and (self.eval_step <= 0 or self.global_step % self.eval_step != 0):
#             overwatch.warning("Evaluating...")
#             eval_meter, eval_time = self.eval_fn(eval_loader, use_tqdm=use_tqdm)
#             if self.accelerator.is_main_process:
#                 overwatch.info(f"[Rank {self.rank}] Valid Step: {self.global_step}, Time: {eval_time}\n{eval_meter.avg}")

#         self._close_tensorboard_writer()

#     def _set_model_progress_bar(self, disable: bool) -> None:
#         model = self.unwrap_model()
#         model._progress_bar_config = {"disable": disable, "leave": False}

#     def eval_fn(self, eval_loader, use_tqdm=True):
#         self.model.eval()
#         self._set_model_progress_bar(disable=not (use_tqdm and self.accelerator.is_main_process))
#         eval_meter = Meter()
#         eval_timer = Timer()

#         label_imgs = []
#         pred_imgs = []

#         n_batches = self._shared_eval_batches(eval_loader)
#         show_progress = bool(use_tqdm and self.accelerator.is_main_process)
#         iterator = tqdm(eval_loader, total=n_batches, disable=not show_progress, dynamic_ncols=True)

#         with torch.no_grad():
#             for batch_idx, inputs in enumerate(iterator):
#                 if batch_idx >= n_batches:
#                     break
#                 inputs = self.prepare_batch(inputs)
#                 outputs = self.forward_step(inputs)

#                 raw_metric_and_loss = {
#                     k: v for k, v in outputs.items() if k.split("_")[0] in {"metric", "loss"}
#                 }
#                 if raw_metric_and_loss:
#                     eval_meter.update({k: self.reduce_mean(v) for k, v in raw_metric_and_loss.items()})

#                 label_img = inputs["images"].detach().float()
#                 pred_img = self.unwrap_model().inv_vae_transform(outputs["images"])
#                 pred_img = torch.clamp(pred_img, 0, 1).detach().float()

#                 label_imgs.append(label_img.cpu())
#                 pred_imgs.append(pred_img.cpu())

#         if label_imgs and pred_imgs:
#             label_imgs = torch.cat(label_imgs, dim=0)
#             pred_imgs = torch.cat(pred_imgs, dim=0)

#             psnr = calculate_psnr(pred_imgs, label_imgs)
#             ssim = calculate_ssim(pred_imgs, label_imgs)
#             psnr_value = self.reduce_mean(psnr.to(self.device))
#             ssim_value = self.reduce_mean(ssim.to(self.device))

#             eval_meter.update({"validation/psnr": psnr_value, "validation/ssim": ssim_value})

#             if self.accelerator.is_main_process:
#                 overwatch.info(f"PSNR: {psnr_value:.4f}")
#                 overwatch.info(f"SSIM: {ssim_value:.4f}")
#                 writer = self._ensure_tensorboard_writer()
#                 writer.add_scalar(
#                     "validation/psnr",
#                     psnr_value,
#                     self.global_step,
#                 )
#                 writer.add_scalar(
#                     "validation/ssim",
#                     ssim_value,
#                     self.global_step,
#                 )
#                 self._save_eval_images(label_imgs, pred_imgs)
#                 writer.flush()

#         eval_time = eval_timer.elapse(True)
#         self.model.train()
#         return eval_meter, eval_time

#     def _save_eval_images(self, label_imgs: torch.Tensor, pred_imgs: torch.Tensor) -> None:
#         image_path = os.path.join(self.log_dir, "images", str(self.global_step))
#         ensure_directory(image_path)

#         toimg = T.ToPILImage()
#         n_save = min(self.max_eval_images_to_save, pred_imgs.shape[0])
#         for idx in range(n_save):
#             toimg(pred_imgs[idx]).save(
#                 os.path.join(image_path, f"{idx}_pred.jpeg"),
#                 quality=self.eval_jpeg_quality,
#             )
#             toimg(label_imgs[idx]).save(
#                 os.path.join(image_path, f"{idx}_gt.jpeg"),
#                 quality=self.eval_jpeg_quality,
#             )

#         pair_count = min(
#             self.eval_grid_pairs,
#             label_imgs.shape[0],
#             pred_imgs.shape[0],
#         )
#         paired_grid = make_grid(
#             torch.cat(
#                 [label_imgs[:pair_count], pred_imgs[:pair_count]],
#                 dim=0,
#             ),
#             nrow=pair_count,
#             padding=2,
#             pad_value=1.0,
#             normalize=False,
#         )
#         toimg(paired_grid).save(
#             os.path.join(image_path, "gt_top_pred_bottom.jpeg"),
#             quality=self.eval_jpeg_quality,
#         )
#         self.writer.add_image(
#             "validation/gt_top_pred_bottom",
#             paired_grid,
#             self.global_step,
#             dataformats="CHW",
#         )

#     def manually_eval(self, images, batch_size=64):
#         self.model.eval()
#         model = self.unwrap_model()
#         toimg = T.ToPILImage()
#         transforms = T.Compose(
#             [T.Resize((self.img_size, self.img_size), interpolation=T.InterpolationMode.BICUBIC), T.ToTensor()]
#         )

#         image_path = os.path.join(self.log_dir, "images", str(self.global_step))
#         ensure_directory(image_path)

#         with torch.no_grad():
#             for start_idx in range(0, len(images), batch_size):
#                 end_idx = min(start_idx + batch_size, len(images))
#                 batch_images = images[start_idx:end_idx]
#                 tensor_images = torch.stack([transforms(image).to(self.device) for image in batch_images])
#                 inputs = self.prepare_batch({"images": tensor_images})
#                 outputs = self.forward_step(inputs)

#                 pred_imgs = model.inv_vae_transform(outputs["images"])
#                 pred_imgs = torch.clamp(pred_imgs, 0, 1)

#                 overwatch.info(f"PSNR: {calculate_psnr(pred_imgs, tensor_images):.4f}")
#                 overwatch.info(f"SSIM: {calculate_ssim(pred_imgs, tensor_images):.4f}")

#                 pred_imgs = [toimg(pred_img.squeeze().cpu()) for pred_img in pred_imgs]
#                 for idx, pred_img in enumerate(pred_imgs):
#                     pred_img.save(os.path.join(image_path, f"{start_idx + idx}_pred.jpeg"))
#                     batch_images[idx].save(os.path.join(image_path, f"{start_idx + idx}_gt.jpeg"))

#     def interpolation_eval(
#         self,
#         image1,
#         image2,
#         tokens=None,
#         num_interpolation=5,
#         to_video=False,
#         name="interpolation.mp4",
#     ):
#         self.model.eval()
#         model = self.unwrap_model()
#         transforms = T.Compose(
#             [T.Resize((self.img_size, self.img_size), interpolation=T.InterpolationMode.BICUBIC), T.ToTensor()]
#         )

#         with torch.no_grad():
#             image1 = transforms(image1).to(self.device).unsqueeze(0)
#             image2 = transforms(image2).to(self.device).unsqueeze(0)
#             generator = torch.Generator(device=self.device)
#             generator.manual_seed(self.seed)
#             outputs = model.interpolation_eval(
#                 self.prepare_batch(image1),
#                 self.prepare_batch(image2),
#                 generator,
#                 tokens=tokens,
#                 num_interpolation=num_interpolation,
#             )

#         toimg = T.ToPILImage()
#         images = []
#         for pred_image in outputs:
#             pred_image = model.inv_vae_transform(pred_image)
#             pred_image = torch.clamp(pred_image, 0, 1)
#             images.append(toimg(pred_image.cpu()))

#         image_path = os.path.join(self.log_dir, "images", str(self.global_step))
#         ensure_directory(image_path)

#         if to_video:
#             import imageio

#             imageio.mimsave(os.path.join(image_path, name), images, fps=10)
#             return

#         for idx, image in enumerate(images):
#             image.save(os.path.join(image_path, f"interpolation_{idx}.jpeg"))

#         widths, heights = zip(*(img.size for img in images))
#         combined_image = Image.new("RGB", (sum(widths), max(heights)))
#         x_offset = 0
#         for image in images:
#             combined_image.paste(image, (x_offset, 0))
#             x_offset += image.size[0]
#         combined_image.save(os.path.join(image_path, f"combined_step_{self.global_step}.jpeg"))

#     def delta_interpolation(self, image, start, end):
#         self.model.eval()
#         model = self.unwrap_model()

#         toimg = T.ToPILImage()
#         transforms = T.Compose(
#             [T.Resize((self.img_size, self.img_size), interpolation=T.InterpolationMode.BICUBIC), T.ToTensor()]
#         )
#         size = image.size

#         with torch.no_grad():
#             start_inputs = transforms(start).to(self.device).unsqueeze(0)
#             end_inputs = transforms(end).to(self.device).unsqueeze(0)
#             image_inputs = transforms(image).to(self.device).unsqueeze(0)
#             generator = torch.Generator(device=self.device)
#             generator.manual_seed(self.seed)

#             outputs = model.delta_interpolation(
#                 self.prepare_batch(image_inputs),
#                 self.prepare_batch(start_inputs),
#                 self.prepare_batch(end_inputs),
#                 generator,
#             )

#         pred_image = model.inv_vae_transform(outputs).squeeze(0)
#         pred_image = torch.clamp(pred_image, 0, 1)
#         pred_image = toimg(pred_image.cpu())

#         image_path = os.path.join(self.log_dir, "images", str(self.global_step))
#         ensure_directory(image_path)

#         pred_image.save(os.path.join(image_path, f"delta_interpolation_{self.global_step}.jpeg"))

#         images = [start.resize(size), end.resize(size), image, pred_image.resize(size)]
#         widths, heights = zip(*(img.size for img in images))
#         combined_image = Image.new("RGB", (sum(widths), max(heights)))
#         x_offset = 0
#         for img in images:
#             combined_image.paste(img, (x_offset, 0))
#             x_offset += img.size[0]
#         combined_image.save(os.path.join(image_path, f"delta_interpolation_combined_{self.global_step}.jpeg"))


"""Standalone training entry point for StaMo FLUX.2 hand-joint conditioning."""

import os
import random


_DATALOADER_DISTRIBUTED_ENV_KEYS = (
    "RANK",
    "WORLD_SIZE",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "GROUP_RANK",
    "ROLE_RANK",
    "ROLE_WORLD_SIZE",
)


def _isolate_dataloader_spawn_worker_environment() -> None:
    """Keep a spawned DataLoader worker CPU-only and non-distributed."""
    os.environ["STAMO_DATALOADER_SPAWN_CHILD"] = "1"
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    os.environ.pop("DS_ACCELERATOR", None)
    for key in _DATALOADER_DISTRIBUTED_ENV_KEYS:
        os.environ.pop(key, None)


if __name__ == "__mp_main__":
    _isolate_dataloader_spawn_worker_environment()


import numpy as np
import torch


def _install_cpu_worker_musa_storage_predicate() -> None:
    if os.environ.get("STAMO_DATALOADER_SPAWN_CHILD") != "1":
        return
    storage_type = getattr(torch, "UntypedStorage", None)
    if storage_type is not None and not hasattr(storage_type, "is_musa"):
        storage_type.is_musa = property(
            lambda storage: getattr(storage.device, "type", None) == "musa"
        )


_install_cpu_worker_musa_storage_predicate()


if __name__ != "__mp_main__":
    import torch_musa

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.musa.set_device(local_rank)

    from stamo.renderer.model.renderer import RenderNet
    from stamo.renderer.trainer import Trainer
    from stamo.renderer.utils.args import init_args
    from stamo.renderer.utils.data import (
        get_loader_info,
        load_multi_datasets_form_json,
    )
    from stamo.renderer.utils.optim import (
        WarmupLinearConstantLR,
        WarmupLinearLR,
        get_criterion,
        get_optimizer,
    )
    from stamo.renderer.utils.overwatch import initialize_overwatch

    overwatch = initialize_overwatch(__name__)


torch.multiprocessing.set_sharing_strategy("file_system")


def _get(config, key: str, default=None):
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "musa") and torch.musa.is_available():
        torch.musa.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_cuda_device(args) -> None:
    musa = getattr(torch, "musa", None)
    if musa is not None and musa.is_available():
        local_rank = int(getattr(args, "local_rank", os.environ.get("LOCAL_RANK", 0)))
        device_count = musa.device_count()
        if local_rank >= device_count:
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} but only {device_count} MUSA device(s) are visible. "
                "Check MUSA_VISIBLE_DEVICES and the launch command."
            )
        musa.set_device(local_rank)
        return

    if not torch.cuda.is_available():
        overwatch.warning("MUSA/CUDA is not available; falling back to CPU.")
        return

    local_rank = int(getattr(args, "local_rank", os.environ.get("LOCAL_RANK", 0)))
    device_count = torch.cuda.device_count()
    if local_rank >= device_count:
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} but only {device_count} CUDA device(s) are visible. "
            "Check CUDA_VISIBLE_DEVICES and the launch command."
        )
    torch.cuda.set_device(local_rank)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def get_warmup_ratio(args) -> float:
    if "warmup_ratio" in args.train:
        return float(args.train.warmup_ratio)
    return float(getattr(args, "warmup_ratio", 0.00001))


def main(args):
    set_seed(args.seed)
    setup_cuda_device(args)

    pose_config = getattr(args, "pose_condition", None)
    if pose_config is None or not bool(_get(pose_config, "enabled", False)):
        raise ValueError(
            "Hand-pose conditioning requires pose_condition.enabled: true"
        )
    train_pose_sidecar = _get(pose_config, "train_sidecars")
    if train_pose_sidecar is None:
        train_pose_sidecar = _get(pose_config, "train_sidecar")
    eval_pose_sidecar = _get(pose_config, "eval_sidecars")
    if eval_pose_sidecar is None:
        eval_pose_sidecar = _get(pose_config, "eval_sidecar")
    if not train_pose_sidecar:
        raise ValueError(
            "pose_condition.train_sidecar or train_sidecars is required"
        )
    if not eval_pose_sidecar:
        raise ValueError(
            "pose_condition.eval_sidecar or eval_sidecars is required"
        )

    make_single_dataset = bool(
        _get(args.data, "make_single_dataset", True)
    )

    # init models
    overwatch.info("Building models...")
    model = RenderNet(args)
    if args.do_train:
        overwatch.warning("Do training...")
        model.train()
        model.set_trainable_params()

        train_dataloader = load_multi_datasets_form_json(
            args.data.train_json_path,
            flip_p=args.data.flip_p,
            img_size=args.data.img_size,
            local_batch_size=args.train.local_batch_size,
            pose_sidecar_path=train_pose_sidecar,
            num_workers=args.data.num_workers,
            is_infinite=True,
            shuffle=True,
            make_single_dataset=make_single_dataset,
            max_read_attempts=int(
                _get(args.data, "max_image_read_attempts", 8)
            ),
            seed=int(args.seed),
            loader_timeout_seconds=int(
                _get(args.data, "loader_timeout_seconds", 0)
            ),
            persistent_workers=bool(
                _get(args.data, "persistent_workers", False)
            ),
            worker_start_method=_get(args.data, "worker_start_method"),
            prefetch_factor=int(_get(args.data, "prefetch_factor", 2)),
            read_trace_dir=_get(args.data, "read_trace_dir"),
            read_trace_samples=int(
                _get(args.data, "read_trace_samples", 0)
            ),
            pose_flip_swap_hands=bool(
                _get(pose_config, "flip_swap_hands", True)
            ),
            pose_verify_manifest=bool(
                _get(pose_config, "verify_manifest", True)
            ),
        )

        eval_dataloader = load_multi_datasets_form_json(
            args.data.eval_json_path,
            flip_p=0,
            img_size=args.data.img_size,
            local_batch_size=args.train.local_batch_size,
            pose_sidecar_path=eval_pose_sidecar,
            num_workers=int(_get(args.data, "eval_num_workers", 1)),
            is_infinite=False,
            shuffle=False,
            drop_last=False,
            make_single_dataset=make_single_dataset,
            max_read_attempts=int(
                _get(args.data, "max_image_read_attempts", 8)
            ),
            seed=int(args.seed),
            loader_timeout_seconds=int(
                _get(args.data, "loader_timeout_seconds", 0)
            ),
            persistent_workers=bool(
                _get(args.data, "eval_persistent_workers", False)
            ),
            worker_start_method=_get(args.data, "worker_start_method"),
            prefetch_factor=int(
                _get(args.data, "eval_prefetch_factor", 1)
            ),
            read_trace_dir=_get(args.data, "read_trace_dir"),
            read_trace_samples=int(
                _get(args.data, "read_trace_samples", 0)
            ),
            eval_mode=_get(args.data, "eval_mode"),
            fast_eval_num_tasks=int(
                _get(args.data, "fast_eval_num_tasks", 8)
            ),
            fast_eval_task_seed=int(
                _get(args.data, "fast_eval_task_seed", args.seed)
            ),
            pose_flip_swap_hands=bool(
                _get(pose_config, "flip_swap_hands", True)
            ),
            pose_verify_manifest=bool(
                _get(pose_config, "verify_manifest", True)
            ),
        )

        train_info = get_loader_info(
            train_dataloader,
            args.train.epochs,
            args.train.local_batch_size,
            args.train.gradient_accumulate_steps,
        )
        _, images_per_batch, args.train.iter_per_ep, args.train.num_iters = train_info

        projector_params = [
            parameter
            for parameter in model.projector.parameters()
            if parameter.requires_grad
        ]
        transformer_params = [
            parameter
            for parameter in model.DiT.parameters()
            if parameter.requires_grad
        ]

        if not projector_params:
            raise RuntimeError("Q-Former has no trainable parameters.")
        if not transformer_params:
            raise RuntimeError("FLUX Transformer has no trainable parameters.")

        projector_param_ids = {id(parameter) for parameter in projector_params}
        transformer_param_ids = {id(parameter) for parameter in transformer_params}
        overlap = projector_param_ids & transformer_param_ids
        if overlap:
            raise RuntimeError(
                "Q-Former and FLUX Transformer optimizer groups overlap."
            )

        assigned_param_ids = projector_param_ids | transformer_param_ids
        unassigned_trainable = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and id(parameter) not in assigned_param_ids
        ]
        if unassigned_trainable:
            raise RuntimeError(
                "Trainable parameters are missing from the optimizer groups: "
                + ", ".join(unassigned_trainable[:20])
            )

        projector_lr = float(args.train.learning_rate)
        transformer_lr = float(
            args.train.get("full_transformer_learning_rate", projector_lr)
        )
        optimizer_groups = [
            {"params": projector_params, "lr": projector_lr},
            {"params": transformer_params, "lr": transformer_lr},
        ]
        optimizer = get_optimizer(
            optimizer_groups,
            opt_type="AdamW",
            lr=projector_lr,
            betas=(0.9, 0.98),
            weight_decay=args.train.decay,
        )
        overwatch.info(
            "Optimizer parameter groups: "
            f"Q-Former={sum(parameter.numel() for parameter in projector_params):,} "
            f"parameters at lr={projector_lr:.9g}; "
            f"FLUX Transformer={sum(parameter.numel() for parameter in transformer_params):,} "
            f"parameters at lr={transformer_lr:.9g}; "
            f"weight_decay={float(args.train.decay):.9g}."
        )

        criterion = get_criterion(
            loss_type="diffusion",
            reduction="mean",
        )

        if args.train.constant_lr:
            scheduler = WarmupLinearConstantLR(
                optimizer,
                max_iter=args.train.num_iters + 1,
                warmup_ratio=get_warmup_ratio(args),
            )
        else:
            scheduler = WarmupLinearLR(
                optimizer,
                max_iter=args.train.num_iters + 1,
                warmup_ratio=get_warmup_ratio(args),
            )

        trainer = Trainer(args, model, criterion, optimizer, scheduler)
        trainer.setup_model_for_training()
        trainer.iter_per_ep = args.train.iter_per_ep
        trainer.num_iters = args.train.num_iters
        if not isinstance(trainer.global_step, int):
            trainer.global_step = 30000
        overwatch.info(f"Total batch size {images_per_batch}")
        overwatch.info(f"Total training steps {args.train.num_iters}")
        overwatch.info(f"Starting train iter: {trainer.global_step + 1}")
        overwatch.info(f"Training steps per epoch (accumulated) {args.train.iter_per_ep}")
        overwatch.info(f"Training dataloader length {len(train_dataloader)}")
        overwatch.info(f"Evaluation happens every {args.train.eval_step} steps")
        overwatch.info(f"Checkpoint saves every {args.train.save_step} steps")

        trainer.train_eval_by_iter(train_loader=train_dataloader, eval_loader=eval_dataloader)


if __name__ == "__main__":
    config = init_args()
    main(config)
