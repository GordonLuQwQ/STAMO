import os
import re
import time
from typing import Any, Dict

import torch
import torch.distributed as DIST
import torchvision.transforms as T
from accelerate import Accelerator
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

from stamo.renderer.model.renderer import RenderNet
from stamo.renderer.utils.data import fp32_to_bf16, fp32_to_fp16, move_to_cuda
from stamo.renderer.utils.device import get_accelerator_device
from stamo.renderer.utils.files import ensure_directory, ensure_dirname
from stamo.renderer.utils.metrics import Meter, Timer, calculate_psnr, calculate_ssim, get_parameters
from stamo.renderer.utils.overwatch import initialize_overwatch


overwatch = initialize_overwatch(__name__)


class Trainer:
    def __init__(self, args, model: RenderNet, criterion=None, optimizer=None, lr_scheduler=None) -> None:
        self.model: RenderNet = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = None
        self._deferred_deepspeed_resume = False

        self.local_rank = overwatch.local_rank()
        self.rank = overwatch.rank()
        self.device = get_accelerator_device(self.local_rank if overwatch.world_size() > 1 else None)

        self.epoch = -1
        self.global_step = -1
        self.eval_before_train = False

        self.resume = args.resume
        self.resume_path = args.resume_path
        self.reset_global_step = bool(args.train.get("reset_global_step", False))
        self.do_train = args.do_train

        self.num_iters = args.train.num_iters
        self.epochs = args.train.epochs
        self.eval_step = args.train.eval_step
        self.save_step = args.train.save_step
        self.local_batch_size = args.train.local_batch_size
        self.gradient_accumulate_steps = args.train.gradient_accumulate_steps
        self.iter_per_ep = None

        self.mixed_precision = str(args.train.get("mixed_precision", "bf16")).lower()
        if self.mixed_precision in {"none", "no", "false"}:
            self.mixed_precision = "no"
        if self.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError(f"Unsupported mixed_precision={self.mixed_precision!r}")

        self.log_interval = max(1, int(args.train.get("log_interval", 20)))
        self.max_eval_images_to_save = max(1, int(args.train.get("max_eval_images_to_save", 32)))
        self.tensorboard_eval_images = max(1, int(args.train.get("tensorboard_eval_images", 16)))
        self.eval_jpeg_quality = min(100, max(1, int(args.train.get("eval_jpeg_quality", 90))))
        self.save_training_state = bool(args.train.get("save_training_state", False))

        self.seed = args.seed
        self.task_name = args.task_name
        self.img_size = args.data.img_size
        self.log_dir = os.path.join(args.log_dir, args.task_name)
        self.ckpt_save_dir = os.path.join(args.train.ckpt_save_dir, args.task_name)

        if overwatch.is_rank_zero() and args.do_train:
            ensure_directory(self.log_dir)
            ensure_directory(self.ckpt_save_dir)

        OmegaConf.resolve(args)
        if overwatch.is_rank_zero():
            ensure_directory(self.ckpt_save_dir)
            OmegaConf.save(args, os.path.join(self.ckpt_save_dir, "config.yaml"))

    def unwrap_model(self) -> RenderNet:
        if self.accelerator is not None:
            return self.accelerator.unwrap_model(self.model)
        return self.model.module if hasattr(self.model, "module") else self.model

    def move_model_to_cuda(self) -> None:
        self.move_model_to_device()

    def move_model_to_device(self) -> None:
        self.model.to(self.device)

    def prepare_dist_model(self) -> None:
        self.accelerator = Accelerator(
            log_with="tensorboard",
            mixed_precision=self.mixed_precision,
            project_dir=self.log_dir,
            gradient_accumulation_steps=self.gradient_accumulate_steps,
        )
        self.accelerator.init_trackers("train")
        self.device = self.accelerator.device

        if self.lr_scheduler is not None:
            self.accelerator.register_for_checkpointing(self.lr_scheduler)

        if self.accelerator.is_main_process:
            self.writer = self.accelerator.get_tracker("tensorboard").writer

        if self.resume:
            assert os.path.exists(self.resume_path), self.resume_path
            if os.path.exists(os.path.join(self.resume_path, "RenderNet.pth")):
                overwatch.warning(f"Resuming model weights from {self.resume_path}")
                self.load_checkpoint(self.resume_path)
            else:
                self._deferred_deepspeed_resume = True
                overwatch.warning(
                    f"No RenderNet.pth found at {self.resume_path}; "
                    "will try DeepSpeed checkpoint loading after accelerator.prepare()."
                )

        if not self.do_train:
            self.model.eval()

        overwatch.info(f"Successfully built models with {get_parameters(self.model)} parameters")

    def forward_step(self, inputs, **kwargs) -> Dict[str, Any]:
        return self.model(inputs, **kwargs)

    def backward_step(self, loss) -> None:
        if self.accelerator is not None:
            self.accelerator.backward(loss)
        else:
            loss.backward()

    def prepare_batch(self, batch) -> Dict[str, Any]:
        batch = move_to_cuda(batch, device=self.device)
        if self.mixed_precision == "bf16":
            batch = fp32_to_bf16(batch)
        elif self.mixed_precision == "fp16":
            batch = fp32_to_fp16(batch)
        return batch

    def step(self) -> None:
        if self.accelerator is not None and not self.accelerator.sync_gradients:
            return

        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        try:
            self.optimizer.zero_grad(set_to_none=True)
        except TypeError:
            self.optimizer.zero_grad()

    def reduce_mean(self, value) -> float:
        if not torch.is_tensor(value):
            return float(value)

        tensor = value.detach().float()
        if tensor.numel() != 1:
            tensor = tensor.mean()
        tensor = tensor.reshape(1).to(self.device)

        if self.accelerator is not None and self.accelerator.num_processes > 1:
            gathered = self.accelerator.gather(tensor)
            return gathered.float().mean().item()

        if DIST.is_available() and DIST.is_initialized() and overwatch.world_size() > 1:
            DIST.all_reduce(tensor, op=DIST.ReduceOp.SUM)
            tensor /= overwatch.world_size()

        return tensor.item()

    def _shared_eval_batches(self, eval_loader) -> int:
        n_batches = len(eval_loader)
        if self.accelerator is None or self.accelerator.num_processes <= 1:
            return n_batches

        n_batches_tensor = torch.tensor([n_batches], device=self.device, dtype=torch.long)
        if DIST.is_available() and DIST.is_initialized():
            DIST.all_reduce(n_batches_tensor, op=DIST.ReduceOp.MIN)
            return int(n_batches_tensor.item())

        gathered = self.accelerator.gather(n_batches_tensor)
        return int(gathered.min().item())

    @staticmethod
    def _cpu_state_dict(state_dict):
        if torch.is_tensor(state_dict):
            return state_dict.detach().cpu()
        if isinstance(state_dict, dict):
            return {k: Trainer._cpu_state_dict(v) for k, v in state_dict.items()}
        return state_dict

    def _save_rendernet_checkpoint(self, state_dict: Dict[str, torch.Tensor], save_path: str) -> None:
        rendernet_dict = {"model": {}, "global_step": int(self.global_step)}
        projector_dict = {}
        exclude_prefixes = ("vae.", "projector.", "vision_backbone.")

        for key, value in state_dict.items():
            if key.startswith("module."):
                key = key[len("module.") :]
            if key.startswith("projector."):
                projector_dict[key[len("projector.") :]] = value
            elif not key.startswith(exclude_prefixes):
                rendernet_dict["model"][key] = value

        torch.save(self._cpu_state_dict(rendernet_dict), os.path.join(save_path, "RenderNet.pth"))
        torch.save(self._cpu_state_dict(projector_dict), os.path.join(save_path, "Projector.pth"))

    def save_checkpoint(self) -> None:
        save_path = os.path.join(self.ckpt_save_dir, str(self.global_step))
        overwatch.warning(f"Saving models to {save_path}")

        if self.accelerator is None:
            if overwatch.is_rank_zero():
                ensure_directory(save_path)
                self.unwrap_model().save_checkpoint(save_path, self.global_step)
            return

        self.accelerator.wait_for_everyone()
        full_state_dict = self.accelerator.get_state_dict(self.model)

        if self.accelerator.is_main_process:
            ensure_directory(save_path)
            self._save_rendernet_checkpoint(full_state_dict, save_path)

        if self.save_training_state:
            train_state_dir = os.path.join(save_path, "train_state")
            self.accelerator.save_state(train_state_dir, safe_serialization=False)

        self.accelerator.wait_for_everyone()

    def load_checkpoint(self, load_path) -> None:
        global_step = self.unwrap_model().load_checkpoint(load_path)
        if self.reset_global_step:
            overwatch.warning(f"reset_global_step=True: ignoring checkpoint global_step={global_step}")
            self.global_step = 0
        else:
            self.global_step = int(global_step)

    def _resume_training_state(self) -> None:
        if self.accelerator is None:
            return

        train_state_dir = os.path.join(self.resume_path, "train_state")
        if os.path.isdir(train_state_dir):
            self.accelerator.load_state(train_state_dir)
            overwatch.warning(f"Resumed optimizer/scheduler/RNG state from {train_state_dir}")
        elif self.save_training_state:
            overwatch.warning(f"No training state found at {train_state_dir}; model weights were loaded only.")

    def _resume_deepspeed_checkpoint(self) -> None:
        if not hasattr(self.model, "load_checkpoint"):
            raise RuntimeError(
                f"{self.resume_path} does not look like a RenderNet checkpoint, "
                "and the prepared model cannot load DeepSpeed checkpoints."
            )

        result = self.model.load_checkpoint(self.resume_path)
        if not isinstance(result, tuple):
            raise RuntimeError(f"Unexpected DeepSpeed load_checkpoint result: {result!r}")

        loaded_path, client_state = result
        if loaded_path is None:
            raise RuntimeError(f"Failed to load DeepSpeed checkpoint from {self.resume_path}")

        global_step = None
        if isinstance(client_state, dict):
            global_step = client_state.get("global_step")

        if global_step is None:
            for candidate in [str(loaded_path).rstrip("/\\"), str(self.resume_path).rstrip("/\\")]:
                match = re.search(r"(\d+)$", os.path.basename(candidate))
                if match:
                    global_step = int(match.group(1))
                    break

        self.global_step = 0 if self.reset_global_step else int(global_step or 0)
        overwatch.warning(f"Resumed DeepSpeed checkpoint at global_step={self.global_step}")

    def setup_model_for_training(self) -> None:
        if overwatch.is_rank_zero():
            overwatch.warning(f"Existing dirs detected {self.log_dir}")
            ensure_dirname(self.log_dir, override=False)

        self.model.set_trainable_params()
        self.prepare_dist_model()

    def train_eval_by_iter(self, train_loader, eval_loader=None, use_tqdm=True) -> None:
        self.model, self.optimizer, train_loader = self.accelerator.prepare(self.model, self.optimizer, train_loader)

        if self._deferred_deepspeed_resume:
            self._resume_deepspeed_checkpoint()
        elif self.resume:
            self._resume_training_state()

        if not self.num_iters:
            overwatch.warning("Skip train & val phase...")
            return

        val_examples = len(eval_loader.dataset) if eval_loader is not None else 0
        val_batches = len(eval_loader) if eval_loader is not None else 0
        overwatch.warning(
            f"Start train & val phase...\n"
            f"Train examples: {len(train_loader.dataset)},\n"
            f"Val examples: {val_examples}, {val_batches}\n"
            f"epochs: {self.epochs}, iters: {self.num_iters},\n"
            f"eval_step: {self.eval_step}, save_step: {self.save_step}, log_interval: {self.log_interval},\n"
            f"global_batch_size: {self.local_batch_size * overwatch.world_size() * self.gradient_accumulate_steps}, "
            f"local_batch_size: {self.local_batch_size}."
        )

        show_progress = bool(use_tqdm and self.accelerator.is_main_process)
        train_pbar = tqdm(total=self.num_iters, disable=not show_progress, dynamic_ncols=True)
        train_meter = Meter()

        if self.global_step > 0:
            train_pbar.update(min(self.global_step, self.num_iters))
        else:
            self.global_step = 0

        if self.eval_before_train and self.global_step == 0 and eval_loader is not None:
            eval_meter, eval_time = self.eval_fn(eval_loader, use_tqdm=use_tqdm)
            if self.accelerator.is_main_process:
                overwatch.info(f"[Rank {self.rank}] Valid before train. Time: {eval_time}\n{eval_meter.avg}")

        self.model.train()
        last_log_time = time.perf_counter()
        last_log_step = self.global_step

        while self.global_step < self.num_iters:
            train_iter = iter(train_loader)
            while self.global_step < self.num_iters:
                try:
                    inputs = next(train_iter)
                except StopIteration:
                    break

                self.epoch = (self.global_step + 1) // max(1, int(self.iter_per_ep or 1))
                inputs["epoch"] = self.epoch
                inputs["global_step"] = self.global_step

                with self.accelerator.accumulate(self.model):
                    inputs = self.prepare_batch(inputs)
                    outputs = self.forward_step(inputs, criterion=self.criterion)
                    self.backward_step(outputs["loss"])
                    self.step()

                if not self.accelerator.sync_gradients:
                    continue

                self.global_step += 1
                train_pbar.update(1)

                should_log = self.global_step == 1 or self.global_step % self.log_interval == 0
                if should_log:
                    metric_and_loss = {
                        k: self.reduce_mean(v)
                        for k, v in outputs.items()
                        if k.split("_")[0] in {"metric", "loss"}
                    }
                    train_meter.update(metric_and_loss)

                    elapsed = max(time.perf_counter() - last_log_time, 1e-9)
                    step_delta = max(self.global_step - last_log_step, 1)
                    images_per_second = (
                        step_delta
                        * self.local_batch_size
                        * overwatch.world_size()
                        * self.gradient_accumulate_steps
                        / elapsed
                    )

                    if show_progress:
                        train_pbar.set_description(
                            "Metering: " + str(train_meter) + f", {images_per_second:.2f} img/s"
                        )

                    if self.accelerator.is_main_process:
                        for key, value in metric_and_loss.items():
                            self.writer.add_scalar(key, value, self.global_step)
                        self.writer.add_scalar("performance/images_per_second", images_per_second, self.global_step)

                    self.accelerator.log(metric_and_loss, step=self.global_step)
                    last_log_time = time.perf_counter()
                    last_log_step = self.global_step

                if self.save_step > 0 and self.global_step % self.save_step == 0:
                    overwatch.warning("Saving model...")
                    self.save_checkpoint()

                if self.eval_step > 0 and self.global_step % self.eval_step == 0:
                    overwatch.warning("Evaluating...")
                    if eval_loader is not None:
                        eval_meter, eval_time = self.eval_fn(eval_loader, use_tqdm=use_tqdm)
                        if self.accelerator.is_main_process:
                            overwatch.info(
                                f"[Rank {self.rank}] Valid Step: {self.global_step}, "
                                f"Time: {eval_time}\n{eval_meter.avg}"
                            )
                    train_meter = Meter()
                    last_log_time = time.perf_counter()
                    last_log_step = self.global_step

            if self.global_step >= self.num_iters:
                break

        train_pbar.close()

        if self.save_step <= 0 or self.global_step % self.save_step != 0:
            overwatch.warning("Saving model...")
            self.save_checkpoint()

        if eval_loader is not None and (self.eval_step <= 0 or self.global_step % self.eval_step != 0):
            overwatch.warning("Evaluating...")
            eval_meter, eval_time = self.eval_fn(eval_loader, use_tqdm=use_tqdm)
            if self.accelerator.is_main_process:
                overwatch.info(f"[Rank {self.rank}] Valid Step: {self.global_step}, Time: {eval_time}\n{eval_meter.avg}")

    def _set_model_progress_bar(self, disable: bool) -> None:
        model = self.unwrap_model()
        model._progress_bar_config = {"disable": disable, "leave": False}

    def eval_fn(self, eval_loader, use_tqdm=True):
        self.model.eval()
        self._set_model_progress_bar(disable=not (use_tqdm and self.accelerator.is_main_process))
        eval_meter = Meter()
        eval_timer = Timer()

        label_imgs = []
        pred_imgs = []

        n_batches = self._shared_eval_batches(eval_loader)
        show_progress = bool(use_tqdm and self.accelerator.is_main_process)
        iterator = tqdm(eval_loader, total=n_batches, disable=not show_progress, dynamic_ncols=True)

        with torch.no_grad():
            for batch_idx, inputs in enumerate(iterator):
                if batch_idx >= n_batches:
                    break
                inputs = self.prepare_batch(inputs)
                outputs = self.forward_step(inputs)

                raw_metric_and_loss = {
                    k: v for k, v in outputs.items() if k.split("_")[0] in {"metric", "loss"}
                }
                if raw_metric_and_loss:
                    eval_meter.update({k: self.reduce_mean(v) for k, v in raw_metric_and_loss.items()})

                label_img = inputs["images"].detach().float()
                pred_img = self.unwrap_model().inv_vae_transform(outputs["images"])
                pred_img = torch.clamp(pred_img, 0, 1).detach().float()

                label_imgs.append(label_img.cpu())
                pred_imgs.append(pred_img.cpu())

        if label_imgs and pred_imgs:
            label_imgs = torch.cat(label_imgs, dim=0)
            pred_imgs = torch.cat(pred_imgs, dim=0)

            psnr = calculate_psnr(pred_imgs, label_imgs)
            ssim = calculate_ssim(pred_imgs, label_imgs)
            psnr_value = self.reduce_mean(psnr.to(self.device))
            ssim_value = self.reduce_mean(ssim.to(self.device))

            eval_meter.update({"validation/psnr": psnr_value, "validation/ssim": ssim_value})

            if self.accelerator.is_main_process:
                overwatch.info(f"PSNR: {psnr_value:.4f}")
                overwatch.info(f"SSIM: {ssim_value:.4f}")
                self.accelerator.log(
                    {"validation/psnr": psnr_value, "validation/ssim": ssim_value},
                    step=self.global_step,
                )
                self._save_eval_images(label_imgs, pred_imgs)

        eval_time = eval_timer.elapse(True)
        self.model.train()
        return eval_meter, eval_time

    def _save_eval_images(self, label_imgs: torch.Tensor, pred_imgs: torch.Tensor) -> None:
        image_path = os.path.join(self.log_dir, "images", str(self.global_step))
        ensure_directory(image_path)

        toimg = T.ToPILImage()
        n_save = min(self.max_eval_images_to_save, pred_imgs.shape[0])
        for idx in range(n_save):
            toimg(pred_imgs[idx]).save(
                os.path.join(image_path, f"{idx}_pred.jpeg"),
                quality=self.eval_jpeg_quality,
            )
            toimg(label_imgs[idx]).save(
                os.path.join(image_path, f"{idx}_gt.jpeg"),
                quality=self.eval_jpeg_quality,
            )

        tb_count = min(self.tensorboard_eval_images, pred_imgs.shape[0])
        self.writer.add_images("validation/pred", pred_imgs[:tb_count], self.global_step, dataformats="NCHW")
        self.writer.add_images("validation/gt", label_imgs[:tb_count], self.global_step, dataformats="NCHW")

    def manually_eval(self, images, batch_size=64):
        self.model.eval()
        model = self.unwrap_model()
        toimg = T.ToPILImage()
        transforms = T.Compose(
            [T.Resize((self.img_size, self.img_size), interpolation=T.InterpolationMode.BICUBIC), T.ToTensor()]
        )

        image_path = os.path.join(self.log_dir, "images", str(self.global_step))
        ensure_directory(image_path)

        with torch.no_grad():
            for start_idx in range(0, len(images), batch_size):
                end_idx = min(start_idx + batch_size, len(images))
                batch_images = images[start_idx:end_idx]
                tensor_images = torch.stack([transforms(image).to(self.device) for image in batch_images])
                inputs = self.prepare_batch({"images": tensor_images})
                outputs = self.forward_step(inputs)

                pred_imgs = model.inv_vae_transform(outputs["images"])
                pred_imgs = torch.clamp(pred_imgs, 0, 1)

                overwatch.info(f"PSNR: {calculate_psnr(pred_imgs, tensor_images):.4f}")
                overwatch.info(f"SSIM: {calculate_ssim(pred_imgs, tensor_images):.4f}")

                pred_imgs = [toimg(pred_img.squeeze().cpu()) for pred_img in pred_imgs]
                for idx, pred_img in enumerate(pred_imgs):
                    pred_img.save(os.path.join(image_path, f"{start_idx + idx}_pred.jpeg"))
                    batch_images[idx].save(os.path.join(image_path, f"{start_idx + idx}_gt.jpeg"))

    def interpolation_eval(
        self,
        image1,
        image2,
        tokens=None,
        num_interpolation=5,
        to_video=False,
        name="interpolation.mp4",
    ):
        self.model.eval()
        model = self.unwrap_model()
        transforms = T.Compose(
            [T.Resize((self.img_size, self.img_size), interpolation=T.InterpolationMode.BICUBIC), T.ToTensor()]
        )

        with torch.no_grad():
            image1 = transforms(image1).to(self.device).unsqueeze(0)
            image2 = transforms(image2).to(self.device).unsqueeze(0)
            generator = torch.Generator(device=self.device)
            generator.manual_seed(self.seed)
            outputs = model.interpolation_eval(
                self.prepare_batch(image1),
                self.prepare_batch(image2),
                generator,
                tokens=tokens,
                num_interpolation=num_interpolation,
            )

        toimg = T.ToPILImage()
        images = []
        for pred_image in outputs:
            pred_image = model.inv_vae_transform(pred_image)
            pred_image = torch.clamp(pred_image, 0, 1)
            images.append(toimg(pred_image.cpu()))

        image_path = os.path.join(self.log_dir, "images", str(self.global_step))
        ensure_directory(image_path)

        if to_video:
            import imageio

            imageio.mimsave(os.path.join(image_path, name), images, fps=10)
            return

        for idx, image in enumerate(images):
            image.save(os.path.join(image_path, f"interpolation_{idx}.jpeg"))

        widths, heights = zip(*(img.size for img in images))
        combined_image = Image.new("RGB", (sum(widths), max(heights)))
        x_offset = 0
        for image in images:
            combined_image.paste(image, (x_offset, 0))
            x_offset += image.size[0]
        combined_image.save(os.path.join(image_path, f"combined_step_{self.global_step}.jpeg"))

    def delta_interpolation(self, image, start, end):
        self.model.eval()
        model = self.unwrap_model()

        toimg = T.ToPILImage()
        transforms = T.Compose(
            [T.Resize((self.img_size, self.img_size), interpolation=T.InterpolationMode.BICUBIC), T.ToTensor()]
        )
        size = image.size

        with torch.no_grad():
            start_inputs = transforms(start).to(self.device).unsqueeze(0)
            end_inputs = transforms(end).to(self.device).unsqueeze(0)
            image_inputs = transforms(image).to(self.device).unsqueeze(0)
            generator = torch.Generator(device=self.device)
            generator.manual_seed(self.seed)

            outputs = model.delta_interpolation(
                self.prepare_batch(image_inputs),
                self.prepare_batch(start_inputs),
                self.prepare_batch(end_inputs),
                generator,
            )

        pred_image = model.inv_vae_transform(outputs).squeeze(0)
        pred_image = torch.clamp(pred_image, 0, 1)
        pred_image = toimg(pred_image.cpu())

        image_path = os.path.join(self.log_dir, "images", str(self.global_step))
        ensure_directory(image_path)

        pred_image.save(os.path.join(image_path, f"delta_interpolation_{self.global_step}.jpeg"))

        images = [start.resize(size), end.resize(size), image, pred_image.resize(size)]
        widths, heights = zip(*(img.size for img in images))
        combined_image = Image.new("RGB", (sum(widths), max(heights)))
        x_offset = 0
        for img in images:
            combined_image.paste(img, (x_offset, 0))
            x_offset += img.size[0]
        combined_image.save(os.path.join(image_path, f"delta_interpolation_combined_{self.global_step}.jpeg"))
