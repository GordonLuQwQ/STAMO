# import copy
# import inspect
# import os
# from contextlib import nullcontext
# from typing import Any, Dict, List, Optional, Union

# import numpy as np
# import torch
# import torch.nn as nn
# import torchvision.transforms as T
# from diffusers import AutoencoderKLFlux2, Flux2Transformer2DModel
# from diffusers.models.embeddings import get_1d_rotary_pos_embed
# from diffusers.models.transformers.transformer_flux2 import Flux2PosEmbed
# from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
# from diffusers.training_utils import (
#     compute_density_for_timestep_sampling,
#     compute_loss_weighting_for_sd3,
# )
# from diffusers.utils.torch_utils import randn_tensor
# from tqdm.auto import tqdm

# from stamo.renderer.model.backbone import VisionBackbone
# from stamo.renderer.model.flux2_utils import (
#     compute_empirical_mu,
#     denormalize_vae_latents,
#     normalize_vae_latents,
#     pack_latents,
#     patchify_latents,
#     prepare_latent_ids,
#     prepare_text_ids,
#     unpack_latents,
#     unpatchify_latents,
# )
# from stamo.renderer.model.projector import build_projector
# from stamo.renderer.utils.device import get_accelerator_device
# from stamo.renderer.utils.data import check_tensor
# from stamo.renderer.utils.overwatch import initialize_overwatch


# overwatch = initialize_overwatch(__name__)


# def _get(config, key: str, default=None):
#     if config is None:
#         return default
#     if hasattr(config, "get"):
#         return config.get(key, default)
#     return getattr(config, key, default)


# def _torch_dtype(value):
#     if isinstance(value, torch.dtype):
#         return value
#     return {
#         "float32": torch.float32,
#         "fp32": torch.float32,
#         "float16": torch.float16,
#         "fp16": torch.float16,
#         "bfloat16": torch.bfloat16,
#         "bf16": torch.bfloat16,
#     }.get(str(value).lower().replace("torch.", ""))


# class MusaFlux2PosEmbed(Flux2PosEmbed):
#     """Construct FLUX.2 rotary frequencies in FP32 on MUSA."""

#     def forward(self, ids: torch.Tensor):
#         if ids.device.type != "musa":
#             return super().forward(ids)

#         cosines = []
#         sines = []
#         positions = ids.float()
#         for axis, axis_dim in enumerate(self.axes_dim):
#             cosine, sine = get_1d_rotary_pos_embed(
#                 axis_dim,
#                 positions[..., axis],
#                 theta=self.theta,
#                 repeat_interleave_real=True,
#                 use_real=True,
#                 freqs_dtype=torch.float32,
#             )
#             cosines.append(cosine)
#             sines.append(sine)
#         return (
#             torch.cat(cosines, dim=-1).to(ids.device),
#             torch.cat(sines, dim=-1).to(ids.device),
#         )


# def retrieve_timesteps(
#     scheduler,
#     num_inference_steps: Optional[int] = None,
#     device: Optional[Union[str, torch.device]] = None,
#     timesteps: Optional[List[int]] = None,
#     sigmas: Optional[List[float]] = None,
#     **kwargs,
# ):
#     if timesteps is not None and sigmas is not None:
#         raise ValueError(
#             "Only one of `timesteps` or `sigmas` can be passed. "
#             "Please choose one custom schedule."
#         )
#     if timesteps is not None:
#         accepts_timesteps = "timesteps" in set(
#             inspect.signature(scheduler.set_timesteps).parameters
#         )
#         if not accepts_timesteps:
#             raise ValueError(
#                 f"{scheduler.__class__.__name__}.set_timesteps does not "
#                 "support custom timesteps."
#             )
#         scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
#         return scheduler.timesteps, len(timesteps)
#     if sigmas is not None:
#         accepts_sigmas = "sigmas" in set(
#             inspect.signature(scheduler.set_timesteps).parameters
#         )
#         if not accepts_sigmas:
#             raise ValueError(
#                 f"{scheduler.__class__.__name__}.set_timesteps does not "
#                 "support custom sigmas."
#             )
#         scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
#         return scheduler.timesteps, len(sigmas)
#     scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
#     return scheduler.timesteps, num_inference_steps


# class RenderNet(nn.Module):
#     """StaMo visual autoencoding objective with a FLUX.2 4B or 9B backbone."""

#     def __init__(self, args):
#         super().__init__()
#         self.device = get_accelerator_device()
#         self.vision_backbone = VisionBackbone(
#             img_size=args.data.img_size,
#             model_name=args.vision_backbone.model_name,
#             pretrained=args.vision_backbone.pretrained,
#             local_ckpt=args.vision_backbone.local_ckpt,
#         )

#         flux_config = args.render_net.flux
#         self.flux_local_ckpt = str(flux_config.local_ckpt)
#         dtype = _torch_dtype(_get(flux_config, "torch_dtype", "bfloat16"))
#         pretrained_kwargs = {"torch_dtype": dtype} if dtype is not None else {}

#         self.DiT = Flux2Transformer2DModel.from_pretrained(
#             self.flux_local_ckpt,
#             subfolder="transformer",
#             **pretrained_kwargs,
#         )
#         original_pos_embed = self.DiT.pos_embed
#         self.DiT.pos_embed = MusaFlux2PosEmbed(
#             theta=original_pos_embed.theta,
#             axes_dim=list(original_pos_embed.axes_dim),
#         )
#         self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
#             self.flux_local_ckpt,
#             subfolder="scheduler",
#         )
#         self.scheduler_copy = copy.deepcopy(self.scheduler)
#         self.vae = AutoencoderKLFlux2.from_pretrained(
#             self.flux_local_ckpt,
#             subfolder="vae",
#             **pretrained_kwargs,
#         )
#         #？？？？？？？？？
#         if str(_get(flux_config, "musa_vae_attention_backend", "legacy")) == "legacy":
#             self.vae.set_default_attn_processor()

#         self.dtype = next(self.DiT.parameters()).dtype
#         self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
#         self.num_channels_transformer = int(self.DiT.config.in_channels)

#         self.projector = build_projector(
#             args,
#             self.vision_backbone.patches,
#             self.vision_backbone.channels,
#         )
#         self.num_token = int(args.projector.num_token)
#         self.token_dropout = bool(args.render_net.token_dropout)
#         self.height = args.data.img_size
#         self.width = args.data.img_size
#         self.seed = args.seed
#         self.guidance_scale = _get(flux_config, "guidance_scale", 1.0)
#         self.num_inference_steps = args.render_net.num_inference_steps
#         self.weighting_scheme = _get(args.render_net, "weighting_scheme", "none")
#         self.logit_mean = _get(args.render_net, "logit_mean", 0.0)
#         self.logit_std = _get(args.render_net, "logit_std", 1.0)
#         self.mode_scale = _get(args.render_net, "mode_scale", 1.29)

#         self.projector_feature_extractor = self.vision_backbone.transforms
#         self.dit_feature_extractor = T.Normalize(mean=[0.5], std=[0.5])
#         self.inv_vae_transform = T.Compose(
#             [T.Lambda(lambda image: image * 0.5 + 0.5)]
#         )

#     def to(self, *args, **kwargs):
#         converted = super().to(*args, **kwargs)
#         self.device = next(self.parameters()).device
#         self.dtype = next(self.DiT.parameters()).dtype
#         return converted

#     def set_trainable_params(self) -> None:
#         self.DiT.requires_grad_(True)
#         self.projector.requires_grad_(True)
#         self.vae.requires_grad_(False)
#         self.vision_backbone.requires_grad_(False)

#         self._set_submodule_modes(self.training)

#     def _set_submodule_modes(self, mode: bool) -> None:
#         self.DiT.train(mode)
#         self.projector.train(mode)

#         # The feature extractor and VAE are frozen in both training and eval.
#         self.vae.eval()
#         self.vision_backbone.eval()

#     def train(self, mode: bool = True):
#         super().train(mode)
#         self._set_submodule_modes(bool(mode))
#         return self

#     def save_checkpoint(self, save_path: str, global_step: int) -> None:
#         excluded = ["vae", "projector", "vision_backbone"]
#         save_dict = {"model": {}, "global_step": global_step}
#         for key, value in self.state_dict().items():
#             if not any(key.startswith(prefix) for prefix in excluded):
#                 save_dict["model"][key] = value

#         torch.save(save_dict, os.path.join(save_path, "RenderNet.pth"))
#         torch.save(
#             self.projector.state_dict(),
#             os.path.join(save_path, "Projector.pth"),
#         )

#     def load_checkpoint(self, load_path: str) -> int:
#         projector_path = os.path.join(load_path, "Projector.pth")
#         rendernet_path = os.path.join(load_path, "RenderNet.pth")
#         assert os.path.exists(projector_path), (
#             f"Projector.pth not found in {load_path}"
#         )
#         assert os.path.exists(rendernet_path), (
#             f"RenderNet.pth not found in {load_path}"
#         )

#         overwatch.warning(f"loading checkpoints from {load_path}")

#         def _log_missing_unexpected(title, missing_keys, unexpected_keys):
#             def extract_top_level(keys):
#                 return sorted({key.split(".")[0] for key in keys})

#             overwatch.warning(
#                 f"{title} - Missing top-level keys: "
#                 f"{extract_top_level(missing_keys)}"
#             )
#             overwatch.warning(
#                 f"{title} - Unexpected top-level keys: "
#                 f"{extract_top_level(unexpected_keys)}"
#             )

#         rendernet = torch.load(rendernet_path, map_location="cpu")
#         missing, unexpected = self.load_state_dict(
#             rendernet["model"],
#             strict=False,
#         )
#         _log_missing_unexpected("RenderNet", missing, unexpected)

#         projector = torch.load(projector_path, map_location="cpu")
#         missing, unexpected = self.projector.load_state_dict(
#             projector,
#             strict=False,
#         )
#         _log_missing_unexpected("Projector", missing, unexpected)
#         return rendernet["global_step"]

#     def encode(self, images: torch.Tensor):
#         dtype = next(self.projector.parameters()).dtype
#         images = images.to(device=self.device, dtype=dtype)
#         image_embeddings = self.vision_backbone(images)
#         check_tensor(image_embeddings, "vision_backbone")
#         image_embeddings = self.projector(image_embeddings)
#         check_tensor(image_embeddings, "compress_token")
#         pooled_embeddings = image_embeddings.mean(dim=1)

#         if self.training and self.token_dropout:
#             keep = torch.randint(1, self.num_token + 1, ())
#             image_embeddings = image_embeddings[:, :keep]

#         image_embeddings = image_embeddings.to(dtype=self.dtype)
#         pooled_embeddings = pooled_embeddings.to(dtype=self.dtype)
#         return image_embeddings, pooled_embeddings

#     def encode_condition_images(self, images: torch.Tensor):
#         return self.encode(self.projector_feature_extractor(images))

#     def vae_encode(self, images: torch.Tensor) -> torch.Tensor:
#         vae_dtype = next(self.vae.parameters()).dtype
#         raw_latents = self.vae.encode(
#             images.to(self.device, dtype=vae_dtype)
#         ).latent_dist.mode()
#         latents = normalize_vae_latents(
#             self.vae,
#             patchify_latents(raw_latents),
#         )
#         return latents.to(dtype=self.dtype)

#     def vae_decode(self, latents: torch.Tensor) -> torch.Tensor:
#         vae_dtype = next(self.vae.parameters()).dtype
#         latents = denormalize_vae_latents(
#             self.vae,
#             latents.to(self.device, dtype=vae_dtype),
#         )
#         return self.vae.decode(
#             unpatchify_latents(latents),
#             return_dict=False,
#         )[0]

#     def train_step(
#         self,
#         inputs: Dict[str, Any],
#         outputs: Dict[str, Any],
#         criterion: nn.Module,
#     ) -> Dict[str, Any]:
#         images = inputs["images"]
#         batch_size = images.shape[0]

#         projector_images = self.projector_feature_extractor(images)
#         dit_images = self.dit_feature_extractor(images)
#         check_tensor(projector_images, "projector_images")
#         check_tensor(dit_images, "dit_images")
#         image_embeddings, _ = self.encode(projector_images)
#         clean_latents = self.vae_encode(dit_images)
#         check_tensor(clean_latents, "vae_latents")
#         noise = torch.randn_like(clean_latents)

#         # FLUX.2 keeps the training sigma table unshifted. Dynamic shifting is
#         # applied by scheduler.set_timesteps() only for the inference solver.
#         density = compute_density_for_timestep_sampling(
#             weighting_scheme=self.weighting_scheme,
#             batch_size=batch_size,
#             logit_mean=self.logit_mean,
#             logit_std=self.logit_std,
#             mode_scale=self.mode_scale,
#         )
#         indices = (
#             density * self.scheduler_copy.config.num_train_timesteps
#         ).long()
#         timesteps = self.scheduler_copy.timesteps[indices].to(clean_latents.device)
#         sigmas = self.scheduler_copy.sigmas[indices].to(
#             clean_latents.device,
#             clean_latents.dtype,
#         )
#         while sigmas.ndim < clean_latents.ndim:
#             sigmas = sigmas.unsqueeze(-1)

#         noisy_latents = (1.0 - sigmas) * clean_latents + sigmas * noise
#         latent_height, latent_width = clean_latents.shape[-2:]
#         packed_noisy_latents = pack_latents(noisy_latents)
#         model_prediction = self.DiT(
#             hidden_states=packed_noisy_latents,
#             timestep=timesteps / 1000,
#             guidance=None,
#             encoder_hidden_states=image_embeddings,
#             txt_ids=prepare_text_ids(image_embeddings),
#             img_ids=prepare_latent_ids(clean_latents),
#             return_dict=False,
#         )[0]
#         if model_prediction.shape[1] < packed_noisy_latents.shape[1]:
#             raise RuntimeError(
#                 "FLUX.2 returned fewer image tokens than its latent input: "
#                 f"prediction={tuple(model_prediction.shape)}, "
#                 f"latents={tuple(packed_noisy_latents.shape)}."
#             )
#         model_prediction = model_prediction[
#             :, : packed_noisy_latents.shape[1]
#         ]
#         model_prediction = unpack_latents(
#             model_prediction,
#             latent_height,
#             latent_width,
#         )
#         check_tensor(
#             model_prediction,
#             "model_pred",
#             check_bound=100,
#             check_std=10,
#         )
#         target = noise - clean_latents
#         check_tensor(target, "target")
#         weighting = compute_loss_weighting_for_sd3(
#             weighting_scheme=self.weighting_scheme,
#             sigmas=sigmas,
#         )
#         outputs["loss"] = criterion(weighting, model_prediction, target)
#         return outputs

#     def progress_bar(self, iterable=None, total=None):
#         if not hasattr(self, "_progress_bar_config"):
#             self._progress_bar_config = {}
#         elif not isinstance(self._progress_bar_config, dict):
#             raise ValueError(
#                 "self._progress_bar_config must be a dict, got "
#                 f"{type(self._progress_bar_config)}."
#             )

#         if iterable is not None:
#             return tqdm(iterable, **self._progress_bar_config)
#         if total is not None:
#             return tqdm(total=total, **self._progress_bar_config)
#         raise ValueError("Either total or iterable must be defined.")

#     def prepare_latents(
#         self,
#         batch_size,
#         num_channels_latents,
#         height,
#         width,
#         dtype,
#         device,
#         generator,
#         latents=None,
#     ) -> torch.Tensor:
#         if latents is not None:
#             return latents.to(device=device, dtype=dtype)

#         shape = (
#             batch_size,
#             num_channels_latents,
#             int(height) // (self.vae_scale_factor * 2),
#             int(width) // (self.vae_scale_factor * 2),
#         )

#         if isinstance(generator, list) and len(generator) != batch_size:
#             raise ValueError(
#                 f"Generator list has length {len(generator)}, but batch size "
#                 f"is {batch_size}."
#             )

#         return randn_tensor(
#             shape,
#             generator=generator,
#             device=device,
#             dtype=dtype,
#         )

#     def _inference_timesteps(self, packed_latents: torch.Tensor):
#         sigmas = np.linspace(
#             1.0,
#             1.0 / self.num_inference_steps,
#             self.num_inference_steps,
#         )
#         mu = compute_empirical_mu(
#             int(packed_latents.shape[1]),
#             self.num_inference_steps,
#         )
#         timesteps, steps = retrieve_timesteps(
#             self.scheduler,
#             self.num_inference_steps,
#             self.device,
#             sigmas=sigmas,
#             mu=mu,
#         )
#         if hasattr(self.scheduler, "set_begin_index"):
#             self.scheduler.set_begin_index(0)
#         return timesteps, steps

#     def _generate_from_embeddings(
#         self,
#         image_embeddings: torch.Tensor,
#         generator=None,
#     ) -> torch.Tensor:
#         batch_size = image_embeddings.shape[0]
#         latents = self.prepare_latents(
#             batch_size,
#             self.num_channels_transformer,
#             self.height,
#             self.width,
#             self.dtype,
#             self.device,
#             generator,
#             latents=None,
#         )
#         latents = pack_latents(latents)

#         timesteps, num_inference_steps = self._inference_timesteps(latents)
#         num_warmup_steps = max(
#             len(timesteps) - num_inference_steps * self.scheduler.order,
#             0,
#         )
#         latent_height = self.height // (self.vae_scale_factor * 2)
#         latent_width = self.width // (self.vae_scale_factor * 2)
#         spatial_latents = unpack_latents(latents, latent_height, latent_width)
#         image_ids = prepare_latent_ids(spatial_latents)
#         text_ids = prepare_text_ids(image_embeddings)

#         with self.progress_bar(total=num_inference_steps) as progress:
#             for index, timestep_value in enumerate(timesteps):
#                 timestep = timestep_value.expand(batch_size).to(latents.dtype)
#                 cache_context = (
#                     self.DiT.cache_context("cond")
#                     if hasattr(self.DiT, "cache_context")
#                     else nullcontext()
#                 )
#                 with cache_context:
#                     prediction = self.DiT(
#                         hidden_states=latents,
#                         timestep=timestep / 1000,
#                         guidance=None,
#                         encoder_hidden_states=image_embeddings,
#                         txt_ids=text_ids,
#                         img_ids=image_ids,
#                         return_dict=False,
#                     )[0]
#                 if prediction.shape[1] < latents.shape[1]:
#                     raise RuntimeError(
#                         "FLUX.2 returned fewer image tokens than its latent input: "
#                         f"prediction={tuple(prediction.shape)}, "
#                         f"latents={tuple(latents.shape)}."
#                     )
#                 prediction = prediction[:, : latents.shape[1]]
#                 latents = self.scheduler.step(
#                     prediction,
#                     timestep_value,
#                     latents,
#                     return_dict=False,
#                 )[0]
#                 if index == len(timesteps) - 1 or (
#                     (index + 1) > num_warmup_steps
#                     and (index + 1) % self.scheduler.order == 0
#                 ):
#                     progress.update()

#         latents = latents.to(dtype=self.dtype)
#         latents = unpack_latents(latents, latent_height, latent_width)
#         return self.vae_decode(latents)

#     @torch.no_grad()
#     def eval_step(
#         self,
#         inputs: Dict[str, Any],
#         outputs: Dict[str, Any],
#     ) -> Dict[str, Any]:
#         images = inputs["images"]
#         image_embeddings, _ = self.encode_condition_images(images)
#         outputs["images"] = self._generate_from_embeddings(
#             image_embeddings,
#             inputs["generator"],
#         )
#         return outputs

#     @torch.no_grad()
#     def interpolation_eval(
#         self,
#         image1: torch.Tensor,
#         image2: torch.Tensor,
#         generator,
#         tokens=None,
#         num_interpolation: int = 5,
#     ) -> torch.Tensor:
#         embeddings1, _ = self.encode_condition_images(image1)
#         embeddings2, _ = self.encode_condition_images(image2)
#         images = []
#         for alpha in torch.linspace(
#             0,
#             1,
#             steps=num_interpolation,
#             device=self.device,
#             dtype=embeddings1.dtype,
#         ):
#             embeddings = embeddings1 * (1 - alpha) + embeddings2 * alpha
#             if tokens:
#                 embeddings[:, tokens] = embeddings1[:, tokens]
#             images.append(
#                 self._generate_from_embeddings(
#                     embeddings,
#                     generator,
#                 )
#             )
#         return torch.cat(images)

#     @torch.no_grad()
#     def get_delta_action(self, start: torch.Tensor, end: torch.Tensor):
#         start_embeddings, _ = self.encode_condition_images(start)
#         end_embeddings, _ = self.encode_condition_images(end)
#         return end_embeddings - start_embeddings

#     @torch.no_grad()
#     def delta_interpolation(self, image, start, end, generator):
#         embeddings, _ = self.encode_condition_images(image)
#         return self._generate_from_embeddings(
#             embeddings + self.get_delta_action(start, end),
#             generator,
#         )

#     def forward(
#         self,
#         inputs: Dict[str, Any],
#         criterion: nn.Module = None,
#         generator=None,
#     ) -> Dict[str, Any]:
#         outputs = {}
#         if generator is None:
#             generator = torch.Generator(device=self.device)
#             generator.manual_seed(self.seed)
#         inputs["generator"] = generator

#         if self.training:
#             outputs = self.train_step(inputs, outputs, criterion)
#         else:
#             outputs = self.eval_step(inputs, outputs)

#         inputs.pop("generator", None)
#         return outputs

# version 1
# """Standalone StaMo + FLUX.2 renderer with EgoVerse joint conditioning."""

# import copy
# import inspect
# import os
# from contextlib import contextmanager, nullcontext
# from typing import Any, Dict, List, Optional, Union

# import numpy as np
# import torch
# import torch.nn as nn
# import torchvision.transforms as T
# from diffusers import AutoencoderKLFlux2, Flux2Transformer2DModel
# from diffusers.models.embeddings import get_1d_rotary_pos_embed
# from diffusers.models.transformers.transformer_flux2 import Flux2PosEmbed
# from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
# from diffusers.training_utils import (
#     compute_density_for_timestep_sampling,
#     compute_loss_weighting_for_sd3,
# )
# from diffusers.utils.torch_utils import randn_tensor
# from tqdm.auto import tqdm

# from stamo.renderer.model.backbone import VisionBackbone
# from stamo.renderer.model.flux2_utils import (
#     compute_empirical_mu,
#     denormalize_vae_latents,
#     normalize_vae_latents,
#     pack_latents,
#     patchify_latents,
#     prepare_latent_ids,
#     prepare_text_ids,
#     unpack_latents,
#     unpatchify_latents,
# )
# from stamo.renderer.model.projector import build_projector
# from stamo.renderer.model.pose_bond import (
#     JOINTS_PER_HAND as JOINTS_PER_HAND,
#     NUM_HANDS as NUM_HANDS,
#     PoseConditionProjector as HandPoseConditionProjector,
# )
# from stamo.renderer.utils.device import get_accelerator_device
# from stamo.renderer.utils.data import check_tensor
# from stamo.renderer.utils.overwatch import initialize_overwatch


# overwatch = initialize_overwatch(__name__)


# def _get(config, key: str, default=None):
#     if config is None:
#         return default
#     if hasattr(config, "get"):
#         return config.get(key, default)
#     return getattr(config, key, default)


# def _torch_dtype(value):
#     if isinstance(value, torch.dtype):
#         return value
#     return {
#         "float32": torch.float32,
#         "fp32": torch.float32,
#         "float16": torch.float16,
#         "fp16": torch.float16,
#         "bfloat16": torch.bfloat16,
#         "bf16": torch.bfloat16,
#     }.get(str(value).lower().replace("torch.", ""))


# class MusaFlux2PosEmbed(Flux2PosEmbed):
#     """Construct FLUX.2 rotary frequencies in FP32 on MUSA."""

#     def forward(self, ids: torch.Tensor):
#         if ids.device.type != "musa":
#             return super().forward(ids)

#         cosines = []
#         sines = []
#         positions = ids.float()
#         for axis, axis_dim in enumerate(self.axes_dim):
#             cosine, sine = get_1d_rotary_pos_embed(
#                 axis_dim,
#                 positions[..., axis],
#                 theta=self.theta,
#                 repeat_interleave_real=True,
#                 use_real=True,
#                 freqs_dtype=torch.float32,
#             )
#             cosines.append(cosine)
#             sines.append(sine)
#         return (
#             torch.cat(cosines, dim=-1).to(ids.device),
#             torch.cat(sines, dim=-1).to(ids.device),
#         )


# def retrieve_timesteps(
#     scheduler,
#     num_inference_steps: Optional[int] = None,
#     device: Optional[Union[str, torch.device]] = None,
#     timesteps: Optional[List[int]] = None,
#     sigmas: Optional[List[float]] = None,
#     **kwargs,
# ):
#     if timesteps is not None and sigmas is not None:
#         raise ValueError(
#             "Only one of `timesteps` or `sigmas` can be passed. "
#             "Please choose one custom schedule."
#         )
#     if timesteps is not None:
#         accepts_timesteps = "timesteps" in set(
#             inspect.signature(scheduler.set_timesteps).parameters
#         )
#         if not accepts_timesteps:
#             raise ValueError(
#                 f"{scheduler.__class__.__name__}.set_timesteps does not "
#                 "support custom timesteps."
#             )
#         scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
#         return scheduler.timesteps, len(timesteps)
#     if sigmas is not None:
#         accepts_sigmas = "sigmas" in set(
#             inspect.signature(scheduler.set_timesteps).parameters
#         )
#         if not accepts_sigmas:
#             raise ValueError(
#                 f"{scheduler.__class__.__name__}.set_timesteps does not "
#                 "support custom sigmas."
#             )
#         scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
#         return scheduler.timesteps, len(sigmas)
#     scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
#     return scheduler.timesteps, num_inference_steps


# class RenderNet(nn.Module):
#     """StaMo FLUX.2 renderer whose Q-Former memory includes 42 hand joints."""

#     def __init__(self, args):
#         super().__init__()
#         self.device = get_accelerator_device()
#         self.vision_backbone = VisionBackbone(
#             img_size=args.data.img_size,
#             model_name=args.vision_backbone.model_name,
#             pretrained=args.vision_backbone.pretrained,
#             local_ckpt=args.vision_backbone.local_ckpt,
#         )

#         flux_config = args.render_net.flux
#         self.flux_local_ckpt = str(flux_config.local_ckpt)
#         dtype = _torch_dtype(_get(flux_config, "torch_dtype", "bfloat16"))
#         pretrained_kwargs = {"torch_dtype": dtype} if dtype is not None else {}

#         self.DiT = Flux2Transformer2DModel.from_pretrained(
#             self.flux_local_ckpt,
#             subfolder="transformer",
#             **pretrained_kwargs,
#         )
#         original_pos_embed = self.DiT.pos_embed
#         self.DiT.pos_embed = MusaFlux2PosEmbed(
#             theta=original_pos_embed.theta,
#             axes_dim=list(original_pos_embed.axes_dim),
#         )
#         self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
#             self.flux_local_ckpt,
#             subfolder="scheduler",
#         )
#         self.scheduler_copy = copy.deepcopy(self.scheduler)
#         self.vae = AutoencoderKLFlux2.from_pretrained(
#             self.flux_local_ckpt,
#             subfolder="vae",
#             **pretrained_kwargs,
#         )
#         # MUSA uses the unfused VAE attention backend for compatibility.
#         if str(_get(flux_config, "musa_vae_attention_backend", "legacy")) == "legacy":
#             self.vae.set_default_attn_processor()

#         self.dtype = next(self.DiT.parameters()).dtype
#         self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
#         self.num_channels_transformer = int(self.DiT.config.in_channels)

#         qformer_projector = build_projector(
#             args,
#             self.vision_backbone.patches,
#             self.vision_backbone.channels,
#         )

#         pose_config = getattr(args, "pose_condition", None)
#         if pose_config is None or not bool(_get(pose_config, "enabled", False)):
#             raise ValueError(
#                 "The renderer requires pose_condition.enabled: true"
#             )
#         image_size = int(args.data.img_size)
#         if image_size != 224:
#             raise ValueError(
#                 "The hand-joint renderer requires data.img_size=224; "
#                 f"got {image_size}"
#             )
#         self.projector = HandPoseConditionProjector(
#             projector=qformer_projector,
#             token_dim=int(self.vision_backbone.channels),
#             image_size=image_size,
#             joint_patch_sigma=float(_get(pose_config, "patch_sigma", 0.75)),
#             joint_depth_mean=float(_get(pose_config, "depth_mean", 0.0)),
#             joint_depth_scale=float(_get(pose_config, "depth_scale", 1.0)),
#             joint_depth_hidden_dim=int(
#                 _get(pose_config, "depth_hidden_dim", 64)
#             ),
#         )
#         self._hand_pose_uvz = None
#         self.num_token = int(args.projector.num_token)
#         self.token_dropout = bool(args.render_net.token_dropout)
#         self.height = args.data.img_size
#         self.width = args.data.img_size
#         self.seed = args.seed
#         self.guidance_scale = _get(flux_config, "guidance_scale", 1.0)
#         self.num_inference_steps = args.render_net.num_inference_steps
#         self.weighting_scheme = _get(args.render_net, "weighting_scheme", "none")
#         self.logit_mean = _get(args.render_net, "logit_mean", 0.0)
#         self.logit_std = _get(args.render_net, "logit_std", 1.0)
#         self.mode_scale = _get(args.render_net, "mode_scale", 1.29)

#         self.projector_feature_extractor = self.vision_backbone.transforms
#         self.dit_feature_extractor = T.Normalize(mean=[0.5], std=[0.5])
#         self.inv_vae_transform = T.Compose(
#             [T.Lambda(lambda image: image * 0.5 + 0.5)]
#         )

#     def to(self, *args, **kwargs):
#         converted = super().to(*args, **kwargs)
#         self.device = next(self.parameters()).device
#         self.dtype = next(self.DiT.parameters()).dtype
#         return converted

#     def set_trainable_params(self) -> None:
#         self.DiT.requires_grad_(True)
#         self.projector.requires_grad_(True)
#         self.vae.requires_grad_(False)
#         self.vision_backbone.requires_grad_(False)

#         self._set_submodule_modes(self.training)

#     def _set_submodule_modes(self, mode: bool) -> None:
#         self.DiT.train(mode)
#         self.projector.train(mode)

#         # The feature extractor and VAE are frozen in both training and eval.
#         self.vae.eval()
#         self.vision_backbone.eval()

#     def train(self, mode: bool = True):
#         super().train(mode)
#         self._set_submodule_modes(bool(mode))
#         return self

#     def save_checkpoint(self, save_path: str, global_step: int) -> None:
#         excluded = ["vae", "projector", "vision_backbone"]
#         save_dict = {"model": {}, "global_step": global_step}
#         for key, value in self.state_dict().items():
#             if not any(key.startswith(prefix) for prefix in excluded):
#                 save_dict["model"][key] = value

#         torch.save(save_dict, os.path.join(save_path, "RenderNet.pth"))
#         torch.save(
#             self.projector.state_dict(),
#             os.path.join(save_path, "Projector.pth"),
#         )

#     def load_checkpoint(self, load_path: str) -> int:
#         projector_path = os.path.join(load_path, "Projector.pth")
#         rendernet_path = os.path.join(load_path, "RenderNet.pth")
#         assert os.path.exists(projector_path), (
#             f"Projector.pth not found in {load_path}"
#         )
#         assert os.path.exists(rendernet_path), (
#             f"RenderNet.pth not found in {load_path}"
#         )

#         overwatch.warning(f"loading checkpoints from {load_path}")

#         def _log_missing_unexpected(title, missing_keys, unexpected_keys):
#             def extract_top_level(keys):
#                 return sorted({key.split(".")[0] for key in keys})

#             overwatch.warning(
#                 f"{title} - Missing top-level keys: "
#                 f"{extract_top_level(missing_keys)}"
#             )
#             overwatch.warning(
#                 f"{title} - Unexpected top-level keys: "
#                 f"{extract_top_level(unexpected_keys)}"
#             )

#         rendernet = torch.load(rendernet_path, map_location="cpu")
#         missing, unexpected = self.load_state_dict(
#             rendernet["model"],
#             strict=False,
#         )
#         _log_missing_unexpected("RenderNet", missing, unexpected)

#         projector = torch.load(projector_path, map_location="cpu")
#         missing, unexpected = self.projector.load_state_dict(
#             projector,
#             strict=False,
#         )
#         _log_missing_unexpected("Projector", missing, unexpected)
#         return rendernet["global_step"]

#     @contextmanager
#     def _pose_context(self, inputs: Dict[str, Any]):
#         """Validate and expose the current batch pose only during one forward."""
#         pose_uvz = inputs.get("pose_uvz")
#         if pose_uvz is None:
#             raise KeyError(
#                 "Hand-pose conditioning is enabled but the batch has no 'pose_uvz'."
#             )
#         expected_pose_shape = (
#             NUM_HANDS,
#             JOINTS_PER_HAND,
#             3,
#         )
#         if (
#             pose_uvz.ndim != 4
#             or tuple(pose_uvz.shape[1:]) != expected_pose_shape
#         ):
#             raise ValueError(
#                 "Invalid pose_uvz shape: "
#                 f"got={tuple(pose_uvz.shape)}, expected=[B,2,21,3]"
#             )

#         images = inputs.get("images")
#         if images is None or images.ndim != 4:
#             raise ValueError(
#                 "Hand-pose conditioning requires images shaped [B,C,H,W]."
#             )
#         if tuple(images.shape[-2:]) != (224, 224):
#             raise ValueError(
#                 "Joint conditioning requires the complete RGB image at "
#                 f"224x224, got {tuple(images.shape[-2:])}."
#             )
#         if pose_uvz.shape[0] != images.shape[0]:
#             raise ValueError(
#                 "Pose UVZ/image batch mismatch: "
#                 f"pose={pose_uvz.shape[0]}, image={images.shape[0]}"
#             )
#         if self._hand_pose_uvz is not None:
#             raise RuntimeError("Nested hand-pose context is not supported")

#         self._hand_pose_uvz = pose_uvz
#         try:
#             yield
#         finally:
#             self._hand_pose_uvz = None

#     def encode(self, images: torch.Tensor):
#         dtype = next(self.projector.parameters()).dtype
#         images = images.to(device=self.device, dtype=dtype)
#         image_embeddings = self.vision_backbone(images)
#         check_tensor(image_embeddings, "vision_backbone")
#         if self._hand_pose_uvz is None:
#             raise RuntimeError(
#                 "Renderer encode() was called without an active hand-pose batch."
#             )
#         image_embeddings = self.projector(
#             image_embeddings,
#             self._hand_pose_uvz,
#         )
#         check_tensor(image_embeddings, "compress_token")
#         pooled_embeddings = image_embeddings.mean(dim=1)

#         if self.training and self.token_dropout:
#             keep = torch.randint(1, self.num_token + 1, ())
#             image_embeddings = image_embeddings[:, :keep]

#         image_embeddings = image_embeddings.to(dtype=self.dtype)
#         pooled_embeddings = pooled_embeddings.to(dtype=self.dtype)
#         return image_embeddings, pooled_embeddings

#     def encode_condition_images(self, images: torch.Tensor):
#         return self.encode(self.projector_feature_extractor(images))

#     def vae_encode(self, images: torch.Tensor) -> torch.Tensor:
#         vae_dtype = next(self.vae.parameters()).dtype
#         raw_latents = self.vae.encode(
#             images.to(self.device, dtype=vae_dtype)
#         ).latent_dist.mode()
#         latents = normalize_vae_latents(
#             self.vae,
#             patchify_latents(raw_latents),
#         )
#         return latents.to(dtype=self.dtype)

#     def vae_decode(self, latents: torch.Tensor) -> torch.Tensor:
#         vae_dtype = next(self.vae.parameters()).dtype
#         latents = denormalize_vae_latents(
#             self.vae,
#             latents.to(self.device, dtype=vae_dtype),
#         )
#         return self.vae.decode(
#             unpatchify_latents(latents),
#             return_dict=False,
#         )[0]

#     def train_step(
#         self,
#         inputs: Dict[str, Any],
#         outputs: Dict[str, Any],
#         criterion: nn.Module,
#     ) -> Dict[str, Any]:
#         images = inputs["images"]
#         batch_size = images.shape[0]

#         projector_images = self.projector_feature_extractor(images)
#         dit_images = self.dit_feature_extractor(images)
#         check_tensor(projector_images, "projector_images")
#         check_tensor(dit_images, "dit_images")
#         image_embeddings, _ = self.encode(projector_images)
#         clean_latents = self.vae_encode(dit_images)
#         check_tensor(clean_latents, "vae_latents")
#         noise = torch.randn_like(clean_latents)

#         # FLUX.2 keeps the training sigma table unshifted. Dynamic shifting is
#         # applied by scheduler.set_timesteps() only for the inference solver.
#         density = compute_density_for_timestep_sampling(
#             weighting_scheme=self.weighting_scheme,
#             batch_size=batch_size,
#             logit_mean=self.logit_mean,
#             logit_std=self.logit_std,
#             mode_scale=self.mode_scale,
#         )
#         indices = (
#             density * self.scheduler_copy.config.num_train_timesteps
#         ).long()
#         timesteps = self.scheduler_copy.timesteps[indices].to(clean_latents.device)
#         sigmas = self.scheduler_copy.sigmas[indices].to(
#             clean_latents.device,
#             clean_latents.dtype,
#         )
#         while sigmas.ndim < clean_latents.ndim:
#             sigmas = sigmas.unsqueeze(-1)

#         noisy_latents = (1.0 - sigmas) * clean_latents + sigmas * noise
#         latent_height, latent_width = clean_latents.shape[-2:]
#         packed_noisy_latents = pack_latents(noisy_latents)
#         model_prediction = self.DiT(
#             hidden_states=packed_noisy_latents,
#             timestep=timesteps / 1000,
#             guidance=None,
#             encoder_hidden_states=image_embeddings,
#             txt_ids=prepare_text_ids(image_embeddings),
#             img_ids=prepare_latent_ids(clean_latents),
#             return_dict=False,
#         )[0]
#         if model_prediction.shape[1] < packed_noisy_latents.shape[1]:
#             raise RuntimeError(
#                 "FLUX.2 returned fewer image tokens than its latent input: "
#                 f"prediction={tuple(model_prediction.shape)}, "
#                 f"latents={tuple(packed_noisy_latents.shape)}."
#             )
#         model_prediction = model_prediction[
#             :, : packed_noisy_latents.shape[1]
#         ]
#         model_prediction = unpack_latents(
#             model_prediction,
#             latent_height,
#             latent_width,
#         )
#         check_tensor(
#             model_prediction,
#             "model_pred",
#             check_bound=100,
#             check_std=10,
#         )
#         target = noise - clean_latents
#         check_tensor(target, "target")
#         weighting = compute_loss_weighting_for_sd3(
#             weighting_scheme=self.weighting_scheme,
#             sigmas=sigmas,
#         )
#         outputs["loss"] = criterion(weighting, model_prediction, target)
#         return outputs

#     def progress_bar(self, iterable=None, total=None):
#         if not hasattr(self, "_progress_bar_config"):
#             self._progress_bar_config = {}
#         elif not isinstance(self._progress_bar_config, dict):
#             raise ValueError(
#                 "self._progress_bar_config must be a dict, got "
#                 f"{type(self._progress_bar_config)}."
#             )

#         if iterable is not None:
#             return tqdm(iterable, **self._progress_bar_config)
#         if total is not None:
#             return tqdm(total=total, **self._progress_bar_config)
#         raise ValueError("Either total or iterable must be defined.")

#     def prepare_latents(
#         self,
#         batch_size,
#         num_channels_latents,
#         height,
#         width,
#         dtype,
#         device,
#         generator,
#         latents=None,
#     ) -> torch.Tensor:
#         if latents is not None:
#             return latents.to(device=device, dtype=dtype)

#         shape = (
#             batch_size,
#             num_channels_latents,
#             int(height) // (self.vae_scale_factor * 2),
#             int(width) // (self.vae_scale_factor * 2),
#         )

#         if isinstance(generator, list) and len(generator) != batch_size:
#             raise ValueError(
#                 f"Generator list has length {len(generator)}, but batch size "
#                 f"is {batch_size}."
#             )

#         return randn_tensor(
#             shape,
#             generator=generator,
#             device=device,
#             dtype=dtype,
#         )

#     def _inference_timesteps(self, packed_latents: torch.Tensor):
#         sigmas = np.linspace(
#             1.0,
#             1.0 / self.num_inference_steps,
#             self.num_inference_steps,
#         )
#         mu = compute_empirical_mu(
#             int(packed_latents.shape[1]),
#             self.num_inference_steps,
#         )
#         timesteps, steps = retrieve_timesteps(
#             self.scheduler,
#             self.num_inference_steps,
#             self.device,
#             sigmas=sigmas,
#             mu=mu,
#         )
#         if hasattr(self.scheduler, "set_begin_index"):
#             self.scheduler.set_begin_index(0)
#         return timesteps, steps

#     def _generate_from_embeddings(
#         self,
#         image_embeddings: torch.Tensor,
#         generator=None,
#     ) -> torch.Tensor:
#         batch_size = image_embeddings.shape[0]
#         latents = self.prepare_latents(
#             batch_size,
#             self.num_channels_transformer,
#             self.height,
#             self.width,
#             self.dtype,
#             self.device,
#             generator,
#             latents=None,
#         )
#         latents = pack_latents(latents)

#         timesteps, num_inference_steps = self._inference_timesteps(latents)
#         num_warmup_steps = max(
#             len(timesteps) - num_inference_steps * self.scheduler.order,
#             0,
#         )
#         latent_height = self.height // (self.vae_scale_factor * 2)
#         latent_width = self.width // (self.vae_scale_factor * 2)
#         spatial_latents = unpack_latents(latents, latent_height, latent_width)
#         image_ids = prepare_latent_ids(spatial_latents)
#         text_ids = prepare_text_ids(image_embeddings)

#         with self.progress_bar(total=num_inference_steps) as progress:
#             for index, timestep_value in enumerate(timesteps):
#                 timestep = timestep_value.expand(batch_size).to(latents.dtype)
#                 cache_context = (
#                     self.DiT.cache_context("cond")
#                     if hasattr(self.DiT, "cache_context")
#                     else nullcontext()
#                 )
#                 with cache_context:
#                     prediction = self.DiT(
#                         hidden_states=latents,
#                         timestep=timestep / 1000,
#                         guidance=None,
#                         encoder_hidden_states=image_embeddings,
#                         txt_ids=text_ids,
#                         img_ids=image_ids,
#                         return_dict=False,
#                     )[0]
#                 if prediction.shape[1] < latents.shape[1]:
#                     raise RuntimeError(
#                         "FLUX.2 returned fewer image tokens than its latent input: "
#                         f"prediction={tuple(prediction.shape)}, "
#                         f"latents={tuple(latents.shape)}."
#                     )
#                 prediction = prediction[:, : latents.shape[1]]
#                 latents = self.scheduler.step(
#                     prediction,
#                     timestep_value,
#                     latents,
#                     return_dict=False,
#                 )[0]
#                 if index == len(timesteps) - 1 or (
#                     (index + 1) > num_warmup_steps
#                     and (index + 1) % self.scheduler.order == 0
#                 ):
#                     progress.update()

#         latents = latents.to(dtype=self.dtype)
#         latents = unpack_latents(latents, latent_height, latent_width)
#         return self.vae_decode(latents)

#     @torch.no_grad()
#     def eval_step(
#         self,
#         inputs: Dict[str, Any],
#         outputs: Dict[str, Any],
#     ) -> Dict[str, Any]:
#         images = inputs["images"]
#         image_embeddings, _ = self.encode_condition_images(images)
#         outputs["images"] = self._generate_from_embeddings(
#             image_embeddings,
#             inputs["generator"],
#         )
#         return outputs

#     @torch.no_grad()
#     def interpolation_eval(
#         self,
#         image1: torch.Tensor,
#         image2: torch.Tensor,
#         generator,
#         tokens=None,
#         num_interpolation: int = 5,
#     ) -> torch.Tensor:
#         embeddings1, _ = self.encode_condition_images(image1)
#         embeddings2, _ = self.encode_condition_images(image2)
#         images = []
#         for alpha in torch.linspace(
#             0,
#             1,
#             steps=num_interpolation,
#             device=self.device,
#             dtype=embeddings1.dtype,
#         ):
#             embeddings = embeddings1 * (1 - alpha) + embeddings2 * alpha
#             if tokens:
#                 embeddings[:, tokens] = embeddings1[:, tokens]
#             images.append(
#                 self._generate_from_embeddings(
#                     embeddings,
#                     generator,
#                 )
#             )
#         return torch.cat(images)

#     @torch.no_grad()
#     def get_delta_action(self, start: torch.Tensor, end: torch.Tensor):
#         start_embeddings, _ = self.encode_condition_images(start)
#         end_embeddings, _ = self.encode_condition_images(end)
#         return end_embeddings - start_embeddings

#     @torch.no_grad()
#     def delta_interpolation(self, image, start, end, generator):
#         embeddings, _ = self.encode_condition_images(image)
#         return self._generate_from_embeddings(
#             embeddings + self.get_delta_action(start, end),
#             generator,
#         )

#     def forward(
#         self,
#         inputs: Dict[str, Any],
#         criterion: nn.Module = None,
#         generator=None,
#     ) -> Dict[str, Any]:
#         outputs = {}
#         if generator is None:
#             generator = torch.Generator(device=self.device)
#             generator.manual_seed(self.seed)
#         inputs["generator"] = generator

#         try:
#             with self._pose_context(inputs):
#                 if self.training:
#                     outputs = self.train_step(inputs, outputs, criterion)
#                 else:
#                     outputs = self.eval_step(inputs, outputs)
#         finally:
#             inputs.pop("generator", None)
#         return outputs

# version 2
"""StaMo + FLUX.2 with hand maps concatenated before FLUX x_embedder."""

import copy
import inspect
import os
from contextlib import contextmanager, nullcontext
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from diffusers import AutoencoderKLFlux2, Flux2Transformer2DModel
from diffusers.models.embeddings import get_1d_rotary_pos_embed
from diffusers.models.transformers.transformer_flux2 import Flux2PosEmbed
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from diffusers.utils.torch_utils import randn_tensor
from tqdm.auto import tqdm

from stamo.renderer.model.backbone import VisionBackbone
from stamo.renderer.model.flux2_utils import (
    compute_empirical_mu,
    denormalize_vae_latents,
    normalize_vae_latents,
    pack_latents,
    patchify_latents,
    prepare_latent_ids,
    prepare_text_ids,
    unpack_latents,
    unpatchify_latents,
)
from stamo.renderer.model.projector import build_projector
from stamo.renderer.model.pose_bond import (
    JOINTS_PER_HAND,
    NUM_HANDS,
    PoseConditionProjector as HandPoseConditionProjector,
)
from stamo.renderer.utils.device import get_accelerator_device
from stamo.renderer.utils.data import check_tensor
from stamo.renderer.utils.overwatch import initialize_overwatch


overwatch = initialize_overwatch(__name__)


def _get(config, key: str, default=None):
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def _torch_dtype(value):
    if isinstance(value, torch.dtype):
        return value
    return {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }.get(str(value).lower().replace("torch.", ""))


class MusaFlux2PosEmbed(Flux2PosEmbed):
    """Construct FLUX.2 rotary frequencies in FP32 on MUSA."""

    def forward(self, ids: torch.Tensor):
        if ids.device.type != "musa":
            return super().forward(ids)

        cosines = []
        sines = []
        positions = ids.float()
        for axis, axis_dim in enumerate(self.axes_dim):
            cosine, sine = get_1d_rotary_pos_embed(
                axis_dim,
                positions[..., axis],
                theta=self.theta,
                repeat_interleave_real=True,
                use_real=True,
                freqs_dtype=torch.float32,
            )
            cosines.append(cosine)
            sines.append(sine)
        return (
            torch.cat(cosines, dim=-1).to(ids.device),
            torch.cat(sines, dim=-1).to(ids.device),
        )


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be passed. "
            "Please choose one custom schedule."
        )
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters
        )
        if not accepts_timesteps:
            raise ValueError(
                f"{scheduler.__class__.__name__}.set_timesteps does not "
                "support custom timesteps."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        return scheduler.timesteps, len(timesteps)
    if sigmas is not None:
        accepts_sigmas = "sigmas" in set(
            inspect.signature(scheduler.set_timesteps).parameters
        )
        if not accepts_sigmas:
            raise ValueError(
                f"{scheduler.__class__.__name__}.set_timesteps does not "
                "support custom sigmas."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        return scheduler.timesteps, len(sigmas)
    scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
    return scheduler.timesteps, num_inference_steps


class RenderNet(nn.Module):
    """Keep Q-Former RGB-only and concatenate spatial hand maps to FLUX latents."""

    def __init__(self, args):
        super().__init__()
        self.device = get_accelerator_device()
        self.vision_backbone = VisionBackbone(
            img_size=args.data.img_size,
            model_name=args.vision_backbone.model_name,
            pretrained=args.vision_backbone.pretrained,
            local_ckpt=args.vision_backbone.local_ckpt,
        )

        flux_config = args.render_net.flux
        self.flux_local_ckpt = str(flux_config.local_ckpt)
        dtype = _torch_dtype(_get(flux_config, "torch_dtype", "bfloat16"))
        pretrained_kwargs = {"torch_dtype": dtype} if dtype is not None else {}

        self.DiT = Flux2Transformer2DModel.from_pretrained(
            self.flux_local_ckpt,
            subfolder="transformer",
            **pretrained_kwargs,
        )
        original_pos_embed = self.DiT.pos_embed
        self.DiT.pos_embed = MusaFlux2PosEmbed(
            theta=original_pos_embed.theta,
            axes_dim=list(original_pos_embed.axes_dim),
        )
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.flux_local_ckpt,
            subfolder="scheduler",
        )
        self.scheduler_copy = copy.deepcopy(self.scheduler)
        self.vae = AutoencoderKLFlux2.from_pretrained(
            self.flux_local_ckpt,
            subfolder="vae",
            **pretrained_kwargs,
        )
        # MUSA uses the unfused VAE attention backend for compatibility.
        if str(_get(flux_config, "musa_vae_attention_backend", "legacy")) == "legacy":
            self.vae.set_default_attn_processor()

        self.dtype = next(self.DiT.parameters()).dtype
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.num_channels_transformer = int(self.DiT.config.in_channels)

        qformer_projector = build_projector(
            args,
            self.vision_backbone.patches,
            self.vision_backbone.channels,
        )

        pose_config = getattr(args, "pose_condition", None)
        if pose_config is None or not bool(_get(pose_config, "enabled", False)):
            raise ValueError(
                "The renderer requires pose_condition.enabled: true"
            )
        image_size = int(args.data.img_size)
        if image_size != 224:
            raise ValueError(
                "The hand-joint renderer requires data.img_size=224; "
                f"got {image_size}"
            )
        self.hand_condition_channels = int(
            _get(pose_config, "output_channels", 16)
        )
        if self.hand_condition_channels <= 0:
            raise ValueError("pose_condition.output_channels must be positive")
        self._expand_flux_input_embedder(self.hand_condition_channels)
        self.projector = HandPoseConditionProjector(
            projector=qformer_projector,
            token_dim=int(self.vision_backbone.channels),
            image_size=image_size,
            joint_patch_sigma=float(_get(pose_config, "patch_sigma", 0.75)),
            joint_depth_mean=float(_get(pose_config, "depth_mean", 0.0)),
            joint_depth_scale=float(_get(pose_config, "depth_scale", 1.0)),
            joint_depth_hidden_dim=int(
                _get(pose_config, "depth_hidden_dim", 64)
            ),
            hand_output_channels=self.hand_condition_channels,
        )
        self._hand_pose_uvz = None
        self.num_token = int(args.projector.num_token)
        self.token_dropout = bool(args.render_net.token_dropout)
        self.height = args.data.img_size
        self.width = args.data.img_size
        self.seed = args.seed
        self.guidance_scale = _get(flux_config, "guidance_scale", 1.0)
        self.num_inference_steps = args.render_net.num_inference_steps
        self.weighting_scheme = _get(args.render_net, "weighting_scheme", "none")
        self.logit_mean = _get(args.render_net, "logit_mean", 0.0)
        self.logit_std = _get(args.render_net, "logit_std", 1.0)
        self.mode_scale = _get(args.render_net, "mode_scale", 1.29)

        self.projector_feature_extractor = self.vision_backbone.transforms
        self.dit_feature_extractor = T.Normalize(mean=[0.5], std=[0.5])
        self.inv_vae_transform = T.Compose(
            [T.Lambda(lambda image: image * 0.5 + 0.5)]
        )

    def _expand_flux_input_embedder(self, extra_channels: int) -> None:
        """Expand FLUX input projection with zero-initialized hand columns."""
        input_embedder = getattr(self.DiT, "x_embedder", None)
        if not isinstance(input_embedder, nn.Linear):
            raise TypeError(
                "FLUX.2 x_embedder must be nn.Linear for spatial hand concat, "
                f"got {type(input_embedder)!r}"
            )
        latent_channels = self.num_channels_transformer
        if input_embedder.in_features != latent_channels:
            raise ValueError(
                "FLUX.2 x_embedder/config input mismatch before hand expansion: "
                f"x_embedder={input_embedder.in_features}, "
                f"config.in_channels={latent_channels}"
            )

        model_input_channels = latent_channels + int(extra_channels)
        expanded = nn.Linear(
            model_input_channels,
            input_embedder.out_features,
            bias=input_embedder.bias is not None,
        ).to(
            device=input_embedder.weight.device,
            dtype=input_embedder.weight.dtype,
        )
        with torch.no_grad():
            expanded.weight.zero_()
            expanded.weight[:, :latent_channels].copy_(input_embedder.weight)
            if input_embedder.bias is not None:
                expanded.bias.copy_(input_embedder.bias)
        self.DiT.x_embedder = expanded
        self.model_input_channels = model_input_channels

    def to(self, *args, **kwargs):
        converted = super().to(*args, **kwargs)
        self.device = next(self.parameters()).device
        self.dtype = next(self.DiT.parameters()).dtype
        return converted

    def set_trainable_params(self) -> None:
        self.DiT.requires_grad_(True)
        self.projector.requires_grad_(True)
        self.vae.requires_grad_(False)
        self.vision_backbone.requires_grad_(False)

        self._set_submodule_modes(self.training)

    def _set_submodule_modes(self, mode: bool) -> None:
        self.DiT.train(mode)
        self.projector.train(mode)

        # The feature extractor and VAE are frozen in both training and eval.
        self.vae.eval()
        self.vision_backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self._set_submodule_modes(bool(mode))
        return self

    def save_checkpoint(self, save_path: str, global_step: int) -> None:
        excluded = ["vae", "projector", "vision_backbone"]
        save_dict = {"model": {}, "global_step": global_step}
        for key, value in self.state_dict().items():
            if not any(key.startswith(prefix) for prefix in excluded):
                save_dict["model"][key] = value

        torch.save(save_dict, os.path.join(save_path, "RenderNet.pth"))
        torch.save(
            self.projector.state_dict(),
            os.path.join(save_path, "Projector.pth"),
        )

    def load_checkpoint(self, load_path: str) -> int:
        projector_path = os.path.join(load_path, "Projector.pth")
        rendernet_path = os.path.join(load_path, "RenderNet.pth")
        assert os.path.exists(projector_path), (
            f"Projector.pth not found in {load_path}"
        )
        assert os.path.exists(rendernet_path), (
            f"RenderNet.pth not found in {load_path}"
        )

        overwatch.warning(f"loading checkpoints from {load_path}")

        def _log_missing_unexpected(title, missing_keys, unexpected_keys):
            def extract_top_level(keys):
                return sorted({key.split(".")[0] for key in keys})

            overwatch.warning(
                f"{title} - Missing top-level keys: "
                f"{extract_top_level(missing_keys)}"
            )
            overwatch.warning(
                f"{title} - Unexpected top-level keys: "
                f"{extract_top_level(unexpected_keys)}"
            )

        rendernet = torch.load(rendernet_path, map_location="cpu")
        missing, unexpected = self.load_state_dict(
            rendernet["model"],
            strict=False,
        )
        _log_missing_unexpected("RenderNet", missing, unexpected)

        projector = torch.load(projector_path, map_location="cpu")
        missing, unexpected = self.projector.load_state_dict(
            projector,
            strict=False,
        )
        _log_missing_unexpected("Projector", missing, unexpected)
        return rendernet["global_step"]

    @contextmanager
    def _pose_context(self, inputs: Dict[str, Any]):
        """Validate and expose the current batch pose only during one forward."""
        pose_uvz = inputs.get("pose_uvz")
        if pose_uvz is None:
            raise KeyError(
                "Hand-pose conditioning is enabled but the batch has no 'pose_uvz'."
            )
        expected_pose_shape = (
            NUM_HANDS,
            JOINTS_PER_HAND,
            3,
        )
        if (
            pose_uvz.ndim != 4
            or tuple(pose_uvz.shape[1:]) != expected_pose_shape
        ):
            raise ValueError(
                "Invalid pose_uvz shape: "
                f"got={tuple(pose_uvz.shape)}, expected=[B,2,21,3]"
            )

        images = inputs.get("images")
        if images is None or images.ndim != 4:
            raise ValueError(
                "Hand-pose conditioning requires images shaped [B,C,H,W]."
            )
        if tuple(images.shape[-2:]) != (224, 224):
            raise ValueError(
                "Joint conditioning requires the complete RGB image at "
                f"224x224, got {tuple(images.shape[-2:])}."
            )
        if pose_uvz.shape[0] != images.shape[0]:
            raise ValueError(
                "Pose UVZ/image batch mismatch: "
                f"pose={pose_uvz.shape[0]}, image={images.shape[0]}"
            )
        if self._hand_pose_uvz is not None:
            raise RuntimeError("Nested hand-pose context is not supported")

        self._hand_pose_uvz = pose_uvz
        try:
            yield
        finally:
            self._hand_pose_uvz = None

    def encode(self, images: torch.Tensor):
        dtype = next(self.projector.parameters()).dtype
        images = images.to(device=self.device, dtype=dtype)
        image_embeddings = self.vision_backbone(images)
        check_tensor(image_embeddings, "vision_backbone")
        if self._hand_pose_uvz is None:
            raise RuntimeError(
                "Renderer encode() was called without an active hand-pose batch."
            )
        image_embeddings, hand_map = self.projector(
            image_embeddings,
            self._hand_pose_uvz,
        )
        check_tensor(image_embeddings, "compress_token")
        pooled_embeddings = image_embeddings.mean(dim=1)

        if self.training and self.token_dropout:
            keep = torch.randint(1, self.num_token + 1, ())
            image_embeddings = image_embeddings[:, :keep]

        image_embeddings = image_embeddings.to(dtype=self.dtype)
        pooled_embeddings = pooled_embeddings.to(dtype=self.dtype)
        hand_map = hand_map.to(dtype=self.dtype)
        return image_embeddings, pooled_embeddings, hand_map

    def encode_condition_images(self, images: torch.Tensor):
        return self.encode(self.projector_feature_extractor(images))

    def vae_encode(self, images: torch.Tensor) -> torch.Tensor:
        vae_dtype = next(self.vae.parameters()).dtype
        raw_latents = self.vae.encode(
            images.to(self.device, dtype=vae_dtype)
        ).latent_dist.mode()
        latents = normalize_vae_latents(
            self.vae,
            patchify_latents(raw_latents),
        )
        return latents.to(dtype=self.dtype)

    def vae_decode(self, latents: torch.Tensor) -> torch.Tensor:
        vae_dtype = next(self.vae.parameters()).dtype
        latents = denormalize_vae_latents(
            self.vae,
            latents.to(self.device, dtype=vae_dtype),
        )
        return self.vae.decode(
            unpatchify_latents(latents),
            return_dict=False,
        )[0]

    def train_step(
        self,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        criterion: nn.Module,
    ) -> Dict[str, Any]:
        images = inputs["images"]
        batch_size = images.shape[0]

        projector_images = self.projector_feature_extractor(images)
        dit_images = self.dit_feature_extractor(images)
        check_tensor(projector_images, "projector_images")
        check_tensor(dit_images, "dit_images")
        image_embeddings, _, hand_map = self.encode(projector_images)
        clean_latents = self.vae_encode(dit_images)
        check_tensor(clean_latents, "vae_latents")
        noise = torch.randn_like(clean_latents)

        # FLUX.2 keeps the training sigma table unshifted. Dynamic shifting is
        # applied by scheduler.set_timesteps() only for the inference solver.
        density = compute_density_for_timestep_sampling(
            weighting_scheme=self.weighting_scheme,
            batch_size=batch_size,
            logit_mean=self.logit_mean,
            logit_std=self.logit_std,
            mode_scale=self.mode_scale,
        )
        indices = (
            density * self.scheduler_copy.config.num_train_timesteps
        ).long()
        timesteps = self.scheduler_copy.timesteps[indices].to(clean_latents.device)
        sigmas = self.scheduler_copy.sigmas[indices].to(
            clean_latents.device,
            clean_latents.dtype,
        )
        while sigmas.ndim < clean_latents.ndim:
            sigmas = sigmas.unsqueeze(-1)

        noisy_latents = (1.0 - sigmas) * clean_latents + sigmas * noise
        latent_height, latent_width = clean_latents.shape[-2:]
        packed_noisy_latents = pack_latents(noisy_latents)
        packed_hand_map = pack_latents(hand_map)
        model_input = torch.cat(
            (packed_noisy_latents, packed_hand_map),
            dim=-1,
        )
        model_prediction = self.DiT(
            hidden_states=model_input,
            timestep=timesteps / 1000,
            guidance=None,
            encoder_hidden_states=image_embeddings,
            txt_ids=prepare_text_ids(image_embeddings),
            img_ids=prepare_latent_ids(clean_latents),
            return_dict=False,
        )[0]
        if model_prediction.shape[1] < packed_noisy_latents.shape[1]:
            raise RuntimeError(
                "FLUX.2 returned fewer image tokens than its latent input: "
                f"prediction={tuple(model_prediction.shape)}, "
                f"latents={tuple(packed_noisy_latents.shape)}."
            )
        model_prediction = model_prediction[
            :, : packed_noisy_latents.shape[1]
        ]
        model_prediction = unpack_latents(
            model_prediction,
            latent_height,
            latent_width,
        )
        check_tensor(
            model_prediction,
            "model_pred",
            check_bound=100,
            check_std=10,
        )
        target = noise - clean_latents
        check_tensor(target, "target")
        weighting = compute_loss_weighting_for_sd3(
            weighting_scheme=self.weighting_scheme,
            sigmas=sigmas,
        )
        outputs["loss"] = criterion(weighting, model_prediction, target)
        return outputs

    def progress_bar(self, iterable=None, total=None):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        elif not isinstance(self._progress_bar_config, dict):
            raise ValueError(
                "self._progress_bar_config must be a dict, got "
                f"{type(self._progress_bar_config)}."
            )

        if iterable is not None:
            return tqdm(iterable, **self._progress_bar_config)
        if total is not None:
            return tqdm(total=total, **self._progress_bar_config)
        raise ValueError("Either total or iterable must be defined.")

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ) -> torch.Tensor:
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        shape = (
            batch_size,
            num_channels_latents,
            int(height) // (self.vae_scale_factor * 2),
            int(width) // (self.vae_scale_factor * 2),
        )

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"Generator list has length {len(generator)}, but batch size "
                f"is {batch_size}."
            )

        return randn_tensor(
            shape,
            generator=generator,
            device=device,
            dtype=dtype,
        )

    def _inference_timesteps(self, packed_latents: torch.Tensor):
        sigmas = np.linspace(
            1.0,
            1.0 / self.num_inference_steps,
            self.num_inference_steps,
        )
        mu = compute_empirical_mu(
            int(packed_latents.shape[1]),
            self.num_inference_steps,
        )
        timesteps, steps = retrieve_timesteps(
            self.scheduler,
            self.num_inference_steps,
            self.device,
            sigmas=sigmas,
            mu=mu,
        )
        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(0)
        return timesteps, steps

    def _generate_from_embeddings(
        self,
        image_embeddings: torch.Tensor,
        hand_map: torch.Tensor,
        generator=None,
    ) -> torch.Tensor:
        batch_size = image_embeddings.shape[0]
        latents = self.prepare_latents(
            batch_size,
            self.num_channels_transformer,
            self.height,
            self.width,
            self.dtype,
            self.device,
            generator,
            latents=None,
        )
        latents = pack_latents(latents)
        packed_hand_map = pack_latents(hand_map.to(dtype=latents.dtype))

        timesteps, num_inference_steps = self._inference_timesteps(latents)
        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order,
            0,
        )
        latent_height = self.height // (self.vae_scale_factor * 2)
        latent_width = self.width // (self.vae_scale_factor * 2)
        spatial_latents = unpack_latents(latents, latent_height, latent_width)
        image_ids = prepare_latent_ids(spatial_latents)
        text_ids = prepare_text_ids(image_embeddings)

        with self.progress_bar(total=num_inference_steps) as progress:
            for index, timestep_value in enumerate(timesteps):
                timestep = timestep_value.expand(batch_size).to(latents.dtype)
                cache_context = (
                    self.DiT.cache_context("cond")
                    if hasattr(self.DiT, "cache_context")
                    else nullcontext()
                )
                with cache_context:
                    model_input = torch.cat(
                        (latents, packed_hand_map),
                        dim=-1,
                    )
                    prediction = self.DiT(
                        hidden_states=model_input,
                        timestep=timestep / 1000,
                        guidance=None,
                        encoder_hidden_states=image_embeddings,
                        txt_ids=text_ids,
                        img_ids=image_ids,
                        return_dict=False,
                    )[0]
                if prediction.shape[1] < latents.shape[1]:
                    raise RuntimeError(
                        "FLUX.2 returned fewer image tokens than its latent input: "
                        f"prediction={tuple(prediction.shape)}, "
                        f"latents={tuple(latents.shape)}."
                    )
                prediction = prediction[:, : latents.shape[1]]
                latents = self.scheduler.step(
                    prediction,
                    timestep_value,
                    latents,
                    return_dict=False,
                )[0]
                if index == len(timesteps) - 1 or (
                    (index + 1) > num_warmup_steps
                    and (index + 1) % self.scheduler.order == 0
                ):
                    progress.update()

        latents = latents.to(dtype=self.dtype)
        latents = unpack_latents(latents, latent_height, latent_width)
        return self.vae_decode(latents)

    @torch.no_grad()
    def eval_step(
        self,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        images = inputs["images"]
        image_embeddings, _, hand_map = self.encode_condition_images(images)
        outputs["images"] = self._generate_from_embeddings(
            image_embeddings,
            hand_map,
            inputs["generator"],
        )
        return outputs

    @torch.no_grad()
    def interpolation_eval(
        self,
        image1: torch.Tensor,
        image2: torch.Tensor,
        generator,
        tokens=None,
        num_interpolation: int = 5,
    ) -> torch.Tensor:
        embeddings1, _, hand_map1 = self.encode_condition_images(image1)
        embeddings2, _, hand_map2 = self.encode_condition_images(image2)
        images = []
        for alpha in torch.linspace(
            0,
            1,
            steps=num_interpolation,
            device=self.device,
            dtype=embeddings1.dtype,
        ):
            embeddings = embeddings1 * (1 - alpha) + embeddings2 * alpha
            hand_map = hand_map1 * (1 - alpha) + hand_map2 * alpha
            if tokens:
                embeddings[:, tokens] = embeddings1[:, tokens]
            images.append(
                self._generate_from_embeddings(
                    embeddings,
                    hand_map,
                    generator,
                )
            )
        return torch.cat(images)

    @torch.no_grad()
    def get_delta_action(self, start: torch.Tensor, end: torch.Tensor):
        start_embeddings, _, _ = self.encode_condition_images(start)
        end_embeddings, _, _ = self.encode_condition_images(end)
        return end_embeddings - start_embeddings

    @torch.no_grad()
    def delta_interpolation(self, image, start, end, generator):
        embeddings, _, hand_map = self.encode_condition_images(image)
        return self._generate_from_embeddings(
            embeddings + self.get_delta_action(start, end),
            hand_map,
            generator,
        )

    def forward(
        self,
        inputs: Dict[str, Any],
        criterion: nn.Module = None,
        generator=None,
    ) -> Dict[str, Any]:
        outputs = {}
        if generator is None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(self.seed)
        inputs["generator"] = generator

        try:
            with self._pose_context(inputs):
                if self.training:
                    outputs = self.train_step(inputs, outputs, criterion)
                else:
                    outputs = self.eval_step(inputs, outputs)
        finally:
            inputs.pop("generator", None)
        return outputs
