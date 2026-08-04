"""Active STAMO renderer with FLUX.1-dev as the diffusion decoder.

The visual condition still comes from VisionBackbone + Projector. Its compressed
tokens replace FLUX's normal text-encoder context; no CLIP/T5 text encoder is
loaded by this module.
"""

import copy
import inspect
import math
import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from diffusers.models import AutoencoderKL, FluxTransformer2DModel
from diffusers.models.embeddings import FluxPosEmbed, get_1d_rotary_pos_embed
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor
from tqdm.auto import tqdm

from stamo.renderer.model.backbone import DiTConditionHead, VisionBackbone
from stamo.renderer.model.projector import build_projector
from stamo.renderer.utils.device import get_accelerator_device
from stamo.renderer.utils.overwatch import initialize_overwatch


overwatch = initialize_overwatch(__name__)


class MusaFluxPosEmbed(FluxPosEmbed):
    """FLUX RoPE with FP32 frequency construction on MUSA."""

    def forward(self, ids: torch.Tensor):
        if ids.device.type != "musa":
            return super().forward(ids)

        cos_out = []
        sin_out = []
        positions = ids.float()
        for axis_index in range(ids.shape[-1]):
            cos, sin = get_1d_rotary_pos_embed(
                self.axes_dim[axis_index],
                positions[:, axis_index],
                theta=self.theta,
                repeat_interleave_real=True,
                use_real=True,
                freqs_dtype=torch.float32,
            )
            cos_out.append(cos)
            sin_out.append(sin)
        return (
            torch.cat(cos_out, dim=-1).to(ids.device),
            torch.cat(sin_out, dim=-1).to(ids.device),
        )


def _cfg_get(config, key: str, default=None):
    """Read from OmegaConf DictConfig or from a regular Python object."""
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


# Same helper used by the official Diffusers FLUX pipeline.
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed.")

    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(f"{scheduler.__class__.__name__}.set_timesteps does not accept custom timesteps.")
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accepts_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_sigmas:
            raise ValueError(f"{scheduler.__class__.__name__}.set_timesteps does not accept custom sigmas.")
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps

    return timesteps, num_inference_steps


def calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    """Calculate FLUX's resolution-dependent FlowMatch shift."""
    slope = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    intercept = base_shift - slope * base_seq_len
    return image_seq_len * slope + intercept


def sample_flow_timestep_density(
    weighting_scheme: str,
    batch_size: int,
    logit_mean: float,
    logit_std: float,
    mode_scale: float,
) -> torch.Tensor:
    """Sample continuous flow timesteps on CPU."""
    if weighting_scheme == "logit_normal":
        values = torch.normal(
            mean=logit_mean,
            std=logit_std,
            size=(batch_size,),
        )
        return torch.sigmoid(values)
    if weighting_scheme == "mode":
        values = torch.rand(batch_size)
        return 1 - values - mode_scale * (
            torch.cos(math.pi * values / 2) ** 2 - 1 + values
        )
    return torch.rand(batch_size)


def compute_flow_loss_weighting(
    weighting_scheme: str,
    sigmas: torch.Tensor,
) -> torch.Tensor:
    """Return per-sample flow-matching loss weights."""
    if weighting_scheme == "sigma_sqrt":
        return (sigmas**-2.0).float()
    if weighting_scheme == "cosmap":
        denominator = 1 - 2 * sigmas + 2 * sigmas**2
        return 2 / (math.pi * denominator)
    return torch.ones_like(sigmas)


class RenderNet(nn.Module):
    """STAMO visual-condition renderer using only FLUX.1-dev as decoder."""

    def __init__(self, args):
        super().__init__()
        self.device = get_accelerator_device()

        # STAMO visual-condition branch remains unchanged.
        self.vision_backbone = VisionBackbone(
            img_size=args.data.img_size,
            model_name=args.vision_backbone.model_name,
            pretrained=args.vision_backbone.pretrained,
            local_ckpt=args.vision_backbone.local_ckpt,
        )

        flux_config = args.render_net.flux
        local_ckpt_value = _cfg_get(flux_config, "local_ckpt", None)
        if local_ckpt_value is None or not str(local_ckpt_value).strip():
            raise ValueError("render_net.flux.local_ckpt must be set.")
        self.flux_local_ckpt = str(local_ckpt_value)

        torch_dtype = self._resolve_torch_dtype(_cfg_get(flux_config, "torch_dtype", None))
        pretrained_kwargs = {}
        if torch_dtype is not None:
            pretrained_kwargs["torch_dtype"] = torch_dtype

        self.DiT = FluxTransformer2DModel.from_pretrained(
            self.flux_local_ckpt,
            subfolder="transformer",
            **pretrained_kwargs,
        )
        original_pos_embed = self.DiT.pos_embed
        if not isinstance(original_pos_embed, FluxPosEmbed):
            raise TypeError(
                "The loaded FLUX transformer has an unsupported positional "
                f"embedding module: {type(original_pos_embed)!r}."
            )
        self.DiT.pos_embed = MusaFluxPosEmbed(
            theta=original_pos_embed.theta,
            axes_dim=list(original_pos_embed.axes_dim),
        )
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.flux_local_ckpt,
            subfolder="scheduler",
        )
        self.vae = AutoencoderKL.from_pretrained(
            self.flux_local_ckpt,
            subfolder="vae",
            **pretrained_kwargs,
        )

        self.dtype = next(self.DiT.parameters()).dtype
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.scheduler_copy = copy.deepcopy(self.scheduler)

        self.projector = build_projector(
            args,
            self.vision_backbone.patches,
            self.vision_backbone.channels,
        )

        # FLUX pooled_projections must be 768-dimensional.
        self.dit_condition_head = DiTConditionHead(pooled_dim=768)

        self.token_dropout = bool(args.render_net.token_dropout)
        self.num_token = int(args.projector.num_token)

        self.height = int(args.data.img_size)
        self.width = int(args.data.img_size)

        self.seed = int(args.seed)
        self.guidance_scale = float(_cfg_get(flux_config, "guidance_scale", 3.5))
        self.num_inference_steps = int(args.render_net.num_inference_steps)
        self.weighting_scheme = str(_cfg_get(args.render_net, "weighting_scheme", "none"))
        self.logit_mean = float(_cfg_get(args.render_net, "logit_mean", 0.0))
        self.logit_std = float(_cfg_get(args.render_net, "logit_std", 1.0))
        self.mode_scale = float(_cfg_get(args.render_net, "mode_scale", 1.29))

        self.train_transformer = bool(_cfg_get(flux_config, "train_transformer", False))
        self.use_gradient_checkpointing = bool(_cfg_get(flux_config, "gradient_checkpointing", True))

        self.projector_feature_extractor = self.vision_backbone.transforms
        self.dit_feature_extractor = T.Normalize(mean=[0.5], std=[0.5])
        self.inv_vae_transform = T.Compose([T.Lambda(lambda image: image * 0.5 + 0.5)])

        self._validate_flux_model()
        self.set_trainable_params()

    @staticmethod
    def _resolve_torch_dtype(value):
        if value is None:
            return None
        if isinstance(value, torch.dtype):
            return value

        normalized = str(value).lower().replace("torch.", "")
        mapping = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        if normalized not in mapping:
            raise ValueError(f"Unsupported torch_dtype={value!r}.")
        return mapping[normalized]

    def _validate_flux_resolution(self, height: int, width: int) -> None:
        required_multiple = self.vae_scale_factor * 2
        if height % required_multiple != 0 or width % required_multiple != 0:
            raise ValueError(
                f"FLUX image height/width must be divisible by {required_multiple}; "
                f"got {height}x{width}."
            )

    def _vae_scaling_values(self):
        scaling_factor = float(_cfg_get(self.vae.config, "scaling_factor", 1.0))
        shift_factor = _cfg_get(self.vae.config, "shift_factor", 0.0)
        shift_factor = 0.0 if shift_factor is None else float(shift_factor)
        return scaling_factor, shift_factor

    def _validate_flux_model(self) -> None:
        self._validate_flux_resolution(self.height, self.width)

        context_dim = int(_cfg_get(self.DiT.config, "joint_attention_dim", -1))
        pooled_dim = int(_cfg_get(self.DiT.config, "pooled_projection_dim", -1))
        packed_channels = int(_cfg_get(self.DiT.config, "in_channels", -1))
        configured_out_channels = _cfg_get(self.DiT.config, "out_channels", None)
        packed_out_channels = packed_channels if configured_out_channels is None else int(configured_out_channels)
        vae_latent_channels = int(_cfg_get(self.vae.config, "latent_channels", -1))

        if int(self.projector.output_align_dim) != 4096:
            raise ValueError(
                f"Projector must output 4096 for FLUX, got {self.projector.output_align_dim}."
            )
        if context_dim != 4096 or pooled_dim != 768:
            raise ValueError(
                f"Expected FLUX context=4096 and pooled=768, got {context_dim} and {pooled_dim}."
            )
        if not bool(_cfg_get(self.DiT.config, "guidance_embeds", False)):
            raise ValueError("This reference expects FLUX.1-dev with guidance_embeds=True.")
        if packed_channels != 64 or packed_out_channels != 64:
            raise ValueError(
                f"Expected FLUX transformer in/out channels 64, got {packed_channels}/{packed_out_channels}."
            )

        self.num_channels_latents = packed_channels // 4
        if self.num_channels_latents != 16 or vae_latent_channels != 16:
            raise ValueError(
                "Expected 16-channel raw FLUX VAE latents, got "
                f"transformer raw={self.num_channels_latents}, VAE={vae_latent_channels}."
            )
        if self.vae_scale_factor != 8:
            raise ValueError(f"Expected FLUX VAE spatial scale 8, got {self.vae_scale_factor}.")

        scaling_factor, shift_factor = self._vae_scaling_values()
        if abs(scaling_factor - 0.3611) > 1e-5 or abs(shift_factor - 0.1159) > 1e-5:
            raise ValueError(
                "Loaded VAE is not the expected FLUX VAE: "
                f"scaling_factor={scaling_factor}, shift_factor={shift_factor}."
            )

    def to(self, *args, **kwargs):
        model_converted = super().to(*args, **kwargs)
        self.device = next(self.parameters()).device
        self.dtype = next(self.DiT.parameters()).dtype
        return model_converted

    def _set_submodule_modes(self, mode: bool) -> None:
        # DiT remains in training mode during training even when frozen, so
        # Diffusers gradient checkpointing can still reduce activation memory.
        self.DiT.train(mode)
        self.projector.train(mode)
        self.dit_condition_head.train(mode)
        self.vae.eval()
        self.vision_backbone.eval()

    def set_trainable_params(self) -> None:
        if self.use_gradient_checkpointing:
            self.DiT.enable_gradient_checkpointing()
        elif hasattr(self.DiT, "disable_gradient_checkpointing"):
            self.DiT.disable_gradient_checkpointing()

        self.DiT.requires_grad_(self.train_transformer)
        self.projector.requires_grad_(True)
        self.dit_condition_head.requires_grad_(True)
        self.vae.requires_grad_(False)
        self.vision_backbone.requires_grad_(False)
        self._set_submodule_modes(self.training)

    def train(self, mode: bool = True):
        super().train(mode)
        self._set_submodule_modes(mode)
        return self

    def save_checkpoint(self, save_path: str, global_step: int) -> None:
        exclude_prefixes = ["vae", "projector", "vision_backbone"]
        # A frozen FLUX transformer is reloaded from flux.local_ckpt and does
        # not belong in every lightweight STAMO export.
        if not self.train_transformer:
            exclude_prefixes.append("DiT")
        save_dict = {
            "model": {},
            "global_step": int(global_step),
            "backend": "flux",
            "checkpoint_version": 2,
        }
        for key, value in self.state_dict().items():
            if not any(key.startswith(prefix) for prefix in exclude_prefixes):
                save_dict["model"][key] = value

        torch.save(save_dict, os.path.join(save_path, "RenderNet.pth"))
        torch.save(self.projector.state_dict(), os.path.join(save_path, "Projector.pth"))

    def load_checkpoint(self, load_path: str) -> int:
        projector_path = os.path.join(load_path, "Projector.pth")
        rendernet_path = os.path.join(load_path, "RenderNet.pth")
        if not os.path.exists(projector_path):
            raise FileNotFoundError(f"Projector.pth not found in {load_path}")
        if not os.path.exists(rendernet_path):
            raise FileNotFoundError(f"RenderNet.pth not found in {load_path}")

        overwatch.warning(f"loading checkpoints from {load_path}")
        rendernet_ckpt = torch.load(rendernet_path, map_location="cpu")
        saved_backend = rendernet_ckpt.get("backend")
        if saved_backend != "flux":
            raise ValueError(f"Cannot load backend={saved_backend!r} checkpoint into FLUX.")
        if int(rendernet_ckpt.get("checkpoint_version", -1)) != 2:
            raise ValueError(
                "Unsupported lightweight FLUX checkpoint version: "
                f"{rendernet_ckpt.get('checkpoint_version')!r}."
            )

        checkpoint_model = rendernet_ckpt["model"]

        missing, unexpected = self.load_state_dict(checkpoint_model, strict=False)
        overwatch.warning(f"RenderNet missing keys: {sorted({key.split('.')[0] for key in missing})}")
        overwatch.warning(f"RenderNet unexpected keys: {sorted({key.split('.')[0] for key in unexpected})}")

        projector_ckpt = torch.load(projector_path, map_location="cpu")
        missing, unexpected = self.projector.load_state_dict(projector_ckpt, strict=False)
        overwatch.warning(f"Projector missing keys: {missing}")
        overwatch.warning(f"Projector unexpected keys: {unexpected}")
        return int(rendernet_ckpt["global_step"])

    def encode(self, images: torch.Tensor):
        """Turn DINO-normalized images into FLUX context and pooled conditions."""
        if not isinstance(images, torch.Tensor):
            raise TypeError(f"images must be a torch.Tensor, got {images.__class__.__name__}.")

        vision_dtype = next(self.vision_backbone.parameters()).dtype
        images = images.to(device=self.device, dtype=vision_dtype)

        # Frozen vision backbone does not need an autograd graph.
        with torch.no_grad():
            image_embeds = self.vision_backbone(images)
        projector_dtype = next(self.projector.parameters()).dtype
        image_embeds = image_embeds.to(dtype=projector_dtype)
        image_embeds = self.projector(image_embeds)

        if self.training and self.token_dropout:
            kept_tokens = int(torch.randint(1, self.num_token + 1, ()).item())
            image_embeds = image_embeds[:, :kept_tokens]

        # FLUX.1-dev uses its distilled guidance embedding directly.
        condition_head_dtype = next(self.dit_condition_head.parameters()).dtype
        pooled_embeds = self.dit_condition_head(
            image_embeds.to(dtype=condition_head_dtype)
        )

        if image_embeds.shape[-1] != 4096:
            raise ValueError(f"FLUX context must be 4096-D, got {image_embeds.shape[-1]}.")
        if pooled_embeds.shape[-1] != 768:
            raise ValueError(f"FLUX pooled condition must be 768-D, got {pooled_embeds.shape[-1]}.")

        return image_embeds.to(dtype=self.dtype), pooled_embeds.to(dtype=self.dtype)

    def encode_condition_images(self, images: torch.Tensor):
        """Normalize raw [0,1] inputs for DINO and then encode them."""
        return self.encode(self.projector_feature_extractor(images))

    def progress_bar(self, iterable=None, total=None):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        elif not isinstance(self._progress_bar_config, dict):
            raise ValueError(
                f"self._progress_bar_config must be a dict, got {type(self._progress_bar_config)}."
            )

        if iterable is not None:
            return tqdm(iterable, **self._progress_bar_config)
        if total is not None:
            return tqdm(total=total, **self._progress_bar_config)
        raise ValueError("Either total or iterable must be supplied.")

    def get_sigmas(self, timesteps, n_dim=4, dtype=torch.float32):
        sigmas = self.scheduler_copy.sigmas.to(device=self.device, dtype=dtype)
        schedule_timesteps = self.scheduler_copy.timesteps.to(self.device)
        timesteps = timesteps.to(self.device)
        step_indices = [(schedule_timesteps == timestep).nonzero().item() for timestep in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

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
    ):
        """Create raw FLUX VAE noise: [B,16,H/8,W/8]."""
        self._validate_flux_resolution(int(height), int(width))
        shape = (
            int(batch_size),
            int(num_channels_latents),
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )

        if latents is not None:
            if tuple(latents.shape) != shape:
                raise ValueError(f"Provided raw latents have shape {tuple(latents.shape)}, expected {shape}.")
            return latents.to(device=device, dtype=dtype)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"Generator list has length {len(generator)}, but batch_size is {batch_size}."
            )
        return randn_tensor(shape, generator=generator, device=device, dtype=dtype)

    # ========================== FLUX  =============================
    @staticmethod
    def _pack_flux_latents(latents: torch.Tensor) -> torch.Tensor:
        """[B,16,h,w] -> [B,(h/2)*(w/2),64]."""
        if latents.ndim != 4:
            raise ValueError(f"Raw FLUX latents must be 4-D, got {tuple(latents.shape)}.")
        batch_size, channels, height, width = latents.shape
        if height % 2 != 0 or width % 2 != 0:
            raise ValueError(f"Raw latent height/width must be even, got {height}x{width}.")

        latents = latents.reshape(batch_size, channels, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        return latents.reshape(batch_size, (height // 2) * (width // 2), channels * 4)
    @staticmethod
    def _unpack_flux_latents(
        latents: torch.Tensor,
        latent_height: int,
        latent_width: int,
    ) -> torch.Tensor:
        """[B,N,64] -> [B,16,h,w]."""
        if latents.ndim != 3:
            raise ValueError(f"Packed FLUX latents must be 3-D, got {tuple(latents.shape)}.")
        if latent_height % 2 != 0 or latent_width % 2 != 0:
            raise ValueError(f"Raw latent height/width must be even, got {latent_height}x{latent_width}.")

        batch_size, num_patches, packed_channels = latents.shape
        expected_patches = (latent_height // 2) * (latent_width // 2)
        if num_patches != expected_patches:
            raise ValueError(f"Packed sequence N={num_patches}, expected {expected_patches}.")
        if packed_channels % 4 != 0:
            raise ValueError(f"Packed channels {packed_channels} are not divisible by four.")

        channels = packed_channels // 4
        latents = latents.reshape(
            batch_size,
            latent_height // 2,
            latent_width // 2,
            channels,
            2,
            2,
        )
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        return latents.reshape(batch_size, channels, latent_height, latent_width)

    @staticmethod
    def _prepare_flux_image_ids(latent_height: int, latent_width: int, device, dtype):
        """Create FLUX image RoPE IDs with shape [N,3], without a batch axis."""
        token_height = latent_height // 2
        token_width = latent_width // 2
        image_ids = torch.zeros(token_height, token_width, 3, device=device, dtype=dtype)
        image_ids[..., 1] = torch.arange(token_height, device=device, dtype=dtype)[:, None]
        image_ids[..., 2] = torch.arange(token_width, device=device, dtype=dtype)[None, :]
        return image_ids.reshape(token_height * token_width, 3)

    @staticmethod
    def _prepare_flux_text_ids(num_tokens: int, device, dtype):
        """Visual Projector tokens occupy FLUX's text stream; their IDs are zero."""
        return torch.zeros(int(num_tokens), 3, device=device, dtype=dtype)

    def _flux_guidance(self, batch_size: int, device):
        if not bool(_cfg_get(self.DiT.config, "guidance_embeds", False)):
            return None
        return torch.full(
            (int(batch_size),),
            self.guidance_scale,
            device=device,
            dtype=torch.float32,
        )

    def vae_encode(self, images):
        vae_dtype = next(self.vae.parameters()).dtype
        images = images.to(device=self.device, dtype=vae_dtype)
        with torch.no_grad():
            latents = self.vae.encode(images).latent_dist.sample()
        scaling_factor, shift_factor = self._vae_scaling_values()
        latents = (latents - shift_factor) * scaling_factor
        return latents.to(dtype=self.dtype)

    def vae_decode(self, latents):
        scaling_factor, shift_factor = self._vae_scaling_values()
        latents = latents / scaling_factor + shift_factor
        vae_dtype = next(self.vae.parameters()).dtype
        latents = latents.to(device=self.device, dtype=vae_dtype)
        return self.vae.decode(latents, return_dict=False)[0]

    def train_step(
        self,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        criterion: nn.Module,
    ) -> Dict[str, Any]:
        if criterion is None:
            raise ValueError("train_step requires the diffusion criterion.")

        images = inputs["images"]
        self._validate_flux_resolution(images.shape[-2], images.shape[-1])
        batch_size = images.shape[0]

        image_embeddings, pooled_projections = self.encode_condition_images(images)
        clean_latents = self.vae_encode(self.dit_feature_extractor(images)).to(dtype=self.dtype)
        noise = torch.randn_like(clean_latents)

        density = sample_flow_timestep_density(
            weighting_scheme=self.weighting_scheme,
            batch_size=batch_size,
            logit_mean=self.logit_mean,
            logit_std=self.logit_std,
            mode_scale=self.mode_scale,
        )
        indices = (density * self.scheduler_copy.config.num_train_timesteps).long()
        indices = indices.clamp(max=self.scheduler_copy.config.num_train_timesteps - 1)
        # `indices` live on CPU. Index the scheduler's CPU tensors once and
        # transfer only the current batch, avoiding per-sample MUSA
        # nonzero().item() synchronization in get_sigmas().
        schedule_indices = indices.to(
            device=self.scheduler_copy.timesteps.device,
            dtype=torch.long,
        )
        timesteps = self.scheduler_copy.timesteps[schedule_indices].to(
            device=clean_latents.device
        )
        sigma_indices = indices.to(
            device=self.scheduler_copy.sigmas.device,
            dtype=torch.long,
        )
        sigmas = self.scheduler_copy.sigmas[sigma_indices].to(
            device=clean_latents.device,
            dtype=clean_latents.dtype,
        )
        while sigmas.ndim < clean_latents.ndim:
            sigmas = sigmas.unsqueeze(-1)

        # Flow-matching forward process remains the same.
        noisy_latents = (1.0 - sigmas) * clean_latents + sigmas * noise

        latent_height, latent_width = clean_latents.shape[-2:]
        packed_noisy_latents = self._pack_flux_latents(noisy_latents)
        image_ids = self._prepare_flux_image_ids(
            latent_height,
            latent_width,
            device=clean_latents.device,
            dtype=image_embeddings.dtype,
        )
        text_ids = self._prepare_flux_text_ids(
            image_embeddings.shape[1],
            device=clean_latents.device,
            dtype=image_embeddings.dtype,
        )

        packed_model_pred = self.DiT(
            hidden_states=packed_noisy_latents,           # B x N x 64
            timestep=timesteps.to(packed_noisy_latents.dtype) / 1000,
            guidance=self._flux_guidance(batch_size, clean_latents.device),
            encoder_hidden_states=image_embeddings,       # B x T x 4096
            pooled_projections=pooled_projections,         # B x 768
            txt_ids=text_ids,                              # T x 3
            img_ids=image_ids,                             # N x 3
            return_dict=False,
        )[0]

        # Official FLUX training unpacks prediction before the raw-latent loss.
        model_pred = self._unpack_flux_latents(
            packed_model_pred,
            latent_height,
            latent_width,
        )
        target = noise - clean_latents
        if model_pred.shape != target.shape:
            raise RuntimeError(
                f"Prediction/target mismatch: {tuple(model_pred.shape)} vs {tuple(target.shape)}."
            )

        weighting = compute_flow_loss_weighting(
            weighting_scheme=self.weighting_scheme,
            sigmas=sigmas,
        )
        outputs["loss"] = criterion(weighting, model_pred, target)
        return outputs

    def _prepare_generation_latents(self, batch_size: int, generator):
        raw_latents = self.prepare_latents(
            batch_size,
            self.num_channels_latents,
            self.height,
            self.width,
            self.dtype,
            self.device,
            generator,
        )
        return self._pack_flux_latents(raw_latents)

    def _prepare_inference_timesteps(self, packed_latents):
        sigmas = np.linspace(
            1.0,
            1.0 / self.num_inference_steps,
            self.num_inference_steps,
        )
        image_seq_len = int(packed_latents.shape[1])
        mu = calculate_shift(
            image_seq_len,
            int(_cfg_get(self.scheduler.config, "base_image_seq_len", 256)),
            int(_cfg_get(self.scheduler.config, "max_image_seq_len", 4096)),
            float(_cfg_get(self.scheduler.config, "base_shift", 0.5)),
            float(_cfg_get(self.scheduler.config, "max_shift", 1.15)),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            self.num_inference_steps,
            self.device,
            sigmas=sigmas,
            mu=mu,
        )
        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(0)
        return timesteps, num_inference_steps

    def _generate_from_embeddings(
        self,
        image_embeddings,
        pooled_projections,
        batch_size,
        generator,
        initial_latents=None,
    ):
        if image_embeddings.shape[0] != batch_size or pooled_projections.shape[0] != batch_size:
            raise ValueError(
                "FLUX condition batch must equal generation batch."
            )

        if initial_latents is None:
            latents = self._prepare_generation_latents(batch_size, generator)
        else:
            latents = initial_latents.clone().to(device=self.device, dtype=self.dtype)

        expected_shape = (
            batch_size,
            (self.height // 16) * (self.width // 16),
            64,
        )
        if tuple(latents.shape) != expected_shape:
            raise ValueError(f"Packed noise has shape {tuple(latents.shape)}, expected {expected_shape}.")

        timesteps, num_inference_steps = self._prepare_inference_timesteps(latents)
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        latent_height = self.height // self.vae_scale_factor
        latent_width = self.width // self.vae_scale_factor
        image_ids = self._prepare_flux_image_ids(
            latent_height,
            latent_width,
            device=self.device,
            dtype=image_embeddings.dtype,
        )
        text_ids = self._prepare_flux_text_ids(
            image_embeddings.shape[1],
            device=self.device,
            dtype=image_embeddings.dtype,
        )
        guidance = self._flux_guidance(batch_size, self.device)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for index, timestep_value in enumerate(timesteps):
                timestep = timestep_value.expand(batch_size).to(dtype=latents.dtype)
                model_pred = self.DiT(
                    hidden_states=latents,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    encoder_hidden_states=image_embeddings,
                    pooled_projections=pooled_projections,
                    txt_ids=text_ids,
                    img_ids=image_ids,
                    return_dict=False,
                )[0]

                latents_dtype = latents.dtype
                latents = self.scheduler.step(
                    model_pred,
                    timestep_value,
                    latents,
                    return_dict=False,
                )[0]
                if latents.dtype != latents_dtype:
                    latents = latents.to(latents_dtype)

                if index == len(timesteps) - 1 or (
                    (index + 1) > num_warmup_steps
                    and (index + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

        raw_latents = self._unpack_flux_latents(latents, latent_height, latent_width)
        return self.vae_decode(raw_latents.to(dtype=self.dtype))

    @torch.no_grad()
    def eval_step(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> Dict[str, Any]:
        images = inputs["images"]
        image_embeddings, pooled_projections = self.encode_condition_images(images)
        outputs["images"] = self._generate_from_embeddings(
            image_embeddings,
            pooled_projections,
            batch_size=images.shape[0],
            generator=inputs["generator"],
        )
        return outputs

    @torch.no_grad()
    def interpolation_eval(self, image1, image2, generator, tokens=None, num_interpolation=5):
        if image1.shape[0] != 1 or image2.shape[0] != 1:
            raise ValueError("interpolation_eval accepts one image in each endpoint batch.")

        embeds1, pooled1 = self.encode_condition_images(image1)
        embeds2, pooled2 = self.encode_condition_images(image2)

        # One shared initial noise makes alpha the only changing quantity.
        shared_initial_latents = self._prepare_generation_latents(1, generator)
        generated_images = []
        alphas = torch.linspace(
            0,
            1,
            steps=int(num_interpolation),
            device=self.device,
            dtype=embeds1.dtype,
        )

        for alpha in alphas:
            image_embeddings = embeds1 * (1 - alpha) + embeds2 * alpha
            pooled_projections = pooled1 * (1 - alpha) + pooled2 * alpha
            if tokens is not None and len(tokens) > 0:
                image_embeddings[:, tokens, :] = embeds1[:, tokens, :]

            generated_images.append(
                self._generate_from_embeddings(
                    image_embeddings,
                    pooled_projections,
                    batch_size=1,
                    generator=generator,
                    initial_latents=shared_initial_latents,
                )
            )

        return torch.cat(generated_images, dim=0)

    @torch.no_grad()
    def get_delta_action(self, start, end):
        embeds_start, pooled_start = self.encode_condition_images(start)
        embeds_end, pooled_end = self.encode_condition_images(end)
        return embeds_end - embeds_start, pooled_end - pooled_start

    @torch.no_grad()
    def delta_interpolation(self, image, start, end, generator):
        image_embeddings, pooled_projections = self.encode_condition_images(image)
        delta_embeddings, delta_pooled = self.get_delta_action(start, end)
        return self._generate_from_embeddings(
            image_embeddings + delta_embeddings,
            pooled_projections + delta_pooled,
            batch_size=image.shape[0],
            generator=generator,
        )

    def forward(self, inputs, **kwargs):
        outputs = {}
        # Training samples noise directly in train_step and never consumes this
        # generator. Creating a MUSA Generator every optimizer step adds an
        # unnecessary device operation; generation/evaluation still gets it.
        created_generator = not self.training and "generator" not in inputs
        if created_generator:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(self.seed)
            inputs["generator"] = generator

        try:
            if self.training:
                outputs = self.train_step(inputs, outputs, **kwargs)
            else:
                outputs = self.eval_step(inputs, outputs)
        finally:
            if created_generator:
                inputs.pop("generator", None)
        return outputs


if __name__ == "__main__":
    from omegaconf import OmegaConf

    args = OmegaConf.load("./configs/debug_flux.yaml")
    device = get_accelerator_device()
