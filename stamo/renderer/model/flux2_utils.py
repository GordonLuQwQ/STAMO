# """Pure FLUX.2 [klein] tensor helpers and architecture contracts.

# The functions in this module intentionally do not import Diffusers or PEFT.  This
# keeps the shape/normalization contract testable on CPU without downloading a
# multi-billion-parameter checkpoint and makes accidental FLUX.1 behavior easy
# to detect.
# """

# from __future__ import annotations

# from dataclasses import dataclass
# from typing import Any, Iterable

# import torch


# FLUX2_KLEIN_4B_JOINT_ATTENTION_DIM = 7_680
# FLUX2_KLEIN_4B_IN_CHANNELS = 128
# FLUX2_KLEIN_4B_RAW_LATENT_CHANNELS = 32
# FLUX2_KLEIN_4B_DOUBLE_BLOCKS = 5
# FLUX2_KLEIN_4B_SINGLE_BLOCKS = 20
# FLUX2_KLEIN_4B_ATTENTION_HEAD_DIM = 128
# FLUX2_KLEIN_4B_ATTENTION_HEADS = 24
# FLUX2_KLEIN_4B_AXES_DIMS_ROPE = (32, 32, 32, 32)

# FLUX2_KLEIN_9B_JOINT_ATTENTION_DIM = 12_288
# FLUX2_KLEIN_9B_IN_CHANNELS = 128
# FLUX2_KLEIN_9B_RAW_LATENT_CHANNELS = 32
# FLUX2_KLEIN_9B_DOUBLE_BLOCKS = 8
# FLUX2_KLEIN_9B_SINGLE_BLOCKS = 24
# FLUX2_KLEIN_9B_ATTENTION_HEAD_DIM = 128
# FLUX2_KLEIN_9B_ATTENTION_HEADS = 32
# FLUX2_KLEIN_9B_AXES_DIMS_ROPE = (32, 32, 32, 32)


# @dataclass(frozen=True)
# class Flux2KleinVariantContract:
#     """Exact Diffusers architecture signature for one supported Base model."""

#     model_variant: str
#     joint_attention_dim: int
#     in_channels: int
#     raw_latent_channels: int
#     num_layers: int
#     num_single_layers: int
#     attention_head_dim: int
#     num_attention_heads: int
#     patch_size: int = 1
#     axes_dims_rope: tuple[int, ...] = (32, 32, 32, 32)
#     mlp_ratio: float = 3.0
#     eps: float = 1e-6
#     rope_theta: int = 2_000
#     timestep_guidance_channels: int = 256
#     out_channels: int | None = None
#     vae_patch_size: tuple[int, ...] = (2, 2)


# FLUX2_KLEIN_VARIANTS = {
#     "base-4b": Flux2KleinVariantContract(
#         model_variant="base-4b",
#         joint_attention_dim=FLUX2_KLEIN_4B_JOINT_ATTENTION_DIM,
#         in_channels=FLUX2_KLEIN_4B_IN_CHANNELS,
#         raw_latent_channels=FLUX2_KLEIN_4B_RAW_LATENT_CHANNELS,
#         num_layers=FLUX2_KLEIN_4B_DOUBLE_BLOCKS,
#         num_single_layers=FLUX2_KLEIN_4B_SINGLE_BLOCKS,
#         attention_head_dim=FLUX2_KLEIN_4B_ATTENTION_HEAD_DIM,
#         num_attention_heads=FLUX2_KLEIN_4B_ATTENTION_HEADS,
#         axes_dims_rope=FLUX2_KLEIN_4B_AXES_DIMS_ROPE,
#     ),
#     "base-9b": Flux2KleinVariantContract(
#         model_variant="base-9b",
#         joint_attention_dim=FLUX2_KLEIN_9B_JOINT_ATTENTION_DIM,
#         in_channels=FLUX2_KLEIN_9B_IN_CHANNELS,
#         raw_latent_channels=FLUX2_KLEIN_9B_RAW_LATENT_CHANNELS,
#         num_layers=FLUX2_KLEIN_9B_DOUBLE_BLOCKS,
#         num_single_layers=FLUX2_KLEIN_9B_SINGLE_BLOCKS,
#         attention_head_dim=FLUX2_KLEIN_9B_ATTENTION_HEAD_DIM,
#         num_attention_heads=FLUX2_KLEIN_9B_ATTENTION_HEADS,
#         axes_dims_rope=FLUX2_KLEIN_9B_AXES_DIMS_ROPE,
#     ),
# }


# def config_value(config: Any, key: str, default: Any = None) -> Any:
#     """Read a value from a dict, Diffusers FrozenDict, or config object."""

#     if config is None:
#         return default
#     if hasattr(config, "get"):
#         return config.get(key, default)
#     return getattr(config, key, default)


# def _transformer_signature(config: Any) -> dict[str, Any]:
#     """Normalize the fields that distinguish supported Klein architectures."""

#     return {
#         "joint_attention_dim": int(config_value(config, "joint_attention_dim", -1)),
#         "in_channels": int(config_value(config, "in_channels", -1)),
#         "num_layers": int(config_value(config, "num_layers", -1)),
#         "num_single_layers": int(config_value(config, "num_single_layers", -1)),
#         "attention_head_dim": int(config_value(config, "attention_head_dim", -1)),
#         "num_attention_heads": int(config_value(config, "num_attention_heads", -1)),
#         "patch_size": int(config_value(config, "patch_size", -1)),
#         "axes_dims_rope": tuple(
#             int(value) for value in config_value(config, "axes_dims_rope", ())
#         ),
#         "mlp_ratio": float(config_value(config, "mlp_ratio", -1.0)),
#         "eps": float(config_value(config, "eps", -1.0)),
#         "rope_theta": int(config_value(config, "rope_theta", -1)),
#         "timestep_guidance_channels": int(
#             config_value(config, "timestep_guidance_channels", -1)
#         ),
#         "out_channels": config_value(config, "out_channels", "<missing>"),
#     }


# def _contract_signature(contract: Flux2KleinVariantContract) -> dict[str, Any]:
#     return {
#         "joint_attention_dim": contract.joint_attention_dim,
#         "in_channels": contract.in_channels,
#         "num_layers": contract.num_layers,
#         "num_single_layers": contract.num_single_layers,
#         "attention_head_dim": contract.attention_head_dim,
#         "num_attention_heads": contract.num_attention_heads,
#         "patch_size": contract.patch_size,
#         "axes_dims_rope": contract.axes_dims_rope,
#         "mlp_ratio": contract.mlp_ratio,
#         "eps": contract.eps,
#         "rope_theta": contract.rope_theta,
#         "timestep_guidance_channels": contract.timestep_guidance_channels,
#         "out_channels": contract.out_channels,
#     }


# def resolve_flux2_klein_variant(transformer_config: Any) -> Flux2KleinVariantContract:
#     """Identify Base 4B or Base 9B from its complete architecture, not its path."""

#     actual = _transformer_signature(transformer_config)
#     matches = [
#         contract
#         for contract in FLUX2_KLEIN_VARIANTS.values()
#         if actual == _contract_signature(contract)
#     ]
#     if len(matches) == 1:
#         return matches[0]

#     supported = {
#         name: _contract_signature(contract)
#         for name, contract in FLUX2_KLEIN_VARIANTS.items()
#     }
#     raise ValueError(
#         "Unsupported FLUX.2 Klein transformer architecture: "
#         f"actual={actual!r}; supported={supported!r}."
#     )


# def validate_flux2_klein_config_contract(
#     transformer_config: Any,
#     vae_config: Any,
#     projector_output_dim: int,
# ) -> Flux2KleinVariantContract:
#     """Validate config-only Base 4B/9B and projector compatibility."""

#     contract = resolve_flux2_klein_variant(transformer_config)
#     if bool(config_value(transformer_config, "guidance_embeds", True)):
#         raise ValueError(
#             f"FLUX.2 Klein {contract.model_variant} training requires "
#             "guidance_embeds=False from an undistilled Base checkpoint."
#         )
#     checks = {
#         "vae.latent_channels": (
#             int(config_value(vae_config, "latent_channels", -1)),
#             contract.raw_latent_channels,
#         ),
#         "vae.patch_size": (
#             tuple(int(value) for value in config_value(vae_config, "patch_size", ())),
#             contract.vae_patch_size,
#         ),
#         "projector.output_align_dim": (
#             int(projector_output_dim),
#             contract.joint_attention_dim,
#         ),
#     }
#     mismatches = [
#         f"{name}={actual!r} (expected {expected!r})"
#         for name, (actual, expected) in checks.items()
#         if actual != expected
#     ]
#     if mismatches:
#         raise ValueError(
#             f"Invalid FLUX.2 Klein {contract.model_variant} contract: "
#             + "; ".join(mismatches)
#         )
#     return contract


# def validate_flux2_klein_contract(
#     transformer: Any,
#     vae: Any,
#     projector_output_dim: int,
# ) -> Flux2KleinVariantContract:
#     """Validate a loaded Diffusers Base 4B/9B model and return its variant.

#     The distilled and Base transformers share dimensions within each size. The
#     caller separately rejects ``model_index.json`` with
#     ``is_distilled=true``; STAMO needs the undistilled Base checkpoint for
#     training.
#     """

#     transformer_config = getattr(transformer, "config", None)
#     vae_config = getattr(vae, "config", None)

#     expected_transformer_class = "Flux2Transformer2DModel"
#     if transformer.__class__.__name__ != expected_transformer_class:
#         raise TypeError(
#             "Expected a Flux2Transformer2DModel for FLUX.2 Klein, got "
#             f"{transformer.__class__.__module__}.{transformer.__class__.__name__}."
#         )
#     expected_vae_class = "AutoencoderKLFlux2"
#     if vae.__class__.__name__ != expected_vae_class:
#         raise TypeError(
#             "Expected an AutoencoderKLFlux2 for FLUX.2 Klein, got "
#             f"{vae.__class__.__module__}.{vae.__class__.__name__}."
#         )

#     contract = validate_flux2_klein_config_contract(
#         transformer_config,
#         vae_config,
#         projector_output_dim,
#     )
#     if not hasattr(vae, "bn"):
#         raise ValueError("AutoencoderKLFlux2 is missing the checkpoint BN latent statistics.")
#     return contract


# def validate_flux2_klein_4b_contract(
#     transformer: Any,
#     vae: Any,
#     projector_output_dim: int,
# ) -> None:
#     """Backward-compatible strict Base 4B validator."""

#     contract = validate_flux2_klein_contract(transformer, vae, projector_output_dim)
#     if contract.model_variant != "base-4b":
#         raise ValueError(
#             "The checkpoint is not FLUX.2 Klein Base 4B: "
#             f"resolved {contract.model_variant}."
#         )


# def validate_flux2_klein_9b_contract(
#     transformer: Any,
#     vae: Any,
#     projector_output_dim: int,
# ) -> None:
#     """Strict Base 9B validator for callers that require exactly 9B."""

#     contract = validate_flux2_klein_contract(transformer, vae, projector_output_dim)
#     if contract.model_variant != "base-9b":
#         raise ValueError(
#             "The checkpoint is not FLUX.2 Klein Base 9B: "
#             f"resolved {contract.model_variant}."
#         )


# def validate_flux2_scheduler_contract(scheduler: Any) -> None:
#     """Validate the Klein Base flow scheduler used by both supported sizes."""

#     scheduler_config = getattr(scheduler, "config", None)
#     checks = {
#         "num_train_timesteps": (
#             int(config_value(scheduler_config, "num_train_timesteps", -1)),
#             1_000,
#         ),
#         "use_dynamic_shifting": (
#             bool(config_value(scheduler_config, "use_dynamic_shifting", False)),
#             True,
#         ),
#         "time_shift_type": (
#             str(config_value(scheduler_config, "time_shift_type", "")),
#             "exponential",
#         ),
#         "stochastic_sampling": (
#             bool(config_value(scheduler_config, "stochastic_sampling", True)),
#             False,
#         ),
#         "invert_sigmas": (
#             bool(config_value(scheduler_config, "invert_sigmas", True)),
#             False,
#         ),
#     }
#     mismatches = [
#         f"{name}={actual!r} (expected {expected!r})"
#         for name, (actual, expected) in checks.items()
#         if actual != expected
#     ]
#     if mismatches:
#         raise ValueError("Invalid FLUX.2 Klein Base scheduler: " + "; ".join(mismatches))


# def patchify_latents(latents: torch.Tensor) -> torch.Tensor:
#     """Convert raw VAE latents [B,C,H,W] to [B,4C,H/2,W/2]."""

#     if latents.ndim != 4:
#         raise ValueError(f"Raw FLUX.2 latents must be 4-D, got {tuple(latents.shape)}.")
#     batch_size, channels, height, width = latents.shape
#     if height % 2 or width % 2:
#         raise ValueError(f"Raw FLUX.2 latent height/width must be even, got {height}x{width}.")
#     latents = latents.reshape(batch_size, channels, height // 2, 2, width // 2, 2)
#     latents = latents.permute(0, 1, 3, 5, 2, 4)
#     return latents.reshape(batch_size, channels * 4, height // 2, width // 2)


# def unpatchify_latents(latents: torch.Tensor) -> torch.Tensor:
#     """Invert :func:`patchify_latents`."""

#     if latents.ndim != 4:
#         raise ValueError(f"Patchified FLUX.2 latents must be 4-D, got {tuple(latents.shape)}.")
#     batch_size, channels, height, width = latents.shape
#     if channels % 4:
#         raise ValueError(f"Patchified FLUX.2 channels must be divisible by four, got {channels}.")
#     latents = latents.reshape(batch_size, channels // 4, 2, 2, height, width)
#     latents = latents.permute(0, 1, 4, 2, 5, 3)
#     return latents.reshape(batch_size, channels // 4, height * 2, width * 2)


# def pack_latents(latents: torch.Tensor) -> torch.Tensor:
#     """Flatten patchified latents [B,C,H,W] into transformer tokens [B,HW,C]."""

#     if latents.ndim != 4:
#         raise ValueError(f"Patchified FLUX.2 latents must be 4-D, got {tuple(latents.shape)}.")
#     batch_size, channels, height, width = latents.shape
#     return latents.reshape(batch_size, channels, height * width).permute(0, 2, 1)


# def unpack_latents(latents: torch.Tensor, latent_height: int, latent_width: int) -> torch.Tensor:
#     """Invert standard-grid FLUX.2 token packing into [B,C,H,W]."""

#     if latents.ndim != 3:
#         raise ValueError(f"Packed FLUX.2 latents must be 3-D, got {tuple(latents.shape)}.")
#     batch_size, sequence_length, channels = latents.shape
#     expected_length = int(latent_height) * int(latent_width)
#     if sequence_length != expected_length:
#         raise ValueError(
#             f"Packed FLUX.2 sequence length is {sequence_length}, expected {expected_length}."
#         )
#     return latents.permute(0, 2, 1).reshape(
#         batch_size,
#         channels,
#         int(latent_height),
#         int(latent_width),
#     )


# def prepare_latent_ids(latents: torch.Tensor) -> torch.Tensor:
#     """Return batched FLUX.2 image RoPE coordinates in (T,H,W,L) order."""

#     if latents.ndim != 4:
#         raise ValueError(f"FLUX.2 latent IDs require a 4-D tensor, got {tuple(latents.shape)}.")
#     batch_size, _, height, width = latents.shape
#     device = latents.device
#     # Match Diffusers: build this tiny coordinate table on CPU first. MUSA's
#     # PrivateUse1 backend does not consistently register cartesian_prod.
#     ids = torch.cartesian_prod(
#         torch.arange(1, dtype=torch.int64),
#         torch.arange(height, dtype=torch.int64),
#         torch.arange(width, dtype=torch.int64),
#         torch.arange(1, dtype=torch.int64),
#     )
#     ids = ids.to(device=device)
#     return ids.unsqueeze(0).expand(batch_size, -1, -1)


# def prepare_text_ids(embeddings: torch.Tensor) -> torch.Tensor:
#     """Return visual-condition token IDs [0,0,0,L] with shape [B,L,4]."""

#     if embeddings.ndim != 3:
#         raise ValueError(
#             "FLUX.2 condition embeddings must be [batch,tokens,channels], got "
#             f"{tuple(embeddings.shape)}."
#         )
#     batch_size, sequence_length, _ = embeddings.shape
#     device = embeddings.device
#     ids = torch.zeros(1, sequence_length, 4, dtype=torch.int64)
#     ids[..., 3] = torch.arange(sequence_length, dtype=torch.int64)
#     ids = ids.to(device=device)
#     return ids.expand(batch_size, -1, -1)


# def _vae_bn_statistics(vae: Any, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
#     if not hasattr(vae, "bn"):
#         raise ValueError("AutoencoderKLFlux2 is missing its latent BatchNorm module.")
#     mean = vae.bn.running_mean.reshape(1, -1, 1, 1).to(reference.device, reference.dtype)
#     eps = float(config_value(getattr(vae, "config", None), "batch_norm_eps", 1e-4))
#     # Preserve the official order: calculate sqrt in the checkpoint buffer's
#     # dtype first, then cast the completed standard deviation to the latents.
#     std = torch.sqrt(vae.bn.running_var.reshape(1, -1, 1, 1) + eps).to(
#         reference.device,
#         reference.dtype,
#     )
#     if mean.shape[1] != reference.shape[1]:
#         raise ValueError(
#             "FLUX.2 VAE BN channel mismatch: "
#             f"stats={mean.shape[1]}, latents={reference.shape[1]}."
#         )
#     return mean, std


# def normalize_vae_latents(vae: Any, patchified_latents: torch.Tensor) -> torch.Tensor:
#     """Apply the checkpoint's post-patchify latent BatchNorm statistics."""

#     mean, std = _vae_bn_statistics(vae, patchified_latents)
#     return (patchified_latents - mean) / std


# def denormalize_vae_latents(vae: Any, normalized_latents: torch.Tensor) -> torch.Tensor:
#     """Invert :func:`normalize_vae_latents`."""

#     mean, std = _vae_bn_statistics(vae, normalized_latents)
#     return normalized_latents * std + mean


# def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
#     """Official FLUX.2 Klein dynamic-shift fit used during inference."""

#     a1, b1 = 8.73809524e-05, 1.89833333
#     a2, b2 = 0.00016927, 0.45666666
#     if image_seq_len > 4_300:
#         return float(a2 * image_seq_len + b2)
#     m_200 = a2 * image_seq_len + b2
#     m_10 = a1 * image_seq_len + b1
#     slope = (m_200 - m_10) / 190.0
#     intercept = m_200 - 200.0 * slope
#     return float(slope * num_steps + intercept)


# def resolve_flux2_lora_targets(transformer: Any) -> list[str]:
#     """Build the Diffusers Klein LoRA target list from the loaded block count."""

#     single_blocks = getattr(transformer, "single_transformer_blocks", None)
#     if not isinstance(single_blocks, torch.nn.ModuleList):
#         raise TypeError("FLUX.2 transformer must expose single_transformer_blocks as ModuleList.")
#     if not single_blocks:
#         raise ValueError("FLUX.2 transformer contains no single-stream blocks.")

#     targets = ["to_k", "to_q", "to_v", "to_out.0", "to_qkv_mlp_proj"]
#     targets.extend(
#         f"single_transformer_blocks.{index}.attn.to_out"
#         for index in range(len(single_blocks))
#     )
#     return targets


# def trainable_parameter_summary(named_parameters: Iterable[tuple[str, torch.nn.Parameter]]) -> dict[str, int]:
#     """Return deterministic trainable/frozen element counts for logging/tests."""

#     trainable = 0
#     frozen = 0
#     for _, parameter in named_parameters:
#         if parameter.requires_grad:
#             trainable += parameter.numel()
#         else:
#             frozen += parameter.numel()
#     return {"trainable": int(trainable), "frozen": int(frozen), "total": int(trainable + frozen)}

"""Small FLUX.2 tensor helpers used by the StaMo-shaped renderer."""

from typing import Any

import torch


def patchify_latents(latents: torch.Tensor) -> torch.Tensor:
    batch, channels, height, width = latents.shape
    latents = latents.reshape(batch, channels, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 1, 3, 5, 2, 4)
    return latents.reshape(batch, channels * 4, height // 2, width // 2)


def unpatchify_latents(latents: torch.Tensor) -> torch.Tensor:
    batch, channels, height, width = latents.shape
    latents = latents.reshape(batch, channels // 4, 2, 2, height, width)
    latents = latents.permute(0, 1, 4, 2, 5, 3)
    return latents.reshape(batch, channels // 4, height * 2, width * 2)


def pack_latents(latents: torch.Tensor) -> torch.Tensor:
    batch, channels, height, width = latents.shape
    return latents.reshape(batch, channels, height * width).permute(0, 2, 1)


def unpack_latents(
    latents: torch.Tensor,
    latent_height: int,
    latent_width: int,
) -> torch.Tensor:
    batch, _, channels = latents.shape
    return latents.permute(0, 2, 1).reshape(
        batch,
        channels,
        latent_height,
        latent_width,
    )


def prepare_latent_ids(latents: torch.Tensor) -> torch.Tensor:
    batch, _, height, width = latents.shape
    # Build the coordinate table on CPU because MUSA does not implement every
    # cartesian-product kernel used by the official Diffusers helper.
    ids = torch.cartesian_prod(
        torch.arange(1, dtype=torch.int64),
        torch.arange(height, dtype=torch.int64),
        torch.arange(width, dtype=torch.int64),
        torch.arange(1, dtype=torch.int64),
    ).to(latents.device)
    return ids.unsqueeze(0).expand(batch, -1, -1)


# def prepare_text_ids(embeddings: torch.Tensor) -> torch.Tensor:
#     batch, sequence_length, _ = embeddings.shape
#     ids = torch.zeros(1, sequence_length, 4, dtype=torch.int64)
#     ids[..., 3] = torch.arange(sequence_length, dtype=torch.int64)
#     return ids.to(embeddings.device).expand(batch, -1, -1)
def prepare_text_ids(embeddings: torch.Tensor) -> torch.Tensor:
    """Return neutral RoPE IDs for the unordered Q-Former context tokens."""
    batch, sequence_length, _ = embeddings.shape
    return torch.zeros(
        batch,
        sequence_length,
        4,
        dtype=torch.int64,
        device=embeddings.device,
    )

def _vae_statistics(
    vae: Any,
    reference: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    mean = vae.bn.running_mean.reshape(1, -1, 1, 1).to(
        reference.device,
        reference.dtype,
    )
    eps = float(getattr(vae.config, "batch_norm_eps", 1e-4))
    std = torch.sqrt(
        vae.bn.running_var.reshape(1, -1, 1, 1) + eps
    ).to(reference.device, reference.dtype)
    return mean, std


def normalize_vae_latents(
    vae: Any,
    patchified_latents: torch.Tensor,
) -> torch.Tensor:
    mean, std = _vae_statistics(vae, patchified_latents)
    return (patchified_latents - mean) / std


def denormalize_vae_latents(
    vae: Any,
    normalized_latents: torch.Tensor,
) -> torch.Tensor:
    mean, std = _vae_statistics(vae, normalized_latents)
    return normalized_latents * std + mean


def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666
    if image_seq_len > 4_300:
        return float(a2 * image_seq_len + b2)
    mu_200 = a2 * image_seq_len + b2
    mu_10 = a1 * image_seq_len + b1
    slope = (mu_200 - mu_10) / 190.0
    return float(slope * num_steps + mu_200 - 200.0 * slope)
