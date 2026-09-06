"""Stress the production Q-Former BF16 backward on every MUSA rank."""

from __future__ import annotations

import argparse
import os
import time
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn.functional as functional
from diffusers.models.attention_processor import AttnProcessor
from omegaconf import OmegaConf

from STAMO.stamo.renderer.model.projector2 import QformerProjector

try:
    import torch_musa  # noqa: F401
except ImportError as exc:  # pragma: no cover - runs only on the MUSA host
    raise RuntimeError("torch_musa is required for this smoke test") from exc


EXPECTED_PARAMETER_COUNT = 53_165_568


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--expected-world-size", type=int, default=None)
    parser.add_argument("--local_rank", type=int, default=None)
    args, _ = parser.parse_known_args()
    return args


def _all_attention_gradients(model: QformerProjector):
    for layer_index, block in enumerate(model.qformer_layers):
        for attention_name, attention in (
            ("attn1", block.cross_attn.attn1),
            ("attn2", block.cross_attn.attn2),
        ):
            for projection_name in ("to_q", "to_k", "to_v"):
                projection = getattr(attention, projection_name)
                yield (
                    f"layer{layer_index}.{attention_name}.{projection_name}.weight",
                    projection.weight.grad,
                )


def main() -> None:
    args = parse_args()
    if args.iterations <= 0:
        raise ValueError("iterations must be positive")
    if args.timeout_seconds < 60:
        raise ValueError("timeout-seconds must be at least 60")
    if args.expected_world_size is not None and args.expected_world_size <= 0:
        raise ValueError("expected-world-size must be positive")

    local_rank = (
        int(args.local_rank)
        if args.local_rank is not None
        else int(os.environ.get("LOCAL_RANK", 0))
    )
    musa = getattr(torch, "musa", None)
    if musa is None or not musa.is_available():
        raise RuntimeError("MUSA is not available")
    musa.set_device(local_rank)
    device = torch.device("musa", local_rank)

    dist.init_process_group(
        backend="mccl",
        timeout=timedelta(seconds=args.timeout_seconds),
    )
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if args.expected_world_size is not None and world_size != args.expected_world_size:
        raise RuntimeError(
            f"expected world_size={args.expected_world_size}, got {world_size}"
        )

    original_sdpa = getattr(functional, "scaled_dot_product_attention", None)
    sdpa_calls = 0

    def forbidden_sdpa(*unused_args, **unused_kwargs):
        nonlocal sdpa_calls
        sdpa_calls += 1
        raise RuntimeError("Q-Former unexpectedly called fused SDPA on MUSA")

    functional.scaled_dot_product_attention = forbidden_sdpa

    try:
        # Keep model initialization identical across ranks while giving every
        # rank and iteration a different, reproducible training sample.
        torch.manual_seed(20260806)
        config = OmegaConf.load(args.config)
        projector = QformerProjector(
            config,
            patches=int(config.projector.input_token_count),
            channels=int(config.projector.input_dim),
        )
        contract = projector.architecture_contract()
        required_contract = {
            "architecture_version": "dino_flux2_qformer_v2_4l768",
            "legacy_v1": False,
            "input_token_count": 196,
            "input_dim": 768,
            "num_token": 4,
            "num_attn_layers": 4,
            "hidden_dim": 768,
            "num_attention_heads": 12,
            "attention_head_dim": 64,
            "attention_backend": "legacy_upcast",
            "attention_dropout": 0.0,
            "block_norm_eps": 1e-5,
            "input_norm_eps": 1e-6,
            "query_norm_eps": 1e-6,
            "output_norm_eps": 1e-6,
            "norm_elementwise_affine": True,
            "input_norm_elementwise_affine": True,
            "query_norm_elementwise_affine": True,
            "output_norm_elementwise_affine": True,
            "output_branch_dims": [2560, 2560, 2560],
            "segmented_output": True,
            "output_align_dim": 7680,
            "input_pre_norm": True,
            "query_post_norm": True,
            "fp32_layer_norm": True,
            "parameter_count": EXPECTED_PARAMETER_COUNT,
        }
        for key, expected in required_contract.items():
            if contract.get(key) != expected:
                raise RuntimeError(
                    f"Q-Former contract mismatch for {key}: "
                    f"expected={expected!r}, actual={contract.get(key)!r}."
                )

        processor_count = 0
        for block in projector.qformer_layers:
            for attention in (block.cross_attn.attn1, block.cross_attn.attn2):
                if type(attention.get_processor()) is not AttnProcessor:
                    raise RuntimeError(
                        "Q-Former smoke requires the exact legacy AttnProcessor."
                    )
                if not attention.upcast_attention:
                    raise RuntimeError("Q-Former attention did not enable FP32 upcast.")
                processor_count += 1
        if processor_count != 8:
            raise RuntimeError(
                f"Expected 8 stable Q-Former attention processors, got {processor_count}."
            )

        projector.train()
        projector.to(device=device, dtype=torch.bfloat16)

        sample_generator = torch.Generator(device="cpu")
        sample_generator.manual_seed(20260806 + rank * 1009)
        started = time.perf_counter()

        for iteration in range(args.iterations):
            print(
                f"QFORMER_MUSA_BEGIN rank={rank} "
                f"iteration={iteration + 1}/{args.iterations}",
                flush=True,
            )
            # A scalar phase would be erased by LayerNorm.  Fresh random
            # token-by-channel structure genuinely changes cross-attention on
            # every iteration and avoids legitimate all-zero Q/K gradients.
            feature_values = torch.randn(
                1,
                196,
                768,
                generator=sample_generator,
                dtype=torch.float32,
            ).to(device=device, dtype=torch.bfloat16)
            target = torch.randn(
                1,
                4,
                7680,
                generator=sample_generator,
                dtype=torch.float32,
            ).to(device=device)

            health = torch.zeros(2, device=device, dtype=torch.int32)
            local_error = None
            try:
                projector.zero_grad(set_to_none=True)
                output = projector(feature_values)
                loss = (output.float() - target).square().mean()
                loss.backward()
                musa.synchronize()

                finite_flags = [torch.isfinite(output).all(), torch.isfinite(loss).all()]
                nonzero_flags = []
                missing_gradients = []
                attention_gradients = list(_all_attention_gradients(projector))
                for gradient_name, gradient in attention_gradients:
                    if gradient is None:
                        missing_gradients.append(gradient_name)
                        continue
                    finite_flags.append(torch.isfinite(gradient).all())
                    nonzero_flags.append(
                        gradient.detach().float().abs().amax() > 0
                    )

                if missing_gradients:
                    local_error = "missing gradients: " + ", ".join(missing_gradients)
                else:
                    if iteration == 0:
                        finite_flags.extend(
                            torch.isfinite(parameter).all()
                            for parameter in projector.parameters()
                        )
                    if iteration in {0, args.iterations - 1}:
                        finite_flags.extend(
                            torch.isfinite(parameter.grad).all()
                            for parameter in projector.parameters()
                            if parameter.grad is not None
                        )
                    health[0] = torch.stack(finite_flags).all().to(torch.int32)
                    health[1] = torch.stack(nonzero_flags).all().to(torch.int32)
                    if sdpa_calls != 0:
                        health[1].zero_()
            except Exception as exc:  # keep every healthy rank moving to consensus
                local_error = f"{exc.__class__.__name__}: {exc}"

            if local_error is not None:
                print(
                    f"QFORMER_MUSA_LOCAL_ERROR rank={rank} "
                    f"iteration={iteration + 1} error={local_error}",
                    flush=True,
                )
            print(
                f"QFORMER_MUSA_COMPUTE_END rank={rank} "
                f"iteration={iteration + 1}/{args.iterations}",
                flush=True,
            )
            dist.all_reduce(health, op=dist.ReduceOp.MIN)
            global_health = health.cpu().tolist()
            if global_health != [1, 1]:
                raise FloatingPointError(
                    "Q-Former BF16 backward failed finite/nonzero/SDPA checks at "
                    f"iteration {iteration + 1}: health={global_health}; "
                    f"local_error={local_error!r}."
                )
            print(
                f"QFORMER_MUSA_END rank={rank} "
                f"iteration={iteration + 1}/{args.iterations}",
                flush=True,
            )

        if rank == 0:
            elapsed = max(time.perf_counter() - started, 1e-9)
            print(
                "QFORMER_MUSA_SMOKE_PASS "
                f"world_size={world_size} iterations={args.iterations} "
                "backend=legacy_upcast processors=8 head_dim=64 "
                f"params={EXPECTED_PARAMETER_COUNT} shape=1x4x7680 "
                f"sdpa_calls={sdpa_calls} elapsed_seconds={elapsed:.6f}",
                flush=True,
            )
    finally:
        if original_sdpa is None:
            delattr(functional, "scaled_dot_product_attention")
        else:
            functional.scaled_dot_product_attention = original_sdpa
        # A final barrier would hide the first rank that failed.
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
