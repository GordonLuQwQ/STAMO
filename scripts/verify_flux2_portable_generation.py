#!/usr/bin/env python3
"""Strictly load a portable FLUX.2 renderer export and generate one image."""

from __future__ import annotations

import argparse

import torch
from omegaconf import OmegaConf

try:
    import torch_musa  # noqa: F401
except ImportError as exc:
    raise ImportError("This verifier must run inside the torch-MUSA environment.") from exc

from stamo.renderer.model.renderer import RenderNet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    cli = parser.parse_args()

    if not hasattr(torch, "musa") or not torch.musa.is_available():
        raise RuntimeError("MUSA is unavailable.")
    torch.musa.set_device(0)

    config = OmegaConf.load(cli.config)
    config.world_size = 1
    config.local_rank = 0
    config.do_train = False
    config.deepspeed = False
    config.fabric = False
    model = RenderNet(config)
    restored_step = model.load_checkpoint(cli.checkpoint)
    model.to(torch.device("musa", 0))
    model.eval()
    model._progress_bar_config = {"disable": True, "leave": False}

    image_size = int(config.data.img_size)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(config.seed))
    inputs = {
        "images": torch.rand(
            (1, 3, image_size, image_size),
            generator=generator,
            device="cpu",
            dtype=torch.float32,
        )
    }
    with torch.no_grad():
        generated = model(inputs)["images"]
    expected_shape = (1, 3, image_size, image_size)
    if tuple(generated.shape) != expected_shape:
        raise RuntimeError(
            f"Generated tensor has shape {tuple(generated.shape)}, expected {expected_shape}."
        )
    generated_cpu = generated.detach().float().cpu()
    if not bool(torch.isfinite(generated_cpu).all().item()):
        raise FloatingPointError("Portable FLUX.2 generation produced non-finite values.")
    print(
        "STAMO_PORTABLE_GENERATION_PASS "
        f"step={restored_step} shape=1x3x{image_size}x{image_size} "
        f"min={float(generated_cpu.min().item()):.6g} "
        f"max={float(generated_cpu.max().item()):.6g}",
        flush=True,
    )


if __name__ == "__main__":
    main()
