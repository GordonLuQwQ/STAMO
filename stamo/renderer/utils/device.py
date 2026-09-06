import os

import torch


def _resolve_local_rank(local_rank=None):
    if local_rank is None:
        local_rank = os.environ.get("LOCAL_RANK", 0)
    local_rank = int(local_rank)
    return max(local_rank, 0)


def get_accelerator_device(local_rank=None):
    musa = getattr(torch, "musa", None)
    if musa is None or not musa.is_available():
        return torch.device("cpu")

    local_rank = _resolve_local_rank(local_rank)
    device_count = musa.device_count()
    if local_rank >= device_count:
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} but only {device_count} MUSA device(s) are visible. "
            "Check MUSA_VISIBLE_DEVICES and the launch command."
        )
    musa.set_device(local_rank)
    return torch.device("musa", local_rank)