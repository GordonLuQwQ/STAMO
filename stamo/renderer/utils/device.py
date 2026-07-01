import os

import torch


def _resolve_local_rank(local_rank=None):
    if local_rank is None:
        local_rank = os.environ.get("LOCAL_RANK", 0)
    local_rank = int(local_rank)
    return max(local_rank, 0)


def get_accelerator_device(local_rank=None):
    if not torch.cuda.is_available():
        return torch.device("cpu")

    local_rank = _resolve_local_rank(local_rank)
    device_count = torch.cuda.device_count()
    if local_rank >= device_count:
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} but only {device_count} CUDA device(s) are visible. "
            "Check CUDA_VISIBLE_DEVICES and the launch command."
        )
    torch.cuda.set_device(local_rank)
    return torch.device("cuda", local_rank)
