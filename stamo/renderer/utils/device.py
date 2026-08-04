import os

import torch

try:
    import torch_musa  # noqa: F401
except ImportError:
    torch_musa = None


def _resolve_local_rank(local_rank=None):
    if local_rank is None:
        local_rank = os.environ.get("LOCAL_RANK", 0)
    local_rank = int(local_rank)
    return max(local_rank, 0)


def get_accelerator_device(local_rank=None):
    local_rank = _resolve_local_rank(local_rank)

    musa = getattr(torch, "musa", None)
    if musa is not None and musa.is_available():
        device_count = int(musa.device_count())
        if local_rank >= device_count:
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} but only {device_count} MUSA device(s) are visible. "
                "Check MUSA_VISIBLE_DEVICES and the launch command."
            )
        musa.set_device(local_rank)
        return torch.device("musa", local_rank)

    if not torch.cuda.is_available():
        return torch.device("cpu")

    device_count = torch.cuda.device_count()
    if local_rank >= device_count:
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} but only {device_count} CUDA device(s) are visible. "
            "Check CUDA_VISIBLE_DEVICES and the launch command."
        )
    torch.cuda.set_device(local_rank)
    return torch.device("cuda", local_rank)
