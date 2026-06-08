import torch


def get_accelerator_device(local_rank=None):
    try:
        import torch_musa  # noqa: F401  # edit for musa
    except ImportError:
        pass

    if hasattr(torch, "musa"):
        try:
            if torch.musa.is_available():
                if local_rank is not None:
                    torch.musa.set_device(local_rank)
                    return torch.device("musa", local_rank)
                return torch.device("musa")
        except RuntimeError:
            pass

    if torch.cuda.is_available():
        if local_rank is not None:
            torch.cuda.set_device(local_rank)
            return torch.device("cuda", local_rank)
        return torch.device("cuda")

    return torch.device("cpu")
