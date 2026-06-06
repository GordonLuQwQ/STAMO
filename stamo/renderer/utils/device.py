import torch


def get_accelerator_device():
    try:
        import torch_musa  # noqa: F401  # edit for musa
    except ImportError:
        pass

    if hasattr(torch, "musa"):
        try:
            if torch.musa.is_available():
                return torch.device("musa")
        except RuntimeError:
            pass

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")
