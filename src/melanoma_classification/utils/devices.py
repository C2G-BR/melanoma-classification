import torch


def get_device() -> torch.device:
    """Get pytorch device.

    Returns:
        Pytorch device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
