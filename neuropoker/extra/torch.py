"""Helper functions for PyTorch."""

import torch


def get_device() -> str:
    """Get the device to train on.

    Returns:
        device: str
            The device to train on.
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"
