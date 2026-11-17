"""Choose the compute device: NVIDIA GPU, Apple GPU (MPS) or CPU."""

from __future__ import annotations

import torch


def select_device(preference: str = "auto") -> torch.device:
    """Return the device to run on.

    Args:
        preference: "auto" picks CUDA, then Apple MPS, then CPU. Any other
            value ("cuda", "cuda:1", "mps", "cpu") is used as given.

    Raises:
        RuntimeError: if an explicitly requested CUDA or MPS device is not
            available on this machine.
    """
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(preference)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"device '{preference}' requested but CUDA is not available")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(f"device '{preference}' requested but Apple MPS is not available")
    return device


def describe_device(device: torch.device) -> str:
    """One-line human description, e.g. 'cuda (<GPU name>, 17.2 GB)'."""
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        return f"{device} ({props.name}, {props.total_memory / 1e9:.1f} GB)"
    if device.type == "mps":
        return f"{device} (Apple GPU)"
    return f"{device} (CPU only: training will be slow)"
