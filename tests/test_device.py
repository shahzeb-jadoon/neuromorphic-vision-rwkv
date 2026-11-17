"""Device selection must pick CUDA, then Apple MPS, then CPU, on any machine."""

from __future__ import annotations

import pytest
import torch

from src.utils.device import describe_device, select_device


def _fake_availability(monkeypatch: pytest.MonkeyPatch, cuda: bool, mps: bool) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)


@pytest.mark.parametrize(
    ("cuda", "mps", "expected"),
    [(True, True, "cuda"), (True, False, "cuda"), (False, True, "mps"), (False, False, "cpu")],
)
def test_auto_prefers_cuda_then_mps_then_cpu(
    monkeypatch: pytest.MonkeyPatch, cuda: bool, mps: bool, expected: str
) -> None:
    _fake_availability(monkeypatch, cuda=cuda, mps=mps)
    assert select_device("auto").type == expected


def test_explicit_cpu_is_always_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_availability(monkeypatch, cuda=False, mps=False)
    assert select_device("cpu") == torch.device("cpu")


@pytest.mark.parametrize("requested", ["cuda", "cuda:0", "mps"])
def test_unavailable_accelerator_raises(monkeypatch: pytest.MonkeyPatch, requested: str) -> None:
    _fake_availability(monkeypatch, cuda=False, mps=False)
    with pytest.raises(RuntimeError, match="not available"):
        select_device(requested)


def test_describe_cpu_warns_about_speed() -> None:
    assert "CPU only" in describe_device(torch.device("cpu"))
