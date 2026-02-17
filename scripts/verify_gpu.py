"""Verify GPU access and PyTorch CUDA functionality."""
import sys
import torch


def main():
    print(f"Python:         {sys.version.split()[0]}")
    print(f"PyTorch:        {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available — check driver and PyTorch install")
        sys.exit(1)

    print(f"CUDA version:   {torch.version.cuda}")
    print(f"cuDNN version:  {torch.backends.cudnn.version()}")
    print(f"GPU:            {torch.cuda.get_device_name(0)}")

    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / 1e9
    print(f"VRAM:           {vram_gb:.1f} GB")
    print(f"Compute cap:    {props.major}.{props.minor}")

    # Smoke test
    x = torch.randn(2048, 2048, device="cuda")
    y = x @ x.T
    assert y.shape == (2048, 2048), f"Unexpected shape: {y.shape}"
    print(f"MatMul test:    PASSED ({y.shape})")

    # Memory test
    allocated = torch.cuda.memory_allocated(0) / 1e6
    print(f"VRAM used:      {allocated:.0f} MB")

    del x, y
    torch.cuda.empty_cache()
    print("Cleanup:        PASSED")


if __name__ == "__main__":
    main()
