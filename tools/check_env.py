"""
Run this after installing requirements-torch.txt and requirements.txt to ensure the torch install has succeeded and can
properly use your GPU
"""

import torch

def main() -> None:
    # Print a bunch of torch information to verify install
    print(f"Torch version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"Torch CUDA version: {torch.version.cuda}")
    # Only check if torch actually sees the GPU
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        tmp = torch.ones(3, device="cuda")
        tmp = tmp * 2
        tmp = tmp + tmp
        print(f"Tensor sum: {tmp.sum()}")


if __name__ == "__main__":
    main()
