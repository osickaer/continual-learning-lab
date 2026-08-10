from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed each RNG used by this project and configure deterministic kernels."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Required by deterministic CUDA matrix multiplications on recent CUDA versions.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)


def seed_worker(worker_id: int) -> None:
    """Give NumPy and Python RNGs the seed assigned to a DataLoader worker."""
    del worker_id  # The worker-specific value is already in torch.initial_seed().
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator

