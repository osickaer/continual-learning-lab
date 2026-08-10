import random

import numpy as np
import torch

from continual_learning_lab.reproducibility import seed_everything


def test_seed_everything_replays_python_numpy_and_torch_rngs() -> None:
    deterministic_was_enabled = torch.are_deterministic_algorithms_enabled()
    try:
        seed_everything(123, deterministic=True)
        first = (random.random(), np.random.rand(), torch.rand(3))

        seed_everything(123, deterministic=True)
        second = (random.random(), np.random.rand(), torch.rand(3))

        assert first[0] == second[0]
        assert first[1] == second[1]
        assert torch.equal(first[2], second[2])
        assert torch.are_deterministic_algorithms_enabled()
    finally:
        torch.use_deterministic_algorithms(deterministic_was_enabled)
