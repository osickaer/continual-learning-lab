import pytest
import torch
from torch import nn

from continual_learning_lab.config import ModelConfig
from continual_learning_lab.model import SmallCNN
from continual_learning_lab.reproducibility import seed_everything


def _model_config() -> ModelConfig:
    return ModelConfig(channels=(32, 64), hidden_dim=256, dropout=0.2, num_classes=10)


def test_small_cnn_preserves_expected_output_shape() -> None:
    model = SmallCNN(_model_config())

    logits = model(torch.randn(3, 3, 32, 32))

    assert logits.shape == (3, 10)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_small_cnn_supports_strict_deterministic_cuda_backward() -> None:
    deterministic_was_enabled = torch.are_deterministic_algorithms_enabled()
    cudnn_deterministic_was_enabled = torch.backends.cudnn.deterministic
    cudnn_benchmark_was_enabled = torch.backends.cudnn.benchmark
    try:
        seed_everything(123, deterministic=True)
        model = SmallCNN(_model_config()).cuda()
        inputs = torch.randn(2, 3, 32, 32, device="cuda")
        targets = torch.tensor([0, 1], device="cuda")

        nn.CrossEntropyLoss()(model(inputs), targets).backward()
    finally:
        torch.use_deterministic_algorithms(deterministic_was_enabled)
        torch.backends.cudnn.deterministic = cudnn_deterministic_was_enabled
        torch.backends.cudnn.benchmark = cudnn_benchmark_was_enabled
