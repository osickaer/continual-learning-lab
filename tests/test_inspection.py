import json
from pathlib import Path

import torch

from continual_learning_lab.config import ModelConfig
from continual_learning_lab.inspection import capture_activations, write_inspection
from continual_learning_lab.model import SmallCNN


def _small_model() -> SmallCNN:
    return SmallCNN(
        ModelConfig(channels=(4, 8), hidden_dim=16, dropout=0.2, num_classes=10)
    )


def test_capture_activations_records_shapes_and_relu_behavior() -> None:
    torch.manual_seed(12)
    model = _small_model()
    model.train()

    result = capture_activations(model, torch.randn(1, 3, 32, 32))
    summaries = {summary.name: summary for summary in result.summaries}

    assert model.training  # Inspection temporarily uses eval mode, then restores the prior mode.
    assert result.logits.shape == (1, 10)
    assert summaries["features.0"].shape == (1, 4, 32, 32)
    assert summaries["features.4"].shape == (1, 4, 16, 16)
    assert summaries["features.10"].shape == (1, 8, 4, 4)
    assert summaries["classifier.0"].shape == (1, 128)
    assert summaries["classifier.4"].shape == (1, 10)
    assert summaries["features.0"].negative_fraction > 0
    assert summaries["features.1"].negative_fraction == 0
    assert summaries["features.1"].zero_fraction > 0


def test_write_inspection_creates_report_images_and_raw_values(tmp_path: Path) -> None:
    torch.manual_seed(4)
    model = _small_model()
    image = torch.randn(3, 32, 32)
    result = capture_activations(model, image.unsqueeze(0))
    output_dir = tmp_path / "inspection"

    report = write_inspection(
        output_dir,
        model,
        image,
        result,
        checkpoint=tmp_path / "model.pt",
        split="test",
        dataset_index=0,
        target=3,
        max_channels=4,
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
    )

    assert report.is_file()
    assert (output_dir / "input.png").is_file()
    assert (output_dir / "intermediate_values.pt").is_file()
    assert len(list((output_dir / "activation_images").glob("*.png"))) == 16
    metadata = json.loads((output_dir / "prediction.json").read_text(encoding="utf-8"))
    assert metadata["target_name"] == "cat"
    raw = torch.load(output_dir / "intermediate_values.pt", weights_only=True)
    assert raw["activations"]["features.10"].shape == (1, 8, 4, 4)
