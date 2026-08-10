from pathlib import Path

import pytest

from continual_learning_lab.config import load_config


CONFIG_PATH = Path(__file__).parents[1] / "configs" / "naive_split_cifar10.yaml"


def test_baseline_config_loads() -> None:
    config = load_config(CONFIG_PATH)

    assert config.training.method == "naive"
    assert config.data.task_classes == ((0, 1), (2, 3), (4, 5), (6, 7), (8, 9))
    assert config.model.num_classes == 10


def test_duplicate_or_missing_task_classes_are_rejected(tmp_path: Path) -> None:
    text = CONFIG_PATH.read_text(encoding="utf-8").replace("- [8, 9]", "- [8, 8]")
    invalid_path = tmp_path / "invalid.yaml"
    invalid_path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="every class"):
        load_config(invalid_path)

