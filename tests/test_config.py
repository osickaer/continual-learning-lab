from pathlib import Path

import pytest

from continual_learning_lab.config import load_config


CONFIG_PATH = Path(__file__).parents[1] / "configs" / "naive_split_cifar10.yaml"
JOINT_CONFIG_PATH = Path(__file__).parents[1] / "configs" / "joint_cifar10_oracle.yaml"


def test_baseline_config_loads() -> None:
    config = load_config(CONFIG_PATH)

    assert config.training.method == "naive"
    assert config.data.protocol == "sequential"
    assert config.data.task_classes == ((0, 1), (2, 3), (4, 5), (6, 7), (8, 9))
    assert config.model.num_classes == 10


def test_joint_config_changes_only_identity_and_data_protocol() -> None:
    baseline = load_config(CONFIG_PATH).to_dict()
    joint = load_config(JOINT_CONFIG_PATH).to_dict()

    # Construct the complete expected config from the baseline. This catches a
    # future accidental change to any controlled condition, not just the fields
    # we happen to remember to assert individually.
    baseline["experiment"]["name"] = "experiment-001-joint-cifar10-oracle"
    baseline["experiment"]["run_name"] = "exp001-joint-seed-42"
    baseline["data"]["protocol"] = "joint"

    assert joint == baseline


def test_duplicate_or_missing_task_classes_are_rejected(tmp_path: Path) -> None:
    text = CONFIG_PATH.read_text(encoding="utf-8").replace("- [8, 9]", "- [8, 8]")
    invalid_path = tmp_path / "invalid.yaml"
    invalid_path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="every class"):
        load_config(invalid_path)


def test_unknown_data_protocol_is_rejected(tmp_path: Path) -> None:
    text = CONFIG_PATH.read_text(encoding="utf-8").replace(
        "protocol: sequential", "protocol: unsupported"
    )
    invalid_path = tmp_path / "invalid.yaml"
    invalid_path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="data.protocol must be sequential or joint"):
        load_config(invalid_path)
