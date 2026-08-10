from __future__ import annotations

import argparse
from pathlib import Path

from continual_learning_lab.config import load_config
from continual_learning_lab.experiment import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a continual-learning experiment")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a YAML experiment configuration",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run_experiment(config, args.config)


if __name__ == "__main__":
    main()

