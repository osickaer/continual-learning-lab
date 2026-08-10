from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AccuracyMatrix:
    """Stores R[i, j]: task-j accuracy after finishing training task i."""

    num_tasks: int
    values: list[list[float]] = field(init=False)

    def __post_init__(self) -> None:
        if self.num_tasks <= 0:
            raise ValueError("num_tasks must be positive")
        self.values = [[math.nan for _ in range(self.num_tasks)] for _ in range(self.num_tasks)]

    def update(self, after_task: int, evaluated_task: int, accuracy: float) -> None:
        if not 0 <= accuracy <= 1:
            raise ValueError("accuracy must be in [0, 1]")
        self.values[after_task][evaluated_task] = accuracy

    def average_accuracy(self, after_task: int) -> float:
        seen = self.values[after_task][: after_task + 1]
        self._require_complete(seen, after_task)
        return sum(seen) / len(seen)

    def average_forgetting(self, after_task: int) -> float:
        if after_task == 0:
            return 0.0

        forgetting: list[float] = []
        for evaluated_task in range(after_task):
            history = [self.values[row][evaluated_task] for row in range(evaluated_task, after_task)]
            current = self.values[after_task][evaluated_task]
            self._require_complete(history + [current], after_task)
            forgetting.append(max(history) - current)
        return sum(forgetting) / len(forgetting)

    def stage_summary(self, after_task: int) -> dict[str, float]:
        return {
            "average_accuracy": self.average_accuracy(after_task),
            "average_forgetting": self.average_forgetting(after_task),
        }

    def write_csv(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["after_task"] + [f"eval_task_{index}" for index in range(self.num_tasks)])
            for row_index, row in enumerate(self.values):
                writer.writerow([row_index] + ["" if math.isnan(value) else value for value in row])

    @staticmethod
    def _require_complete(values: list[float], after_task: int) -> None:
        if any(math.isnan(value) for value in values):
            raise ValueError(f"Accuracy matrix is incomplete at training stage {after_task}")

