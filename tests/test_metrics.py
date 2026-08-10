import pytest

from continual_learning_lab.metrics import AccuracyMatrix


def test_average_accuracy_and_forgetting() -> None:
    matrix = AccuracyMatrix(3)
    rows = [
        [0.80, 0.10, 0.10],
        [0.50, 0.85, 0.10],
        [0.40, 0.60, 0.90],
    ]
    for after_task, row in enumerate(rows):
        for evaluated_task, accuracy in enumerate(row):
            matrix.update(after_task, evaluated_task, accuracy)

    assert matrix.average_accuracy(0) == pytest.approx(0.80)
    assert matrix.average_accuracy(2) == pytest.approx((0.40 + 0.60 + 0.90) / 3)
    assert matrix.average_forgetting(0) == 0.0
    assert matrix.average_forgetting(1) == pytest.approx(0.30)
    assert matrix.average_forgetting(2) == pytest.approx((0.40 + 0.25) / 2)


def test_metrics_reject_incomplete_seen_task_row() -> None:
    matrix = AccuracyMatrix(2)
    matrix.update(1, 0, 0.5)

    with pytest.raises(ValueError, match="incomplete"):
        matrix.average_accuracy(1)

