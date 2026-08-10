import torch
from torch.utils.data import TensorDataset

from continual_learning_lab.data import ClassSubset


def test_class_subset_keeps_global_labels_and_caps_each_class() -> None:
    inputs = torch.arange(8).unsqueeze(1)
    targets = torch.tensor([0, 1, 2, 1, 2, 2, 3, 2])
    dataset = TensorDataset(inputs, targets)

    subset = ClassSubset(dataset, targets.tolist(), classes=(1, 2), max_samples_per_class=2)

    labels = [int(subset[index][1]) for index in range(len(subset))]
    assert labels == [1, 2, 1, 2]

