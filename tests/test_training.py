import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from continual_learning_lab.config import TrainingConfig
from continual_learning_lab.learners.naive import NaiveSequentialLearner


def test_naive_learner_updates_parameters_and_learns_tiny_dataset() -> None:
    torch.manual_seed(7)
    inputs = torch.cat([torch.full((8, 2), -1.0), torch.full((8, 2), 1.0)])
    targets = torch.tensor([0] * 8 + [1] * 8)
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=4, shuffle=False)
    model = nn.Linear(2, 2)
    initial = model.weight.detach().clone()
    config = TrainingConfig(
        method="naive",
        epochs_per_task=8,
        optimizer="sgd",
        learning_rate=0.2,
        momentum=0.0,
        weight_decay=0.0,
    )

    learner = NaiveSequentialLearner(model, config, torch.device("cpu"))
    results = learner.train_task(task_id=0, loader=loader)

    assert not torch.equal(initial, model.weight)
    assert results[-1].accuracy >= 0.95

