# Repository guidance

## Purpose

This repository studies catastrophic forgetting and neural-memory mechanisms
through controlled, reproducible PyTorch experiments.

## Implementation and experiment standards

- Make one main conceptual change at a time.
- Keep baselines, datasets, architecture, evaluation, seeds, and training
  budgets fixed unless a change is explicit and justified.
- Treat configuration as the experiment contract. Separate data protocol,
  learning method, and shared infrastructure.
- Prefer simple, readable PyTorch and configuration changes over clever
  abstractions or duplicated scripts.
- Add tests for important metrics and non-obvious training behavior.
- Record predictions before official runs and flag confounds.
- Evaluate retention and new-task learning together.
- Keep MLflow sparse and literal: underscore-separated metric names, no
  redundant aliases, and documented step semantics.
- Preserve scientific results, limitations, and negative findings.

## Records and context

- `README.md`: project overview, reproduction commands, and experiment index.
- `research/experiments/`: scientific plans, results, and interpretations.
- `research/findings/`: durable findings.
- MLflow: quantitative run record. Git: code history.

For implementation, read this file plus relevant source and configuration.
For planning or interpretation, add the matching research record. Experiment
status is recorded in each research record; historical procedures are not
current instructions.

Personal notes and assessments belong in ignored local storage and must not be
added to the public repository.
