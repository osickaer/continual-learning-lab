# Experiment 001: Joint-training CIFAR-10 oracle

Status: completed on 2026-08-25; interpretation supports proceeding to replay

MLflow run: `ec350c1fd58045b19a6dddc028c38770`

Baseline: `research/experiments/000_naive_split_cifar10.md`

Configuration: `configs/joint_cifar10_oracle.yaml`

## Implementation progress

As of 2026-08-16, the joint configuration and joint CIFAR-10 loader are
implemented and covered by focused tests. The learner boundary now exposes one
training epoch, which is required to interleave joint training and evaluation.

The sequential runner was adapted on 2026-08-25 to use a one-epoch training
interface. Its original task order, evaluation timing, metric names, and
continuous epoch numbering are preserved and covered by a scheduling test.

Joint protocol dispatch, per-epoch overall evaluation, final pair diagnostics,
mean pair accuracy, example-exposure accounting, dynamic MLflow protocol tags,
and joint-specific summaries are now implemented. Joint summaries deliberately
omit the sequential accuracy matrix and forgetting metric. After the tracking
presentation improvements, the full suite passes with 17 tests and one
expected CUDA-only skip.

A two-epoch capped smoke run completed on MPS on 2026-08-25 using eight training
and four evaluation examples per class. It presented 160 training examples and
successfully wrote the model, resolved config, summary, metrics, and
`protocol=joint` tracking tag. The capped accuracy is not scientific evidence
and must not be interpreted against the preregistered bands.

## Research question

Can the Experiment 000 CNN learn and retain all ten CIFAR-10 classes when all
training examples remain jointly available, or is model capacity/optimization
itself a major limitation?

## Prediction

Recorded 2026-08-09, before implementation or results:

> My prediction is that the CNN's capacity/optimization will be the main
> limitation. I bet we'll see some slight accuracy improvements in the final
> test/eval, but my prediction is that the classifier won't be particularly
> strong at predicting any of the classes.
>
> A result where the model still forgets would make me reconsider my
> prediction.

Pre-implementation clarification: joint training never withdraws old classes,
so it does not directly measure sequential forgetting. Strong final accuracy
across all five class-pair subsets is the result that would most directly
challenge the capacity/optimization prediction.

## Hypothesis

Joint training should produce materially stronger final accuracy across all
five class pairs than naive sequential training. If it does, that supports the
interpretation that Experiment 000 failed primarily because later-task updates
interfered with earlier knowledge, not because the CNN was incapable of
representing the ten-class problem.

## Baseline

Experiment 000 trained the same shared-head CNN sequentially on five disjoint
two-class tasks. Each current task reached `0.7950` to `0.9380` accuracy, but
all earlier tasks fell to zero. Its final seen-task average accuracy was
`0.1767` and average forgetting was `0.873875` for seed 42.

Joint training is an oracle/reference, not a continual-learning solution: it
is allowed to retain and revisit the full training dataset at every update.

## Method

- Reuse the canonical CIFAR-10 training and test splits.
- Mix all ten training classes in one shuffled training loader.
- Use the same global labels and shared ten-output CNN.
- Train for 15 joint epochs using seed 42 and strict determinism.
- Evaluate overall ten-class test performance and each of the five canonical
  class-pair test subsets.
- Log configuration, per-epoch training/evaluation curves, per-pair final
  metrics, and saved artifacts to local MLflow.

## Controlled conditions

Keep these Experiment 000 conditions fixed:

- dataset and predefined train/test split
- normalization and training augmentation
- CNN architecture and 330,538 trainable parameters
- batch sizes and data-loader settings
- SGD learning rate, momentum, and weight decay
- seed and deterministic CUDA behavior
- ten-output shared head and global class labels
- test subsets and evaluation implementation

The only conceptual change is training-data availability and ordering:
all classes remain jointly available rather than arriving as disjoint tasks.

## Execution environment

The oracle ran on an Apple M5 Pro using PyTorch's MPS backend. Experiment 000
was also rerun unchanged on MPS, so the direct results below do not have the
original Windows/CUDA versus macOS/MPS backend confound. Cross-platform exact
reproducibility is still not assumed.

## Training-budget decision

Experiment 000 presented:

```text
10,000 images per task * 15 epochs * 5 tasks = 750,000 examples
```

Experiment 001 presented:

```text
50,000 joint images * 15 epochs = 750,000 examples
```

This exactly matches example exposure. With batch size 128 and the final
partial batch retained, the sequential run performs 5,925 optimizer steps and
the joint run performs 5,865, a difference of about 1%. Record this known
batch-boundary discrepancy rather than adding complexity solely to match it.

## Metrics

- per-epoch joint training loss and accuracy
- per-epoch overall ten-class test loss and accuracy
- final accuracy and loss on each canonical two-class test subset
- final mean of the five task-subset accuracies
- trainable parameter count and total example exposure

Do not report joint-training forgetting as though it were comparable to the
sequential accuracy-matrix metric. No task is withdrawn from joint training.

## Results

The official seed-42 run completed all 15 epochs and presented exactly 750,000
training examples.

| Test view | Classes | Accuracy | Loss |
|---|---|---:|---:|
| Overall | 0–9 | 0.7819 | 0.6245 |
| Pair 0 | 0, 1 | 0.8575 | 0.4068 |
| Pair 1 | 2, 3 | 0.5850 | 1.1070 |
| Pair 2 | 4, 5 | 0.7430 | 0.7390 |
| Pair 3 | 6, 7 | 0.8745 | 0.4049 |
| Pair 4 | 8, 9 | 0.8495 | 0.4647 |

- Final mean pair accuracy: `0.7819`
- Final training accuracy/loss: `0.75946` / `0.6870`
- Trainable parameters: `330,538`
- Device: MPS

The mean pair accuracy equals overall accuracy because the five pair test views
are disjoint and all contain the same number of examples.

## Interpretation criteria

- Preregistered clear joint-capacity evidence: overall accuracy at least 50%
  and every class-pair subset at least 40%.
- Preregistered weak joint capacity: overall accuracy below 30%, or at least
  three pairs below 30%.
- Results between those bands are ambiguous.
- Uniformly poor joint performance points toward capacity, optimization, or
  training-budget limitations that should be diagnosed before replay.
- Strong overall accuracy with one weak class pair suggests class-specific
  difficulty rather than catastrophic forgetting.
- This single-seed oracle will be a reference point, not a variance estimate.

## Interpretation

The result is clear joint-capacity evidence: overall accuracy was 78.19%, and
the weakest pair was still 58.50%. The unchanged CNN can therefore represent
and learn a useful ten-class solution under the matched example-exposure
budget. Compared on the same MPS backend, naive sequential training finished at
16.80% average accuracy with 85.89% forgetting.

This strongly supports sequential update interference as the leading cause of
Experiment 000's collapse. It challenges the preregistered prediction that
capacity or optimization would be the primary limitation. The result does not
show that capacity is unlimited or that 78.19% is optimal; it shows that basic
capacity is not the bottleneck responsible for the near-total sequential
failure.

Pair 1, bird/cat, is materially weaker than the other pairs and deserves
monitoring in later comparisons. The training accuracy being slightly below
test accuracy is not inherently suspicious: training uses random augmentation,
dropout, and within-epoch predictions from a model that is still changing,
whereas test evaluation uses unaugmented inputs and the final epoch model with
dropout disabled.

The next defensible conceptual experiment is simple experience replay while
preserving the sequential protocol, model, optimizer, seed, evaluation data,
and stated budget.

## Implementation constraints

- Keep protocol selection distinct from continual-learning method selection if
  the configuration can do so without obscuring the code. "Joint" describes
  data availability; future "replay" describes a sequential learning method.
- Reuse model, transforms, evaluation, tracking, and artifact infrastructure.
- Do not duplicate the Experiment 000 script or silently alter its behavior.
- Add focused tests for the joint dataset and training path.
- Do not tune architecture or optimizer against the test set in this experiment.

## Possible follow-ups

- If the oracle is strong, implement experience replay as Experiment 002.
- If the oracle is weak, introduce a validation split and diagnose model or
  optimization capacity before evaluating a continual-learning method.
- After the first meaningful method comparison, repeat naive and replay runs
  across multiple seeds.
