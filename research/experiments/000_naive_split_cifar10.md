# Experiment 000: Naive sequential Split CIFAR-10 baseline

Status: completed on 2026-08-08

MLflow run: `d5669187cc274ae1882b36bc1480c6e1`

macOS/MPS replication: `72d7c82f999747e7921fb36f73fa7ba5` (2026-08-25)

## Hypothesis

A small single-head CNN trained sequentially on five disjoint CIFAR-10 class
pairs, without access to old examples, will learn each current task but lose
substantial accuracy on earlier tasks. This should produce a visibly
lower-triangular accuracy pattern and positive average forgetting.

## Method

- Split CIFAR-10 into `(0,1)`, `(2,3)`, `(4,5)`, `(6,7)`, `(8,9)`.
- Use global labels and one shared 10-class head; do not provide task identity.
- Train the small CNN on each task for 15 epochs with persistent SGD state.
- Use seed 42, deterministic PyTorch algorithms, and seeded data loaders.
- Evaluate every task's held-out CIFAR-10 test subset after every training task.

The exact configuration is `configs/naive_split_cifar10.yaml`. It is the
comparison contract for later methods: task order, data transforms, model,
seed, optimizer, epoch budget, and evaluation logic should remain fixed unless
the experiment explicitly studies one of them.

## Baseline

There is no earlier continual-learning result. Useful reference points are
current-task accuracy (plasticity) and chance-level 10-way accuracy (10%). A
joint-training oracle is intentionally deferred until the naive baseline is
established.

## Metrics

- full accuracy matrix `R[i, j]`
- seen-task average accuracy after each task
- average forgetting over earlier tasks after each task
- current-task training and test accuracy/loss

## Results

Accuracy matrix (`R[i, j]` is evaluation accuracy on task `j` after training
task `i`):

| After task | Task 0 | Task 1 | Task 2 | Task 3 | Task 4 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.9380 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 1 | 0.0000 | 0.7950 | 0.0000 | 0.0000 | 0.0000 |
| 2 | 0.0000 | 0.0000 | 0.8545 | 0.0000 | 0.0000 |
| 3 | 0.0000 | 0.0000 | 0.0000 | 0.9080 | 0.0000 |
| 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.8835 |

- Final seen-task average accuracy: `0.1767`
- Final average forgetting: `0.873875`
- Final current-task accuracy: `0.8835`
- Device: CUDA (`NVIDIA GeForce RTX 2070 SUPER`)

The saved result files are under
`outputs/d5669187cc274ae1882b36bc1480c6e1/`.

### macOS/MPS replication

The baseline was rerun unchanged on the same Apple MPS environment used for
Experiment 001. This removes the CUDA-versus-MPS backend difference from the
direct oracle comparison.

| After task | Task 0 | Task 1 | Task 2 | Task 3 | Task 4 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.9385 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 1 | 0.0000 | 0.7700 | 0.0000 | 0.0000 | 0.0000 |
| 2 | 0.0000 | 0.0000 | 0.8375 | 0.0000 | 0.0000 |
| 3 | 0.0000 | 0.0000 | 0.0000 | 0.8895 | 0.0000 |
| 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.8400 |

- Final seen-task average accuracy: `0.1680`
- Final average forgetting: `0.858875`
- Device: MPS

The replication preserves the original qualitative result: every current task
is learned well, and every earlier pair collapses to zero after later training.
The files are under `outputs/72d7c82f999747e7921fb36f73fa7ba5/`.

## Interpretation criteria

The hypothesis is supported if current-task test accuracy rises materially
above chance while earlier-task accuracies later decline. High forgetting with
poor current-task learning would instead indicate a broken or undertrained
baseline, not a useful demonstration of catastrophic forgetting.

## Interpretation

The hypothesis is strongly supported for this seed. Each current task reached
substantial test accuracy (`0.7950` to `0.9380`), showing that the model retained
the ability to learn new tasks. However, every preceding task fell to zero
accuracy immediately after the next task was learned. The low final average
accuracy is therefore caused by catastrophic forgetting rather than a general
failure to learn.

This is a single-seed baseline. It establishes the failure mode but does not
yet estimate run-to-run variance.

## Unexpected observations

- The first CUDA attempt stopped before completing its first parameter update.
  `AdaptiveAvgPool2d` has no deterministic CUDA backward implementation in
  PyTorch 2.13. The layer was replaced with fixed `AvgPool2d(2)`, which is
  appropriate for fixed-size CIFAR-10 inputs and preserves the same 4x4 feature
  shape and parameter count. This was a pre-baseline implementation correction,
  not an experimental result.

## Follow-ups

1. Repeat with several seeds to quantify variance after the single-seed
   pipeline is validated.
2. Add a joint-training oracle to estimate model capacity separately from
   forgetting.
3. Add simple experience replay without changing this experiment's training
   budget or evaluation protocol.
