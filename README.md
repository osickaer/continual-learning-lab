# Continual Learning Lab

A small, explicit PyTorch research repository for studying catastrophic
forgetting and candidate neural-memory mechanisms. Experiments 000 and 001
establish sequential and joint CIFAR-10 references; Experiments 002–004 test
recurrent-memory and persistent-state hypotheses.

## CIFAR-10 results

A controlled, single-seed comparison on the same Apple MPS backend used an
unchanged 330,538-parameter CNN and 750,000 training-example presentations per
condition:

| Training protocol | Final held-out accuracy | Average forgetting |
|---|---:|---:|
| Sequential class pairs | 16.80% | 85.89 percentage points |
| Joint access to all classes | 78.19% | Not applicable |

The joint result supports useful model capacity; sequential update interference
is the leading explanation for the baseline collapse. Joint training is a
reference condition, not a continual-learning solution. These runs do not
estimate variation across seeds. Full methods and limitations are in the
[sequential](research/experiments/000_naive_split_cifar10.md) and
[joint](research/experiments/001_joint_training_oracle.md) records.

## What the baseline measures

CIFAR-10 is split into five tasks in its canonical class order:

1. airplane / automobile
2. bird / cat
3. deer / dog
4. frog / horse
5. ship / truck

The model has one shared 10-class output head. It is trained on one task at a
time without replay, regularization, or task identity at inference. After each
task, it is evaluated on the test set for every task. This is a
class-incremental evaluation: the model must choose among all ten classes, not
just the two classes in the current task.

The run records an accuracy matrix `R`, where `R[i, j]` is accuracy on task
`j` after training task `i`. At stage `i`:

- average accuracy is the mean of `R[i, 0:i+1]`;
- average forgetting is the mean, over earlier tasks, of their best previous
  accuracy minus their current accuracy.

Both matter: a method that retains old tasks but cannot learn the new task is
not successful.

## Setup

PyTorch's supported Python versions can lag the newest Python release. Use
Python 3.11, 3.12, or 3.13 unless the current PyTorch install page confirms a
newer version for your platform.

On macOS or Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The default requirements file is cross-platform and lets pip select PyTorch
for the current operating system. Apple Silicon Macs can use PyTorch's MPS
backend, which `device: auto` selects when available. For a Windows CUDA build,
install the appropriate `torch` and `torchvision` wheels from the PyTorch
selector first, then install this project. The optional
`requirements-windows-cu130.txt` file preserves the exact package versions used
for Experiment 000 on the original NVIDIA machine.

### Activating the virtual environment

Run the activation command from the repository root. In PowerShell:

```powershell
Set-Location path\to\continual-learning-lab
.\.venv\Scripts\Activate.ps1
```

The prompt should gain a `(.venv)` prefix. Confirm that the environment is
active with `python -c "import sys; print(sys.executable)"`; it should print a
path ending in `.venv\Scripts\python.exe`.

If PowerShell reports that script execution is disabled, allow scripts only
for the current terminal session and try again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

For Command Prompt use `.venv\Scripts\activate.bat`. For Git Bash use
`source .venv/Scripts/activate`. On macOS or Linux use
`source .venv/bin/activate`.

Activation is optional. You can always address the environment directly:

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\cl-run.exe --config configs/naive_split_cifar10.yaml
```

## Run the baseline

From the repository root:

```powershell
cl-run --config configs/naive_split_cifar10.yaml
```

The first run downloads CIFAR-10 into `data/`. The default configuration uses
deterministic algorithms and seeded data-loader workers. Exact reproducibility
still requires the same hardware, PyTorch/CUDA versions, and configuration;
these details are logged with each run.

The full baseline (5 tasks x 15 epochs) is intentionally a real experiment,
not a fast test. To check the pipeline quickly, copy the YAML and lower
`training.epochs_per_task` and `data.max_train_samples_per_class`. Do not
compare a shortened run to the baseline as if their training budgets matched.

## Run the Experiment 001 joint-training oracle

After tests pass, run the full joint control from the repository root:

```bash
cl-run --config configs/joint_cifar10_oracle.yaml
```

This uses the same CNN, optimizer, seed, augmentation, test data, and 750,000
example-presentation budget as Experiment 000. Its only conceptual change is
that all ten classes remain available in every training epoch. The run logs
overall ten-class evaluation each epoch and five final class-pair diagnostic
views. It intentionally does not report an accuracy matrix or forgetting
metric, because joint training has no sequence of withdrawn tasks.

On Apple Silicon, `device: auto` selects MPS when PyTorch can access it. Treat a
shortened, sample-capped run as a pipeline smoke test only; use the unchanged
checked-in configuration for the official result.

## Run Experiment 002 delayed recall

Experiment 002 asks whether a hidden state containing fixed fast, medium, and
slow update rates recalls an earlier value better at long delays than a vanilla
RNN and the strongest validation-selected single update rate. A same-width GRU
is included as a secondary learned-gating reference, not as part of the
hypothesis verdict.

Before the official run, record the predicted accuracy curves and hidden-state
behavior in
`research/experiments/002_heterogeneous_leaky_delayed_recall.md`. Then run:

```bash
.venv/bin/pytest -q
.venv/bin/cl-run --config configs/heterogeneous_leaky_delayed_recall.yaml
```

The configuration launches one MLflow parent run containing 18 paired child
runs: three seeds for vanilla RNN, GRU, heterogeneous leak, and each of three
homogeneous alpha values. All models learn one balanced mixture of delays 5, 10,
20, 40, and 80. Test data is evaluated only after the fixed final epoch;
validation results select the official homogeneous alpha.

The parent run and its matching `outputs/<parent-run-id>/aggregate/` directory
contain:

- per-seed and aggregate CSV/JSON results;
- the automated preregistered hypothesis verdict;
- recall accuracy versus delay with mean and sample-standard-deviation bars;
- exact hidden-state tensors and heatmaps for paired delay-80 examples;
- mean activity traces for the fast, medium, and slow heterogeneous groups.

On the parent run's Metrics tab, `test_*` series use the MLflow step axis as
delay length, while `epoch_*` series use it as training epoch. Both mean and
individual-seed accuracy curves are logged so seed instability is not hidden by
one aggregate line.

This synthetic task tests a recurrent-memory mechanism. It does not by itself
measure catastrophic forgetting or establish a continual-learning solution.

The completed first run exposed a timing confound and reached ceiling for the
main leaky comparisons. Its config is preserved as the historical protocol. A
corrected follow-up fixes the value at timestep 0, puts each cue after exactly
its requested delay, uses zero padding only after the cue, and reads the hidden
state at that cue. After recording a fresh prediction in the Experiment 002
research record, run:

```bash
.venv/bin/cl-run --config configs/heterogeneous_leaky_delayed_recall_corrected.yaml
```

## Run Experiment 003 Adding Problem

Experiment 003 tests the same recurrent families on selective continuous-value
memory. Models train on lengths 20, 40, and 80, then receive final test
evaluation at those lengths plus unseen lengths 160 and 320:

```bash
.venv/bin/pytest -q
.venv/bin/cl-run --config configs/adding_problem.yaml
```

The target is the sum of exactly two marked `Uniform(0, 1)` values. The parent
MLflow run contains mean and per-seed MSE curves, the validation-selected
homogeneous alpha, the preregistered exploratory verdict, and length-320 hidden-
state artifacts. Lower MSE is better; the test-only extrapolation lengths do not
participate in optimization or model selection.

## Run Experiment 004 Stream-51 persistent state

Experiment 004 compares stateless and persistent GRUCell classifiers at one and
four internal ticks. It manipulates only the training stream's temporal
structure while matching every condition's observation IDs and state-reset
positions.

Download the official Stream-51 archive manually and place its contents under
`data/stream51`. The directory must contain `Stream-51_meta_train.json`,
`Stream-51_meta_test.json`, and the image paths referenced by those files. Then
run:

```bash
.venv/bin/pytest -q
.venv/bin/cl-run --config configs/stream51_temporal_state.yaml
```

On first use, the runner downloads the configured torchvision ResNet-18 weights
if they are not already cached, crops each frame using Stream-51 metadata, and
writes frozen 512-dimensional features under `data/stream51`. It does not
download Stream-51 itself.

The official suite creates 36 child runs: three seeds, three training orders,
and four state/tick conditions. Every prediction is scored before its label
updates the model. Natural and local shuffle use real trajectory boundaries;
global shuffle uses matched-length pseudo-trajectories, giving all three orders
the exact same reset vector. The static known-class test images reset recurrent
state independently.

## Inspect results with MLflow

Runs use a local SQLite backend (`mlflow.db`) and local artifact directory
(`mlartifacts/`). Start the UI from the repository root:

```powershell
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --host 127.0.0.1 --port 5000
```

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000). Each run logs the
configuration, relevant training/evaluation metrics, a summary, package
versions, and the final model state. Sequential runs additionally save the
accuracy matrix; joint runs save final per-pair diagnostics instead.

Metrics use consistent underscore-separated prefixes:

- `train_*`: augmented training batches measured while weights are changing;
- `test_overall_*`: the full held-out test set after each joint epoch;
- `test_pair_*_accuracy` or `test_task_*_accuracy`: class-pair diagnostics;
- `summary_*`: sequential-stage continual-learning aggregates.

In a joint run, `train_*` and `test_overall_*` both have 15 points because the
overall test set is evaluated after each of the 15 training epochs. Each pair
accuracy has only one point at step 14 because pair diagnostics run only after
the final epoch. The last `test_overall_*` point is the final held-out test
result; detailed final values also live in `summary.json`.

Each run also includes a Markdown description and a
`reports/run_summary.md` artifact explaining its purpose, final results, and
metric interpretation. Evaluating the test set during training does not update
the model. Because these experiments use a preregistered epoch budget, the
curves are diagnostics rather than an early-stopping signal; future model or
hyperparameter selection should use a validation split instead.

Local copies of final artifacts are also written under `outputs/<run-id>/`.

## Inspect what the CNN sees

After completing a training run, inspect one real CIFAR-10 image as it passes
through every layer:

```powershell
cl-inspect --config configs/naive_split_cifar10.yaml --class-id 8 --sample-index 0
```

`--class-id 8` selects the ship class, and `--sample-index 0` selects its first
test example. If `--class-id` is omitted, `--sample-index` refers to the full
CIFAR-10 dataset. The command uses the newest `outputs/<run-id>/final_model_state.pt`
by default; select a particular run with:

```powershell
cl-inspect --config configs/naive_split_cifar10.yaml `
  --checkpoint outputs/<run-id>/final_model_state.pt `
  --class-id 8 --sample-index 0
```

It creates a unique directory under `inspections/` containing:

- `report.html`: input image, prediction, layer-by-layer activation images,
  tensor shapes, numerical ranges, and negative/zero/positive fractions;
- `intermediate_values.pt`: every exact intermediate tensor for further Python
  exploration;
- `activation_summary.json` and `prediction.json`: machine-readable summaries;
- `model_architecture.txt`: the exact PyTorch model structure.

Open the printed report path in a browser. PowerShell can open it with:

```powershell
start inspections/<printed-directory>/report.html
```

The report uses evaluation mode, so dropout is disabled and no weights are
updated. Its orange/blue maps show positive/negative activations; each map is
scaled independently to expose spatial structure, so use the numerical
statistics when comparing magnitude between layers.

Because `cl-inspect` is a new console entry point, an existing environment may
need one editable reinstall before the command appears:

```powershell
python -m pip install -r requirements.txt
```

No dependency was added; this refreshes the repository's installed command
metadata.

## Tests

```powershell
pytest
```

The tests use small synthetic datasets and do not download CIFAR-10 or Stream-51.

### Inspect joint loaders while developing

Experiment 001 includes a fast development harness that uses the real,
already-downloaded CIFAR-10 data with tiny sample caps and no worker processes:

```bash
.venv/bin/python scripts/inspect_joint_loaders.py
```

It checks subset sizes, global label counts, batch tensor shapes and dtypes,
training shuffling, and the five pair-specific evaluation views. Override the
small defaults with `--train-per-class` and `--eval-per-class` when useful. This
command validates data plumbing only; it does not train the model or write an
experiment result.

## Experiment records

| Experiment | Question | Status |
|---|---|---|
| [000: Sequential CIFAR-10](research/experiments/000_naive_split_cifar10.md) | How much does a shared-head CNN forget under sequential training? | Complete |
| [001: Joint CIFAR-10](research/experiments/001_joint_training_oracle.md) | Can the same CNN learn all ten classes with joint access? | Complete |
| [002: Delayed recall](research/experiments/002_heterogeneous_leaky_delayed_recall.md) | Do fixed heterogeneous timescales improve recall? | Complete; advantage not demonstrated |
| [003: Adding Problem](research/experiments/003_adding_problem.md) | Do heterogeneous timescales improve selective memory and extrapolation? | Complete; hypothesis not supported |
| [004: Stream-51](research/experiments/004_stream51_persistent_state.md) | Does persistent activation exploit temporal coherence? | Implemented; official run pending |

See [the research strategy](research/strategy/RESEARCH_STRATEGY.md) for comparison
and evidence standards.

## Repository layout

```text
configs/                    experiment configurations
src/continual_learning_lab/ reusable experiment infrastructure
  learners/                 method-specific training behavior
  inspection.py             layer activation capture and report generation
tests/                      metric, data-split, and training tests
research/                   hypotheses, experiment records, and findings
```

To add a continual-learning method, implement the small learner protocol in
`learners/base.py`, add its configuration, and select it in the learner
factory. Dataset construction, evaluation, metrics, and tracking remain
unchanged.
