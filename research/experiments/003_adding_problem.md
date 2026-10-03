# Experiment 003: Adding Problem recurrent-memory extrapolation

Status: official run complete; primary hypothesis not supported

Origin: exploratory harder test of the heterogeneous-timescale hypothesis

Configuration: `configs/adding_problem.yaml`

MLflow parent run: `c924e2af54814cde8aff15a667811db1`

## Research question

Do fixed heterogeneous recurrent update rates improve selective continuous-value
memory as sequence length and interference increase, particularly beyond the
lengths observed during training?

## Primary hypothesis

The heterogeneous leaky RNN will maintain lower error at longer sequence
lengths than the vanilla RNN and validation-selected best homogeneous leaky RNN,
without materially sacrificing short-sequence performance.

The GRU remains a secondary strong gated-memory reference. It cannot support or
falsify the primary hypothesis.

## Pre-run predictions

**Primary pre-run prediction:** Heterogeneous leak should maintain lower
long-length error than vanilla and the best homogeneous leak while preserving
short-length performance.

**Alternative pre-run prediction:** The Adding Problem should separate the models more
than binary delayed recall because two continuous values cannot be represented
by only two saturated attractors. I expect GRU to extrapolate most reliably. I
expect vanilla error to rise sharply beyond the training range. Heterogeneous
leak should outperform vanilla at length 320, but the best homogeneous alpha may
remain competitive; my preregistered call is that the full strict heterogeneous-
over-both-baselines criterion is uncertain and more likely to fail than pass.

## Data protocol

- Inputs have shape `[batch, sequence_length, 2]` with value and marker channels.
- Values are independent `Uniform(0, 1)` samples.
- Exactly two positions are marked. One is sampled uniformly from the first
  half and one from the second half, ensuring a length-dependent retention span.
- The scalar target is the exact sum of the two marked values and lies in `[0, 2]`.
- Training lengths are 20, 40, and 80.
- Test lengths are 20, 40, 80, 160, and 320. Lengths 160 and 320 are unseen:
  they are absent from training, per-epoch validation, alpha selection, and
  optimizer decisions.
- Each seed uses 4,096 training examples per training length, 512 validation
  examples per training length, and 1,024 test examples per evaluation length.
- Paired architecture settings receive identical tensors, length-batch schedule,
  initialization policy, and within-length batch order for each seed.

Always predicting the expected target of 1.0 has population MSE `1/6`, providing
a literal non-memory reference line.

## Models and controlled conditions

All models use hidden width 96 and a one-output linear regression head read from
the final recurrent state:

- one-layer vanilla tanh `nn.RNN`;
- homogeneous custom leaky RNN at alpha 0.5, 0.1, and 0.01;
- heterogeneous custom leaky RNN with 32 neurons at each alpha;
- one-layer same-width `nn.GRU` as the secondary reference.

The custom candidate and leak equations, dense recurrent connectivity, fixed
alpha buffers, initialization bounds, and paired-seed conventions are unchanged
from Experiment 002. Vanilla and leaky models have comparable parameter counts;
the GRU's larger count is logged explicitly.

## Training and selection

- Seeds 42, 43, and 44.
- Thirty fixed epochs, AdamW, learning rate 0.001, zero weight decay.
- Batch size 128; gradient-norm clipping at 1.0.
- MSE training loss, with MSE and MAE reported separately by length.
- Fixed final epoch; no early stopping or checkpoint selection.
- Homogeneous alpha selected using pooled final validation MSE at trained lengths
  40 and 80. Ties use all trained validation lengths and then lower alpha.
- Test data is evaluated once after training and never selects an alpha or epoch.

## Preregistered exploratory verdict

At both unseen lengths 160 and 320, heterogeneous leak must:

1. reduce mean test MSE by at least 20% relative to vanilla and the selected
   homogeneous model; and
2. achieve lower paired-seed MSE than each primary baseline in at least two of
   three seeds.

At trained short lengths 20 and 40, heterogeneous mean MSE may exceed the
stronger primary baseline by no more than 0.01. Every condition must hold for
exploratory support. Three seeds do not establish formal significance.

## Planned artifacts

- Per-seed and aggregate CSV/JSON/Markdown results with mean ± sample SD.
- MSE-versus-sequence-length plots with unseen lengths and the `1/6` baseline.
- Parent MLflow curves keyed by sequence length and epoch.
- Exact length-320 inputs, targets, predictions, and hidden states for two
  seed-42 examples.
- Hidden-state heatmaps for all displayed models and heterogeneous group traces.

## Results and interpretation

The official unchanged suite completed on 2026-08-26 on CPU. Validation selected
homogeneous alpha 0.5.

| Model | Length 20 | Length 40 | Length 80 | Length 160 unseen | Length 320 unseen |
|---|---:|---:|---:|---:|---:|
| Vanilla RNN | 0.0260 | 0.0298 | 0.0448 | 0.0876 | 0.1268 |
| Homogeneous alpha 0.5 | 0.1456 | 0.1583 | 0.1668 | 0.1753 | 0.1712 |
| Heterogeneous leak | 0.1527 | 0.1605 | 0.1676 | 0.1770 | 0.1815 |
| GRU reference | 0.0025 | 0.0025 | 0.0038 | 0.0063 | 0.0133 |

Entries are mean test MSE across three seeds; lower is better. The constant
predict-1 reference has expected MSE 0.1667.

## Preregistered verdict

The primary hypothesis is not supported. Heterogeneous leak did not achieve the
required 20% long-length improvement over either baseline. It lost to vanilla
in every paired seed at both unseen lengths, and its mean MSE was slightly worse
than selected homogeneous alpha 0.5 at both 160 and 320. Most importantly, it
violated the short-length safeguard by 0.1267 MSE at length 20 and 0.1306 at
length 40 relative to vanilla, far beyond the allowed 0.01.

The single paired-win condition against homogeneous leak at length 160 passed,
but all conditions were preregistered as jointly necessary. GRU's strong result
is descriptive and cannot alter this verdict.

## Interpretation

This result is more informative than Experiment 002's ceiling. The heterogeneous
model did not merely forget as sequences became longer: it failed to learn the
selective addition rule adequately at the shortest trained lengths. Its MSE
stayed close to the constant-mean baseline across the curve.

A post-hoc seed-42 length-320 diagnostic makes the behavior concrete:

| Model | Prediction mean | Prediction SD | Correlation with target |
|---|---:|---:|---:|
| Vanilla RNN | 1.212 | 0.255 | 0.799 |
| Homogeneous alpha 0.5 | 0.940 | 0.075 | 0.028 |
| Heterogeneous leak | 0.963 | 0.079 | 0.065 |
| GRU | 1.011 | 0.401 | 0.991 |

The target SD was 0.406. Homogeneous and heterogeneous predictions barely varied
and were nearly uncorrelated with the correct sum, showing that they mostly
learned the expected target near 1.0. The heterogeneous heatmap likewise shows
little organized state change at the marked positions. GRU activity changes
sharply at both markers, and its output preserves nearly the full target
variation. Vanilla learned a partial continuous algorithm that degraded with
unseen length but remained substantially better than the no-memory baseline.

The full homogeneous sweep adds a caution about interpreting slow leak as safe
memory. Alpha 0.01 remained near baseline on trained lengths but extrapolated
catastrophically: mean MSE rose to 1.255 at length 160 and 10.132 at length 320.
Slow updates can accumulate small input-driven errors or drift over a much longer
unroll; retention alone does not provide selective writing or stable addition.

Under the controlled optimizer and 30-epoch budget, fixed mixed timescales did
not supply the input-dependent write behavior needed by this task. Dense
recurrence was sufficient for vanilla to learn partially, while learned GRU
gates were dramatically more effective. The result rejects the configured
end-to-end hypothesis; it does not prove that every heterogeneous-alpha design
or training method must fail. A follow-up aimed at representational capacity
would need to separate optimization difficulty from final memory, for example
by testing learned input-dependent alpha or a curriculum, and would constitute
a new hypothesis rather than a reinterpretation of this run.
