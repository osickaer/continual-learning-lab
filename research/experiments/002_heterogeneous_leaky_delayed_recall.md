# Experiment 002: Heterogeneous leaky RNN delayed recall

Status: official and corrected runs complete; hypothesis not supported

Origin: exploratory recurrent-memory hypothesis

Configuration: `configs/heterogeneous_leaky_delayed_recall.yaml`

MLflow parent run: `df975eeebe104351995d8592258e85f8`

## Research question

Can a fixed distribution of fast, medium, and slow recurrent update rates retain
designated information over long delays better than a vanilla RNN and the best
validation-selected single update rate?

This tests a possible memory primitive. It is relevant to future continual-
learning architectures, but the delayed-recall task does not itself test
catastrophic forgetting or continual learning.

## Primary hypothesis

Heterogeneous leak should produce better long-delay recall than the best
homogeneous leak and vanilla RNN without substantially harming short-delay
performance.

The GRU is a secondary learned-gating reference. It does not determine whether
the primary hypothesis is supported.

## Prediction gate

The following expectations were recorded before the official run.

My prediction is that the accuracy-versus-delay curve for the vanilla recurrent model will be better for shorter delays but worse for longer range delays. I think the homogeneous leaky model will have better mid-range delay performance but will struggle between short and long range depending on the alpha. Then, I think the heterogeneous leaky recurrent model will have _decent_ or flat performance across all delay tests but none of them will be exceptional performance or accuracy. I think the GRU model will perform similarly to this. For the three heterogeneous leak groups, I believe that the low-alpha neurons will mostly take care of the long-range memory while the high-alpha neurons will take care of the short term memory and the mid-alpha neurons will handle the medium-range memory.

**Alternative pre-run prediction, recorded before the official run:** I expect the GRU
to have the strongest and flattest curve, probably near ceiling, because it can
learn when to write and retain and has more parameters. I expect vanilla to be
strong at delays 5 and 10 and to weaken at 40 and 80, although this one-bit task
may let it learn a more stable recurrent state than we expect. I predict alpha
0.1 will be the selected homogeneous setting: alpha 0.5 writes quickly but may
overwrite quickly, while alpha 0.01 retains strongly but only weakly writes the
single-timestep value. I expect heterogeneous leak to beat vanilla at long
delays without a meaningful short-delay penalty, but I do **not** expect it to
clear the full five-point margin over the best homogeneous model at both 40 and
80; my preregistered call is therefore that the strict hypothesis will not be
supported. In its hidden activity I expect the alpha-0.01 group to change most
smoothly and persistently, the alpha-0.5 group to respond most sharply to the
value and cue, and the alpha-0.1 group to sit between them. Because recurrence is
dense, I expect distributed cooperation rather than perfectly separated
short-, medium-, and long-delay neuron roles.

## Task and data

- Delays: 5, 10, 20, 40, and 80 intervening timesteps.
- Every sequence is 82 timesteps with three input features: designated value,
  irrelevant distractor, and recall cue.
- The cue is fixed at the final timestep. A delay-$D$ value occurs at index
  `80 - D`, leaving exactly $D$ intervening inputs.
- Values are exactly -1 or +1 with balanced class labels. Distractors are drawn
  independently from `Uniform(-1, 1)` in their own input channel.
- Each seed receives 1,024 training, 256 validation, and 512 test examples per
  delay. Paired architecture runs receive identical data and batch order.

The fixed sequence length controls total recurrent computation across delays.
Moving the value position changes only how long it must be retained before the
common final cue.

## Models

All models use hidden width 96 and the same two-output classification head.

- Vanilla one-layer tanh `nn.RNN`.
- Custom homogeneous leaky RNN swept over alpha 0.5, 0.1, and 0.01.
- Custom heterogeneous leaky RNN with 32 neurons at each alpha.
- Same-width `nn.GRU` as a secondary reference.

The explicit leaky update is:

```text
candidate_t = tanh(W_ih x_t + b_ih + W_hh h_prev + b_hh)
h_t = (1 - alpha) h_prev + alpha candidate_t
```

Alpha is fixed, saved as model state, moved with the model, and excluded from
gradient updates. The recurrent matrix is dense. Vanilla and leaky models have
9,890 trainable parameters; the same-width GRU has 29,282, which is a documented
advantage rather than a primary fair-capacity comparison.

## Training contract

- Seeds 42, 43, and 44.
- Thirty fixed epochs; no early stopping or checkpoint selection.
- AdamW, learning rate 0.001, zero weight decay.
- Batch size 128 and gradient-norm clipping at 1.0.
- Common uniform initialization bounded by `1 / sqrt(hidden_size)` with zero
  biases.
- Cross-entropy loss on the two recall labels.
- Validation after every epoch; test evaluation once after the final epoch.

The best homogeneous alpha is selected by pooled final validation accuracy at
delays 40 and 80 across all seeds. Ties use all-delay validation accuracy and
then the lower alpha. Test metrics never select a model or alpha.

## Preregistered interpretation

The hypothesis receives exploratory support only if all conditions hold:

1. At delays 40 and 80, heterogeneous mean test accuracy exceeds vanilla and
   the selected homogeneous model by at least five percentage points.
2. At each long delay, heterogeneous accuracy beats each primary baseline in at
   least two of three paired seeds.
3. At delays 5 and 10, heterogeneous mean accuracy is no more than three points
   below the stronger primary baseline.

If any condition fails, the hypothesis is not supported by this exploratory
experiment. Three seeds show preliminary robustness, not formal statistical
significance. GRU performance is contextual and cannot change the verdict.

## Metrics and artifacts

- Per-epoch training loss, training recall accuracy, synchronized training time,
  and aggregate/per-delay validation metrics.
- Final per-delay test loss and recall accuracy for every child run.
- Per-seed CSV, mean and sample-standard-deviation summaries, an automated
  criterion report, and the selected homogeneous alpha.
- Recall accuracy versus delay for all official curves and a chance reference.
- Exact delay-80 hidden states for paired -1 and +1 examples, model heatmaps,
  and heterogeneous leak-group traces.

## Results

The official unchanged three-seed suite completed on 2026-08-26. Validation
selected homogeneous alpha 0.1.

| Model | Delay 5 | Delay 10 | Delay 20 | Delay 40 | Delay 80 |
|---|---:|---:|---:|---:|---:|
| Vanilla RNN | 90.76% | 74.54% | 76.56% | 79.04% | 100.00% |
| Homogeneous alpha 0.1 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| Heterogeneous leak | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| GRU | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |

The vanilla means hide strong seed variation. Seeds 43 and 44 reached 100% at
every delay. Seed 42 finished at 72.27%, 23.63%, 29.69%, 37.11%, and 100% from
delay 5 through 80, producing sample standard deviations of 16–44 points at the
non-80 delays.

The complete homogeneous sweep also revealed a timescale pattern:

| Alpha | Delay 5 | Delay 10 | Delay 20 | Delay 40 | Delay 80 |
|---:|---:|---:|---:|---:|---:|
| 0.5 | 83.33% | 83.33% | 83.33% | 83.33% | 100.00% |
| 0.1 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| 0.01 | 70.77% | 77.21% | 95.18% | 100.00% | 100.00% |

## Preregistered verdict

The primary hypothesis is not supported by this configured experiment.
Heterogeneous leak incurred no short-delay penalty and exceeded vanilla's mean
at delay 40 by 20.96 points, but it did not beat vanilla in two of three paired
seeds there. At delay 80, vanilla was also perfect. Most decisively, the selected
homogeneous alpha 0.1 tied heterogeneous leak at 100% for every delay and seed,
leaving no five-point heterogeneous advantage at either long delay.

This is not evidence that heterogeneous dynamics cannot remember. They learned
the entire task perfectly and consistently. It is evidence that this task and
configuration do not demonstrate an advantage over one well-chosen homogeneous
timescale. The homogeneous, heterogeneous, and GRU curves are all ceiling-
limited, so their relative memory limits remain unidentified.

## Additional observations

- The alternative predictions that GRU would be near ceiling, alpha 0.1 would be
  selected, and the strict hypothesis would fail were supported. The prediction
  that vanilla would degrade with longer post-value delay was not supported.
- The primary prediction correctly anticipated the qualitative alpha direction. Alpha 0.01
  was strongest when it had many post-value updates and weakest when the value
  arrived shortly before the cue, exposing the tradeoff between slow retention
  and slow writing/integration.
- Some fast-update models were optimization-unstable despite learning the task.
  Vanilla seed 42 reached 100% validation accuracy for many epochs before
  falling to 53.52% at epoch 30. Homogeneous alpha 0.5 seed 44 similarly fell
  from 100% to 60.00% over the last two epochs. The fixed-final-epoch contract
  correctly retains these failures instead of selecting a favorable checkpoint.
- Heterogeneous seed-42 activity did not separate into independent short-,
  medium-, and long-delay modules. Alpha-0.5 neurons changed and saturated
  fastest, alpha-0.1 neurons followed, and alpha-0.01 neurons accumulated slowly;
  dense recurrence produced a distributed persistent state across groups.

## Protocol limitation discovered from the result

Keeping every mixed-delay sequence at length 82 placed the value at timestep
`80 - delay`. Delay-80 examples therefore present the value immediately from a
zero hidden state, while shorter-delay examples first expose the model to up to
75 distractor timesteps. The resulting curve measures sensitivity to the hidden
state before the value arrives as well as memory after it arrives.

A post-hoc diagnostic on vanilla seed 42 confirmed this matters. Removing only
post-value distractors barely changed its final accuracy, while altering the
pre-value distractor prefix changed outcomes radically. This diagnostic does not
change the preregistered verdict, but it prevents interpreting the non-monotonic
vanilla curve as a pure memory-retention curve.

A clean follow-up should keep the value at timestep zero, place the cue at
`delay + 1`, pad only after the cue, and read the hidden state at each example's
cue position. That change should be made before increasing delay or task
difficulty, so timing protocol and memory capacity remain separate questions.

## Corrected follow-up protocol (Experiment 002b)

Status: implemented; pre-run prediction and execution pending

Configuration: `configs/heterogeneous_leaky_delayed_recall_corrected.yaml`

The follow-up keeps the model sweep, widths, alpha values, seeds, optimizer,
epoch budget, initialization, clipping, dataset counts, selection rule, and
support thresholds unchanged. It corrects the task construction:

- Delays are 5, 10, 20, 40, 80, and 120.
- Every tensor has 122 timesteps so mixed delays still batch together.
- The target value always arrives at timestep 0 with a store marker.
- A delay-$D$ example receives exactly $D$ distractor updates at timesteps
  1 through $D$, then its recall cue at timestep $D+1$.
- Timesteps after the cue are zero padding. The output head reads the hidden
  state at that example's cue, so padding adds neither memory updates nor
  evidence to its prediction.
- Target values and distractors share the content channel. The store marker
  distinguishes the one item that should be remembered, preventing a solution
  that simply ignores a permanently irrelevant distractor channel.

The homogeneous alpha is now selected on validation delays 80 and 120, and the
preregistered long-delay checks apply at 80 and 120. This is a new run, not a
replacement or retrospective alteration of the completed first run.

### Corrected-run prediction gate

Before running Experiment 002b, append brief predictions for:

1. the four displayed curves at delays 5 through 120;
2. whether the selected homogeneous alpha will remain 0.1;
3. whether same-channel distractors will prevent ceiling performance; and
4. whether heterogeneous leak will clear the five-point margins at both 80 and
   120.
