# Experiment 004: Persistent recurrent state on Stream-51

Status: implemented; awaiting raw dataset, smoke run, and official run

Origin: exploratory temporal-coherence hypothesis

Configuration: `configs/stream51_temporal_state.yaml`

## Research question

Does recurrent activation carried between observations improve single-pass
online classification when experience has natural temporal structure, compared
with a parameter- and compute-matched learner that resets activation before
every observation?

## Hypotheses and predictions

**Primary hypothesis:** one-tick persistent state will improve prequential macro
accuracy in natural Stream-51 order, and that persistence gain will be larger
than under global shuffle.

**Primary pre-run prediction:** persistent state should exploit coherent
experience better than a comparable stateless learner. Natural, within-
trajectory shuffled, and globally shuffled streams should expose which level of
temporal structure supplies the advantage.

**Alternative pre-run prediction:** frozen semantic features plus repeated views of one
object should make persistence useful in natural and local-shuffle segments.
Local shuffle may remain close to natural if object continuity matters more than
exact frame chronology. State carried through globally shuffled pseudo-segments
should be neutral or harmful. Four blank recurrent ticks may improve state
processing, but I am less confident that their gain will exceed the matched
four-tick stateless control.

## Controlled protocol

- Stream-51 training images are cropped with their bounding boxes and encoded
  once by frozen ImageNet ResNet-18 into 512-dimensional features.
- The original metadata defines trajectories by class, clip, and video; frames
  are chronological within each trajectory.
- One fixed trajectory order supplies the canonical trajectory-length sequence.
- Natural, local-shuffle, and global-shuffle contain identical observation IDs.
- Global shuffle is partitioned into pseudo-trajectories using the canonical
  lengths. All conditions therefore have identical reset counts and positions.
- `original_trajectory_id` remains metadata; only the separate experimental
  segment identifier controls recurrent resets.
- Every learner sees one observation at a time for one pass. Its prediction is
  scored before cross-entropy updates the parameters.
- Persistent activations cross observations as detached tensors. Gradients do
  not cross observation boundaries, but four-tick variants backpropagate through
  all ticks belonging to the current observation.

## Models and fixed conditions

The four conditions share one GRUCell classifier with hidden width 256 and a
51-class head:

1. stateless, one tick;
2. stateless, four ticks;
3. persistent, one tick;
4. persistent, four ticks.

Only the first tick receives the visual feature; later ticks receive zeros.
Within a tick count, stateless and persistent models have exactly the same
parameters and per-observation computation. Seeds are 42, 43, and 44. AdamW,
learning rate 0.001, zero weight decay, and gradient clipping at 1.0 remain
fixed. Static held-out images reset state independently.

## Preregistered verdict

Primary support requires all of the following:

- natural one-tick persistence gain of at least 0.02 mean macro accuracy;
- positive natural gain in at least two of three paired seeds;
- natural-minus-global persistence interaction of at least 0.02 and positive
  in at least two paired seeds; and
- natural persistent held-out macro accuracy no more than 0.02 below its
  stateless match.

The secondary four-tick hypothesis requires a natural-order tick-by-persistence
interaction of at least 0.01 and positive interaction in two paired seeds.

Local shuffle is diagnostic rather than a gating condition: similarity to
natural would point to same-object continuity rather than exact chronology.

## Planned evidence

- final prequential macro/micro accuracy and cross-entropy;
- 1,000-observation prequential windows;
- accuracy by position within the matched reset segment;
- fixed held-out known-class macro/micro accuracy;
- structural adjacency diagnostics and the exact reset-schedule hash;
- paired-seed aggregate tables, plot, and automated verdict.

Novelty detection, encoder fine-tuning, replay, alternate stream seeds, and
cross-observation TBPTT are separate future hypotheses.
