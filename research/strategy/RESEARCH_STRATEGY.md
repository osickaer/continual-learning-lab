# Research Strategy

Status: concise research-method reference. `AGENTS.md` controls assistant
behavior; individual experiment records contain current status.

## Goal

Build the ability to convert important questions into trustworthy experiments,
then use the evidence to improve continual-learning architectures.

```text
question -> prediction -> controlled experiment -> evidence -> updated belief
```

A disproven hypothesis is useful when the experiment is valid and the conclusion
is clear.

## Research loop

1. **Observe:** Identify a specific confusion, failure, or promising behavior.
2. **Ask:** Form one narrow question that an experiment can materially answer.
3. **Predict:** Record the expected result, reasoning, and evidence that would
   change the prediction.
4. **Establish the baseline:** Verify that the comparison run, configuration,
   metrics, and evaluation are trustworthy.
5. **Change one main thing:** Keep unrelated data, architecture, optimization,
   seed, budget, and evaluation conditions fixed when practical.
6. **Run:** Make the command and resolved configuration reproducible.
7. **Measure:** Log only the parameters, metrics, curves, and artifacts needed to
   reproduce and interpret the run. Prefer clear machine records over redundant
   dashboards.
8. **Interpret:** Compare against the prediction and baseline; estimate effect
   size, identify confounds or alternative explanations, and avoid changing the
   experiment before understanding the result.
9. **Record:** Preserve the concise scientific conclusion and decisive evidence.
10. **Decide:** Stop, repeat, refine, or branch.

The previous result should usually determine the next experiment.

## Fair-comparison rules

- State the baseline and the one conceptual change.
- Declare the primary training-budget contract and any unavoidable mismatch.
- Do not silently tune multiple mechanisms at once.
- Do not use the test set as an iterative tuning set.
- Compare retention and new-task learning; reduced forgetting alone is not
  success.
- Replicate across seeds after the first meaningful method result, not before a
  basic signal exists.
- Investigate unexpected results before explaining them away.

## Records

Use each system for one purpose:

- **MLflow:** resolved configuration and quantitative run evidence.
- **Git:** the implementation that produced the result.
- **Research record:** question, prediction, change, decisive results,
  interpretation, limitations, and next decision.

Research notes may summarize the few decisive metrics needed to make the
conclusion readable; they should not duplicate every MLflow series.

## Finished experiment

An experiment is complete when the following are answerable:

- What question did it test?
- What changed relative to the baseline?
- What evidence determined the result?
- What is the current conclusion and its limitation?
- What should happen next: stop, repeat, refine, or branch?

Do not indefinitely polish an experiment after those questions are answered.

## Momentum rules

- Prefer a small valid experiment over an elaborate unexecuted design.
- Shrink the question when stuck.
- Build infrastructure only when current work needs it repeatedly.
- Capture unrelated ideas without abandoning the active experiment.
- Read when it helps design, implement, or interpret the next experiment.
- Negative results count.
- When overwhelmed, return to the baseline and the single next decision.
