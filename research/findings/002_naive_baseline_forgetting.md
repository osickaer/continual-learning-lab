# Naive sequential Split CIFAR-10 exhibits complete old-task collapse

Status: supported by Experiment 000, seed 42

Experiment 000 demonstrated the intended catastrophic-forgetting baseline. The
single-head CNN learned every current task to `0.7950`-`0.9380` test accuracy,
but each older task fell to zero accuracy after training the next task. Final
average accuracy was `0.1767` and final average forgetting was `0.873875`.

The simultaneous presence of high current-task accuracy and zero old-task
accuracy separates catastrophic forgetting from insufficient model plasticity.
Future methods should be compared against the identical task order, model,
training budget, seed, and evaluation protocol. Repeat seeds are still needed
before treating the exact metric values as stable estimates.

Source: Experiment 000, MLflow run
`d5669187cc274ae1882b36bc1480c6e1`.
