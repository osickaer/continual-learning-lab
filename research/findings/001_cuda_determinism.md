# CUDA determinism requires operation-level compatibility

Status: supported by implementation failure and replacement smoke test

Strict `torch.use_deterministic_algorithms(True)` correctly rejected
`AdaptiveAvgPool2d` during CUDA backpropagation in PyTorch 2.13. A model can
therefore run successfully on CPU yet fail on GPU even when seeding and cuDNN
settings are correct.

For fixed 32x32 CIFAR-10 inputs, replacing the adaptive 8x8-to-4x4 reduction
with `AvgPool2d(2)` preserves the architecture's output shape and trainable
parameter count while supporting deterministic CUDA backpropagation. Keep a
CUDA backward smoke test when changing model operations; do not weaken strict
determinism to warnings without making that an explicit experimental choice.
