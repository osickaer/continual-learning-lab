from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from html import escape
from pathlib import Path

import torch
from torch import nn
from torchvision.utils import make_grid, save_image


CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


@dataclass(frozen=True)
class ActivationSummary:
    """Small, readable statistics for one intermediate tensor."""

    name: str
    display_name: str
    module_type: str
    shape: tuple[int, ...]
    parameters: int
    minimum: float
    maximum: float
    mean: float
    standard_deviation: float
    negative_fraction: float
    zero_fraction: float
    positive_fraction: float
    image_path: str | None = None


@dataclass(frozen=True)
class InspectionResult:
    logits: torch.Tensor
    probabilities: torch.Tensor
    activations: dict[str, torch.Tensor]
    summaries: list[ActivationSummary]


def capture_activations(model: nn.Module, inputs: torch.Tensor) -> InspectionResult:
    """Run one forward pass and retain every leaf module's output tensor."""

    activations: dict[str, torch.Tensor] = {}
    module_details: dict[str, tuple[str, int]] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def register(name: str, module: nn.Module) -> None:
        module_details[name] = (
            type(module).__name__,
            sum(parameter.numel() for parameter in module.parameters(recurse=False)),
        )

        def save_output(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: object) -> None:
            if isinstance(output, torch.Tensor):
                activations[name] = output.detach().cpu()

        handles.append(module.register_forward_hook(save_output))

    for name, module in model.named_modules():
        if name and not any(module.children()):
            register(name, module)

    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            logits = model(inputs).detach().cpu()
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)

    summaries = [
        summarize_activation(name, tensor, *module_details[name])
        for name, tensor in activations.items()
    ]
    return InspectionResult(
        logits=logits,
        probabilities=torch.softmax(logits, dim=1),
        activations=activations,
        summaries=summaries,
    )


def summarize_activation(
    name: str,
    tensor: torch.Tensor,
    module_type: str,
    parameters: int,
) -> ActivationSummary:
    values = tensor.detach().float().cpu()
    total = values.numel()
    return ActivationSummary(
        name=name,
        display_name=_display_name(name, module_type),
        module_type=module_type,
        shape=tuple(values.shape),
        parameters=parameters,
        minimum=values.min().item(),
        maximum=values.max().item(),
        mean=values.mean().item(),
        standard_deviation=values.std(unbiased=False).item(),
        negative_fraction=(values < 0).sum().item() / total,
        zero_fraction=(values == 0).sum().item() / total,
        positive_fraction=(values > 0).sum().item() / total,
    )


def write_inspection(
    output_dir: Path,
    model: nn.Module,
    normalized_image: torch.Tensor,
    result: InspectionResult,
    *,
    checkpoint: Path,
    split: str,
    dataset_index: int,
    target: int,
    max_channels: int,
    mean: tuple[float, ...],
    std: tuple[float, ...],
) -> Path:
    """Write exact values, visual summaries, and a browser-readable report."""

    output_dir.mkdir(parents=True, exist_ok=False)
    image_dir = output_dir / "activation_images"
    image_dir.mkdir()

    input_path = output_dir / "input.png"
    save_image(_denormalize(normalized_image, mean, std), input_path)

    summaries: list[ActivationSummary] = []
    for index, summary in enumerate(result.summaries, start=1):
        tensor = result.activations[summary.name]
        image_path = image_dir / f"{index:02d}_{summary.name.replace('.', '_')}.png"
        if tensor.ndim == 4:
            save_image(_feature_map_grid(tensor, max_channels), image_path)
        elif tensor.ndim == 2:
            save_image(_vector_image(tensor), image_path)
        else:
            image_path = None
        summaries.append(
            ActivationSummary(
                **{
                    **asdict(summary),
                    "image_path": image_path.relative_to(output_dir).as_posix()
                    if image_path is not None
                    else None,
                }
            )
        )

    probabilities = result.probabilities[0]
    prediction = int(probabilities.argmax().item())
    metadata = {
        "checkpoint": str(checkpoint.resolve()),
        "split": split,
        "dataset_index": dataset_index,
        "target_id": target,
        "target_name": CIFAR10_CLASSES[target],
        "prediction_id": prediction,
        "prediction_name": CIFAR10_CLASSES[prediction],
        "confidence": probabilities[prediction].item(),
        "logits": result.logits[0].tolist(),
        "probabilities": probabilities.tolist(),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }
    (output_dir / "prediction.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    (output_dir / "activation_summary.json").write_text(
        json.dumps([asdict(summary) for summary in summaries], indent=2),
        encoding="utf-8",
    )
    (output_dir / "model_architecture.txt").write_text(str(model), encoding="utf-8")
    torch.save(
        {
            "normalized_input": normalized_image.detach().cpu(),
            "activations": result.activations,
            "logits": result.logits,
            "probabilities": result.probabilities,
        },
        output_dir / "intermediate_values.pt",
    )

    report_path = output_dir / "report.html"
    report_path.write_text(
        _build_html_report(metadata, summaries, max_channels), encoding="utf-8"
    )
    return report_path


def _display_name(name: str, module_type: str) -> str:
    section, _, position = name.partition(".")
    if position.isdigit():
        return f"{section.title()} {int(position) + 1}: {module_type}"
    return f"{name}: {module_type}"


def _denormalize(
    image: torch.Tensor,
    mean: tuple[float, ...],
    std: tuple[float, ...],
) -> torch.Tensor:
    mean_tensor = torch.tensor(mean, dtype=image.dtype).view(-1, 1, 1)
    std_tensor = torch.tensor(std, dtype=image.dtype).view(-1, 1, 1)
    return (image.detach().cpu() * std_tensor + mean_tensor).clamp(0, 1)


def _feature_map_grid(tensor: torch.Tensor, max_channels: int) -> torch.Tensor:
    maps = tensor[0, :max_channels].float().cpu()
    scale = maps.abs().flatten(1).amax(dim=1).clamp_min(1e-12).view(-1, 1, 1)
    normalized = maps / scale
    positive = normalized.clamp(min=0)
    negative = (-normalized).clamp(min=0)

    # Orange marks positive activations; blue marks negative activations.
    rgb = torch.stack(
        (positive, 0.45 * positive + 0.25 * negative, negative), dim=1
    )
    columns = min(4, len(rgb))
    return make_grid(rgb, nrow=columns, padding=2, pad_value=0.12)


def _vector_image(tensor: torch.Tensor) -> torch.Tensor:
    values = tensor[0].float().cpu()
    columns = math.ceil(math.sqrt(len(values)))
    rows = math.ceil(len(values) / columns)
    padded = torch.zeros(rows * columns)
    padded[: len(values)] = values
    plane = padded.view(rows, columns)
    scale = plane.abs().max().clamp_min(1e-12)
    normalized = plane / scale
    positive = normalized.clamp(min=0)
    negative = (-normalized).clamp(min=0)
    return torch.stack((positive, 0.45 * positive + 0.25 * negative, negative))


def _build_html_report(
    metadata: dict[str, object],
    summaries: list[ActivationSummary],
    max_channels: int,
) -> str:
    probability_rows = "".join(
        f"<tr><td>{escape(name)}</td><td>{float(logit):.4f}</td>"
        f"<td>{float(probability):.2%}</td></tr>"
        for name, logit, probability in zip(
            CIFAR10_CLASSES,
            metadata["logits"],
            metadata["probabilities"],
            strict=True,
        )
    )
    layer_sections = "".join(_layer_html(summary, max_channels) for summary in summaries)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SmallCNN activation inspection</title>
  <style>
    :root {{ color-scheme: light dark; font-family: system-ui, sans-serif; }}
    body {{ max-width: 1100px; margin: 0 auto; padding: 24px; line-height: 1.5; }}
    header, section {{ margin-bottom: 32px; }}
    .overview {{ display: grid; grid-template-columns: minmax(180px, 280px) 1fr; gap: 28px; }}
    .input-image, .activation-image {{ image-rendering: pixelated; width: 100%; border: 1px solid #8888; }}
    .layer {{ border-top: 1px solid #8888; padding-top: 20px; }}
    .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 8px 18px; }}
    .stats div {{ display: grid; }}
    .stats span:first-child {{ opacity: 0.7; font-size: 0.9rem; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ padding: 6px 10px; border-bottom: 1px solid #8885; text-align: left; }}
    code {{ overflow-wrap: anywhere; }}
    .legend {{ display: flex; gap: 18px; flex-wrap: wrap; }}
    .swatch {{ display: inline-block; width: 1em; height: 1em; margin-right: 5px; vertical-align: -0.1em; }}
    .positive {{ background: rgb(255, 115, 0); }}
    .negative {{ background: rgb(0, 64, 255); }}
    .zero {{ background: rgb(0, 0, 0); border: 1px solid #888; }}
    @media (max-width: 650px) {{ .overview {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <header>
    <h1>SmallCNN activation inspection</h1>
    <p>True class: <strong>{escape(str(metadata['target_name']))}</strong>. Predicted:
    <strong>{escape(str(metadata['prediction_name']))}</strong>
    ({float(metadata['confidence']):.2%}).</p>
    <p><code>{escape(str(metadata['checkpoint']))}</code></p>
  </header>
  <section class="overview">
    <div><h2>Input</h2><img class="input-image" src="input.png" alt="CIFAR-10 input image"></div>
    <div>
      <h2>Final class scores</h2>
      <table><thead><tr><th>Class</th><th>Logit</th><th>Probability</th></tr></thead>
      <tbody>{probability_rows}</tbody></table>
    </div>
  </section>
  <section>
    <h2>How to read activation colors</h2>
    <div class="legend">
      <span><i class="swatch positive"></i>positive</span>
      <span><i class="swatch negative"></i>negative</span>
      <span><i class="swatch zero"></i>zero</span>
    </div>
    <p>Each map is scaled independently so its spatial pattern is visible. Use the statistics—not
    color intensity—to compare numerical magnitude between layers. At most {max_channels} channels
    are shown, while the raw file retains every channel.</p>
  </section>
  {layer_sections}
</body>
</html>
"""


def _layer_html(summary: ActivationSummary, max_channels: int) -> str:
    image = ""
    if summary.image_path is not None:
        qualifier = ""
        if len(summary.shape) == 4 and summary.shape[1] > max_channels:
            qualifier = f" (showing {max_channels} of {summary.shape[1]} channels)"
        image = (
            f'<p>{escape(qualifier)}</p><img class="activation-image" '
            f'src="{escape(summary.image_path)}" alt="Activation visualization for '
            f'{escape(summary.display_name)}">'
        )
    shape = " × ".join(str(value) for value in summary.shape)
    return f"""
  <section class="layer">
    <h2>{escape(summary.display_name)}</h2>
    <div class="stats">
      <div><span>Tensor shape</span><strong>{shape}</strong></div>
      <div><span>Layer parameters</span><strong>{summary.parameters:,}</strong></div>
      <div><span>Minimum</span><strong>{summary.minimum:.5f}</strong></div>
      <div><span>Maximum</span><strong>{summary.maximum:.5f}</strong></div>
      <div><span>Mean</span><strong>{summary.mean:.5f}</strong></div>
      <div><span>Standard deviation</span><strong>{summary.standard_deviation:.5f}</strong></div>
      <div><span>Negative</span><strong>{summary.negative_fraction:.1%}</strong></div>
      <div><span>Exactly zero</span><strong>{summary.zero_fraction:.1%}</strong></div>
      <div><span>Positive</span><strong>{summary.positive_fraction:.1%}</strong></div>
    </div>
    {image}
  </section>
"""
