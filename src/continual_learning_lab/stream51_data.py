from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import torch
import torchvision
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18

from continual_learning_lab.config import Stream51Config, Stream51DataConfig
from continual_learning_lab.experiment_utils import write_json
from continual_learning_lab.reproducibility import make_generator


@dataclass(frozen=True)
class Stream51Image:
    observation_id: int
    class_id: int
    original_trajectory_id: str | None
    frame_num: int | None
    bbox: tuple[float, float, float, float]
    file_location: str


@dataclass(frozen=True)
class Stream51Record:
    observation_id: int
    class_id: int
    original_trajectory_id: str
    frame_num: int
    reset_segment_id: int
    position_in_segment: int
    reset_before: bool


@dataclass(frozen=True)
class OrderingDiagnostics:
    observations: int
    segments: int
    resets: int
    adjacent_same_trajectory_rate: float
    adjacent_same_class_rate: float
    chronological_successor_rate: float


@dataclass(frozen=True)
class FeatureCache:
    train_features: torch.Tensor
    train_labels: torch.Tensor
    test_features: torch.Tensor
    test_labels: torch.Tensor
    manifest: dict[str, object]


@dataclass(frozen=True)
class PreparedStream51Data:
    train_features: torch.Tensor
    train_labels: torch.Tensor
    test_features: torch.Tensor
    test_labels: torch.Tensor
    orderings: dict[str, tuple[Stream51Record, ...]]
    diagnostics: dict[str, OrderingDiagnostics]
    segment_lengths: tuple[int, ...]
    reset_schedule_hash: str
    cache_manifest: dict[str, object]


def _load_json_list(path: Path) -> list[list[object]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing Stream-51 metadata: {path}. Download Stream-51 manually "
            "and place it under the configured data.root."
        )
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, list):
        raise ValueError(f"Stream-51 metadata must be a list: {path}")
    return values


def load_train_metadata(path: str | Path) -> tuple[Stream51Image, ...]:
    records: list[Stream51Image] = []
    for observation_id, raw in enumerate(_load_json_list(Path(path))):
        if not isinstance(raw, list) or len(raw) != 7:
            raise ValueError("Each Stream-51 training record must contain 7 fields")
        class_id, clip_num, video_num, frame_num, _shape, bbox, file_location = raw
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError("Each Stream-51 bounding box must contain four values")
        trajectory_id = f"{int(class_id)}:{int(clip_num)}:{int(video_num)}"
        records.append(
            Stream51Image(
                observation_id=observation_id,
                class_id=int(class_id),
                original_trajectory_id=trajectory_id,
                frame_num=int(frame_num),
                bbox=tuple(float(value) for value in bbox),
                file_location=str(file_location),
            )
        )
    if not records:
        raise ValueError("Stream-51 training metadata is empty")
    return tuple(records)


def load_test_metadata(path: str | Path) -> tuple[Stream51Image, ...]:
    records: list[Stream51Image] = []
    for observation_id, raw in enumerate(_load_json_list(Path(path))):
        if not isinstance(raw, list) or len(raw) != 4:
            raise ValueError("Each Stream-51 test record must contain 4 fields")
        class_id, _shape, bbox, file_location = raw
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError("Each Stream-51 bounding box must contain four values")
        records.append(
            Stream51Image(
                observation_id=observation_id,
                class_id=int(class_id),
                original_trajectory_id=None,
                frame_num=None,
                bbox=tuple(float(value) for value in bbox),
                file_location=str(file_location),
            )
        )
    if not records:
        raise ValueError("Stream-51 test metadata is empty")
    return tuple(records)


def _canonical_trajectories(
    observations: Sequence[Stream51Image],
    *,
    seed: int,
    max_trajectories: int | None,
) -> list[list[Stream51Image]]:
    grouped: dict[str, list[Stream51Image]] = defaultdict(list)
    for observation in observations:
        if observation.original_trajectory_id is None or observation.frame_num is None:
            raise ValueError("Training observations require trajectory and frame metadata")
        grouped[observation.original_trajectory_id].append(observation)
    trajectories = [
        sorted(values, key=lambda value: value.frame_num)
        for _key, values in sorted(grouped.items())
    ]
    permutation = torch.randperm(len(trajectories), generator=make_generator(seed)).tolist()
    trajectories = [trajectories[index] for index in permutation]
    if max_trajectories is not None:
        trajectories = trajectories[:max_trajectories]
    return trajectories


def _assign_segments(
    observations: Sequence[Stream51Image], segment_lengths: Sequence[int]
) -> tuple[Stream51Record, ...]:
    if sum(segment_lengths) != len(observations):
        raise ValueError("Segment lengths must cover every observation exactly once")
    records: list[Stream51Record] = []
    cursor = 0
    for segment_id, length in enumerate(segment_lengths):
        for position in range(length):
            observation = observations[cursor]
            if observation.original_trajectory_id is None or observation.frame_num is None:
                raise ValueError("Training observations require trajectory and frame metadata")
            records.append(
                Stream51Record(
                    observation_id=observation.observation_id,
                    class_id=observation.class_id,
                    original_trajectory_id=observation.original_trajectory_id,
                    frame_num=observation.frame_num,
                    reset_segment_id=segment_id,
                    position_in_segment=position,
                    reset_before=position == 0,
                )
            )
            cursor += 1
    return tuple(records)


def build_stream_orderings(
    observations: Sequence[Stream51Image],
    *,
    seed: int,
    max_trajectories: int | None = None,
) -> tuple[dict[str, tuple[Stream51Record, ...]], tuple[int, ...]]:
    trajectories = _canonical_trajectories(
        observations, seed=seed, max_trajectories=max_trajectories
    )
    segment_lengths = tuple(len(trajectory) for trajectory in trajectories)
    natural_images = [image for trajectory in trajectories for image in trajectory]

    local_generator = make_generator(seed + 1)
    local_images: list[Stream51Image] = []
    for trajectory in trajectories:
        permutation = torch.randperm(
            len(trajectory), generator=local_generator
        ).tolist()
        local_images.extend(trajectory[index] for index in permutation)

    global_permutation = torch.randperm(
        len(natural_images), generator=make_generator(seed + 2)
    ).tolist()
    global_images = [natural_images[index] for index in global_permutation]

    orderings = {
        "natural": _assign_segments(natural_images, segment_lengths),
        "local_shuffle": _assign_segments(local_images, segment_lengths),
        "global_shuffle": _assign_segments(global_images, segment_lengths),
    }
    validate_matched_orderings(orderings, segment_lengths)
    return orderings, segment_lengths


def segment_lengths_from_records(records: Sequence[Stream51Record]) -> tuple[int, ...]:
    lengths: list[int] = []
    for record in records:
        if record.reset_before:
            lengths.append(0)
        if not lengths:
            raise ValueError("The first stream observation must reset state")
        lengths[-1] += 1
    return tuple(lengths)


def validate_matched_orderings(
    orderings: dict[str, tuple[Stream51Record, ...]],
    expected_segment_lengths: Sequence[int] | None = None,
) -> None:
    expected_names = {"natural", "local_shuffle", "global_shuffle"}
    if set(orderings) != expected_names:
        raise ValueError(f"Orderings must be exactly {sorted(expected_names)}")
    natural = orderings["natural"]
    expected_ids = sorted(record.observation_id for record in natural)
    expected_reset_vector = tuple(record.reset_before for record in natural)
    expected_reset_positions = tuple(
        index for index, reset in enumerate(expected_reset_vector) if reset
    )
    expected_lengths = segment_lengths_from_records(natural)
    if expected_segment_lengths is not None and expected_lengths != tuple(
        expected_segment_lengths
    ):
        raise ValueError("Natural ordering does not match the canonical segment lengths")
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("Natural ordering contains duplicate observation IDs")

    for name, records in orderings.items():
        ids = sorted(record.observation_id for record in records)
        resets = tuple(record.reset_before for record in records)
        reset_positions = tuple(index for index, reset in enumerate(resets) if reset)
        lengths = segment_lengths_from_records(records)
        if ids != expected_ids:
            raise ValueError(f"{name} does not contain the same observation IDs")
        if len(records) != len(natural):
            raise ValueError(f"{name} has a different number of observations")
        if sum(resets) != sum(expected_reset_vector):
            raise ValueError(f"{name} has a different number of resets")
        if reset_positions != expected_reset_positions:
            raise ValueError(f"{name} has different reset positions")
        if lengths != expected_lengths:
            raise ValueError(f"{name} has a different segment-length distribution")


def ordering_diagnostics(records: Sequence[Stream51Record]) -> OrderingDiagnostics:
    adjacent = max(0, len(records) - 1)
    same_trajectory = 0
    same_class = 0
    chronological = 0
    for previous, current in zip(records, records[1:]):
        same_trajectory += previous.original_trajectory_id == current.original_trajectory_id
        same_class += previous.class_id == current.class_id
        chronological += (
            previous.original_trajectory_id == current.original_trajectory_id
            and current.frame_num == previous.frame_num + 1
        )
    denominator = adjacent if adjacent else 1
    return OrderingDiagnostics(
        observations=len(records),
        segments=sum(record.reset_before for record in records),
        resets=sum(record.reset_before for record in records),
        adjacent_same_trajectory_rate=same_trajectory / denominator,
        adjacent_same_class_rate=same_class / denominator,
        chronological_successor_rate=chronological / denominator,
    )


def reset_schedule_hash(segment_lengths: Sequence[int]) -> str:
    payload = json.dumps(list(segment_lengths), separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def feature_cache_manifest(
    config: Stream51Config,
    train_images: Sequence[Stream51Image],
    test_images: Sequence[Stream51Image],
) -> dict[str, object]:
    root = Path(config.data.root)
    return {
        "schema_version": 1,
        "encoder": "resnet18",
        "encoder_weights": config.model.encoder_weights,
        "torchvision_version": torchvision.__version__,
        "feature_size": config.model.feature_size,
        "bbox_crop": config.data.bbox_crop,
        "bbox_padding_ratio": config.data.bbox_padding_ratio,
        "preprocessing": "ResNet18_Weights.IMAGENET1K_V1.transforms",
        "train_metadata_sha256": _sha256(root / config.data.train_metadata),
        "test_metadata_sha256": _sha256(root / config.data.test_metadata),
        "train_observation_ids_sha256": hashlib.sha256(
            json.dumps([image.observation_id for image in train_images]).encode()
        ).hexdigest(),
        "test_observation_ids_sha256": hashlib.sha256(
            json.dumps([image.observation_id for image in test_images]).encode()
        ).hexdigest(),
        "train_observations": len(train_images),
        "test_observations": len(test_images),
    }


class _Stream51FrameDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(
        self,
        root: Path,
        records: Sequence[Stream51Image],
        transform: Callable[[Image.Image], torch.Tensor],
        *,
        bbox_crop: bool,
        bbox_padding_ratio: float,
    ) -> None:
        self.root = root
        self.records = tuple(records)
        self.transform = transform
        self.bbox_crop = bbox_crop
        self.bbox_padding_ratio = bbox_padding_ratio

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self.records[index]
        image_path = self.root / record.file_location
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing Stream-51 image: {image_path}")
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        if self.bbox_crop:
            xmax, xmin, ymax, ymin = record.bbox
            center_x = xmin + (xmax - xmin) / 2
            center_y = ymin + (ymax - ymin) / 2
            width = (xmax - xmin) * self.bbox_padding_ratio
            height = (ymax - ymin) * self.bbox_padding_ratio
            crop = (
                max(0, int(center_x - width / 2)),
                max(0, int(center_y - height / 2)),
                min(image.width, int(center_x + width / 2)),
                min(image.height, int(center_y + height / 2)),
            )
            image = image.crop(crop)
        return self.transform(image), record.class_id


@torch.no_grad()
def _extract_features(
    records: Sequence[Stream51Image],
    config: Stream51DataConfig,
    encoder: nn.Module,
    transform: Callable[[Image.Image], torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    dataset = _Stream51FrameDataset(
        Path(config.root),
        records,
        transform,
        bbox_crop=config.bbox_crop,
        bbox_padding_ratio=config.bbox_padding_ratio,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.feature_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory and device.type == "cuda",
        persistent_workers=config.num_workers > 0,
    )
    encoder.eval()
    feature_batches: list[torch.Tensor] = []
    label_batches: list[torch.Tensor] = []
    for inputs, labels in loader:
        encoded = encoder(inputs.to(device, non_blocking=True)).flatten(1)
        feature_batches.append(encoded.cpu())
        label_batches.append(labels.long())
    return torch.cat(feature_batches), torch.cat(label_batches)


def load_cached_features(
    path: str | Path, expected_manifest: dict[str, object]
) -> FeatureCache | None:
    cache_path = Path(path)
    if not cache_path.is_file():
        return None
    values = torch.load(cache_path, map_location="cpu", weights_only=True)
    if not isinstance(values, dict) or values.get("manifest") != expected_manifest:
        return None
    return FeatureCache(
        train_features=values["train_features"],
        train_labels=values["train_labels"],
        test_features=values["test_features"],
        test_labels=values["test_labels"],
        manifest=values["manifest"],
    )


def load_or_build_feature_cache(
    config: Stream51Config,
    train_images: Sequence[Stream51Image],
    test_images: Sequence[Stream51Image],
    device: torch.device,
) -> FeatureCache:
    manifest = feature_cache_manifest(config, train_images, test_images)
    cache_path = Path(config.data.feature_cache)
    cached = load_cached_features(cache_path, manifest)
    if cached is not None:
        return cached

    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    encoder = nn.Sequential(*list(model.children())[:-1]).to(device)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    transform = weights.transforms()
    print("Building frozen ResNet-18 Stream-51 feature cache...", flush=True)
    train_features, train_labels = _extract_features(
        train_images, config.data, encoder, transform, device
    )
    test_features, test_labels = _extract_features(
        test_images, config.data, encoder, transform, device
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "train_features": train_features,
            "train_labels": train_labels,
            "test_features": test_features,
            "test_labels": test_labels,
            "manifest": manifest,
        },
        cache_path,
    )
    write_json(cache_path.with_suffix(".manifest.json"), manifest)
    return FeatureCache(
        train_features=train_features,
        train_labels=train_labels,
        test_features=test_features,
        test_labels=test_labels,
        manifest=manifest,
    )


def prepare_stream51_data(
    config: Stream51Config, device: torch.device
) -> PreparedStream51Data:
    root = Path(config.data.root)
    train_images = load_train_metadata(root / config.data.train_metadata)
    test_images = load_test_metadata(root / config.data.test_metadata)
    cache = load_or_build_feature_cache(config, train_images, test_images, device)
    if cache.train_features.shape != (len(train_images), config.model.feature_size):
        raise ValueError("Stream-51 training feature cache has an unexpected shape")
    if cache.test_features.shape != (len(test_images), config.model.feature_size):
        raise ValueError("Stream-51 test feature cache has an unexpected shape")

    orderings, segment_lengths = build_stream_orderings(
        train_images,
        seed=config.data.order_seed,
        max_trajectories=config.data.max_trajectories,
    )
    selected_ids = [record.observation_id for record in orderings["natural"]]
    selected_index = torch.tensor(selected_ids, dtype=torch.long)
    train_features = cache.train_features.index_select(0, selected_index)
    train_labels = cache.train_labels.index_select(0, selected_index)

    # Orderings retain original observation IDs. The runner indexes the full
    # cache directly, so keep the complete tensors when no trajectory cap is in
    # use and otherwise remap records to the compact selected cache.
    if config.data.max_trajectories is not None:
        id_to_compact = {observation_id: index for index, observation_id in enumerate(selected_ids)}
        compact_orderings: dict[str, tuple[Stream51Record, ...]] = {}
        for name, records in orderings.items():
            compact_orderings[name] = tuple(
                Stream51Record(
                    **{
                        **asdict(record),
                        "observation_id": id_to_compact[record.observation_id],
                    }
                )
                for record in records
            )
        orderings = compact_orderings
    else:
        train_features = cache.train_features
        train_labels = cache.train_labels

    known_mask = cache.test_labels.ge(0) & cache.test_labels.lt(config.model.num_classes)
    test_features = cache.test_features[known_mask]
    test_labels = cache.test_labels[known_mask]
    diagnostics = {name: ordering_diagnostics(records) for name, records in orderings.items()}
    return PreparedStream51Data(
        train_features=train_features,
        train_labels=train_labels,
        test_features=test_features,
        test_labels=test_labels,
        orderings=orderings,
        diagnostics=diagnostics,
        segment_lengths=segment_lengths,
        reset_schedule_hash=reset_schedule_hash(segment_lengths),
        cache_manifest=cache.manifest,
    )
