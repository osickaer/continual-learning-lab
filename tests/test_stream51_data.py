import json
from pathlib import Path

import torch

from continual_learning_lab.stream51_data import (
    Stream51Image,
    build_stream_orderings,
    load_cached_features,
    load_test_metadata,
    load_train_metadata,
    ordering_diagnostics,
    segment_lengths_from_records,
)


def _observations() -> tuple[Stream51Image, ...]:
    records = []
    observation_id = 0
    for class_id, length in enumerate((6, 5, 7)):
        for frame_num in range(length):
            records.append(
                Stream51Image(
                    observation_id=observation_id,
                    class_id=class_id,
                    original_trajectory_id=f"{class_id}:0:0",
                    frame_num=frame_num,
                    bbox=(10.0, 0.0, 10.0, 0.0),
                    file_location=f"frame-{observation_id}.jpg",
                )
            )
            observation_id += 1
    return tuple(records)


def test_orderings_share_observations_resets_positions_and_segment_lengths() -> None:
    orderings, lengths = build_stream_orderings(_observations(), seed=42)
    natural = orderings["natural"]
    expected_ids = sorted(record.observation_id for record in natural)
    expected_resets = tuple(record.reset_before for record in natural)
    expected_reset_positions = tuple(
        index for index, reset in enumerate(expected_resets) if reset
    )

    for records in orderings.values():
        assert sorted(record.observation_id for record in records) == expected_ids
        assert len(records) == len(natural)
        assert sum(record.reset_before for record in records) == len(lengths)
        assert tuple(record.reset_before for record in records) == expected_resets
        assert tuple(
            index for index, record in enumerate(records) if record.reset_before
        ) == expected_reset_positions
        assert segment_lengths_from_records(records) == lengths


def test_global_shuffle_uses_pseudo_segments_not_original_trajectory_boundaries() -> None:
    orderings, lengths = build_stream_orderings(_observations(), seed=42)

    for name in ("natural", "local_shuffle"):
        by_segment: dict[int, set[str]] = {}
        for record in orderings[name]:
            by_segment.setdefault(record.reset_segment_id, set()).add(
                record.original_trajectory_id
            )
        assert all(len(trajectories) == 1 for trajectories in by_segment.values())

    global_by_segment: dict[int, set[str]] = {}
    for record in orderings["global_shuffle"]:
        global_by_segment.setdefault(record.reset_segment_id, set()).add(
            record.original_trajectory_id
        )
    assert any(len(trajectories) > 1 for trajectories in global_by_segment.values())
    assert tuple(len(global_by_segment[index]) > 0 for index in range(len(lengths)))


def test_natural_is_chronological_and_structure_diagnostics_separate_orders() -> None:
    orderings, _lengths = build_stream_orderings(_observations(), seed=42)
    natural = orderings["natural"]
    for previous, current in zip(natural, natural[1:]):
        if previous.reset_segment_id == current.reset_segment_id:
            assert current.frame_num == previous.frame_num + 1

    diagnostics = {
        name: ordering_diagnostics(records) for name, records in orderings.items()
    }
    assert diagnostics["natural"].chronological_successor_rate > diagnostics[
        "local_shuffle"
    ].chronological_successor_rate
    assert diagnostics["local_shuffle"].adjacent_same_trajectory_rate > diagnostics[
        "global_shuffle"
    ].adjacent_same_trajectory_rate


def test_metadata_parsers_preserve_official_fields(tmp_path: Path) -> None:
    train_path = tmp_path / "train.json"
    test_path = tmp_path / "test.json"
    train_path.write_text(
        json.dumps([[3, 4, 5, 6, [20, 20], [10, 0, 10, 0], "train.jpg"]]),
        encoding="utf-8",
    )
    test_path.write_text(
        json.dumps([[3, [20, 20], [10, 0, 10, 0], "test.jpg"]]),
        encoding="utf-8",
    )

    train = load_train_metadata(train_path)
    test = load_test_metadata(test_path)

    assert train[0].original_trajectory_id == "3:4:5"
    assert train[0].frame_num == 6
    assert test[0].original_trajectory_id is None


def test_feature_cache_reuse_requires_an_exact_manifest(tmp_path: Path) -> None:
    path = tmp_path / "features.pt"
    manifest = {"schema_version": 1, "encoder": "test"}
    torch.save(
        {
            "train_features": torch.randn(3, 4),
            "train_labels": torch.tensor([0, 1, 2]),
            "test_features": torch.randn(2, 4),
            "test_labels": torch.tensor([0, 1]),
            "manifest": manifest,
        },
        path,
    )

    assert load_cached_features(path, manifest) is not None
    assert load_cached_features(path, {**manifest, "encoder": "changed"}) is None
