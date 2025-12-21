"""
Tests for the KPICalculator class.
Validates the logic for calculating various performance metrics.
"""

import logging
from typing import List, Optional

import pytest

from src.kpi_calculator import KPICalculator
from src.models import BaseFrame, FusedFrame, FusedObject

logger = logging.getLogger(__name__)


@pytest.mark.infrastructure
@pytest.mark.parametrize(
    "data_timestamps, malformed_indices, expected, expected_rate",
    [
        ([], [], [], 0.0),
        ([], [], [1000, 1100, 1200], 1.0),
        ([1000], [], [1000, 1100, 1200], 2 / 3),
        ([], [0], [1000], 1.0),
    ],
    ids=["empty", "all_missing", "partial_missing", "malformed_only"],
)
def test_calculate_data_drop_rate(
    data_timestamps: List[int], malformed_indices: List[int], expected: List[int], expected_rate: float
) -> None:
    """Test data drop rate calculation with various scenarios."""
    data: List[Optional[BaseFrame]] = []

    # Fill with valid frames
    for ts in data_timestamps:
        data.append(BaseFrame(timestamp=ts, latency_ms=10.0))

    # Inject malformed frames (None) at specific indices if needed
    # Note: the current logic of calculate_data_drop_rate counts Nones in 'data'
    # and compares against 'expected_timestamps' length.
    for _ in malformed_indices:
        data.append(None)

    drop_rate = KPICalculator.calculate_data_drop_rate(data, expected)
    assert drop_rate == pytest.approx(expected_rate), f"Expected {expected_rate} drop rate, got {drop_rate}"


@pytest.mark.infrastructure
@pytest.mark.parametrize(
    "fused_objects, expected_consistency",
    [
        ([], 1.0),
        (
            [FusedObject(object_class="car", source_classes={"camera": "car", "radar": "car"})],
            1.0,
        ),
        (
            [FusedObject(object_class="car", source_classes={"camera": "truck", "radar": "pedestrian"})],
            0.0,
        ),
    ],
    ids=["no_objects", "perfect_match", "mismatch"],
)
def test_calculate_decision_consistency(fused_objects: List[FusedObject], expected_consistency: float) -> None:
    """Test decision consistency calculation."""
    frame = FusedFrame(timestamp=1000, latency_ms=50.0, fused_objects=fused_objects)
    consistency = KPICalculator.calculate_decision_consistency(frame)
    assert consistency == pytest.approx(
        expected_consistency
    ), f"Expected {expected_consistency} consistency, got {consistency}"


@pytest.mark.infrastructure
@pytest.mark.parametrize(
    "fused_objects, expected_error",
    [
        ([FusedObject(object_class="car", source_classes={})], 0.0),
        (
            [
                FusedObject(
                    object_class="car",
                    source_classes={},
                    camera_bbox_3d=[{"x": 0, "y": 0}, {"x": 2, "y": 2}],
                    radar_points=[{"x": 1, "y": 1}],
                )
            ],
            0.0,
        ),
        (
            [
                FusedObject(
                    object_class="car",
                    source_classes={},
                    camera_bbox_3d=[{"x": 10, "y": 10}],
                    radar_points=[{"x": 13, "y": 14}],
                )
            ],
            5.0,
        ),
    ],
    ids=["no_geometry", "perfect_alignment", "offset_alignment"],
)
def test_calculate_spatial_alignment_error(fused_objects: List[FusedObject], expected_error: float) -> None:
    """Test spatial alignment error calculation."""
    frame = FusedFrame(timestamp=1000, latency_ms=50.0, fused_objects=fused_objects)
    error = KPICalculator.calculate_spatial_alignment_error(frame)
    assert error == pytest.approx(expected_error), f"Expected {expected_error} spatial error, got {error}"


@pytest.mark.infrastructure
@pytest.mark.parametrize(
    "confidences, expected_stability",
    [
        ([], 0.0),
        ([0.9, 0.9], 0.0),
        ([0.8, 1.0], 0.1),
    ],
    ids=["empty", "stable", "unstable"],
)
def test_calculate_confidence_stability(confidences: List[float], expected_stability: float) -> None:
    """Test confidence stability calculation (standard deviation)."""
    frames = []
    for conf in confidences:
        frames.append(
            FusedFrame(
                timestamp=1000,
                latency_ms=50.0,
                fused_objects=[FusedObject(object_class="car", source_classes={}, fused_confidence=conf)],
            )
        )

    stability = KPICalculator.calculate_confidence_stability(frames)
    assert stability == pytest.approx(expected_stability), f"Expected {expected_stability} stability, got {stability}"
