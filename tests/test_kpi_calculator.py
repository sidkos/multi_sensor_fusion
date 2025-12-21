"""
Tests for the KPICalculator class.
Validates the logic for calculating various performance metrics.
"""

import logging
from typing import List, Optional

import pytest

from src.kpi_calculator import KPICalculator, SensorData

logger = logging.getLogger(__name__)


@pytest.mark.infrastructure
@pytest.mark.parametrize(
    "data, expected, expected_rate",
    [
        ([], [], 0.0),
        ([], [1000, 1100, 1200], 1.0),
        ([{"timestamp": 1000}], [1000, 1100, 1200], 2 / 3),
        ([None], [1000], 2.0),
    ],
    ids=["empty", "all_missing", "partial_missing", "malformed_only"],
)
def test_calculate_data_drop_rate(data: List[Optional[SensorData]], expected: List[int], expected_rate: float) -> None:
    """Test data drop rate calculation with various scenarios."""
    drop_rate = KPICalculator.calculate_data_drop_rate(data, expected)
    assert drop_rate == pytest.approx(expected_rate), f"Expected {expected_rate} drop rate, got {drop_rate}"


@pytest.mark.infrastructure
@pytest.mark.parametrize(
    "frame, expected_consistency",
    [
        ({}, 1.0),
        ({"fused_objects": "invalid"}, 1.0),
        ({"fused_objects": []}, 1.0),
        (
            {
                "fused_objects": [
                    {
                        "class": "car",
                        "source_classes": {"camera": "car", "radar": "car"},
                    }
                ]
            },
            1.0,
        ),
        (
            {
                "fused_objects": [
                    {
                        "class": "car",
                        "source_classes": {"camera": "truck", "radar": "pedestrian"},
                    }
                ]
            },
            0.0,
        ),
    ],
    ids=["empty", "invalid_type", "no_objects", "perfect_match", "mismatch"],
)
def test_calculate_decision_consistency(frame: SensorData, expected_consistency: float) -> None:
    """Test decision consistency calculation."""
    consistency = KPICalculator.calculate_decision_consistency(frame)
    assert consistency == pytest.approx(
        expected_consistency
    ), f"Expected {expected_consistency} consistency, got {consistency}"


@pytest.mark.infrastructure
@pytest.mark.parametrize(
    "frame, expected_error",
    [
        ({"fused_objects": [{"class": "car"}]}, 0.0),
        (
            {
                "fused_objects": [
                    {
                        "camera_bbox_3d": [{"x": 0, "y": 0}, {"x": 2, "y": 2}],
                        "radar_points": [{"x": 1, "y": 1}],
                    }
                ]
            },
            0.0,
        ),
        (
            {
                "fused_objects": [
                    {
                        "camera_bbox_3d": [{"x": 10, "y": 10}],
                        "radar_points": [{"x": 13, "y": 14}],
                    }
                ]
            },
            5.0,
        ),
    ],
    ids=["no_geometry", "perfect_alignment", "offset_alignment"],
)
def test_calculate_spatial_alignment_error(frame: SensorData, expected_error: float) -> None:
    """Test spatial alignment error calculation."""
    error = KPICalculator.calculate_spatial_alignment_error(frame)
    assert error == pytest.approx(expected_error), f"Expected {expected_error} spatial error, got {error}"


@pytest.mark.infrastructure
@pytest.mark.parametrize(
    "frames, expected_stability",
    [
        ([], 0.0),
        ([{"fused_objects": []}], 0.0),
        ([{"fused_objects": [{"fused_confidence": 0.9}, {"fused_confidence": 0.9}]}], 0.0),
        ([{"fused_objects": [{"fused_confidence": 0.8}]}, {"fused_objects": [{"fused_confidence": 1.0}]}], 0.1),
    ],
    ids=["empty", "no_objects", "stable", "unstable"],
)
def test_calculate_confidence_stability(frames: List[SensorData], expected_stability: float) -> None:
    """Test confidence stability calculation (standard deviation)."""
    stability = KPICalculator.calculate_confidence_stability(frames)
    assert stability == pytest.approx(expected_stability), f"Expected {expected_stability} stability, got {stability}"
