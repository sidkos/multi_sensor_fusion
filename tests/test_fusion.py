"""
Tests for Fusion-specific KPIs.
Validates fusion latency, jitter, and decision consistency.
"""

import logging
from typing import Any, Dict, List, Optional

import pytest

from src.data_loader import AlignedFrame, DataLoader
from src.kpi_calculator import KPICalculator
from src.models import CameraFrame, FusedFrame, RadarFrame

logger = logging.getLogger(__name__)

FUSION_LATENCY_MAX = 100
FUSION_JITTER_MAX = 5
FUSION_CONSISTENCY_MIN = 0.95


@pytest.mark.fusion
@pytest.mark.parametrize(
    "kpi_name, threshold",
    [
        ("latency", FUSION_LATENCY_MAX),
        ("jitter", FUSION_JITTER_MAX),
        ("consistency", FUSION_CONSISTENCY_MIN),
    ],
    ids=["fusion_latency", "fusion_jitter", "fusion_consistency"],
)
def test_fusion_per_frame_kpis(
    loaded_data: Dict[str, Any],
    kpi_name: str,
    threshold: float,
) -> None:
    """Test fusion performance metrics per frame."""
    fused_data_raw = loaded_data.get("fused", [])
    if not isinstance(fused_data_raw, list):
        pytest.fail("Invalid data format in loaded_data")

    fused_data: List[FusedFrame] = [frame for frame in fused_data_raw if isinstance(frame, FusedFrame)]

    failures: List[str] = []
    values: List[float] = []
    for frame in fused_data:
        if kpi_name == "latency":
            val = frame.latency_ms
        elif kpi_name == "jitter":
            val = frame.data_alignment_jitter_ms
        else:
            val = KPICalculator.calculate_decision_consistency(frame)

        values.append(float(val))
        if kpi_name == "consistency":
            if val < threshold:
                failures.append(f"Frame {frame.timestamp}: {val} below threshold {threshold}")
        else:
            if val > threshold:
                failures.append(f"Frame {frame.timestamp}: {val} exceeded threshold {threshold}")

    if values:
        avg_val = sum(values) / len(values)
        max_val = max(values)
        min_val = min(values)
        if kpi_name == "consistency":
            logger.info(f"Fusion {kpi_name} - Avg: {avg_val:.4f}, Min: {min_val:.4f} (Threshold: {threshold})")
        else:
            logger.info(f"Fusion {kpi_name} - Avg: {avg_val:.4f}, Max: {max_val:.4f} (Threshold: {threshold})")

    assert not failures, f"Fusion {kpi_name} violations found in {len(failures)} frames:\n" + "\n".join(failures)


@pytest.mark.fusion
def test_fusion_data_presence(
    loaded_data: Dict[str, Any],
) -> None:
    """Checks that fused frames contains data from both sensors for the same timestamp.

    This test verifies the structural integrity of the fusion output by ensuring that:
    1. If an object is marked as having a camera source, 'camera_bbox_3d' is not empty.
    2. If an object is marked as having a radar source, 'radar_points' is not empty.
    3. The data presence in the FusedObject matches the declared 'source_classes'.

    Args:
        loaded_data (Dict[str, Any]): Session-scoped fixture containing fused data.
    """
    fused_data_raw = loaded_data.get("fused", [])
    if not isinstance(fused_data_raw, list):
        pytest.fail("Invalid data format in loaded_data")

    fused_frames: List[FusedFrame] = [frame for frame in fused_data_raw if isinstance(frame, FusedFrame)]

    failures = []
    for frame in fused_frames:
        for obj in frame.fused_objects:
            has_camera_src = "camera" in obj.source_classes
            has_radar_src = "radar" in obj.source_classes

            # 1. Check Camera Data Presence
            if has_camera_src and not obj.camera_bbox_3d:
                failures.append(
                    f"Frame {frame.timestamp}: Object {obj.object_class} has 'camera' "
                    "source but empty 'camera_bbox_3d'."
                )

            # 2. Check Radar Data Presence
            if has_radar_src and not obj.radar_points:
                failures.append(
                    f"Frame {frame.timestamp}: Object {obj.object_class} has 'radar' "
                    "source but empty 'radar_points'."
                )

            # 3. Check for undeclared data
            if obj.camera_bbox_3d and not has_camera_src:
                failures.append(
                    f"Frame {frame.timestamp}: Object {obj.object_class} has "
                    "'camera_bbox_3d' but 'camera' is missing from 'source_classes'."
                )
            if obj.radar_points and not has_radar_src:
                failures.append(
                    f"Frame {frame.timestamp}: Object {obj.object_class} has "
                    "'radar_points' but 'radar' is missing from 'source_classes'."
                )

    assert not failures, f"Fusion data presence violations found in {len(failures)} cases:\n" + "\n".join(failures)


@pytest.mark.fusion
def test_fusion_value_correctness(
    loaded_data: Dict[str, Any],
) -> None:
    """Checks the actual correctness of values in a fused frame.

    This test verifies that for a given timestamp X:
    1. All objects from radar and camera at timestamp X are correctly represented in
       the fused frame at timestamp X+100ms.
    2. The values (coordinates, classes) match exactly and are not mixed or confused.

    Args:
        loaded_data (Dict[str, Any]): Session-scoped fixture containing sensor and fused data.
    """
    radar_data = loaded_data.get("radar", [])
    camera_data = loaded_data.get("camera", [])
    fused_data = loaded_data.get("fused", [])

    radar_frames: List[Optional[RadarFrame]] = [f for f in radar_data if isinstance(f, RadarFrame)]
    camera_frames: List[Optional[CameraFrame]] = [f for f in camera_data if isinstance(f, CameraFrame)]
    fused_frames: List[Optional[FusedFrame]] = [f for f in fused_data if isinstance(f, FusedFrame)]

    aligned_frames: List[AlignedFrame] = DataLoader.align_data(radar_frames, camera_frames, fused_frames)

    failures = []
    for aligned in aligned_frames:
        f_frame = aligned["fused"]
        r_frame = aligned["radar"]
        c_frame = aligned["camera"]

        if not isinstance(f_frame, FusedFrame):
            continue

        # Use the KPICalculator to validate value correctness
        # Note: Mypy might complain about types from aligned map, so we ensure they are correct
        radar_input = r_frame if isinstance(r_frame, RadarFrame) else None
        camera_input = c_frame if isinstance(c_frame, CameraFrame) else None

        frame_failures = KPICalculator.validate_fusion_value_correctness(f_frame, radar_input, camera_input)
        failures.extend(frame_failures)

    assert not failures, f"Fusion value correctness violations found in {len(failures)} cases:\n" + "\n".join(failures)
