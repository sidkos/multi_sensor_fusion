"""
Tests for Camera-specific KPIs.
Validates latency, error rate, and data drop rate for the camera sensor.
"""

import logging
from typing import Any, Dict, List, Optional

import pytest

from src.kpi_calculator import KPICalculator
from src.models import BaseFrame, CameraFrame

logger = logging.getLogger(__name__)

CAMERA_LATENCY_MAX = 50
CAMERA_ERROR_RATE_MAX = 0.001
CAMERA_DATA_DROP_MAX = 0.005


@pytest.mark.camera
@pytest.mark.parametrize(
    "kpi_name, threshold",
    [
        ("latency", CAMERA_LATENCY_MAX),
        ("error_rate", CAMERA_ERROR_RATE_MAX),
    ],
    ids=["camera_latency", "camera_error_rate"],
)
def test_camera_per_frame_kpis(
    loaded_data: Dict[str, Any],
    kpi_name: str,
    threshold: float,
) -> None:
    """Test camera performance metrics per frame."""
    camera_data_raw = loaded_data.get("camera", [])
    if not isinstance(camera_data_raw, list):
        pytest.fail("Invalid data format in loaded_data")

    camera_data: List[CameraFrame] = [frame for frame in camera_data_raw if isinstance(frame, CameraFrame)]

    failures: List[str] = []
    values: List[float] = []
    for frame in camera_data:
        if kpi_name == "latency":
            val = frame.latency_ms
        else:
            val = frame.error_rate_percent if frame.error_rate_percent is not None else 0.0

        values.append(float(val))
        if val > threshold:
            failures.append(f"Frame {frame.timestamp}: {val} exceeded threshold {threshold}")

    if values:
        avg_val = sum(values) / len(values)
        max_val = max(values)
        logger.info(f"Camera {kpi_name} - Avg: {avg_val:.4f}, Max: {max_val:.4f} (Threshold: {threshold})")

    assert not failures, f"Camera {kpi_name} violations found in {len(failures)} frames:\n" + "\n".join(failures)


@pytest.mark.camera
def test_camera_drop_rate(
    loaded_data: Dict[str, Any],
) -> None:
    """Test camera data drop rate aggregate."""
    camera_data_raw = loaded_data.get("camera", [])
    expected_ts_raw = loaded_data.get("expected_radar_ts", [])

    if not isinstance(camera_data_raw, list) or not isinstance(expected_ts_raw, list):
        pytest.fail("Invalid data format in loaded_data")

    camera_data: List[Optional[BaseFrame]] = [
        frame for frame in camera_data_raw if frame is None or isinstance(frame, CameraFrame)
    ]
    expected_ts: List[int] = [ts for ts in expected_ts_raw if isinstance(ts, int)]

    drop_rate = KPICalculator.calculate_data_drop_rate(camera_data, expected_ts)
    logger.info(f"Camera Drop Rate: {drop_rate:.4f} (Threshold: {CAMERA_DATA_DROP_MAX})")
    assert drop_rate < CAMERA_DATA_DROP_MAX, f"Camera drop rate {drop_rate} exceeded threshold {CAMERA_DATA_DROP_MAX}"
