"""
Tests for Camera-specific KPIs.
Validates latency, error rate, and data drop rate for the camera sensor.
"""

import logging
from typing import Dict, List, Optional, Union

import pytest

from src.data_loader import JSONValue
from src.kpi_calculator import KPICalculator

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
    loaded_data: Dict[str, Union[List[Optional[JSONValue]], List[int], List[Optional[Dict[str, JSONValue]]]]],
    kpi_name: str,
    threshold: float,
) -> None:
    """Test camera performance metrics per frame."""
    camera_data_raw = loaded_data.get("camera", [])
    if not isinstance(camera_data_raw, list):
        pytest.fail("Invalid data format in loaded_data")

    camera_data: List[Dict[str, JSONValue]] = [frame for frame in camera_data_raw if isinstance(frame, dict)]

    values: List[float] = []
    for frame in camera_data:
        if kpi_name == "latency":
            val = frame.get("latency_ms")
        else:
            val = frame.get("error_rate_percent")

        if isinstance(val, (int, float)):
            values.append(float(val))
            assert val < threshold

    if values:
        avg_val = sum(values) / len(values)
        max_val = max(values)
        logger.info(f"Camera {kpi_name} - Avg: {avg_val:.4f}, Max: {max_val:.4f} (Threshold: {threshold})")


@pytest.mark.camera
def test_camera_drop_rate(
    loaded_data: Dict[str, Union[List[Optional[JSONValue]], List[int], List[Optional[Dict[str, JSONValue]]]]],
) -> None:
    """Test camera data drop rate aggregate."""
    camera_data_raw = loaded_data.get("camera", [])
    expected_ts_raw = loaded_data.get("expected_radar_ts", [])

    if not isinstance(camera_data_raw, list) or not isinstance(expected_ts_raw, list):
        pytest.fail("Invalid data format in loaded_data")

    camera_data: List[Optional[Dict[str, JSONValue]]] = [
        frame for frame in camera_data_raw if frame is None or isinstance(frame, dict)
    ]
    expected_ts: List[int] = [ts for ts in expected_ts_raw if isinstance(ts, int)]

    drop_rate = KPICalculator.calculate_data_drop_rate(camera_data, expected_ts)
    logger.info(f"Camera Drop Rate: {drop_rate:.4f} (Threshold: {CAMERA_DATA_DROP_MAX})")
    assert drop_rate < CAMERA_DATA_DROP_MAX
