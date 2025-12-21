"""
Tests for Radar-specific KPIs.
Validates latency, error rate, and data drop rate for the radar sensor.
"""

import logging
from typing import Any, Dict, List, Optional

import pytest

from src.kpi_calculator import KPICalculator
from src.models import BaseFrame, RadarFrame

logger = logging.getLogger(__name__)

RADAR_LATENCY_MAX = 50
RADAR_ERROR_RATE_MAX = 0.001  # 0.1%
RADAR_DATA_DROP_MAX = 0.005  # 0.5%


@pytest.mark.radar
@pytest.mark.parametrize(
    "kpi_name, threshold",
    [
        ("latency", RADAR_LATENCY_MAX),
        ("error_rate", RADAR_ERROR_RATE_MAX),
    ],
    ids=["radar_latency", "radar_error_rate"],
)
def test_radar_per_frame_kpis(
    loaded_data: Dict[str, Any],
    kpi_name: str,
    threshold: float,
) -> None:
    """Test radar performance metrics per frame."""
    radar_data_raw = loaded_data.get("radar", [])
    if not isinstance(radar_data_raw, list):
        pytest.fail("Invalid data format in loaded_data")

    radar_data: List[RadarFrame] = [frame for frame in radar_data_raw if isinstance(frame, RadarFrame)]

    failures: List[str] = []
    values: List[float] = []
    for frame in radar_data:
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
        logger.info(f"Radar {kpi_name} - Avg: {avg_val:.4f}, Max: {max_val:.4f} (Threshold: {threshold})")

    assert not failures, f"Radar {kpi_name} violations found in {len(failures)} frames:\n" + "\n".join(failures)


@pytest.mark.radar
def test_radar_drop_rate(
    loaded_data: Dict[str, Any],
) -> None:
    """Test radar data drop rate aggregate."""
    radar_data_raw = loaded_data.get("radar", [])
    expected_ts_raw = loaded_data.get("expected_radar_ts", [])

    if not isinstance(radar_data_raw, list) or not isinstance(expected_ts_raw, list):
        pytest.fail("Invalid data format in loaded_data")

    radar_data: List[Optional[BaseFrame]] = [
        frame for frame in radar_data_raw if frame is None or isinstance(frame, RadarFrame)
    ]
    expected_ts: List[int] = [ts for ts in expected_ts_raw if isinstance(ts, int)]

    # Aggregate drop rate
    drop_rate = KPICalculator.calculate_data_drop_rate(radar_data, expected_ts)
    logger.info(f"Radar Drop Rate: {drop_rate:.4f} (Threshold: {RADAR_DATA_DROP_MAX})")
    assert drop_rate < RADAR_DATA_DROP_MAX, f"Radar drop rate {drop_rate} exceeded threshold {RADAR_DATA_DROP_MAX}"
