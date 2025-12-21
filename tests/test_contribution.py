"""
Tests for the Sensor Contribution Balance KPI.
This KPI tracks how often the fusion result relies on both sensors versus a single sensor.
It helps identify if one sensor is consistently ignored by the fusion logic.
"""

import logging
from typing import Any, Dict, List

import pytest

from src.kpi_calculator import KPICalculator
from src.models import FusedFrame

logger = logging.getLogger(__name__)


@pytest.mark.contribution
@pytest.mark.additional
def test_sensor_contribution_balance(
    loaded_data: Dict[str, Any],
) -> None:
    """Validate the sensor contribution balance.

    The balance shows the distribution of fusion sources: both sensors,
    camera only, or radar only.
    """
    fused_data_raw = loaded_data.get("fused", [])
    if not isinstance(fused_data_raw, list):
        pytest.fail("Invalid data format in loaded_data")

    fused_data: List[FusedFrame] = [frame for frame in fused_data_raw if isinstance(frame, FusedFrame)]
    balance = KPICalculator.calculate_sensor_contribution_balance(fused_data)

    logger.info(f"Additional KPI - Sensor Contribution: {balance}")
    assert (
        balance["both"] + balance["camera_only"] + balance["radar_only"] > 0
    ), "Total sensor contribution balance must be greater than zero"
