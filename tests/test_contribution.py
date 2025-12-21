"""
Tests for the Sensor Contribution Balance KPI.
This KPI tracks how often the fusion result relies on both sensors versus a single sensor.
It helps identify if one sensor is consistently ignored by the fusion logic.
"""

import logging
from typing import Dict, List, Optional, Union

import pytest

from src.data_loader import JSONValue
from src.kpi_calculator import KPICalculator

logger = logging.getLogger(__name__)


@pytest.mark.contribution
@pytest.mark.additional
def test_sensor_contribution_balance(
    loaded_data: Dict[str, Union[List[Optional[JSONValue]], List[int], List[Optional[Dict[str, JSONValue]]]]],
) -> None:
    """Validate the sensor contribution balance.

    The balance shows the distribution of fusion sources: both sensors,
    camera only, or radar only.
    """
    fused_data_raw = loaded_data.get("fused", [])
    if not isinstance(fused_data_raw, list):
        pytest.fail("Invalid data format in loaded_data")

    fused_data: List[Dict[str, JSONValue]] = [frame for frame in fused_data_raw if isinstance(frame, dict)]
    balance = KPICalculator.calculate_sensor_contribution_balance(fused_data)

    logger.info(f"Additional KPI - Sensor Contribution: {balance}")
    assert balance["both"] + balance["camera_only"] + balance["radar_only"] > 0
