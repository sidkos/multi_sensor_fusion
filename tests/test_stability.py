"""
Tests for the Confidence Stability Score KPI.
This KPI measures the variance of fused_confidence across frames for the same object class.
High variance might indicate an unstable fusion algorithm.
"""

import logging
from typing import Dict, List, Optional, Union

import pytest

from src.data_loader import JSONValue
from src.kpi_calculator import KPICalculator

logger = logging.getLogger(__name__)


@pytest.mark.stability
@pytest.mark.additional
def test_confidence_stability(
    loaded_data: Dict[str, Union[List[Optional[JSONValue]], List[int], List[Optional[Dict[str, JSONValue]]]]],
) -> None:
    """Validate the confidence stability score.

    The score is the standard deviation of fusion confidence values.
    """
    fused_data_raw = loaded_data.get("fused", [])
    if not isinstance(fused_data_raw, list):
        pytest.fail("Invalid data format in loaded_data")

    fused_data: List[Dict[str, JSONValue]] = [frame for frame in fused_data_raw if isinstance(frame, dict)]
    stability = KPICalculator.calculate_confidence_stability(fused_data)

    logger.info(f"Additional KPI - Confidence Stability (StdDev): {stability:.4f}")
    assert stability >= 0, f"Confidence stability (StdDev) {stability} cannot be negative"
