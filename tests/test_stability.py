"""
Tests for the Confidence Stability Score KPI.
This KPI measures the variance of fused_confidence across frames for the same object class.
High variance might indicate an unstable fusion algorithm.
"""

import logging
from typing import Any, Dict, List

import pytest

from src.kpi_calculator import KPICalculator
from src.models import FusedFrame

logger = logging.getLogger(__name__)


@pytest.mark.stability
@pytest.mark.extended_kpis
def test_confidence_stability(
    loaded_data: Dict[str, Any],
) -> None:
    """Validate the confidence stability score.

    The score is the standard deviation of fusion confidence values.
    """
    fused_data_raw = loaded_data.get("fused", [])
    if not isinstance(fused_data_raw, list):
        pytest.fail("Invalid data format in loaded_data")

    fused_data: List[FusedFrame] = [frame for frame in fused_data_raw if isinstance(frame, FusedFrame)]
    stability = KPICalculator.calculate_confidence_stability(fused_data)

    logger.info(f"KPI - Confidence Stability (StdDev): {stability:.4f}")
    assert stability >= 0, f"Confidence stability (StdDev) {stability} cannot be negative"
