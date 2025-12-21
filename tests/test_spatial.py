"""
Tests for the Spatial Alignment Error KPI.
This KPI calculates the distance between the radar point center and the camera bounding box center.
It helps detect spatial calibration issues between sensors.
"""

import logging
from typing import Any, Dict, List

import pytest

from src.kpi_calculator import KPICalculator
from src.models import FusedFrame

logger = logging.getLogger(__name__)


@pytest.mark.spatial
@pytest.mark.extended_kpis
def test_spatial_alignment_error(
    loaded_data: Dict[str, Any],
) -> None:
    """Validate the spatial alignment error.

    Calculates the Euclidean distance between camera bounding box centers
    and radar point centers.
    """
    fused_data_raw = loaded_data.get("fused", [])
    if not isinstance(fused_data_raw, list):
        pytest.fail("Invalid data format in loaded_data")

    fused_data: List[FusedFrame] = [frame for frame in fused_data_raw if isinstance(frame, FusedFrame)]

    # Check spatial alignment for the first available frame
    if fused_data:
        spatial_err = KPICalculator.calculate_spatial_alignment_error(fused_data[0])
        logger.info(f"KPI - Spatial Alignment Error (Frame 0): {spatial_err:.4f}")
        assert spatial_err >= 0, f"Spatial alignment error {spatial_err} cannot be negative"
    else:
        pytest.skip("No fused data available to check spatial alignment")
