"""
Tests for Fusion-specific KPIs.
Validates fusion latency, jitter, and decision consistency.
"""

import logging
from typing import Dict, List, Optional, Union

import pytest

from src.data_loader import JSONValue
from src.kpi_calculator import KPICalculator

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
    loaded_data: Dict[str, Union[List[Optional[JSONValue]], List[int], List[Optional[Dict[str, JSONValue]]]]],
    kpi_name: str,
    threshold: float,
) -> None:
    """Test fusion performance metrics per frame."""
    fused_data_raw = loaded_data.get("fused", [])
    if not isinstance(fused_data_raw, list):
        pytest.fail("Invalid data format in loaded_data")

    fused_data: List[Dict[str, JSONValue]] = [frame for frame in fused_data_raw if isinstance(frame, dict)]

    values: List[float] = []
    for frame in fused_data:
        if kpi_name == "latency":
            val = frame.get("fusion_latency_ms")
        elif kpi_name == "jitter":
            val = frame.get("data_alignment_jitter_ms")
        else:  # consistency
            val = KPICalculator.calculate_decision_consistency(frame)

        if isinstance(val, (int, float)):
            values.append(float(val))
            if kpi_name == "consistency":
                assert val >= threshold, f"Fusion {kpi_name} {val} below threshold {threshold}"
            else:
                assert val <= threshold, f"Fusion {kpi_name} {val} exceeded threshold {threshold}"

    if values:
        avg_val = sum(values) / len(values)
        max_val = max(values)
        min_val = min(values)
        if kpi_name == "consistency":
            logger.info(f"Fusion {kpi_name} - Avg: {avg_val:.4f}, Min: {min_val:.4f} (Threshold: {threshold})")
        else:
            logger.info(f"Fusion {kpi_name} - Avg: {avg_val:.4f}, Max: {max_val:.4f} (Threshold: {threshold})")
