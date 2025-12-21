"""
Tests for Fusion-specific KPIs.
Validates fusion latency, jitter, and decision consistency.
"""

from typing import Dict, List, Optional, Union

import pytest

from src.data_loader import JSONValue
from src.kpi_calculator import KPICalculator

FUSION_LATENCY_MAX = 100
FUSION_JITTER_MAX = 5
FUSION_CONSISTENCY_MIN = 0.95


@pytest.mark.fusion
def test_fusion_kpis(
    loaded_data: Dict[str, Union[List[Optional[JSONValue]], List[int], List[Optional[Dict[str, JSONValue]]]]],
) -> None:
    """Test fusion performance metrics.

    Validates that:
    - Fusion latency is < 100ms.
    - Data alignment jitter is <= 5ms.
    - Decision consistency score is >= 0.95.
    """
    fused_data_raw = loaded_data.get("fused", [])
    if not isinstance(fused_data_raw, list):
        pytest.fail("Invalid data format in loaded_data")

    fused_data: List[Dict[str, JSONValue]] = [frame for frame in fused_data_raw if isinstance(frame, dict)]

    consistencies = []
    for frame in fused_data:
        latency = frame.get("fusion_latency_ms")
        jitter = frame.get("data_alignment_jitter_ms")
        if isinstance(latency, (int, float)) and isinstance(jitter, (int, float)):
            assert latency < FUSION_LATENCY_MAX
            assert jitter <= FUSION_JITTER_MAX

            consistency = KPICalculator.calculate_decision_consistency(frame)
            consistencies.append(consistency)

    avg_consistency = sum(consistencies) / len(consistencies) if consistencies else 0
    assert avg_consistency >= FUSION_CONSISTENCY_MIN
