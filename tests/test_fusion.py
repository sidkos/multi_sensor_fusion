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

    latencies: List[float] = []
    jitters: List[float] = []
    consistencies: List[float] = []
    for frame in fused_data:
        latency = frame.get("fusion_latency_ms")
        jitter = frame.get("data_alignment_jitter_ms")
        if isinstance(latency, (int, float)) and isinstance(jitter, (int, float)):
            latencies.append(float(latency))
            jitters.append(float(jitter))
            assert latency < FUSION_LATENCY_MAX, f"Fusion latency {latency} exceeded threshold {FUSION_LATENCY_MAX}"
            assert jitter <= FUSION_JITTER_MAX, f"Fusion jitter {jitter} exceeded threshold {FUSION_JITTER_MAX}"

            consistency = KPICalculator.calculate_decision_consistency(frame)
            consistencies.append(consistency)

    if latencies:
        avg_lat = sum(latencies) / len(latencies)
        max_lat = max(latencies)
        logger.info(f"Fusion Latency - Avg: {avg_lat:.4f}, Max: {max_lat:.4f} (Threshold: {FUSION_LATENCY_MAX})")

    if jitters:
        avg_jitter = sum(jitters) / len(jitters)
        max_jitter = max(jitters)
        logger.info(f"Fusion Jitter - Avg: {avg_jitter:.4f}, Max: {max_jitter:.4f} (Threshold: {FUSION_JITTER_MAX})")

    avg_consistency = sum(consistencies) / len(consistencies) if consistencies else 0
    logger.info(f"Fusion Consistency - Avg: {avg_consistency:.4f} (Threshold: {FUSION_CONSISTENCY_MIN})")
    assert (
        avg_consistency >= FUSION_CONSISTENCY_MIN
    ), f"Average fusion consistency {avg_consistency} below threshold {FUSION_CONSISTENCY_MIN}"
