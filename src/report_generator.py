"""
Module for generating performance reports and visualizations.
Produces latency plots and aggregate summary statistics in JSON format.
"""

import json
import logging
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt

from src.data_loader import DataLoader
from src.kpi_calculator import KPICalculator
from src.models import CameraFrame, FusedFrame, RadarFrame

logger = logging.getLogger(__name__)


def generate_report(
    radar_data: Optional[List[RadarFrame]] = None,
    camera_data: Optional[List[CameraFrame]] = None,
    fused_data: Optional[List[FusedFrame]] = None,
    test_results: Optional[Dict[str, Any]] = None,
) -> None:
    """Generates visual and text reports from provided or loaded data.

    If data is not provided, it loads it from default paths.
    Calculates aggregate metrics and saves them to:
    - latency_report.png: Plot of sensor and fusion latencies.
    - performance_summary.png: Distribution of fusion consistency and spatial errors.
    - summary_stats.json: JSON file with average latencies and stability scores.
    """
    if radar_data is None:
        radar_raw = DataLoader.load_radar_data("data/radar_data_with_kpis.jsonl")
        radar_data = [f for f in radar_raw if f]
    if camera_data is None:
        camera_raw = DataLoader.load_camera_data("data/camera_data_with_kpis.jsonl")
        camera_data = [f for f in camera_raw if f]
    if fused_data is None:
        fused_raw = DataLoader.load_fused_data("data/fused_data_with_kpis.jsonl")
        fused_data = [f for f in fused_raw if f]

    # 1. Plot Latencies
    plt.figure(figsize=(12, 6))
    plt.plot([f.latency_ms for f in radar_data], label="Radar Latency", alpha=0.7)
    plt.plot([f.latency_ms for f in camera_data], label="Camera Latency", alpha=0.7)
    plt.plot([f.latency_ms for f in fused_data], label="Fusion Latency", linewidth=2)
    plt.axhline(y=50, color="r", linestyle="--", label="Sensor Limit (50ms)")
    plt.axhline(y=100, color="g", linestyle="--", label="Fusion Limit (100ms)")
    plt.xlabel("Frame index")
    plt.ylabel("Latency (ms)")
    plt.title("Sensor and Fusion Latency Performance Over Time")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.savefig("latency_report.png")
    plt.close()
    logger.info("Latency report saved as latency_report.png")

    # 2. Plot Performance Summary (Consistency and Spatial Error)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Consistency scores
    consistency_scores = [KPICalculator.calculate_decision_consistency(f) for f in fused_data]
    ax1.hist(consistency_scores, bins=10, color="skyblue", edgecolor="black")
    ax1.set_title("Fusion Decision Consistency Distribution")
    ax1.set_xlabel("Consistency Score")
    ax1.set_ylabel("Frequency")

    # Spatial alignment errors
    spatial_errors = [KPICalculator.calculate_spatial_alignment_error(f) for f in fused_data]
    ax2.plot(spatial_errors, marker="o", linestyle="-", color="coral", markersize=4)
    ax2.set_title("Spatial Alignment Error Over Time")
    ax2.set_xlabel("Frame index")
    ax2.set_ylabel("Distance Error (units)")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("performance_summary.png")
    plt.close()
    logger.info("Performance summary saved as performance_summary.png")

    # 3. Summary Statistics
    radar_latencies = [f.latency_ms for f in radar_data]
    camera_latencies = [f.latency_ms for f in camera_data]
    fusion_latencies = [f.latency_ms for f in fused_data]

    summary = {
        "radar_avg_latency": sum(radar_latencies) / len(radar_latencies) if radar_latencies else 0.0,
        "camera_avg_latency": sum(camera_latencies) / len(camera_latencies) if camera_latencies else 0.0,
        "fusion_avg_latency": sum(fusion_latencies) / len(fusion_latencies) if fusion_latencies else 0.0,
        "avg_spatial_error": sum(spatial_errors) / len(spatial_errors) if spatial_errors else 0.0,
        "fusion_stability": KPICalculator.calculate_confidence_stability(fused_data),
        "sensor_contribution": KPICalculator.calculate_sensor_contribution_balance(fused_data),
    }

    if test_results:
        summary["test_results"] = test_results

    with open("summary_stats.json", "w") as f:
        json.dump(summary, f, indent=4)
    logger.info("Summary stats saved as summary_stats.json")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    generate_report()
