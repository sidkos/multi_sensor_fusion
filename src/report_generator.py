"""
Module for generating performance reports and visualizations.
Produces latency plots and aggregate summary statistics in JSON format.
"""

import json
import logging
import os
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
    output_dir: str = "reports",
) -> None:
    """Generates visual and text reports from provided or loaded data.

    This function produces a set of Matplotlib plots and a JSON summary file
    representing the system's performance. If data lists are not provided,
    it defaults to loading them from the standard dataset paths.

    Args:
        radar_data (Optional[List[RadarFrame]]): List of radar frames. Defaults to None.
        camera_data (Optional[List[CameraFrame]]): List of camera frames. Defaults to None.
        fused_data (Optional[List[FusedFrame]]): List of fused frames. Defaults to None.
        test_results (Optional[Dict[str, Any]]): Statistics from pytest execution. Defaults to None.
        output_dir (str): Directory where artifacts will be saved. Defaults to "reports".

    Returns:
        None
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if radar_data is None:
        radar_raw = DataLoader.load_radar_data("data/radar_data_with_kpis.jsonl")
        radar_data = [f for f in radar_raw if f]
    if camera_data is None:
        camera_raw = DataLoader.load_camera_data("data/camera_data_with_kpis.jsonl")
        camera_data = [f for f in camera_raw if f]
    if fused_data is None:
        fused_raw = DataLoader.load_fused_data("data/fused_data_with_kpis.jsonl")
        fused_data = [f for f in fused_raw if f]

    # 1. Radar Latency
    plt.figure(figsize=(10, 5))
    plt.plot([f.latency_ms for f in radar_data], label="Radar Latency", color="blue")
    plt.axhline(y=50, color="r", linestyle="--", label="Limit (50ms)")
    plt.xlabel("Frame index")
    plt.ylabel("Latency (ms)")
    plt.title("Radar Latency Performance")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.savefig(os.path.join(output_dir, "radar_latency.png"))
    plt.close()

    # 2. Camera Latency
    plt.figure(figsize=(10, 5))
    plt.plot([f.latency_ms for f in camera_data], label="Camera Latency", color="orange")
    plt.axhline(y=50, color="r", linestyle="--", label="Limit (50ms)")
    plt.xlabel("Frame index")
    plt.ylabel("Latency (ms)")
    plt.title("Camera Latency Performance")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.savefig(os.path.join(output_dir, "camera_latency.png"))
    plt.close()

    # 3. Fusion Latency
    plt.figure(figsize=(10, 5))
    plt.plot([f.latency_ms for f in fused_data], label="Fusion Latency", color="green")
    plt.axhline(y=100, color="r", linestyle="--", label="Limit (100ms)")
    plt.xlabel("Frame index")
    plt.ylabel("Latency (ms)")
    plt.title("Fusion Latency Performance")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.savefig(os.path.join(output_dir, "fusion_latency.png"))
    plt.close()

    # 4. Combined Camera and Radar Latency
    plt.figure(figsize=(10, 5))
    plt.plot([f.latency_ms for f in camera_data], label="Camera Latency", color="orange", alpha=0.7)
    plt.plot([f.latency_ms for f in radar_data], label="Radar Latency", color="blue", alpha=0.7)
    plt.axhline(y=50, color="r", linestyle="--", label="Limit (50ms)")
    plt.xlabel("Frame index")
    plt.ylabel("Latency (ms)")
    plt.title("Combined Camera and Radar Latency Performance")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.savefig(os.path.join(output_dir, "camera_radar_latency.png"))
    plt.close()

    # 5. Decision Consistency Distribution
    plt.figure(figsize=(10, 5))
    consistency_scores = [KPICalculator.calculate_decision_consistency(f) for f in fused_data]
    plt.hist(consistency_scores, bins=10, color="skyblue", edgecolor="black")
    plt.title("Fusion Decision Consistency Distribution")
    plt.xlabel("Consistency Score")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(output_dir, "decision_consistency.png"))
    plt.close()

    # 5. Spatial Alignment Error Over Time
    plt.figure(figsize=(10, 5))
    spatial_errors = [KPICalculator.calculate_spatial_alignment_error(f) for f in fused_data]
    plt.plot(spatial_errors, marker="o", linestyle="-", color="coral", markersize=4)
    plt.title("Spatial Alignment Error Over Time")
    plt.xlabel("Frame index")
    plt.ylabel("Distance Error (units)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(os.path.join(output_dir, "spatial_alignment.png"))
    plt.close()

    logger.info(f"Individual reports saved in {output_dir}/")

    # 6. Summary Statistics
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

    with open(os.path.join(output_dir, "summary_stats.json"), "w") as f:
        json.dump(summary, f, indent=4)
    logger.info(f"Summary stats saved in {output_dir}/summary_stats.json")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    generate_report()
