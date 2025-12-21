"""
Module for generating performance reports and visualizations.
Produces latency plots and aggregate summary statistics in JSON format.
"""

import json
import logging
from typing import List

import matplotlib.pyplot as plt

from src.data_loader import DataLoader, SensorData
from src.kpi_calculator import KPICalculator

logger = logging.getLogger(__name__)


def generate_report() -> None:
    """Loads all datasets and generates visual and text reports.

    Calculates aggregate metrics and saves them to:
    - latency_report.png: Plot of sensor and fusion latencies.
    - summary_stats.json: JSON file with average latencies and stability scores.
    """
    radar_data = DataLoader.load_jsonl("data/radar_data.jsonl")
    camera_data = DataLoader.load_jsonl("data/camera_data.jsonl")
    fused_data = DataLoader.load_jsonl("data/fused_data.jsonl")

    # Calculate metrics
    radar_latencies: List[float] = []
    for f in radar_data:
        if f:
            lat = f.get("latency_ms")
            if isinstance(lat, (int, float)):
                radar_latencies.append(float(lat))

    camera_latencies: List[float] = []
    for f in camera_data:
        if f:
            lat = f.get("latency_ms")
            if isinstance(lat, (int, float)):
                camera_latencies.append(float(lat))

    fusion_latencies: List[float] = []
    for f in fused_data:
        if f:
            lat = f.get("fusion_latency_ms")
            if isinstance(lat, (int, float)):
                fusion_latencies.append(float(lat))

    valid_fused_data: List[SensorData] = [f for f in fused_data if f]

    # Plot Latencies
    plt.figure(figsize=(10, 6))
    plt.plot(radar_latencies, label="Radar Latency")
    plt.plot(camera_latencies, label="Camera Latency")
    plt.plot(fusion_latencies, label="Fusion Latency")
    plt.axhline(y=50, color="r", linestyle="--", label="Sensor Limit (50ms)")
    plt.axhline(y=100, color="g", linestyle="--", label="Fusion Limit (100ms)")
    plt.xlabel("Frame index")
    plt.ylabel("Latency (ms)")
    plt.title("Sensor and Fusion Latency Performance")
    plt.legend()
    plt.savefig("latency_report.png")
    logger.info("Report saved as latency_report.png")

    # Summary Statistics
    summary = {
        "radar_avg_latency": sum(radar_latencies) / len(radar_latencies) if radar_latencies else 0.0,
        "camera_avg_latency": sum(camera_latencies) / len(camera_latencies) if camera_latencies else 0.0,
        "fusion_avg_latency": sum(fusion_latencies) / len(fusion_latencies) if fusion_latencies else 0.0,
        "fusion_stability": KPICalculator.calculate_confidence_stability(valid_fused_data),
    }

    with open("summary_stats.json", "w") as f:
        json.dump(summary, f, indent=4)
    logger.info("Summary stats saved as summary_stats.json")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    generate_report()
