"""
Module for generating performance reports and visualizations.
Produces latency plots and aggregate summary statistics in JSON format.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from src.data_loader import DataLoader
from src.kpi_calculator import KPICalculator
from src.models import CameraFrame, FusedFrame, RadarFrame

logger = logging.getLogger(__name__)


def plot_kpi_compliance_heatmap(
    radar_data: List[RadarFrame],
    camera_data: List[CameraFrame],
    fused_data: List[FusedFrame],
    output_path: str,
) -> None:
    """Generates a heatmap showing KPI compliance across different sensor modalities.

    Args:
        radar_data (List[RadarFrame]): List of radar data frames.
        camera_data (List[CameraFrame]): List of camera data frames.
        fused_data (List[FusedFrame]): List of fused data frames.
        output_path (str): File path to save the generated heatmap.
    """
    kpis = ["Latency", "Error Rate", "Data Drop", "Consistency", "Jitter"]
    sensors = ["Radar", "Camera", "Fusion"]
    data_matrix = np.zeros((len(kpis), len(sensors)))

    # Thresholds
    THRESHOLDS = {
        "Radar": {"Latency": 50, "Error Rate": 0.001, "Data Drop": 0.005},
        "Camera": {"Latency": 50, "Error Rate": 0.001, "Data Drop": 0.005},
        "Fusion": {"Latency": 100, "Consistency": 0.95, "Jitter": 5},
    }

    # Radar
    if radar_data:
        data_matrix[0, 0] = 1 if all(f.latency_ms < THRESHOLDS["Radar"]["Latency"] for f in radar_data) else 0
        data_matrix[1, 0] = (
            1 if all((f.error_rate_percent or 0) < THRESHOLDS["Radar"]["Error Rate"] for f in radar_data) else 0
        )
    # Camera
    if camera_data:
        data_matrix[0, 1] = 1 if all(f.latency_ms < THRESHOLDS["Camera"]["Latency"] for f in camera_data) else 0
        data_matrix[1, 1] = (
            1 if all((f.error_rate_percent or 0) < THRESHOLDS["Camera"]["Error Rate"] for f in camera_data) else 0
        )
    # Fusion
    if fused_data:
        data_matrix[0, 2] = 1 if all(f.latency_ms < THRESHOLDS["Fusion"]["Latency"] for f in fused_data) else 0
        data_matrix[3, 2] = (
            1
            if all(
                KPICalculator.calculate_decision_consistency(f) >= THRESHOLDS["Fusion"]["Consistency"]
                for f in fused_data
            )
            else 0
        )
        data_matrix[4, 2] = (
            1 if all(f.data_alignment_jitter_ms <= THRESHOLDS["Fusion"]["Jitter"] for f in fused_data) else 0
        )

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(data_matrix, cmap="RdYlGn", aspect="auto")

    ax.set_xticks(np.arange(len(sensors)))
    ax.set_yticks(np.arange(len(kpis)))
    ax.set_xticklabels(sensors)
    ax.set_yticklabels(kpis)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(len(kpis)):
        for j in range(len(sensors)):
            val = "Pass" if data_matrix[i, j] == 1 else "Fail"
            # Disable markers for N/A cells
            if sensors[j] == "Radar" and kpis[i] in ["Consistency", "Jitter"]:
                val = "N/A"
            if sensors[j] == "Camera" and kpis[i] in ["Consistency", "Jitter"]:
                val = "N/A"
            if sensors[j] == "Fusion" and kpis[i] in ["Error Rate", "Data Drop"]:
                val = "N/A"
            ax.text(j, i, val, ha="center", va="center", color="black")

    ax.set_title("KPI Compliance Heatmap")
    fig.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_confidence_stability_trend(fused_data: List[FusedFrame], output_path: str) -> None:
    """Tracks the stability of the fusion module's confidence over time.

    Args:
        fused_data (List[FusedFrame]): List of fused data frames.
        output_path (str): File path to save the generated trend plot.
    """
    plt.figure(figsize=(10, 5))
    for i, frame in enumerate(fused_data):
        for j, obj in enumerate(frame.fused_objects):
            plt.scatter(i, obj.fused_confidence, color="blue", alpha=0.5, s=10)

    plt.title("Fusion Confidence Stability Trend")
    plt.xlabel("Frame Index")
    plt.ylabel("Fused Confidence")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig(output_path)
    plt.close()


def plot_sensor_contribution_pie(fused_data: List[FusedFrame], output_path: str) -> None:
    """Visualizes the balance of sensor data used in the final fusion output.

    Args:
        fused_data (List[FusedFrame]): List of fused data frames.
        output_path (str): File path to save the generated pie chart.
    """
    balance = KPICalculator.calculate_sensor_contribution_balance(fused_data)
    labels = ["Both Sensors", "Camera-Only", "Radar-Only"]
    sizes = [balance["both"], balance["camera_only"], balance["radar_only"]]
    colors = ["#ff9999", "#66b3ff", "#99ff99"]

    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=140)
    plt.title("Sensor Contribution Balance")
    plt.savefig(output_path)
    plt.close()


def plot_spatial_alignment_scatter(fused_data: List[FusedFrame], output_path: str) -> None:
    """Maps the spatial error in a 2D plane (X-Y drift).

    Args:
        fused_data (List[FusedFrame]): List of fused data frames.
        output_path (str): File path to save the generated scatter plot.
    """
    x_offsets = []
    y_offsets = []

    for frame in fused_data:
        for obj in frame.fused_objects:
            cam_bbox = obj.camera_bbox_3d
            radar_pts = obj.radar_points
            if cam_bbox and radar_pts:
                cam_x = sum(float(p["x"]) for p in cam_bbox) / len(cam_bbox)
                cam_y = sum(float(p["y"]) for p in cam_bbox) / len(cam_bbox)
                radar_x = sum(float(p["x"]) for p in radar_pts) / len(radar_pts)
                radar_y = sum(float(p["y"]) for p in radar_pts) / len(radar_pts)
                x_offsets.append(cam_x - radar_x)
                y_offsets.append(cam_y - radar_y)

    plt.figure(figsize=(8, 8))
    plt.scatter(x_offsets, y_offsets, alpha=0.6, color="purple")
    plt.axhline(0, color="black", linestyle="--", alpha=0.3)
    plt.axvline(0, color="black", linestyle="--", alpha=0.3)
    plt.xlabel("X Offset (Camera - Radar)")
    plt.ylabel("Y Offset (Camera - Radar)")
    plt.title("Spatial Alignment Scatter (X-Y Drift)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.savefig(output_path)
    plt.close()


def plot_cumulative_data_drop(
    radar_data: List[RadarFrame],
    camera_data: List[CameraFrame],
    output_path: str,
) -> None:
    """Shows when data drops occur throughout the recording.

    Args:
        radar_data (List[RadarFrame]): List of radar data frames.
        camera_data (List[CameraFrame]): List of camera data frames.
        output_path (str): File path to save the generated step chart.
    """
    expected_ts = list(range(1000, 1000 + 100 * 100, 100))

    def get_cumulative_drops(data: List[Any], expected: List[int]) -> List[int]:
        """Calculates cumulative count of missing frames."""
        received_ts = {f.timestamp for f in data if f}
        drops = []
        count = 0
        for ts in expected:
            if ts not in received_ts:
                count += 1
            drops.append(count)
        return drops

    radar_drops = get_cumulative_drops(radar_data, expected_ts)
    camera_drops = get_cumulative_drops(camera_data, expected_ts)

    plt.figure(figsize=(10, 5))
    plt.step(range(len(expected_ts)), radar_drops, label="Radar Drops", where="post", color="blue")
    plt.step(range(len(expected_ts)), camera_drops, label="Camera Drops", where="post", color="orange")
    plt.title("Cumulative Data Drop Rate (Step Chart)")
    plt.xlabel("Frame Index")
    plt.ylabel("Total Drops")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.savefig(output_path)
    plt.close()


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
    spatial_errors = [KPICalculator.calculate_spatial_alignment_error(f) for f in fused_data]
    plt.figure(figsize=(10, 5))
    plt.plot(spatial_errors, marker="o", linestyle="-", color="coral", markersize=4)
    plt.title("Spatial Alignment Error Over Time")
    plt.xlabel("Frame index")
    plt.ylabel("Distance Error (units)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(os.path.join(output_dir, "spatial_alignment.png"))
    plt.close()

    # 6. KPI Compliance Heatmap
    plot_kpi_compliance_heatmap(radar_data, camera_data, fused_data, os.path.join(output_dir, "kpi_heatmap.png"))

    # 7. Fusion Confidence Stability Trend
    plot_confidence_stability_trend(fused_data, os.path.join(output_dir, "confidence_stability.png"))

    # 8. Sensor Contribution Pie Chart
    plot_sensor_contribution_pie(fused_data, os.path.join(output_dir, "sensor_contribution.png"))

    # 9. Spatial Alignment Scatter (X-Y Drift)
    plot_spatial_alignment_scatter(fused_data, os.path.join(output_dir, "spatial_drift_scatter.png"))

    # 10. Cumulative Data Drop Rate (Step Chart)
    plot_cumulative_data_drop(radar_data, camera_data, os.path.join(output_dir, "data_drop_step.png"))

    logger.info(f"Individual reports saved in {output_dir}/")

    # 11. Summary Statistics
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
