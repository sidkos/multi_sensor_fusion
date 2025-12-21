"""
Script to generate synthetic sensor and fusion datasets for testing purposes.
Produces radar_data.jsonl, camera_data.jsonl, and fused_data.jsonl with realistic values.
"""

import json
import logging
import os
import random

logger = logging.getLogger(__name__)


def generate_radar_data(filename: str, num_frames: int = 100, start_ts: int = 1000) -> None:
    """Generates synthetic radar data with timestamps, latency, and points.

    Args:
        filename (str): Output filename.
        num_frames (int): Number of frames to generate. Defaults to 100.
        start_ts (int): Starting timestamp in milliseconds. Defaults to 1000.
    """
    with open(filename, "w") as f:
        for i in range(num_frames):
            ts = start_ts + i * 100
            frame = {
                "timestamp": ts,
                "latency_ms": random.uniform(30, 49),
                "error_rate_percent": random.uniform(0, 0.0009),
                "points": [
                    {
                        "x": random.uniform(10, 20),
                        "y": random.uniform(-5, 5),
                        "z": random.uniform(0, 2),
                        "signal_strength": random.uniform(0.5, 1.0),
                        "doppler_velocity": random.uniform(0, 5),
                        "object_class": "car",
                    }
                    for _ in range(random.randint(1, 3))
                ],
            }
            f.write(f"{json.dumps(frame)}\n")


def generate_camera_data(filename: str, num_frames: int = 100, start_ts: int = 1000) -> None:
    """Generates synthetic camera data with bounding boxes.

    Args:
        filename (str): Output filename.
        num_frames (int): Number of frames to generate. Defaults to 100.
        start_ts (int): Starting timestamp in milliseconds. Defaults to 1000.
    """
    with open(filename, "w") as f:
        for i in range(num_frames):
            ts = start_ts + i * 100
            frame = {
                "timestamp": ts,
                "latency_ms": random.uniform(20, 50),
                "error_rate_percent": random.uniform(0, 0.001),
                "frame_id": f"frame_{ts}.jpg",
                "objects": [
                    {
                        "class": "car",
                        "bbox_3d": [
                            {"x": 11.0, "y": -4.0, "z": 0.0},
                            {"x": 13.0, "y": -4.0, "z": 0.0},
                            {"x": 13.0, "y": -2.0, "z": 0.0},
                            {"x": 11.0, "y": -2.0, "z": 0.0},
                        ],
                    }
                ],
            }
            f.write(f"{json.dumps(frame)}\n")


def generate_fused_data(filename: str, num_frames: int = 100, start_ts: int = 1100) -> None:
    """Generates synthetic fusion data, aligned at t+1 with sensor data.

    Args:
        filename (str): Output filename.
        num_frames (int): Number of frames to generate. Defaults to 100.
        start_ts (int): Starting timestamp in milliseconds. Defaults to 1100.
    """
    with open(filename, "w") as f:
        for i in range(num_frames):
            ts = start_ts + i * 100
            frame = {
                "timestamp": ts,
                "fusion_latency_ms": random.uniform(50, 99),
                "data_alignment_jitter_ms": random.uniform(1, 5),
                "fused_objects": [
                    {
                        "class": "car",
                        "source_classes": {"camera": "car", "radar": "car"},
                        "camera_bbox_3d": [
                            {"x": 11.0, "y": -4.0, "z": 0.0},
                            {"x": 13.0, "y": -4.0, "z": 0.0},
                            {"x": 13.0, "y": -2.0, "z": 0.0},
                            {"x": 11.0, "y": -2.0, "z": 0.0},
                        ],
                        "radar_points": [{"x": 12.0, "y": -3.0, "z": 0.5}],
                        "fused_confidence": random.uniform(0.9, 0.99),
                    }
                ],
            }
            f.write(f"{json.dumps(frame)}\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    os.makedirs("data", exist_ok=True)
    generate_radar_data("data/radar_data.jsonl")
    generate_camera_data("data/camera_data.jsonl")
    generate_fused_data("data/fused_data.jsonl")
    logger.info("Synthetic data generated in data/ folder.")
