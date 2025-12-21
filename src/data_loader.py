"""
Module for loading and aligning sensor data.
Provides functionality to read JSONL files and synchronize radar, camera, and fusion data.
"""

import json
import os
from typing import Dict, List, Optional, Union

# Define types for sensor data
JSONValue = Union[str, int, float, bool, None, Dict[str, "JSONValue"], List["JSONValue"]]
SensorData = Dict[str, JSONValue]
AlignedFrame = Dict[str, Optional[JSONValue]]


class DataLoader:
    """
    Handles data ingestion and temporal alignment for multiple sensor modalities.
    """

    @staticmethod
    def load_jsonl(filepath: str) -> List[Optional[SensorData]]:
        """Loads a JSONL file and returns a list of dictionaries.

        Args:
            filepath (str): Path to the .jsonl file.

        Returns:
            list: List of data dictionaries, with None for malformed lines.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        data: List[Optional[SensorData]] = []
        with open(filepath, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        # Mark as malformed for data_drop_rate calculation
                        data.append(None)
        return data

    @staticmethod
    def align_data(
        radar_data: List[Optional[SensorData]],
        camera_data: List[Optional[SensorData]],
        fused_data: List[Optional[SensorData]],
    ) -> List[AlignedFrame]:
        """Aligns radar and camera frames with fusion output using timestamp logic.

        Fusion at t+1 corresponds to sensors at t.

        Args:
            radar_data (list): List of radar data frames.
            camera_data (list): List of camera data frames.
            fused_data (list): List of fused data frames.

        Returns:
            list: List of aligned frames containing fused, radar, and camera data.
        """
        # Fusion at t+1 corresponds to sensors at t
        # We'll create a mapping by timestamp
        radar_map = {item["timestamp"]: item for item in radar_data if item}
        camera_map = {item["timestamp"]: item for item in camera_data if item}

        aligned_frames: List[AlignedFrame] = []
        for fused_frame in fused_data:
            if not fused_frame:
                continue

            timestamp = fused_frame.get("timestamp")
            if not isinstance(timestamp, int):
                continue

            target_ts = timestamp - 100  # Assuming 10 FPS (100ms interval)

            aligned_frames.append(
                {
                    "fused": fused_frame,
                    "radar": radar_map.get(target_ts),
                    "camera": camera_map.get(target_ts),
                    "timestamp": fused_frame["timestamp"],
                }
            )

        return aligned_frames
