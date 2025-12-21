"""
Utility for loading test data, shared between pytest fixtures and manual test execution.
"""

import os
from typing import Dict, List, Optional, Union

from src.data_loader import DataLoader
from src.models import CameraFrame, FusedFrame, RadarFrame


def load_test_data() -> (
    Dict[str, Union[List[Optional[RadarFrame]], List[Optional[CameraFrame]], List[Optional[FusedFrame]], List[int]]]
):
    """Loads radar, camera, and fusion data from the data directory.

    This utility attempts to locate the .jsonl datasets in the project's 'data/' folder.
    It returns the parsed objects for radar points, camera bounding boxes, and
    fused objects, along with the theoretical expected timestamps for drop rate analysis.

    Returns:
        Dict[str, Union[List[Optional[RadarFrame]], List[Optional[CameraFrame]], List[Optional[FusedFrame]], List[int]]]:
            A dictionary containing:
            - "radar": List of RadarFrame objects or None for malformed lines.
            - "camera": List of CameraFrame objects or None.
            - "fused": List of FusedFrame objects or None.
            - "expected_radar_ts": Expected timestamps for sensors (1000-10900).
            - "expected_fused_ts": Expected timestamps for fusion (1100-11000).
    """

    radar_path = "data/radar_data_with_kpis.jsonl"
    camera_path = "data/camera_data_with_kpis.jsonl"
    fused_path = "data/fused_data_with_kpis.jsonl"

    # If not found, try to find it relative to this file (in case tests are run from a different CWD)
    if not os.path.exists(radar_path):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # If we are in src/, the data folder is one level up
        project_root = os.path.dirname(base_dir)
        radar_path = os.path.join(project_root, "data/radar_data_with_kpis.jsonl")
        camera_path = os.path.join(project_root, "data/camera_data_with_kpis.jsonl")
        fused_path = os.path.join(project_root, "data/fused_data_with_kpis.jsonl")

    radar = DataLoader.load_radar_data(radar_path)
    camera = DataLoader.load_camera_data(camera_path)
    fused = DataLoader.load_fused_data(fused_path)

    # Expected timestamps (10 FPS for 10 seconds = 100 frames)
    # Radar/Camera start at 1000, Fused at 1100
    expected_radar_ts = list(range(1000, 1000 + 100 * 100, 100))
    expected_fused_ts = list(range(1100, 1100 + 100 * 100, 100))

    return {
        "radar": radar,
        "camera": camera,
        "fused": fused,
        "expected_radar_ts": expected_radar_ts,
        "expected_fused_ts": expected_fused_ts,
    }
