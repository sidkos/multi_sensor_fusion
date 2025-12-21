"""
Utility for loading test data, shared between pytest fixtures and manual test execution.
"""

import os
from typing import Dict, List, Optional, Union

from src.data_loader import DataLoader, JSONValue


def load_test_data() -> Dict[str, Union[List[Optional[JSONValue]], List[int], List[Optional[Dict[str, JSONValue]]]]]:
    """Loads radar, camera, and fusion data for testing.

    Returns:
        dict: Dictionary containing loaded data and expected timestamp lists.
    """
    # Use paths relative to the project root
    radar_path = "data/radar_data.jsonl"
    camera_path = "data/camera_data.jsonl"
    fused_path = "data/fused_data.jsonl"

    # If not found, try to find it relative to this file (in case tests are run from a different CWD)
    if not os.path.exists(radar_path):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # If we are in src/, the data folder is one level up
        project_root = os.path.dirname(base_dir)
        radar_path = os.path.join(project_root, "data/radar_data.jsonl")
        camera_path = os.path.join(project_root, "data/camera_data.jsonl")
        fused_path = os.path.join(project_root, "data/fused_data.jsonl")

    radar = DataLoader.load_jsonl(radar_path)
    camera = DataLoader.load_jsonl(camera_path)
    fused = DataLoader.load_jsonl(fused_path)

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
