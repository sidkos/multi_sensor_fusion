"""
Module for calculating Key Performance Indicators (KPIs).
Includes calculations for mandatory KPIs and additional performance metrics.
"""

import math
from typing import Dict, List, Optional, Union

# Import SensorData and AlignedFrame if needed, but they are defined in data_loader.
# To avoid circular imports, I'll define common types or use the ones from data_loader if possible.
# Actually, I'll just define the types here as well if they are needed for calculation.

JSONValue = Union[str, int, float, bool, None, Dict[str, "JSONValue"], List["JSONValue"]]
SensorData = Dict[str, JSONValue]


class KPICalculator:
    """
    Collection of static methods for sensor and fusion KPI calculations.
    """

    @staticmethod
    def calculate_data_drop_rate(data: List[Optional[SensorData]], expected_timestamps: List[int]) -> float:
        """Calculates the data drop rate based on missing and malformed frames.

        Args:
            data (list): List of received data frames.
            expected_timestamps (list): List of timestamps that should have been received.

        Returns:
            float: Drop rate as a float between 0.0 and 1.0.
        """
        if not expected_timestamps:
            return 0.0

        received_timestamps = {item["timestamp"] for item in data if item and isinstance(item.get("timestamp"), int)}
        missing_count = sum(1 for ts in expected_timestamps if ts not in received_timestamps)
        malformed_count = sum(1 for item in data if item is None)

        return (missing_count + malformed_count) / len(expected_timestamps)

    @staticmethod
    def calculate_decision_consistency(fused_frame: SensorData) -> float:
        """Calculates consistency between fused class and source sensor classes.

        Args:
            fused_frame (dict): A single aligned frame containing fusion results.

        Returns:
            float: Consistency score as a float (ratio of consistent objects).
        """
        fused_objects_raw = fused_frame.get("fused_objects", [])
        if not isinstance(fused_objects_raw, list):
            return 1.0
        fused_objects: List[SensorData] = [obj for obj in fused_objects_raw if isinstance(obj, dict)]
        if not fused_objects:
            return 1.0  # No objects to disagree on

        consistent_count = 0
        for obj in fused_objects:
            fused_class = obj.get("class")
            source_classes_raw = obj.get("source_classes", {})
            if not isinstance(source_classes_raw, dict):
                continue
            source_classes: SensorData = source_classes_raw
            camera_class = source_classes.get("camera")
            radar_class = source_classes.get("radar")

            # Simple logic: fused class should match at least one available source class
            if (camera_class and fused_class == camera_class) or (radar_class and fused_class == radar_class):
                consistent_count += 1
            elif not camera_class and not radar_class:
                # If no source provided, it's inconsistent unless fusion created it (out of scope?)
                pass

        return consistent_count / len(fused_objects)

    @staticmethod
    def calculate_spatial_alignment_error(fused_frame: SensorData) -> float:
        """Calculates the average Euclidean distance between camera and radar centers.

        Args:
            fused_frame (dict): A single aligned frame containing fusion results.

        Returns:
            float: Average spatial error distance.
        """
        # Additional KPI 2: Spatial Alignment Error
        fused_objects_raw = fused_frame.get("fused_objects", [])
        if not isinstance(fused_objects_raw, list):
            return 0.0
        fused_objects: List[SensorData] = [obj for obj in fused_objects_raw if isinstance(obj, dict)]
        errors = []
        for obj in fused_objects:
            # Calculate distance between camera bbox center and radar points center
            cam_bbox_raw = obj.get("camera_bbox_3d", [])
            radar_pts_raw = obj.get("radar_points", [])

            if isinstance(cam_bbox_raw, list) and isinstance(radar_pts_raw, list):
                cam_bbox: List[Dict[str, float]] = []
                for p in cam_bbox_raw:
                    if (
                        isinstance(p, dict)
                        and isinstance(p.get("x"), (int, float))
                        and isinstance(p.get("y"), (int, float))
                    ):
                        x = p.get("x")
                        y = p.get("y")
                        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                            cam_bbox.append({"x": float(x), "y": float(y)})

                radar_pts: List[Dict[str, float]] = []
                for p in radar_pts_raw:
                    if isinstance(p, dict):
                        x = p.get("x")
                        y = p.get("y")
                        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                            radar_pts.append({"x": float(x), "y": float(y)})

                if cam_bbox and radar_pts:
                    cam_center = {
                        "x": sum(float(p["x"]) for p in cam_bbox) / len(cam_bbox),
                        "y": sum(float(p["y"]) for p in cam_bbox) / len(cam_bbox),
                    }
                    radar_center = {
                        "x": sum(float(p["x"]) for p in radar_pts) / len(radar_pts),
                        "y": sum(float(p["y"]) for p in radar_pts) / len(radar_pts),
                    }
                    dist = math.sqrt(
                        (cam_center["x"] - radar_center["x"]) ** 2 + (cam_center["y"] - radar_center["y"]) ** 2
                    )
                    errors.append(dist)

        return sum(errors) / len(errors) if errors else 0.0

    @staticmethod
    def calculate_confidence_stability(fused_frames: List[SensorData]) -> float:
        """Calculates the standard deviation of fusion confidence across frames.

        Args:
            fused_frames (list): List of fused data frames.

        Returns:
            float: Standard deviation of fused confidence scores.
        """
        # Additional KPI 1: Confidence Stability Score (Variance)
        confidences = []
        for frame in fused_frames:
            fused_objects_raw = frame.get("fused_objects", [])
            if not isinstance(fused_objects_raw, list):
                continue
            fused_objects: List[SensorData] = [obj for obj in fused_objects_raw if isinstance(obj, dict)]
            for obj in fused_objects:
                confidence = obj.get("fused_confidence", 0)
                if isinstance(confidence, (int, float)):
                    confidences.append(float(confidence))

        if len(confidences) < 2:
            return 0.0

        mean = sum(confidences) / len(confidences)
        variance = sum((x - mean) ** 2 for x in confidences) / len(confidences)
        return math.sqrt(variance)  # Returning standard deviation

    @staticmethod
    def calculate_sensor_contribution_balance(fused_frames: List[SensorData]) -> Dict[str, float]:
        """Analyzes the source distribution of fused objects.

        Args:
            fused_frames (list): List of fused data frames.

        Returns:
            dict: Dictionary containing the percentage distribution of fusion sources.
        """
        # Additional KPI 3: Sensor Contribution Balance
        both = 0
        camera_only = 0
        radar_only = 0
        total = 0

        for frame in fused_frames:
            fused_objects_raw = frame.get("fused_objects", [])
            if not isinstance(fused_objects_raw, list):
                continue
            fused_objects: List[SensorData] = [obj for obj in fused_objects_raw if isinstance(obj, dict)]
            for obj in fused_objects:
                total += 1
                sources_raw = obj.get("source_classes", {})
                if not isinstance(sources_raw, dict):
                    continue
                sources: SensorData = sources_raw
                has_cam = sources.get("camera") is not None
                has_rad = sources.get("radar") is not None

                if has_cam and has_rad:
                    both += 1
                elif has_cam:
                    camera_only += 1
                elif has_rad:
                    radar_only += 1

        if total == 0:
            return {"both": 0.0, "camera_only": 0.0, "radar_only": 0.0}

        return {"both": both / total, "camera_only": camera_only / total, "radar_only": radar_only / total}
