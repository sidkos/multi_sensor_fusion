"""
Module for calculating Key Performance Indicators (KPIs).
Includes calculations for mandatory KPIs and advanced performance metrics.
"""

import math
from typing import Dict, List, Optional, Union

from src.models import BaseFrame, FusedFrame

# Define types for backward compatibility or internal use if needed
JSONValue = Union[str, int, float, bool, None, Dict[str, "JSONValue"], List["JSONValue"]]
SensorData = Dict[str, JSONValue]


class KPICalculator:
    """
    Collection of static methods for sensor and fusion KPI calculations.
    """

    @staticmethod
    def calculate_data_drop_rate(data: List[Optional[BaseFrame]], expected_timestamps: List[int]) -> float:
        """Calculates the data drop rate based on missing and malformed frames.

        Args:
            data (list): List of received data frames (objects or None).
            expected_timestamps (list): List of timestamps that should have been received.

        Returns:
            float: Drop rate as a float between 0.0 and 1.0.
        """
        if not expected_timestamps:
            return 0.0

        received_timestamps = {item.timestamp for item in data if item}
        missing_count = sum(1 for ts in expected_timestamps if ts not in received_timestamps)
        malformed_count = sum(1 for item in data if item is None)

        # Drop rate is ratio of missing or malformed to total expected
        return min(1.0, (missing_count + malformed_count) / len(expected_timestamps))

    @staticmethod
    def calculate_decision_consistency(fused_frame: FusedFrame) -> float:
        """Calculates consistency between fused class and source sensor classes.

        Implements logic that monitors classification results per frame for each modality
        before fusion and compares these with the fused classification.
        Assumes the fusion module may rely on a single trusted modality if the other
        modality does not detect an object.

        Args:
            fused_frame (FusedFrame): A single aligned frame containing fusion results.

        Returns:
            float: Consistency score as a float (ratio of consistent objects).
        """
        fused_objects = fused_frame.fused_objects
        if not fused_objects:
            return 1.0  # No objects to disagree on

        consistent_count = 0
        for obj in fused_objects:
            fused_class = obj.object_class
            source_classes = obj.source_classes
            camera_class = source_classes.get("camera")
            radar_class = source_classes.get("radar")

            # Simple logic: fused class should match at least one available source class
            available_sources = [cls for cls in [camera_class, radar_class] if cls]
            if not available_sources:
                continue
            if fused_class in available_sources:
                consistent_count += 1

        return consistent_count / len(fused_objects)

    @staticmethod
    def calculate_spatial_alignment_error(fused_frame: FusedFrame) -> float:
        """Calculates the average Euclidean distance between camera and radar centers.

        Args:
            fused_frame (FusedFrame): A single aligned frame containing fusion results.

        Returns:
            float: Average spatial error distance.
        """
        fused_objects = fused_frame.fused_objects
        errors = []
        for obj in fused_objects:
            # Calculate distance between camera bbox center and radar points center
            cam_bbox = obj.camera_bbox_3d
            radar_pts = obj.radar_points

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
    def calculate_confidence_stability(fused_frames: List[FusedFrame]) -> float:
        """Calculates the standard deviation of fusion confidence across frames.

        Args:
            fused_frames (list): List of fused data frames.

        Returns:
            float: Standard deviation of fused confidence scores.
        """
        confidences = []
        for frame in fused_frames:
            for obj in frame.fused_objects:
                confidences.append(float(obj.fused_confidence))

        if len(confidences) < 2:
            return 0.0

        mean = sum(confidences) / len(confidences)
        variance = sum((x - mean) ** 2 for x in confidences) / len(confidences)
        return math.sqrt(variance)  # Returning standard deviation

    @staticmethod
    def calculate_sensor_contribution_balance(fused_frames: List[FusedFrame]) -> Dict[str, float]:
        """Analyzes the source distribution of fused objects.

        Args:
            fused_frames (list): List of fused data frames.

        Returns:
            dict: Dictionary containing the percentage distribution of fusion sources.
        """
        both = 0
        camera_only = 0
        radar_only = 0
        total = 0

        for frame in fused_frames:
            for obj in frame.fused_objects:
                total += 1
                sources = obj.source_classes
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
