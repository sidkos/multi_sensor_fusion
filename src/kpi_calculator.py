"""
Module for calculating Key Performance Indicators (KPIs).
Includes calculations for mandatory KPIs and advanced performance metrics.
"""

import math
from typing import Dict, List, Optional

from src.models import BaseFrame, CameraFrame, FusedFrame, RadarFrame


class KPICalculator:
    """
    Collection of static methods for sensor and fusion KPI calculations.

    This class provides a suite of tools to evaluate the performance of a multi-sensor
    fusion system. It includes mandatory KPIs like drop rate and consistency,
    as well as advanced metrics like spatial alignment error and confidence stability.
    """

    @staticmethod
    def calculate_data_drop_rate(data: List[Optional[BaseFrame]], expected_timestamps: List[int]) -> float:
        """Calculates the data drop rate based on missing and malformed frames.

        The drop rate is defined as the ratio of (missing frames + malformed frames)
        to the total number of expected frames over a given period.
        - A frame is "missing" if its timestamp is in expected_timestamps but not in data.
        - A frame is "malformed" if it appears as None in the data list (indicating a parsing error).

        Args:
            data (List[Optional[BaseFrame]]): List of received data frames (objects or None).
            expected_timestamps (List[int]): List of timestamps that should have been received.

        Returns:
            float: Drop rate as a float between 0.0 (no drops) and 1.0 (all drops).
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

        This KPI monitors the classification result per frame for each modality
        before fusion and compares these with the final fused classification.
        A detection is considered consistent if the fused class matches at least one
        of the source modalities (camera or radar).

        If one modality is missing (e.g., radar didn't detect the object), the logic
        assumes the fusion module may rely on the single available trusted modality.

        Args:
            fused_frame (FusedFrame): A single aligned frame containing fusion results.

        Returns:
            float: Consistency score as a float between 0.0 and 1.0, where 1.0 means
                all fused objects are consistent with their sources.
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

            # Fused class should match at least one available source class
            available_sources = [cls for cls in [camera_class, radar_class] if cls]
            if not available_sources:
                # If no source class is provided, we cannot verify consistency
                # In this specific context, we skip it or consider it inconsistent
                continue
            if fused_class in available_sources:
                consistent_count += 1

        return consistent_count / len(fused_objects)

    @staticmethod
    def calculate_spatial_alignment_error(fused_frame: FusedFrame) -> float:
        """Calculates the average Euclidean distance between camera and radar centers.

        This metric measures the spatial synchronization and calibration accuracy
        between the camera and radar sensors. For each fused object, it computes
         the Euclidean distance between:
        1. The geometric center of the 3D bounding box provided by the camera.
        2. The centroid of the point cluster provided by the radar.

        Args:
            fused_frame (FusedFrame): A single aligned frame containing fusion results.

        Returns:
            float: Average spatial error distance in the sensor's coordinate units.
                Returns 0.0 if no fused objects with both modalities are present.
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

        High variance (high standard deviation) in fusion confidence might indicate
        an unstable fusion algorithm or inconsistent sensor data, which could lead
        to "flickering" detections. This KPI aggregates all confidence scores
        from all objects across the provided frames.

        Args:
            fused_frames (List[FusedFrame]): List of fused data frames to analyze.

        Returns:
            float: Standard deviation of fused confidence scores. Returns 0.0
                if there are fewer than 2 confidence values to compare.
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
        """Analyzes the distribution of sensor sources for fused objects.

        This KPI tracks how often the fusion result relies on both sensors versus
        a single sensor (camera-only or radar-only). It helps identify if one
        sensor is consistently being ignored or if the system is properly
        utilizing the complementary information from both modalities.

        Args:
            fused_frames (List[FusedFrame]): List of fused data frames to analyze.

        Returns:
            Dict[str, float]: A dictionary containing the percentage distribution:
                - "both": Percentage of objects fused from both modalities.
                - "camera_only": Percentage of objects based only on camera.
                - "radar_only": Percentage of objects based only on radar.
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

    @staticmethod
    def validate_fusion_value_correctness(
        fused_frame: FusedFrame,
        radar_frame: Optional[RadarFrame],
        camera_frame: Optional[CameraFrame],
    ) -> List[str]:
        """Performs a deep value-level validation of the fusion output.

        This method verifies that the data within a fused frame correctly reflects
        the raw sensor inputs from the corresponding timestamp. It checks for:
        1. Coordinate preservation: Bounding boxes and radar points must match
           exactly between the sensor source and the fused object.
        2. Classification integrity: The source classes declared in the fusion
           result must match the original sensor detections.
        3. Data isolation: Ensures no data mixing or "ghost" data is introduced.

        Args:
            fused_frame (FusedFrame): The fusion output frame at timestamp T.
            radar_frame (Optional[RadarFrame]): The raw radar frame at T-100ms.
            camera_frame (Optional[CameraFrame]): The raw camera frame at T-100ms.

        Returns:
            List[str]: A list of error messages describing any value mismatches
                found. Returns an empty list if validation passes.
        """
        failures = []

        for f_obj in fused_frame.fused_objects:
            found_in_source = False

            # 1. Validate Camera Source Data
            if "camera" in f_obj.source_classes and camera_frame:
                c_class_expected = f_obj.source_classes["camera"]
                # Find matching object in camera frame by class and exact bbox coordinates
                matching_c_obj = next(
                    (
                        obj
                        for obj in camera_frame.objects
                        if obj.object_class == c_class_expected and obj.bbox_3d == f_obj.camera_bbox_3d
                    ),
                    None,
                )
                if not matching_c_obj:
                    failures.append(
                        f"Frame {fused_frame.timestamp}: Fused object {f_obj.object_class} "
                        f"(from camera {c_class_expected}) has bbox data that doesn't match "
                        f"any object in raw camera frame {camera_frame.timestamp}."
                    )
                else:
                    found_in_source = True

            # 2. Validate Radar Source Data
            if "radar" in f_obj.source_classes and radar_frame:
                r_class_expected = f_obj.source_classes["radar"]

                # Convert radar points to list of dicts for comparison
                r_frame_pts_dicts = [
                    {
                        "x": p.x,
                        "y": p.y,
                        "z": p.z,
                        "signal_strength": p.signal_strength,
                        "doppler_velocity": p.doppler_velocity,
                    }
                    for p in radar_frame.points
                    if p.object_class == r_class_expected
                ]

                # Check if all points in fused object exist exactly in the radar source
                all_pts_match = True
                for p_fused in f_obj.radar_points:
                    if p_fused not in r_frame_pts_dicts:
                        all_pts_match = False
                        break

                if not all_pts_match:
                    failures.append(
                        f"Frame {fused_frame.timestamp}: Fused object {f_obj.object_class} "
                        f"(from radar {r_class_expected}) has radar points that don't match "
                        f"raw radar frame {radar_frame.timestamp}."
                    )
                else:
                    found_in_source = True

            # 3. Source Attribution Check
            if not found_in_source:
                failures.append(
                    f"Frame {fused_frame.timestamp}: Fused object {f_obj.object_class} "
                    "has no matching source data in sensors from corresponding timestamp."
                )

        return failures
