"""
Module for loading and aligning sensor data.
Provides functionality to read JSONL files and synchronize radar, camera, and fusion data.
"""

import json
import os
from typing import Dict, List, Optional, Union

from src.models import (
    CameraFrame,
    CameraObject,
    FusedFrame,
    FusedObject,
    RadarFrame,
    RadarPoint,
)
from src.models.types import SensorData

AlignedFrame = Dict[str, Optional[Union[RadarFrame, CameraFrame, FusedFrame, int]]]


class DataLoader:
    """
    Handles data ingestion and temporal alignment for multiple sensor modalities.

    This class provides tools to load raw sensor data from JSONL files and convert
    them into structured Python objects. It also handles the temporal alignment
    between raw sensor frames and late fusion output.
    """

    @staticmethod
    def load_jsonl(filepath: str) -> List[Optional[SensorData]]:
        """Loads a JSONL file and returns a list of dictionaries.

        Args:
            filepath (str): Path to the .jsonl file to be read.

        Returns:
            List[Optional[SensorData]]: List of data dictionaries, where each
                dictionary represents a line in the file. Returns None for lines
                that are not valid JSON.

        Raises:
            FileNotFoundError: If the specified file does not exist.
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
                        data.append(None)
        return data

    @staticmethod
    def load_radar_data(filepath: str) -> List[Optional[RadarFrame]]:
        """Loads radar data and returns a list of RadarFrame objects.

        Parses the JSONL file and maps each valid entry to a RadarFrame,
        including all nested RadarPoint objects.

        Args:
            filepath (str): Path to the radar_data.jsonl file.

        Returns:
            List[Optional[RadarFrame]]: A list of RadarFrame objects, with None
                at indices where the input was malformed.
        """
        raw_data = DataLoader.load_jsonl(filepath)
        frames: List[Optional[RadarFrame]] = []
        for item in raw_data:
            if item is None:
                frames.append(None)
                continue

            points_raw = item.get("points", [])
            points = []
            if isinstance(points_raw, list):
                for p in points_raw:
                    if isinstance(p, dict):
                        # Ensure all fields are present and of correct type for RadarPoint
                        val_x = p.get("x", 0.0)
                        val_y = p.get("y", 0.0)
                        val_z = p.get("z", 0.0)
                        val_ss = p.get("signal_strength", 0.0)
                        val_dv = p.get("doppler_velocity", 0.0)
                        points.append(
                            RadarPoint(
                                x=float(val_x) if isinstance(val_x, (int, float)) else 0.0,
                                y=float(val_y) if isinstance(val_y, (int, float)) else 0.0,
                                z=float(val_z) if isinstance(val_z, (int, float)) else 0.0,
                                signal_strength=float(val_ss) if isinstance(val_ss, (int, float)) else 0.0,
                                doppler_velocity=float(val_dv) if isinstance(val_dv, (int, float)) else 0.0,
                                object_class=str(p.get("object_class", "unknown")),
                            )
                        )

            ts_val = item.get("timestamp", 0)
            lat_val = item.get("latency_ms", 0.0)
            err_val = item.get("error_rate_percent", 0.0)
            frames.append(
                RadarFrame(
                    timestamp=int(ts_val) if isinstance(ts_val, int) else 0,
                    latency_ms=float(lat_val) if isinstance(lat_val, (int, float)) else 0.0,
                    error_rate_percent=float(err_val) if isinstance(err_val, (int, float)) else 0.0,
                    points=points,
                )
            )
        return frames

    @staticmethod
    def load_camera_data(filepath: str) -> List[Optional[CameraFrame]]:
        """Loads camera data and returns a list of CameraFrame objects.

        Parses the JSONL file and maps each valid entry to a CameraFrame,
        including 3D bounding boxes for all detected objects.

        Args:
            filepath (str): Path to the camera_data.jsonl file.

        Returns:
            List[Optional[CameraFrame]]: A list of CameraFrame objects, with None
                at indices where the input was malformed.
        """
        raw_data = DataLoader.load_jsonl(filepath)
        frames: List[Optional[CameraFrame]] = []
        for item in raw_data:
            if item is None:
                frames.append(None)
                continue

            objects_raw = item.get("objects", [])
            objects = []
            if isinstance(objects_raw, list):
                for obj in objects_raw:
                    if isinstance(obj, dict):
                        bbox_raw = obj.get("bbox_3d", [])
                        bbox: List[Dict[str, float]] = []
                        if isinstance(bbox_raw, list):
                            for p in bbox_raw:
                                if isinstance(p, dict):
                                    bbox.append({k: float(v) for k, v in p.items() if isinstance(v, (int, float))})

                        objects.append(CameraObject(object_class=str(obj.get("class", "unknown")), bbox_3d=bbox))

            ts_val = item.get("timestamp", 0)
            lat_val = item.get("latency_ms", 0.0)
            err_val = item.get("error_rate_percent", 0.0)
            frames.append(
                CameraFrame(
                    timestamp=int(ts_val) if isinstance(ts_val, int) else 0,
                    latency_ms=float(lat_val) if isinstance(lat_val, (int, float)) else 0.0,
                    error_rate_percent=float(err_val) if isinstance(err_val, (int, float)) else 0.0,
                    frame_id=str(item.get("frame_id", "")),
                    objects=objects,
                )
            )
        return frames

    @staticmethod
    def load_fused_data(filepath: str) -> List[Optional[FusedFrame]]:
        """Loads fused data and returns a list of FusedFrame objects.

        Parses the JSONL file and maps each entry to a FusedFrame. This includes
        the fused object classification, confidence scores, and reference to
        the source data from both sensors.

        Args:
            filepath (str): Path to the fused_data.jsonl file.

        Returns:
            List[Optional[FusedFrame]]: A list of FusedFrame objects, with None
                at indices where the input was malformed.
        """
        raw_data = DataLoader.load_jsonl(filepath)
        frames: List[Optional[FusedFrame]] = []
        for item in raw_data:
            if item is None:
                frames.append(None)
                continue

            fused_objects = []
            objects_raw = item.get("fused_objects", [])
            if isinstance(objects_raw, list):
                for obj in objects_raw:
                    if isinstance(obj, dict):
                        source_classes_raw = obj.get("source_classes", {})
                        source_classes = {}
                        if isinstance(source_classes_raw, dict):
                            source_classes = {str(k): str(v) for k, v in source_classes_raw.items()}

                        cam_bbox_raw = obj.get("camera_bbox_3d", [])
                        cam_bbox: List[Dict[str, float]] = []
                        if isinstance(cam_bbox_raw, list):
                            for p in cam_bbox_raw:
                                if isinstance(p, dict):
                                    cam_bbox.append({k: float(v) for k, v in p.items() if isinstance(v, (int, float))})

                        radar_pts_raw = obj.get("radar_points", [])
                        radar_pts: List[Dict[str, float]] = []
                        if isinstance(radar_pts_raw, list):
                            for p in radar_pts_raw:
                                if isinstance(p, dict):
                                    radar_pts.append({k: float(v) for k, v in p.items() if isinstance(v, (int, float))})

                        conf_val = obj.get("fused_confidence", 0.0)
                        fused_objects.append(
                            FusedObject(
                                object_class=str(obj.get("class", "unknown")),
                                source_classes=source_classes,
                                camera_bbox_3d=cam_bbox,
                                radar_points=radar_pts,
                                fused_confidence=float(conf_val) if isinstance(conf_val, (int, float)) else 0.0,
                            )
                        )
            ts_val = item.get("timestamp", 0)
            lat_val = item.get("fusion_latency_ms", 0.0)
            jit_val = item.get("data_alignment_jitter_ms", 0.0)
            frames.append(
                FusedFrame(
                    timestamp=int(ts_val) if isinstance(ts_val, int) else 0,
                    latency_ms=float(lat_val) if isinstance(lat_val, (int, float)) else 0.0,
                    data_alignment_jitter_ms=float(jit_val) if isinstance(jit_val, (int, float)) else 0.0,
                    fused_objects=fused_objects,
                )
            )
        return frames

    @staticmethod
    def align_data(
        radar_data: List[Optional[RadarFrame]],
        camera_data: List[Optional[CameraFrame]],
        fused_data: List[Optional[FusedFrame]],
    ) -> List[AlignedFrame]:
        """Aligns radar and camera frames with fusion output using t+1 logic.

        The fusion module output at timestamp T corresponds to the sensor inputs
        from timestamp T - 100ms (the previous frame at 10 FPS). This method
        creates a synchronized view where each entry contains the fusion result
        and its corresponding source sensor frames.

        Args:
            radar_data (List[Optional[RadarFrame]]): List of loaded radar frames.
            camera_data (List[Optional[CameraFrame]]): List of loaded camera frames.
            fused_data (List[Optional[FusedFrame]]): List of loaded fusion frames.

        Returns:
            List[AlignedFrame]: A list of dictionaries, where each dictionary
                contains 'timestamp', 'fused', 'radar', and 'camera' keys.
        """
        radar_map = {item.timestamp: item for item in radar_data if item}
        camera_map = {item.timestamp: item for item in camera_data if item}

        aligned_frames: List[AlignedFrame] = []
        for fused_frame in fused_data:
            if not fused_frame:
                continue

            target_ts = fused_frame.timestamp - 100  # Assuming 10 FPS

            aligned_frames.append(
                {
                    "fused": fused_frame,
                    "radar": radar_map.get(target_ts),
                    "camera": camera_map.get(target_ts),
                    "timestamp": fused_frame.timestamp,
                }
            )

        return aligned_frames
