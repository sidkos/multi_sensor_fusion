"""
Fusion-specific models for the sensor fusion project.
"""

from dataclasses import dataclass, field
from typing import Dict, List

from src.models.base import BaseFrame


@dataclass
class FusedObject:
    """Represents an object after sensor fusion.

    Attributes:
        object_class (str): The final classification determined by the fusion module.
        source_classes (Dict[str, str]): A mapping of sensor modality (camera/radar)
            to its original classification result.
        camera_bbox_3d (List[Dict[str, float]]): The 3D bounding box projection
            from the camera sensor.
        radar_points (List[Dict[str, float]]): The clustered 3D points from
            the radar sensor.
        fused_confidence (float): The final confidence score of the fusion result.
    """

    object_class: str
    source_classes: Dict[str, str]
    camera_bbox_3d: List[Dict[str, float]] = field(default_factory=list)
    radar_points: List[Dict[str, float]] = field(default_factory=list)
    fused_confidence: float = 0.0


@dataclass
class FusedFrame(BaseFrame):
    """Data frame containing late fusion results and synchronization metrics.

    Attributes:
        data_alignment_jitter_ms (float): The measured temporal deviation in
            data synchronization.
        fused_objects (List[FusedObject]): List of objects after the fusion process.
    """

    data_alignment_jitter_ms: float = 0.0
    fused_objects: List[FusedObject] = field(default_factory=list)
