"""
Camera-specific models for the sensor fusion project.
"""

from dataclasses import dataclass, field
from typing import Dict, List

from src.models.base import BaseFrame


@dataclass
class CameraObject:
    """Represents a single object detected by the camera.

    Attributes:
        object_class (str): The predicted class of the object (e.g., 'car').
        bbox_3d (List[Dict[str, float]]): A list of 4 points representing the
            corners of the 3D bounding box projection.
    """

    object_class: str
    bbox_3d: List[Dict[str, float]]


@dataclass
class CameraFrame(BaseFrame):
    """Data frame containing camera detections and metadata.

    Attributes:
        frame_id (str): Reference to the physical image file.
        objects (List[CameraObject]): List of objects detected in this frame.
    """

    frame_id: str = ""
    objects: List[CameraObject] = field(default_factory=list)
