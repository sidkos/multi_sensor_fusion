from dataclasses import dataclass, field
from typing import Dict, List

from src.models.base import BaseFrame


@dataclass
class CameraObject:
    object_class: str
    bbox_3d: List[Dict[str, float]]


@dataclass
class CameraFrame(BaseFrame):
    """Specific to visual detections and bounding boxes."""

    frame_id: str = ""
    objects: List[CameraObject] = field(default_factory=list)
