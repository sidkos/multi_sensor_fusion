from dataclasses import dataclass, field
from typing import Dict, List

from src.models.base import BaseFrame


@dataclass
class FusedObject:
    object_class: str
    source_classes: Dict[str, str]
    camera_bbox_3d: List[Dict[str, float]] = field(default_factory=list)
    radar_points: List[Dict[str, float]] = field(default_factory=list)
    fused_confidence: float = 0.0


@dataclass
class FusedFrame(BaseFrame):
    """Specific to fusion logic and alignment jitter."""

    data_alignment_jitter_ms: float = 0.0
    fused_objects: List[FusedObject] = field(default_factory=list)
