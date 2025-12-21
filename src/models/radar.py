from dataclasses import dataclass, field
from typing import List

from src.models.base import BaseFrame


@dataclass
class RadarPoint:
    x: float
    y: float
    z: float
    signal_strength: float
    doppler_velocity: float
    object_class: str


@dataclass
class RadarFrame(BaseFrame):
    """Specific to radar points and object classification."""

    points: List[RadarPoint] = field(default_factory=list)
