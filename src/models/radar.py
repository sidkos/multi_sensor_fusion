"""
Radar-specific models for the sensor fusion project.
"""

from dataclasses import dataclass, field
from typing import List

from src.models.base import BaseFrame


@dataclass
class RadarPoint:
    """Represents a single clustered 3D point from the radar.

    Attributes:
        x (float): X-coordinate in the sensor frame.
        y (float): Y-coordinate in the sensor frame.
        z (float): Z-coordinate in the sensor frame.
        signal_strength (float): The reflected power of the point.
        doppler_velocity (float): The relative velocity of the detected cluster.
        object_class (str): The classification result from the radar sensor.
    """

    x: float
    y: float
    z: float
    signal_strength: float
    doppler_velocity: float
    object_class: str


@dataclass
class RadarFrame(BaseFrame):
    """Data frame containing radar point clouds and metadata.

    Attributes:
        points (List[RadarPoint]): List of radar points detected in this frame.
    """

    points: List[RadarPoint] = field(default_factory=list)
