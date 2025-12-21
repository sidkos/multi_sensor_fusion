"""
Base models for the sensor fusion project.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class BaseFrame:
    """Common attributes for all sensor and fusion outputs.

    Attributes:
        timestamp (int): The synchronization timestamp for the data frame.
        latency_ms (float): The processing time taken to produce this frame.
        error_rate_percent (Optional[float]): The reported error probability
            for the sensor measurement.
    """

    timestamp: int
    latency_ms: float
    error_rate_percent: Optional[float] = None
