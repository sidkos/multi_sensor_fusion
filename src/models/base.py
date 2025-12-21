from dataclasses import dataclass
from typing import Optional


@dataclass
class BaseFrame:
    """Common attributes for all sensor and fusion outputs."""

    timestamp: int
    latency_ms: float
    error_rate_percent: Optional[float] = None
