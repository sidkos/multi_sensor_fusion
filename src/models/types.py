"""
Shared type definitions for the sensor fusion project.
"""

from typing import Dict, List, Union

# Generic JSON-compatible type
JSONValue = Union[str, int, float, bool, None, Dict[str, "JSONValue"], List["JSONValue"]]

# Representation of a single sensor data frame as a dictionary
SensorData = Dict[str, JSONValue]
