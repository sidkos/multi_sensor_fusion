"""
Tests for the DataLoader class.
Validates data ingestion and temporal alignment.
"""

import logging
import os
import tempfile
from typing import Generator, List, Optional

import pytest

from src.data_loader import DataLoader, SensorData

logger = logging.getLogger(__name__)


@pytest.fixture
def temp_jsonl_file() -> Generator[str, None, None]:
    """Fixture to create a temporary JSONL file and delete it after use."""
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.remove(path)


@pytest.mark.infrastructure
def test_load_non_existent_file() -> None:
    """Test that loading a non-existent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="File not found"):
        DataLoader.load_jsonl("non_existent_file.jsonl")


@pytest.mark.infrastructure
@pytest.mark.parametrize(
    "content, expected_len, valid_indices",
    [
        (
            '{"timestamp": 1000, "valid": true}\n{"timestamp": 1100, "malformed": }\n{"timestamp": 1200, "valid": true}\n',
            3,
            [0, 2],
        ),
        (
            'invalid\n{"timestamp": 1000}\n',
            2,
            [1],
        ),
    ],
    ids=["malformed_middle", "malformed_start"],
)
def test_load_malformed_jsonl(temp_jsonl_file: str, content: str, expected_len: int, valid_indices: List[int]) -> None:
    """Test that malformed JSON lines are handled correctly."""
    with open(temp_jsonl_file, "w") as f:
        f.write(content)

    data = DataLoader.load_jsonl(temp_jsonl_file)
    assert len(data) == expected_len, f"Expected {expected_len} lines, got {len(data)}"
    for i in range(expected_len):
        if i in valid_indices:
            assert data[i] is not None, f"Expected line {i} to be valid"
        else:
            assert data[i] is None, f"Expected line {i} to be malformed (None)"
    logger.info("Successfully handled malformed JSON lines")


@pytest.mark.infrastructure
@pytest.mark.parametrize(
    "radar_data, camera_data, fused_data, expected_len, expected_matches",
    [
        (
            [{"timestamp": 1000, "val": 1}],
            [{"timestamp": 1000, "val": 2}],
            [{"timestamp": 1100}],
            1,
            [True],
        ),
        (
            [{"timestamp": 1000, "val": 1}],
            [{"timestamp": 1000, "val": 2}],
            [{"timestamp": 1100}, {"timestamp": 1200}],
            2,
            [True, False],
        ),
        (
            [],
            [],
            [{"timestamp": 1100}],
            1,
            [False],
        ),
    ],
    ids=["perfect_match", "partial_match", "no_match"],
)
def test_align_data_parametrization(
    radar_data: List[Optional[SensorData]],
    camera_data: List[Optional[SensorData]],
    fused_data: List[Optional[SensorData]],
    expected_len: int,
    expected_matches: List[bool],
) -> None:
    """Test data alignment with various scenarios."""
    aligned = DataLoader.align_data(radar_data, camera_data, fused_data)
    assert len(aligned) == expected_len, f"Expected {expected_len} aligned frames, got {len(aligned)}"

    for i, is_matched in enumerate(expected_matches):
        if is_matched:
            assert aligned[i]["radar"] is not None, f"Frame {i} should have radar data"
            assert aligned[i]["camera"] is not None, f"Frame {i} should have camera data"
        else:
            assert aligned[i]["radar"] is None, f"Frame {i} should NOT have radar data"
            assert aligned[i]["camera"] is None, f"Frame {i} should NOT have camera data"


@pytest.mark.infrastructure
def test_align_data_missing_timestamp_field() -> None:
    """Test that frames missing the timestamp field are skipped."""
    fused_data: List[Optional[SensorData]] = [{"invalid": "no_timestamp"}]
    aligned = DataLoader.align_data([], [], fused_data)
    assert len(aligned) == 0, f"Expected 0 aligned frames for missing timestamp, got {len(aligned)}"
