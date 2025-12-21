"""
Shared pytest fixtures for the multi-sensor fusion test suite.
Loads datasets once per session to be used by all test modules.
"""

from typing import Any, Dict

import pytest

from src.data_utils import load_test_data


@pytest.fixture(scope="session")
def loaded_data() -> Dict[str, Any]:
    """Session-scoped fixture that loads radar, camera, and fusion data.

    Provides expected timestamps for data drop rate calculations.

    Returns:
        dict: Dictionary containing loaded data and expected timestamp lists.
    """
    return load_test_data()
