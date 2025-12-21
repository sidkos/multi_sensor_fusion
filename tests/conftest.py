"""
Shared pytest fixtures for the multi-sensor fusion test suite.
Loads datasets once per session to be used by all test modules.
"""

from typing import Dict, List, Optional, Union

import pytest

from src.data_loader import JSONValue
from src.data_utils import load_test_data


@pytest.fixture(scope="session")
def loaded_data() -> Dict[str, Union[List[Optional[JSONValue]], List[int], List[Optional[Dict[str, JSONValue]]]]]:
    """Session-scoped fixture that loads radar, camera, and fusion data.

    Provides expected timestamps for data drop rate calculations.

    Returns:
        dict: Dictionary containing loaded data and expected timestamp lists.
    """
    return load_test_data()
