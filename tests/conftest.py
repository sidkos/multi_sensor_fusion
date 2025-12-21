"""
Shared pytest fixtures for the multi-sensor fusion test suite.
Loads datasets once per session to be used by all test modules.
"""

import logging
import os
from typing import Any, Dict

import pytest

from src.data_utils import load_test_data
from src.models import CameraFrame, FusedFrame, RadarFrame
from src.report_generator import generate_report

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def loaded_data() -> Dict[str, Any]:
    """Session-scoped fixture that loads radar, camera, and fusion data.

    Provides expected timestamps for data drop rate calculations.

    Returns:
        dict: Dictionary containing loaded data and expected timestamp lists.
    """
    return load_test_data()


# Global list to store test results
_test_results: Any = []


def pytest_sessionstart(session: pytest.Session) -> None:
    """Hook called before the test session starts.

    Ensures a clean environment by removing old report files.
    """
    report_files = ["latency_report.png", "performance_summary.png", "summary_stats.json"]
    for f in report_files:
        if os.path.exists(f):
            try:
                os.remove(f)
                logger.info(f"Removed old report file: {f}")
            except Exception as e:
                logger.warning(f"Failed to remove {f}: {e}")


def pytest_runtest_logreport(report: Any) -> None:
    """Collect test results as they happen."""
    if report.when == "call":
        _test_results.append(report)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Hook called after the entire test session finishes.

    Generates the performance report using the loaded datasets and test statistics.
    """
    logger.info("Test session finished. Generating performance report...")

    passed = sum(1 for r in _test_results if r.passed)
    failed = sum(1 for r in _test_results if r.failed)
    skipped = sum(1 for r in _test_results if r.skipped)
    total = len(_test_results)

    test_results = {
        "exit_status": exitstatus,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "total": total,
        "status": "PASS" if exitstatus == 0 else "FAIL",
    }

    try:
        data = load_test_data()
        radar_data = [f for f in data.get("radar", []) if isinstance(f, RadarFrame)]
        camera_data = [f for f in data.get("camera", []) if isinstance(f, CameraFrame)]
        fused_data = [f for f in data.get("fused", []) if isinstance(f, FusedFrame)]

        generate_report(
            radar_data=radar_data,
            camera_data=camera_data,
            fused_data=fused_data,
            test_results=test_results,
        )
        logger.info("Report generation complete.")
    except Exception as e:
        logger.error(f"Failed to generate report after tests: {e}")
