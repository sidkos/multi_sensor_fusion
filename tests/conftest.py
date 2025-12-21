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
REPORTS_DIR = "reports"
_log_file = os.path.join(REPORTS_DIR, "test_execution.log")


def pytest_sessionstart(session: pytest.Session) -> None:
    """Hook called before the test session starts.

    Ensures a clean environment by removing old report files and initializing the log file.
    """
    global _test_results
    _test_results = []

    # Ensure reports directory exists and is clean
    if os.path.exists(REPORTS_DIR):
        try:
            import shutil

            shutil.rmtree(REPORTS_DIR)
            print(f"Cleaned up old reports directory: {REPORTS_DIR}")
        except Exception as e:
            print(f"Warning: Failed to cleanup {REPORTS_DIR}: {e}")

    os.makedirs(REPORTS_DIR, exist_ok=True)

    # Initialize test_execution.log with a header
    with open(_log_file, "w") as f:
        f.write("============================= test session starts ==============================\n")


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item: Any) -> None:
    """Store the docstring of the test item before it runs."""
    doc = getattr(item.obj, "__doc__", "")
    item.user_properties.append(("docstring", str(doc) if doc else ""))


def pytest_runtest_logreport(report: Any) -> None:
    """Collect test results as they happen and write them to the log file."""
    if report.when == "call":
        _test_results.append(report)

    # Write detailed report info to test_execution.log
    with open(_log_file, "a") as f:
        if report.when == "setup":
            f.write(f"\nTestcase: {str(report.nodeid)}\n")

        if report.when == "call":
            # Extract docstring from report if available
            doc_str = ""
            for prop in report.user_properties:
                if isinstance(prop, tuple) and len(prop) >= 2 and prop[0] == "docstring":
                    doc_str = str(prop[1])
                    break
            if doc_str:
                f.write(f"Description: {doc_str.strip()}\n")

            status = str(report.outcome).upper()
            f.write(f"Result: {status}\n")

            # Capture logs from the call if any
            for section in report.sections:
                if "live log call" in section[0] or "Captured log call" in section[0]:
                    f.write(f"-------------------------------- {section[0]} ---------------------------------\n")
                    f.write(f"{section[1]}\n")

            if report.failed:
                f.write("___________________________________ FAILURES ___________________________________\n")
                # Extract only the assertion message, avoiding the full traceback and diffs
                if hasattr(report.longrepr, "reprcrash"):
                    f.write(f"{report.longrepr.reprcrash.message}\n")
                else:
                    # Fallback to a cleaner version of the long representation
                    f.write(f"{str(report.longrepr).split('AssertionError: ')[-1]}\n")


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
            output_dir=REPORTS_DIR,
        )
        logger.info("Report generation complete.")

        # Append summary to test_execution.log
        with open(_log_file, "a") as f:
            f.write(
                f"\n========================== {passed} passed, {failed} failed in summary =========================\n"
            )
            f.write(f"Status: {test_results['status']}\n")

    except Exception as e:
        logger.error(f"Failed to generate report after tests: {e}")
