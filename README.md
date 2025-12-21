# Sensor Fusion Automation Test Suite

This project validates the performance and consistency of a sensor fusion module using simulated radar, camera, and fusion datasets.

## Features
- **Object-Oriented Domain Models**: Uses Python `dataclasses` and inheritance to model `RadarFrame`, `CameraFrame`, and `FusedFrame`, ensuring type safety and code clarity.
- **Robust Data Loading**: Handles `.jsonl` files with built-in resilience for malformed lines and missing fields.
- **Temporal Data Alignment**: Automatically aligns sensor data with fusion output using $t+1$ alignment logic (fusion at $T$ corresponds to sensors at $T-100ms$).
- **Strict KPI Validation**:
    - **Latency**: Monitors per-frame processing time (Radar/Camera < 50ms; Fusion < 100ms).
    - **Error Rate**: Validates sensor-reported error probabilities (< 0.1%).
    - **Data Drop Rate**: Detects missing or corrupted frames over the 10s recording window (< 0.5%).
    - **Alignment Jitter**: Measures the temporal precision of data synchronization (<= 5ms).
    - **Decision Consistency**: Compares fused classifications against source sensor modality decisions (>= 0.95).
- **Advanced Performance Metrics (Additional KPIs)**:
    - **Confidence Stability Score**: Calculates the standard deviation of fusion confidence to detect algorithmic instability.
    - **Spatial Alignment Error**: Measures the Euclidean distance between camera 3D bounding box centers and radar point clusters to detect calibration drift.
    - **Sensor Contribution Balance**: Analyzes the ratio of objects fused from both sensors vs. single-sensor detections to identify "sensor starvation."
- **Automated Reporting**: Generates high-resolution visualizations and structured JSON statistics for system-level health assessment.

## Project Structure
- `src/models/`: Strongly typed data structures for all sensor modalities.
- `src/data_loader.py`: Logic for ingestion, parsing, and temporal alignment.
- `src/kpi_calculator.py`: Implementation of all mandatory and additional KPI algorithms.
- `src/report_generator.py`: Visualization engine and statistics aggregator.
- `tests/`: Comprehensive Pytest suite, categorized into infrastructure (unit) and validation (performance) tests.
- `scripts/`: Quality assurance tools, including the `precommit.sh` static analysis suite.
- `data/`: Recording directory containing the `*_with_kpis.jsonl` datasets.

## How to Run Locally

### 1. Environment Setup
The project requires **Python 3.11+**. It is recommended to use a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt -r requirements-dev.txt
```

### 2. IDE Configuration (PyCharm / VS Code)
This project is optimized for **pytest**.
- **PyCharm**: Go to `Settings` > `Tools` > `Python Integrated Tools` and set `Default test runner` to `pytest`.
- **VS Code**: The project includes a `pytest.ini` which should be automatically detected by the Python extension.

### 3. Executing the Validation Suite
To run the full battery of tests and capture the results for analysis:

```bash
# Run all tests and save detailed per-frame violations to a log file
# The -s flag allows seeing live logs, and tee captures everything
PYTHONPATH=. pytest tests/ -s | tee test_execution.log
```

**Understanding the Output**:
- **Infrastructure Tests**: Fast unit tests that validate the calculation logic and data loader resilience.
- **Validation Tests**: Performance tests that check the actual datasets against the exercise thresholds.
- **Failures**: If a test fails (e.g., `test_camera_latency`), it will list **every frame** that violated the threshold, not just the first one.

### 4. Running Specific KPI Groups
You can target specific areas using pytest markers:
```bash
pytest -m radar          # Only Radar KPIs
pytest -m camera         # Only Camera KPIs
pytest -m fusion         # Only Fusion KPIs
pytest -m infrastructure # Only core logic unit tests
pytest -m additional     # Only the 3 proposed additional KPIs
```

## Visualization and Reporting

The reporting engine is automatically triggered after the test execution. To run the tests and generate the reports:

```bash
PYTHONPATH=. pytest tests/
```

### Generated Artifacts:
1.  **`latency_report.png`**:
    - A temporal line chart comparing Radar, Camera, and Fusion latencies.
    - Includes horizontal dashed lines for the 50ms (sensor) and 100ms (fusion) mandatory limits.
2.  **`performance_summary.png`**:
    - **Left**: A histogram of the Decision Consistency score distribution across the session.
    - **Right**: A temporal plot of the Spatial Alignment Error, helping identify calibration drift.
3.  **`summary_stats.json`**:
    - A structured summary of the session, including average latencies, stability scores, sensor contribution balance, and **test execution results** (pass/fail status).

## Static Code Analysis
Before contributing, run the pre-commit suite to ensure compliance with the project's strict quality standards (Strict Mypy, Black, Flake8, Bandit):

```bash
./scripts/precommit.sh
```

## KPI Test Frequency Justification
To optimize the multidisciplinary development lifecycle at Niart, we apply a tiered testing strategy:

| Frequency | KPIs Included | Justification |
| :--- | :--- | :--- |
| **Every Commit** | Latency, Drop Rate, Consistency | **Safety Critical**: These metrics catch immediate regressions in real-time performance and core decision logic. |
| **Nightly** | Error Rate, Alignment Jitter | **Resource/Noise**: These metrics require longer durations or are sensitive to environmental noise; nightly runs provide stable trends. |
| **Pre-release** | Spatial Error, Stability, Balance | **System Tuning**: These catch long-term drift or architectural imbalances that are typically addressed during integration phases. |
