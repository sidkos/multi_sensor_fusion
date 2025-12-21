# Sensor Fusion Automation Test Suite

## Project Overview
This project provides a robust, Python-based automation framework designed to validate the performance, consistency, and spatial integrity of a multidisciplinary sensor fusion module. By processing high-fidelity datasets from Radar, Camera, and Late Fusion outputs, the suite identifies performance bottlenecks, temporal drifts, and semantic mismatches in real-time perception systems.

The framework is built with scalability and CI/CD integration at its core, utilizing an object-oriented domain model and a tiered testing strategy to provide rapid feedback for safety-critical metrics while maintaining deep diagnostic visibility.

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
- **Advanced Performance Metrics (Extended KPIs)**:
    - **Confidence Stability Score**: Calculates the standard deviation of fusion confidence to detect algorithmic instability.
    - **Spatial Alignment Error**: Measures the Euclidean distance between camera 3D bounding box centers and radar point clusters to detect calibration drift.
    - **Sensor Contribution Balance**: Analyzes the ratio of objects fused from both sensors vs. single-sensor detections to identify "sensor starvation."
- **Automated Reporting**: Generates high-resolution visualizations and structured JSON statistics for system-level health assessment.
- **Unified Test Logging**: Automatically captures all test execution details, including per-frame performance violations and captured logs, into a centralized `test_execution.log` file.

## CI/CD Pipeline
The project features a fully automated CI/CD pipeline implemented via GitHub Actions, designed to enforce high code quality and validate performance on every change.

### Workflow Stages:
1.  **Static Analysis**: Runs a comprehensive suite of linters and security scanners (`Black`, `Isort`, `Flake8`, `Bandit`, `Mypy`, `Radon`, `Yamllint`).
2.  **Unit Tests (Infrastructure)**: Validates the core logic of the data loader and KPI calculators in a clean environment.
3.  **Docker Build**: Packages the application into a production-ready container and stores it in the **GitHub Container Registry (GHCR)**.
4.  **Validation Tests**: Pulls the latest image and executes the performance test suite against the real datasets, ensuring the environment is fully operational.

### Accessing Results & Artifacts:
Performance reports and detailed logs are automatically generated and uploaded for every CI run:
- **Remote Results**: You can access the artifacts from the latest runs on the [Actions CI/CD Page](https://github.com/sidkos/multi_sensor_fusion/actions/workflows/ci_cd.yml). Select a specific run and scroll down to the "Artifacts" section to download the `validation-test-reports`.
- **Local Results**: After running the suite locally, all artifacts (plots, JSON stats, and the execution log) are available in the `reports/` directory.

## Project Structure
- `src/models/`: Strongly typed data structures for all sensor modalities.
- `src/data_loader.py`: Logic for ingestion, parsing, and temporal alignment.
- `src/kpi_calculator.py`: Implementation of all mandatory and extended KPI algorithms.
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
To run the full battery of tests and automatically generate performance reports:

```bash
PYTHONPATH=. pytest tests/
```

**Understanding the Output**:
- **Automatic Management**: The test suite automatically cleans up previous reports before starting and generates fresh ones upon completion.
- **Infrastructure Tests**: Fast unit tests that validate the calculation logic and data loader resilience.
- **Validation Tests**: Performance tests that check the actual datasets against the exercise thresholds.
- **Failures**: If a test fails (e.g., `test_camera_latency`), it will list **every frame** that violated the threshold, providing a complete violation log in `test_execution.log`.
- **Full Execution Log**: Detailed logs, including captured `logger.info()` messages and full failure tracebacks, are stored in `test_execution.log` for post-run analysis.

### 4. Running Specific KPI Groups
You can target specific areas using pytest markers:
```bash
pytest -m radar          # Only Radar KPIs
pytest -m camera         # Only Camera KPIs
pytest -m fusion         # Only Fusion KPIs
pytest -m infrastructure # Only core logic unit tests
pytest -m extended_kpis # Only the 3 proposed advanced KPIs
```

## Visualization and Reporting

The reporting engine is automatically triggered after the test execution. To run the tests and generate the reports:

```bash
PYTHONPATH=. pytest tests/
```

### Generated Artifacts (in `reports/` folder):

The following reports are automatically generated to provide a multi-dimensional view of system health:

1.  **`radar_latency.png`**: 
    - **What it represents**: A temporal analysis of the time taken (in milliseconds) by the radar sensor to process each frame.
    - **Key Indicators**: The red dashed line at 50ms represents the mandatory performance threshold. Spikes above this line indicate processing bottlenecks.
2.  **`camera_latency.png`**:
    - **What it represents**: A temporal analysis of the camera sensor's per-frame processing latency.
    - **Key Indicators**: Similar to radar, it monitors adherence to the 50ms real-time constraint.
3.  **`fusion_latency.png`**:
    - **What it represents**: The end-to-end processing time for the late fusion module.
    - **Key Indicators**: The performance limit is set at 100ms. This includes the time taken to align sensor data and perform classification fusion.
4.  **`camera_radar_latency.png`**:
    - **What it represents**: A comparative view of both primary sensors on a single axis.
    - **Key Indicators**: Helps identify if latency spikes are correlated across sensors (suggesting system-wide resource contention) or isolated to a specific modality.
5.  **`decision_consistency.png`**:
    - **What it represents**: A distribution (histogram) of the consistency scores across all fused objects.
    - **Key Indicators**: Shows how often the fusion module's final classification aligns with its source sensors (Camera/Radar). High frequency near 1.0 indicates a stable and reliable decision logic.
6.  **`spatial_alignment.png`**:
    - **What it represents**: The Euclidean distance between camera 3D bounding box centers and radar point clusters over time.
    - **Key Indicators**: Monitors "spatial drift." An increasing trend or high variance may indicate calibration misalignment between the sensors.
7.  **`summary_stats.json`**:
    - **What it represents**: A structured JSON file containing aggregate metrics (average latencies, stability scores, contribution balance) and the overall test session exit status.
8.  **`test_execution.log`**:
    - **What it represents**: A comprehensive audit trail of the test session, documenting every test case, its description, and a detailed list of every per-frame violation detected.

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

## Future Roadmap & Improvements
To further evolve the automation framework and perception system, the following enhancements are planned:
1.  **Semantic Synchronization**: Enhance the `Decision Consistency` KPI to include object-level tracking, allowing for "Object History" validation across multiple frames.
2.  **3D Geometry Visualization**: Implement an interactive 3D playback tool (e.g., using `Open3D` or `PyVista`) to visualize bounding box overlaps and radar point clouds in the sensor's coordinate system.
3.  **Stress Testing Engine**: Integrate a synthetic data perturbation tool to simulate extreme weather conditions (e.g., sensor noise, occlusion) and validate fusion robustness under stress.
4.  **Hardware-in-the-loop (HIL) Integration**: Adapt the validation suite to interface with real-time hardware targets for low-level latency profiling and memory usage monitoring.
5.  **AI-Driven Anomaly Detection**: Implement unsupervised learning models to automatically detect "unusual" sensor patterns or corner cases that violate safety-critical assumptions but may pass static KPI thresholds.
6.  **Real-time Event Streaming**: Transition from file-based ingestion to processing live streams of events (e.g., via Kafka) to enable real-time performance monitoring.
