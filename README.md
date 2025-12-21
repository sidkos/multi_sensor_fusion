# Sensor Fusion Automation Test Suite

This project validates the performance and consistency of a sensor fusion module using simulated radar, camera, and fusion datasets.

## Features
- **Data Loading**: Handles `.jsonl` files and aligns sensor data with fusion output ($t+1$ alignment).
- **KPI Validation**:
    - Latency (Radar, Camera < 50ms; Fusion < 100ms)
    - Error Rate (< 0.1%)
    - Data Drop Rate (< 0.5%)
    - Alignment Jitter (<= 5ms)
    - Decision Consistency (>= 0.95)
- **Additional KPIs**:
    - **Confidence Stability Score**: Measures variance in fused confidence.
    - **Spatial Alignment Error**: Measures distance between camera and radar object centers.
    - **Sensor Contribution Balance**: Percentage of objects fused from both vs single sensors.
- **Reporting**: Generates a latency performance plot and summary statistics.

## Project Structure
- `src/`: Core logic (Data loader, KPI calculator, Report generator).
- `tests/`: Pytest suite for KPI validation.
- `scripts/`: Utility scripts (e.g., data generation).
- `data/`: Placeholder for datasets.

## How to Run Locally

### 1. IDE Configuration (PyCharm / VS Code)
This project uses **pytest**. If you are using PyCharm, ensure it is configured to use pytest as the default test runner:
1. Go to **Settings/Preferences** > **Tools** > **Python Integrated Tools**.
2. Under **Testing**, set **Default test runner** to **pytest**.

### 2. Direct Execution
Ensure you have Python 3.11 installed.

```bash
# Install dependencies
pip install -r requirements.txt -r requirements-dev.txt

# (Optional) Generate synthetic data
python scripts/generate_data.py

# Run tests
pytest tests/ -s

# Run specific KPI tests
pytest -m radar
pytest -m camera
pytest -m fusion
pytest -m additional
pytest -m stability
pytest -m contribution
pytest -m spatial

# Generate report
python src/report_generator.py
```

### 2. Using Docker
You can run the entire suite inside a container.

```bash
# Build the image
docker build -t fusion-test .

# Run the tests
docker run fusion-test
```

## KPI Test Frequency Justification
- **Every Commit**: Latency, Data Drop Rate, Decision Consistency. These are critical for immediate feedback on regressions.
- **Nightly**: Error Rate, Alignment Jitter. These may fluctuate slightly due to environment noise.
- **Pre-release**: Spatial Alignment, Stability. These catch long-term drift or calibration issues.
