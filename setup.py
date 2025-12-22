from setuptools import find_packages, setup

with open("requirements.txt") as f:
    install_requires = f.read().splitlines()

setup(
    name="multi_sensor_fusion",
    version="0.1.0",
    description="Sensor Fusion Automation Test Suite for validating performance and consistency.",
    packages=find_packages(where="."),
    install_requires=install_requires,
    python_requires=">=3.11",
    include_package_data=True,
)
