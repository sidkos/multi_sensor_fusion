#!/bin/bash

# Pre-commit script to run all checks

set -e  # Exit on first failure

echo "Fixing imports with isort..."
isort .

echo "Formatting with black..."
black .

echo "Running black check..."
black --check .

echo "Running isort..."
isort . --check-only

echo "Running pycodestyle..."
pycodestyle .

echo "Running flake8..."
flake8 .

echo "Running bandit..."
bandit -r .

echo "Running radon..."
radon cc **/*.py

echo "Running yamllint..."
yamllint . --no-warnings

echo "Running mypy..."
mypy .

echo "All checks passed!"
