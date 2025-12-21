# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements files into the container at /app
COPY requirements.txt requirements-dev.txt ./

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt -r requirements-dev.txt

# Copy the current directory contents into the container at /app
COPY . .

# Generate synthetic data if not present (optional, but good for demo)
RUN python scripts/generate_data.py

# Run pytest when the container launches
ENV PYTHONPATH=/app
CMD ["pytest", "tests/", "-s"]
