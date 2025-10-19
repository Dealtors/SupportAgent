FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y git unzip && rm -rf /var/lib/apt/lists/*

# Copy app code
COPY app /app/app
COPY scripts /app/scripts
COPY ml_orchestrator.zip /app/ml_orchestrator.zip
COPY reglament_ops_bundle.zip /app/reglament_ops_bundle.zip
COPY README.md /app/README.md

# Install reqs
RUN pip install --no-cache-dir -r app/requirements.txt

# Prepare data
RUN bash scripts/setup.sh

EXPOSE 8000
