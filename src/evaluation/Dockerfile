FROM python:3.11-slim

WORKDIR /app

# Install system deps just like pipeline container
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY src /app/src
COPY models /app/models

ENV PYTHONPATH="/app"
ENV MLFLOW_TRACKING_URI="http://mlflow:5002"

CMD ["python", "src/evaluation/evaluate_xgb.py"]

