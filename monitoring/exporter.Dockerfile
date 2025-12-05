FROM python:3.10-slim

WORKDIR /app

COPY monitoring/exporter.py /app/exporter.py
COPY requirements.txt /app/

RUN pip install --no-cache-dir mlflow prometheus_client

CMD ["python", "exporter.py"]
