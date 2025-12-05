FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
  && rm -rf /var/lib/apt/lists/*

COPY validator/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

#  Correct copy for GE config
COPY validator/gx /app/gx

# Debug check (shows gx folder at build time)
RUN ls -R /app/gx

COPY validator/validate_inventory_with_gx.py /app/
COPY validator/clean_preprocess.py /app/
COPY validator/openlineage_emitter.py /app/
COPY validator/data_ingestion.py /app/
COPY validator/clickhouse_feature3.py /app/
COPY validator/clickhouse_feature_top10.py /app/

COPY orchestrate_pipeline.py /app/


ENV PYTHONUNBUFFERED=1

CMD ["python", "orchestrate_pipeline.py"]

