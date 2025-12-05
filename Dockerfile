FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir \
    mlflow \
    xgboost \
    clickhouse-connect \
    pandas \
    scikit-learn

CMD ["python", "src/model/train_xgb_model.py"]

