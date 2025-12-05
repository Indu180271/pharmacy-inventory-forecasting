FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# REQUIRED for FastAPI file uploads
RUN pip install --no-cache-dir python-multipart
# REQUIRED for Monitoring 
RUN pip install prometheus-fastapi-instrumentator


COPY . .

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

