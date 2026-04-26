# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY serve/ ./serve/
COPY model_onnx/ ./model_onnx/

EXPOSE 8000 3000

CMD ["uvicorn", "serve.fastapi_app_onnx:app", "--host", "0.0.0.0", "--port", "8000"]