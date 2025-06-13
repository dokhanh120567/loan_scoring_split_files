FROM tiangolo/uvicorn-gunicorn-fastapi:python3.10

WORKDIR /app

# Dockerfile
COPY . .
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH=/app

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
