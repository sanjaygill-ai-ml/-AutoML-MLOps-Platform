FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --prefer-binary

COPY src/ src/
COPY config/ config/
COPY artifacts/ artifacts/
COPY pipeline/ pipeline/

EXPOSE 8000

CMD ["uvicorn", "src.deployment.api:app", "--host", "0.0.0.0", "--port", "8000"]