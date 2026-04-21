FROM python:3.10-slim

WORKDIR /app

COPY ml_api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY core/ ./core/
COPY utils/ ./utils/
COPY models/ ./models/
COPY ml_api/main.py ./main.py

EXPOSE 8080

# Cloud Run sets PORT (often 8080); must bind to $PORT per https://cloud.google.com/run/docs/container-contract
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
