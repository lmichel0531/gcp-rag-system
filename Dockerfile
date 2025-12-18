FROM python:3.11-slim

WORKDIR /app

# Install deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY app ./app

# Cloud Run sets PORT
ENV PORT=8080

# Start FastAPI
CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
