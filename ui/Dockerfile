FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

ENV PORT=8080
CMD exec streamlit run app.py --server.port=${PORT} --server.address=0.0.0.0
