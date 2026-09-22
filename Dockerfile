FROM python:3.14-slim

# xgboost's prebuilt wheel dynamically links against libgomp (OpenMP).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY models/ models/
COPY reports/metrics/ reports/metrics/

RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

ENV PYTHONUNBUFFERED=1
EXPOSE 8080

# Render's free tier is 512MB, so one worker; --threads handles concurrency.
# exec replaces the shell with gunicorn (PID 1) so it receives SIGTERM directly.
CMD ["sh", "-c", "exec gunicorn \"src.api:create_app()\" --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 60"]
