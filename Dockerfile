FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY backend ./backend
COPY templates ./templates
COPY static ./static

ENV PORT=5000
ENV FLASK_ENV=production
ENV MODEL_ASSET_DIR=/models
ENV MAX_UPLOAD_MB=8

EXPOSE 5000

CMD ["sh", "-c", "gunicorn --workers=1 --threads=2 --bind 0.0.0.0:${PORT} 'backend.app:create_app()'"]
