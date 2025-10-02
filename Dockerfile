FROM python:3.11-slim

# 1) base setup
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app

# 2) system deps (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# 3) install python deps first to leverage docker layer cache
COPY requirements.txt .
RUN pip install -r requirements.txt

# 4) copy app code
COPY . .

# 5) start
# Use sh -c so ${PORT} is expanded; default to 8000 for local docker runs
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
