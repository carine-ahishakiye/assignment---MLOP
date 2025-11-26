#  Builder - Install dependencies and compile packages
FROM python:3.9-slim as builder

# Set build-time environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Runtime - Minimal production image

FROM python:3.9-slim

# Metadata
LABEL maintainer="your.email@example.com"
LABEL version="1.0"
LABEL description="Bird Sound Classification ML API with retraining capabilities"

# Set runtime environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FLASK_APP=api.py \
    FLASK_ENV=production \
    DEBIAN_FRONTEND=noninteractive \
    PATH="/opt/venv/bin:$PATH"

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Audio processing libraries
    libsndfile1 \
    ffmpeg \
    libsndfile1-dev \
    # Monitoring and health checks
    curl \
    wget \
    # Timezone data
    tzdata \
    # Cleanup
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && apt-get autoremove -y

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Set working directory
WORKDIR /app

# Create application directories with proper structure
RUN mkdir -p \
    /app/src \
    /app/models \
    /app/templates \
    /app/static \
    /app/uploads \
    /app/retrain_data \
    /app/logs \
    /app/data/train \
    /app/data/test \
    /app/data/raw \
    && chmod -R 755 /app

# Copy application files
COPY src/ ./src/
COPY templates/ ./templates/
COPY api.py .


# Create non-root user for security
RUN groupadd -r appuser && \
    useradd -r -g appuser -u 1000 -m -s /bin/bash appuser && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose application port
EXPOSE 5000

# Health check configuration
HEALTHCHECK --interval=30s \
    --timeout=10s \
    --start-period=60s \
    --retries=3 \
    CMD curl -f http://localhost:5000/api/health || exit 1

# Startup Options


# Development with auto-reload (
# CMD ["python", "api.py"]

#  Production with Gunicorn (recommended)
CMD ["gunicorn", \
    "--bind", "0.0.0.0:5000", \
    "--workers", "4", \
    "--threads", "2", \
    "--worker-class", "gthread", \
    "--worker-tmp-dir", "/dev/shm", \
    "--timeout", "120", \
    "--graceful-timeout", "30", \
    "--keep-alive", "5", \
    "--log-level", "info", \
    "--access-logfile", "-", \
    "--error-logfile", "-", \
    "--access-logformat", "%({X-Forwarded-For}i)s %(l)s %(u)s %(t)s '%(r)s' %(s)s %(b)s '%(f)s' '%(a)s' %(D)s", \
    "api:app"]

#  Production with uWSGI 
# CMD ["uwsgi", "--http", "0.0.0.0:5000", "--wsgi-file", "api.py", "--callable", "app", "--processes", "4", "--threads", "2"]

# Usage Instructions

# 
# BUILD:
#   docker build -t bird-classifier:latest .
#   docker build -t bird-classifier:v1.0 --build-arg VERSION=1.0 .
#
# RUN:
#   docker run -d -p 5000:5000 \
#     -v $(pwd)/models:/app/models \
#     -v $(pwd)/uploads:/app/uploads \
#     -v $(pwd)/retrain_data:/app/retrain_data \
#     --name bird-classifier \
#     bird-classifier:latest
#
# RUN WITH ENVIRONMENT VARIABLES:
#   docker run -d -p 5000:5000 \
#     -e MODEL_PATH=/app/models/final_model.h5 \
#     -e SCALER_PATH=/app/models/scaler.pkl \
#     -e ENCODER_PATH=/app/models/label_encoder.pkl \
#     -v $(pwd)/models:/app/models \
#     --name bird-classifier \
#     bird-classifier:latest
#
# DEBUGGING:
#   docker run -it --rm \
#     -p 5000:5000 \
#     -v $(pwd)/models:/app/models \
#     bird-classifier:latest /bin/bash
#
# LOGS:
#   docker logs -f bird-classifier
#
# HEALTH CHECK:
#   docker inspect --format='{{json .State.Health}}' bird-classifier
#
# RESOURCE LIMITS:
#   docker run -d -p 5000:5000 \
#     --memory="4g" \
#     --cpus="2" \
#     -v $(pwd)/models:/app/models \
#     bird-classifier:latest