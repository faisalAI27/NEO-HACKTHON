# MOSAIC Backend Dockerfile
# Multi-stage build for survival prediction API

# ==============================================================================
# Stage 1: Builder - Install Python dependencies
# ==============================================================================
FROM python:3.10-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ==============================================================================
# Stage 2: Runtime - Slim production image
# ==============================================================================
FROM python:3.10-slim as runtime

WORKDIR /app

# Install runtime system dependencies
# CRITICAL: OpenSlide for WSI processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    openslide-tools \
    libopenslide-dev \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install openslide-python in runtime (needs headers)
RUN pip install --no-cache-dir openslide-python

# Copy application code
COPY src/ ./src/
COPY configs/ ./configs/

# Create directories for data and checkpoints (to be mounted in production)
RUN mkdir -p data/raw data/processed outputs checkpoints

# Copy model checkpoints if available (optional - will be mounted in production)
# Note: checkpoints directory is created above, volumes can override it
COPY --chown=root:root configs/ ./configs/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV MOSAIC_CHECKPOINT_DIR=/app/checkpoints
ENV MOSAIC_WSI_DIR=/app/data/raw/svs

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Run the API server
ENTRYPOINT ["uvicorn", "src.serving.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Default arguments (can be overridden)
CMD ["--workers", "1"]
