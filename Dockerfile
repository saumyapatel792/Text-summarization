# Use slim Python base image
FROM python:3.10-slim

# Set environment variables (Hugging Face Spaces runs on port 7860)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install PyTorch CPU version first to minimize image size
RUN pip install --no-cache-dir torch --extra-index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and files
COPY src/ /app/src/

# Expose the API port for Hugging Face Spaces
EXPOSE 7860

# Health check using the health endpoint, dynamically looking at PORT or fallback to 7860
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:${PORT:-7860}/health || exit 1

# Start FastAPI application, binding to port 7860 for Hugging Face Spaces
CMD uvicorn src.app:app --host 0.0.0.0 --port ${PORT:-7860}
