# Use slim Python base image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install PyTorch CPU version first to minimize image size
RUN pip install --no-cache-dir torch --extra-index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and files
COPY src/ /app/src/

# Expose the API port
EXPOSE 8000

# Health check using the health endpoint, dynamically looking at PORT or fallback to 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request, os; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\", 8000)}/health')" || exit 1

# Start FastAPI application, binding to the dynamic PORT environment variable
CMD uvicorn src.app:app --host 0.0.0.0 --port ${PORT:-8000}

