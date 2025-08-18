# OpenScholar API - GPU-Optimized Dockerfile for RunPod
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    build-essential \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create application directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --upgrade pip setuptools wheel
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install -r requirements.txt

# Install RunPod SDK for serverless support
RUN pip3 install runpod

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/model_cache /app/logs

# Set permissions
RUN chmod +x /app/main.py /app/handler.py

# Initialize system on build (warm start)
RUN python3 -c "import os; os.environ['OPENSCHOLAR_API_KEY'] = 'build-key'; from main import safe_initialize_system; import asyncio; asyncio.run(safe_initialize_system())" || echo "Warm start failed, will initialize at runtime"

# Expose port (for persistent deployment)
EXPOSE 8002

# Health check (for persistent deployment) 
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8002/health || exit 1

# Default to handler for serverless, can override for persistent
CMD ["python3", "-c", "import runpod; from handler import runpod_handler; runpod.serverless.start({'handler': runpod_handler})"]