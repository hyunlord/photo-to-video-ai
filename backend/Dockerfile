# ============================================
# Photo to Video Backend Dockerfile
# Multi-stage build for development and production
# ============================================

# Stage 1: Base image with system dependencies
FROM python:3.11-slim AS base

# Prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Configure DNS for apt
RUN echo "nameserver 8.8.8.8" > /etc/resolv.conf && \
    echo "nameserver 1.1.1.1" >> /etc/resolv.conf

# Install system dependencies (with retry for network issues)
RUN for i in 1 2 3; do \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && break || sleep 5; \
    done

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ============================================
# Stage 2: Development image (with hot reload)
# ============================================
FROM base AS development

# Install development tools for hot reload
RUN pip install --no-cache-dir watchfiles

# Copy application code (will be overridden by volume mount in dev)
COPY . .

# Expose port
EXPOSE 8000

# Development command with hot reload
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# ============================================
# Stage 3: Production image
# ============================================
FROM base AS production

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Production command with multiple workers
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
