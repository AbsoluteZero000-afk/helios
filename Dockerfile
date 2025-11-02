# Multi-stage Dockerfile for Helios v3
# Production-ready Python 3.11 with minimal attack surface

# Build stage
FROM python:3.11-slim as builder

# Set build arguments
ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHONUNBUFFERED=1
ARG PYTHONDONTWRITEBYTECODE=1

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    make \
    pkg-config \
    libffi-dev \
    libssl-dev \
    libpq-dev \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib from source
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz \
    && tar -xzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib \
    && ./configure --prefix=/usr/local \
    && make \
    && make install \
    && cd .. \
    && rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements.txt

# Runtime stage
FROM python:3.11-slim as runtime

# Set runtime arguments
ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHONUNBUFFERED=1
ARG PYTHONDONTWRITEBYTECODE=1
ARG USER_ID=1000
ARG GROUP_ID=1000

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    libpq5 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy TA-Lib from builder
COPY --from=builder /usr/local/lib/libta_lib.* /usr/local/lib/
COPY --from=builder /usr/local/include/ta-lib/ /usr/local/include/ta-lib/
RUN ldconfig

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create helios user and group
RUN groupadd -g ${GROUP_ID} helios \
    && useradd -u ${USER_ID} -g helios -m -s /bin/bash helios

# Create application directories
RUN mkdir -p /app /logs /data \
    && chown -R helios:helios /app /logs /data

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=helios:helios . /app/

# Set environment variables
ENV PYTHONPATH=/app
ENV LOG_FILE_PATH=/logs/helios.log
ENV DATA_STORAGE_PATH=/data
ENV CONTAINER_TIMEZONE=UTC

# Create logs and data directories with proper permissions
RUN mkdir -p /logs /data \
    && chown -R helios:helios /logs /data \
    && chmod 755 /logs /data

# Switch to helios user
USER helios

# Health check
HEALTHCHEK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from utils.health import health_check; health_check()" || exit 1

# Expose port for potential API
EXPOSE 8000

# Default command
CMD ["python", "main.py"]