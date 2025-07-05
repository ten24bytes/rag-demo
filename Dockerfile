# Multi-stage build for optimal image size and performance
FROM python:3.13.5-alpine AS builder

# Install system dependencies needed for building Python packages
RUN apk add --no-cache \
    gcc \
    musl-dev \
    libffi-dev \
    openssl-dev \
    cargo \
    rust

# Install uv (much faster than pip)
RUN pip install --no-cache-dir uv

# Set working directory
WORKDIR /app

# Copy dependency files first (better layer caching)
COPY pyproject.toml uv.lock ./

# Install dependencies in a virtual environment
RUN uv sync --frozen --no-dev

# Production stage
FROM python:3.13.5-alpine AS runtime

# Install runtime dependencies only
RUN apk add --no-cache \
    curl \
    && rm -rf /var/cache/apk/*

# Create non-root user for security
RUN addgroup -g 1001 -S appgroup && \
    adduser -S appuser -u 1001 -G appgroup

# Set working directory
WORKDIR /app

# Copy Python environment from builder stage
COPY --from=builder /app/.venv /app/.venv

# Make sure we use venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy source code with proper ownership
COPY --chown=appuser:appgroup src/ ./src/
COPY --chown=appuser:appgroup .env.example ./.env

# Create uploads and logs directories with proper permissions
RUN mkdir -p uploads logs && chown appuser:appgroup uploads logs

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8501

# Optimized health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl --fail --silent --show-error http://localhost:8501/_stcore/health || exit 1

# Use exec form for proper signal handling
CMD ["python", "-m", "streamlit", "run", "src/app.py", "--server.address", "0.0.0.0", "--server.port", "8501", "--server.headless", "true", "--server.enableCORS", "false"]
