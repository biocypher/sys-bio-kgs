# Dockerfile for sys-bio-kgs

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
# Note: momapy is optional and requires additional system libraries (cairo, pango, gobject-introspection)
# If you want to use momapy, uncomment the following lines and install momapy with:
#   pip install -e ".[momapy]"
# RUN apt-get update && apt-get install -y \
#     libcairo2-dev \
#     libpango1.0-dev \
#     libgirepository1.0-dev \
#     gir1.2-pango-1.0 \
#     && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    clang \
    pkg-config \
    ninja-build \
    git \
    curl \
    libcairo2-dev \
    libglib2.0-dev \
    gobject-introspection \
    libpango1.0-dev \
    libffi-dev \
    cmake \
    libgirepository-2.0-dev \
    python3-gi \
    gir1.2-pango-1.0 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libpangoft2-1.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN mkdir -p /etc/pip && printf "[global]\nno-binary = bezier\n" > /etc/pip.conf

ENV BEZIER_NO_EXTENSION=true
RUN pip install --no-cache-dir bezier

# Use CPU-only PyTorch to reduce image size (from ~2-3GB to ~200MB)
ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu

# Now install your package + momapy, which will reuse that PyGObject
RUN pip install --no-cache-dir -e .

# Copy remaining source code
COPY . .

# Create logs directory
RUN mkdir -p logs

# Set environment variables
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python", "create_knowledge_graph.py"]
