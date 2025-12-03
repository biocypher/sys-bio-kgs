# Dockerfile for sys-bio-kgs

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set working directory
RUN apt-get update && apt-get install -y \
    git \
    curl \
    gcc \
    pkg-config \
    libcairo2-dev \
    libglib2.0-dev \
    libffi-dev \
    gobject-introspection \
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
