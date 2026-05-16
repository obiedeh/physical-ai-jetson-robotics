# Jetson deployment image for physical-ai-jetson-robotics
#
# Build on host (not inside container):
#   docker build -t physical-ai-lab:latest .
#
# Run on Jetson (JetPack 6.x, L4T r36):
#   docker run --runtime nvidia --gpus all \
#     -v /path/to/models:/models \
#     -v /path/to/reports:/app/reports \
#     physical-ai-lab:latest

FROM nvcr.io/nvidia/l4t-pytorch:r36.2.0-pth2.1-py3

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps only — keep layer cacheable
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install package deps before copying source (layer cache efficiency)
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e ".[dev]" || true

# Copy source — changes here don't invalidate the dep layer
COPY physical_ai_lab/ ./physical_ai_lab/
COPY tests/ ./tests/
COPY reports/ ./reports/

RUN pip install --no-cache-dir -e .

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD physical-ai-lab profile || exit 1

USER root
RUN useradd -m appuser && chown -R appuser /app
USER appuser

CMD ["physical-ai-lab", "--help"]
