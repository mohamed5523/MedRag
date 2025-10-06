# syntax=docker/dockerfile:1

# FROM python:3.11-slim AS base

FROM astral/uv:python3.11-trixie-slim AS base
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps for scientific/python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first for better caching
# Install uv (fast Python package installer and runner)
# RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
#     && echo 'export PATH="/root/.local/bin:$PATH"' >> /root/.bashrc
ENV PATH="/root/.local/bin:${PATH}"

# Copy project metadata and resolve dependencies via uv from pyproject
COPY pyproject.toml ./pyproject.toml
COPY README.md ./README.md
RUN uv sync --no-dev

# Copy application code
COPY app ./app
COPY run.py ./run.py
COPY connect2supabase.py ./connect2supabase.py

# Create data directories (mounted via volumes in compose)
RUN mkdir -p /app/data/chroma_db /app/uploads

EXPOSE 8000

# Prefer running via uv to ensure the synced virtualenv is used
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
