# syntax=docker/dockerfile:1
# Stage 1: install Python dependencies into a user-local prefix
FROM python:3.10-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-docker.txt .

# Full requirements.txt pins black+dvc with incompatible pathspec; this file is for a working train+dvc image.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --user -r requirements-docker.txt

# Stage 2: runtime image (no compilers)
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

COPY . .

# Default: run training when prepared data exists (e.g. after bind-mount + dvc repro prepare)
CMD ["python", "src/train.py", "data/prepared", "data/models"]
