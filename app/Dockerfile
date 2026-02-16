# syntax=docker/dockerfile:1
# --- STAGE 1: BUILDER ---
FROM nvidia/cuda:13.1.1-base-ubuntu22.04 AS builder

ENV CONDA_DIR=/opt/conda \
    PATH=/opt/conda/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget ca-certificates && rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O miniforge.sh && \
    /bin/bash miniforge.sh -b -p /opt/conda && rm miniforge.sh

WORKDIR /app

COPY environment.yaml .

# Leverage BuildKit cache for mamba packages
RUN --mount=type=cache,target=/opt/conda/pkgs \
    mamba env create -f environment.yaml

# --- STAGE 2: RUNTIME ---
FROM nvidia/cuda:13.1.1-runtime-ubuntu22.04 AS runtime

ENV PATH=/opt/conda/envs/multi-MTRB/bin:$PATH \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copy the env from builder
COPY --from=builder /opt/conda/envs/multi-MTRB /opt/conda/envs/multi-MTRB

ENTRYPOINT ["python"]
