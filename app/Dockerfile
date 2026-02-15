# --- STAGE 1: BUILDER ---
FROM nvidia/cuda:13.1.1-base-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    CONDA_DIR=/opt/conda \
    PATH=/opt/conda/bin:$PATH \
    PIP_NO_CACHE_DIR=1

# Install only what is needed to download and run the installer
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Miniforge
RUN wget --quiet https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O miniforge.sh && \
    /bin/bash miniforge.sh -b -p /opt/conda && \
    rm miniforge.sh

# Install dependencies and clean up cache
WORKDIR /app
COPY environment.yaml .
RUN mamba env create -f environment.yaml && \
    mamba clean -afy && \
    find /opt/conda/envs/multi-MTRB -follow -type f -name '*.a' -delete && \
    find /opt/conda/envs/multi-MTRB -follow -type f -name '*.pyc' -delete

# --- STAGE 2: RUNTIME ---
FROM nvidia/cuda:13.1.1-runtime-ubuntu22.04 AS runtime

# Re-establish environment variables
ENV PATH=/opt/conda/envs/multi-MTRB/bin:$PATH \
    CONDA_DEFAULT_ENV=multi-MTRB \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copy only the finalized environment from the builder
COPY --from=builder /opt/conda/envs/multi-MTRB /opt/conda/envs/multi-MTRB

# Copy code last
COPY . .

# Use the full path to the environment's python for the best reliability
ENTRYPOINT ["python"]
CMD ["-c", "import torch; print(f'Cuda available: {torch.cuda.is_available()}')"]
