# ML Training Image for Dynamical-SIL
# Includes: PyTorch, robomimic, LeRobot, Opacus, privacy libs

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    python3.10 \
    python3.10-dev \
    python3-pip \
    libhdf5-dev \
    libopencv-dev \
    libjpeg-dev \
    libpng-dev \
    ffmpeg \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Copy and install Python dependencies
WORKDIR /app
COPY pyproject.toml .
RUN pip3 install -e ".[dev,cuda]"

# Install additional ML frameworks
RUN pip3 install \
    robomimic \
    lerobot \
    opacus \
    Pyfhel \
    crypten \
    mmpose \
    mmdet \
    mmcv

# Create workspace directories
RUN mkdir -p /app/ml /app/swarm /app/outputs /app/artifacts

# Copy application code
COPY ml/ /app/ml/
COPY swarm/ /app/swarm/
COPY configs/ /app/configs/

WORKDIR /app

CMD ["python3", "-m", "ml.training.train_cooperative_bc"]
