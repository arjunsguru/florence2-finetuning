# Start with NVIDIA PyTorch image
FROM nvcr.io/nvidia/pytorch:22.03-py3

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda

# Update GPG keys and install system dependencies
RUN apt-get update && apt-get install -y gnupg && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub && \
    apt-get update -y && \
    apt-get install -y --no-install-recommends \
    git \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Upgrade pip and install Python packages
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir \
    packaging \
    tqdm \
    transformers \
    einops \
    pandas \
    datasets \
    timm \
    levenshtein \
    matplotlib \
    pycocotools \
    scikit-image \
    pypng \
    friendlywords \
    wandb

RUN pip3 install peft --no-dependencies

# Set the working directory in the container
WORKDIR /app

# Command to run when starting the container
CMD ["bash"]