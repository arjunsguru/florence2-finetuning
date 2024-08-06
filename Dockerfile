FROM nvidia/cuda:12.1.0-base-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && \
    apt-get install -y \
        git \
        python3-pip \
        python3-dev \
        python3-opencv \
        libglib2.0-0

RUN python3 -m pip install --upgrade pip

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

RUN apt-get update && \
    apt-get install -y cuda-toolkit-12-1

RUN which nvcc

COPY requirements.txt /tmp/requirements.txt

RUN pip install uv
RUN uv venv
RUN . .venv/bin/activate
RUN uv pip install torch
RUN uv pip install --upgrade numpy
RUN uv pip install setuptools
RUN uv pip install psutil
RUN uv pip install -U flash-attn==2.5.9.post1 --no-build-isolation
RUN uv pip install -r /tmp/requirements.txt

ENV PATH="/.venv/bin:$PATH"

CMD ["bash"]