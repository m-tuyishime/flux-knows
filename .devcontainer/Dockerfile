FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

COPY . .

# Install wget, git, gcc, and development libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    python3.9 \
    python3-pip \
    python3.9-dev \
    && rm -rf /var/lib/apt/lists/*

# set python3.9 as default python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

# set python3.9 as default pip
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# set up pip
RUN pip install --upgrade pip

# install dependencies
RUN pip install -r requirements.txt

# Login to Hugging Face
RUN huggingface-cli login --token $HF_TOKEN
