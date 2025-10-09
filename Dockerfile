FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV CONDA_DIR=/opt/conda
ENV PATH="${CONDA_DIR}/bin:${PATH}"

#  NVIDIA repository and CUDA toolkit
RUN apt-get update && apt-get install -y wget gnupg curl && \
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb && \
dpkg -i cuda-keyring_1.0-1_all.deb && \
apt-get update && apt-get install -y \
cuda-toolkit-11-7

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libboost-all-dev \
    libjpeg-dev \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -u -p ${CONDA_DIR} && \
    rm /tmp/miniconda.sh && \
    ${CONDA_DIR}/bin/conda clean -afy


# NVM + Node.js v22
ENV NVM_DIR=/root/.nvm
RUN mkdir -p $NVM_DIR
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash && \
    bash -c "source $NVM_DIR/nvm.sh && nvm install 22 && nvm alias default 22" && \
    bash -c "source $NVM_DIR/nvm.sh && npm install -g npm@latest"

# Add NVM and Node to PATH
ENV NODE_VERSION=22
ENV PATH=$NVM_DIR/versions/node/v$NODE_VERSION/bin:$PATH

SHELL ["/bin/bash", "--login", "-c"]

COPY environment/environment.yml ./environment.yml

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda env create -f environment.yml && \
    echo "conda activate nose-ai" >> ~/.bashrc

RUN conda clean -afy

ENV CONDA_DEFAULT_ENV=nose-ai
ENV PATH="${CONDA_DIR}/envs/nose-ai/bin:${PATH}"

WORKDIR /nose-ai

EXPOSE 3000 3001

CMD ["bash", "--login"]