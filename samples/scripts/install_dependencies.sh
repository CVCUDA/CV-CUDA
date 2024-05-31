#!/bin/bash -e

# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This script installs all the dependencies required to run the CVCUDA samples.
# It uses the /tmp folder to download temporary data and libraries.

# SCRIPT_DIR is the directory where this script is located.
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

# Check CUDA version. Begin by checking if nvcc command exists.
if command -v nvcc >/dev/null 2>&1; then
    # Get CUDA version from nvcc output
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}')

    # Extract major version number
    CUDA_MAJOR_VERSION=$(echo "$CUDA_VERSION" | cut -d. -f1)

    # Check major version to determine CUDA version
    if [ "$CUDA_MAJOR_VERSION" -eq 11 ]; then
        echo "CUDA 11 is installed."
    elif [ "$CUDA_MAJOR_VERSION" -eq 12 ]; then
        echo "CUDA 12 is installed."
    else
        echo "Unknown/Unsupported CUDA version."
        exit 1
    fi
else
    echo "CUDA is not installed."
    exit 1
fi

set -e  # Exit script if any command fails

# Install basic packages first.
cd /tmp
apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    yasm \
    unzip \
    cmake \
    git \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Add repositories and install g++
add-apt-repository -y ppa:ubuntu-toolchain-r/test
apt-get update && apt-get install -y --no-install-recommends \
    gcc-11 g++-11 \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 11
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 11
update-alternatives --set gcc /usr/bin/gcc-11
update-alternatives --set g++ /usr/bin/g++-11

# Install Python and gtest
apt-get update && apt-get install -y --no-install-recommends \
    libgtest-dev \
    libgmock-dev \
    python3-pip \
    ninja-build ccache \
    mlocate && updatedb \
    && rm -rf /var/lib/apt/lists/*

# Install ffmpeg and other libraries needed for VPF.
# Note: We are not installing either libnv-encode or decode libraries here.
apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libavfilter-dev \
    libavformat-dev \
    libavcodec-dev \
    libswresample-dev \
    libavutil-dev\
    && rm -rf /var/lib/apt/lists/*

# Install libssl 1.1.1
cd /tmp
wget http://archive.ubuntu.com/ubuntu/pool/main/o/openssl/libssl1.1_1.1.0g-2ubuntu4_amd64.deb
dpkg -i libssl1.1_1.1.0g-2ubuntu4_amd64.deb

# Install tao-converter which parses the .etlt model file, and generates an optimized TensorRT engine
wget 'https://api.ngc.nvidia.com/v2/resources/nvidia/tao/tao-converter/versions/v4.0.0_trt8.5.1.7_x86/files/tao-converter' --directory-prefix=/usr/local/bin
chmod a+x /usr/local/bin/tao-converter

# Install NVIDIA NSIGHT 2023.2.1
cd /tmp
wget https://developer.download.nvidia.com/devtools/nsight-systems/nsight-systems-2024.2.1_2024.2.1.106-1_amd64.deb
apt-get update && apt-get install -y \
    libsm6 \
    libxrender1 \
    libfontconfig1 \
    libxext6 \
    libx11-dev \
    libxkbfile-dev \
    /tmp/nsight-systems-2024.2.1_2024.2.1.106-1_amd64.deb \
    && rm -rf /var/lib/apt/lists/*

echo "export PATH=$PATH:/opt/tensorrt/bin" >> ~/.bashrc

# Upgrade pip and install all required Python packages.
pip3 install --upgrade pip
pip3 install -r "$SCRIPT_DIR/requirements.txt"

# Install VPF
cd /tmp
[ ! -d 'VideoProcessingFramework' ] && git clone https://github.com/NVIDIA/VideoProcessingFramework.git
# HotFix: Must change the PyTorch version used by PytorchNvCodec to match the one we are using.
# Since we are using 2.2.0 we must use that.
sed -i 's/torch/torch==2.2.0/g' /tmp/VideoProcessingFramework/src/PytorchNvCodec/pyproject.toml
sed -i 's/"torch"/"torch==2.2.0"/g' /tmp/VideoProcessingFramework/src/PytorchNvCodec/setup.py
pip3 install /tmp/VideoProcessingFramework
pip3 install /tmp/VideoProcessingFramework/src/PytorchNvCodec

# Install NvImageCodec
pip3 install nvidia-nvimgcodec-cu${CUDA_MAJOR_VERSION}
pip3 install nvidia-pyindex
pip3 install nvidia-nvjpeg-cu${CUDA_MAJOR_VERSION}

# Install NvPyVideoCodec
cd /tmp
wget --content-disposition https://api.ngc.nvidia.com/v2/resources/nvidia/py_nvvideocodec/versions/0.0.9/zip -O py_nvvideocodec_0.0.9.zip
pip3 install py_nvvideocodec_0.0.9.zip

# Done
