#!/bin/bash -e

# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
add-apt-repository ppa:ubuntu-toolchain-r/test
apt-get update && apt-get install -y --no-install-recommends \
    gcc-11 g++-11 \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 11
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 11
update-alternatives --set gcc /usr/bin/gcc-11
update-alternatives --set g++ /usr/bin/g++-11

# Install python and gtest
apt-get update && apt-get install -y --no-install-recommends \
    libgtest-dev \
    libgmock-dev \
    python3-pip \
    ninja-build ccache \
    mlocate && updatedb \
    && rm -rf /var/lib/apt/lists/*

# Install pip and all the python packages.
pip3 install --upgrade pip
pip3 install torch==1.13.0 torchvision==0.14.0 av==10.0.0 pycuda==2022.1 nvtx==0.2.5
cd /tmp
git clone https://github.com/itsliupeng/torchnvjpeg.git
cd torchnvjpeg && python setup.py bdist_wheel && cd dist && pip3 install torchnvjpeg-0.1.0-cp38-cp38-linux_x86_64.whl
echo "export PATH=$PATH:/opt/tensorrt/bin" >> ~/.bashrc

# Install VPF and its dependencies.
# 1. ffmpeg and other libraries needed for VPF.
# Note: We are not installing either libnv-encode or decode libraries here.
apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libavfilter-dev \
    libavformat-dev \
    libavcodec-dev \
    libswresample-dev \
    libavutil-dev\
    && rm -rf /var/lib/apt/lists/*
cd /tmp
git clone https://github.com/NVIDIA/VideoProcessingFramework.git
pip3 install /tmp/VideoProcessingFramework
pip3 install /tmp/VideoProcessingFramework/src/PytorchNvCodec

# Install tao-converter which parses the .etlt model file, and generates an optimized TensorRT engine
cd /tmp
mkdir tao_binaries
cd tao_binaries
wget --content-disposition 'https://api.ngc.nvidia.com/v2/resources/nvidia/tao/tao-converter/versions/v4.0.0_trt8.5.1.7_x86/files/tao-converter'
chmod a+x tao-converter
echo "export PATH=$PATH:/tmp/tao_binaries" >> ~/.bashrc
# Done
