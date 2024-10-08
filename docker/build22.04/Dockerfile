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

# Docker image used to build CV-CUDA on linux-x64

ARG VER_CUDA=?
ARG VER_UBUNTU=?
ARG VER_NUMPY=?

FROM nvidia/cuda:$VER_CUDA-devel-ubuntu$VER_UBUNTU

ARG DEBIAN_FRONTEND=noninteractive

# need to update and install in one go, or else installation might use
# stale data from server stored in docker cache, with packages that don't exist anymore.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    git git-lfs \
    g++-11 \
    # need to also build with gcc-9.4.0, our minimum supported compiler for the library
    gcc-9=9.4.0-5ubuntu1 cpp-9=9.4.0-5ubuntu1 gcc-9-base=9.4.0-5ubuntu1 libgcc-9-dev=9.4.0-5ubuntu1 libasan5=9.4.0-5ubuntu1 g++-9=9.4.0-5ubuntu1 libstdc++-9-dev=9.4.0-5ubuntu1 \
    # Compilers to which public headers must be compatible
    clang-14 clang-11 \
    ninja-build \
    ccache \
    libgtest-dev libgmock-dev \
    pre-commit shellcheck \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && curl -L https://cmake.org/files/v3.20/cmake-3.20.1-linux-x86_64.tar.gz --output /tmp/cmake-3.20.1.tar.gz \
    && tar -xzf /tmp/cmake-3.20.1.tar.gz -C /tmp/ && cd /tmp/cmake-3.20.1-linux-x86_64/ \
    && cp bin/ share/ doc/ /usr/local/ -r && rm -rf /tmp/cmake-3.20.1*

# Configure ccache
RUN mkdir -p /cache
COPY ccache.conf /etc/ccache.conf
ENV CCACHE_CONFIGPATH=/etc/ccache.conf
ENV PRE_COMMIT_HOME=/cache/pre-commit

# Documentation ======================================

# Allow using this image in systems without proper CUDA runtime/driver support.
# We'll be using this image only for building, don't need strict CUDA checks.
ENV NVIDIA_DISABLE_REQUIRE=true

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev doxygen && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install sphinx-rtd-theme sphinx==4.5.0
RUN python3 -m pip install breathe exhale recommonmark graphviz
# Needed for python sphinx docs and Python wheels
RUN python3 -m pip install numpy patchelf==0.17.2.1

# Python bindings ======================================

# Add deadsnakes apt repo to fetch older pythonv versions
ADD deadsnakes-ubuntu-ppa-jammy.list /etc/apt/sources.list.d
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys BA6932366A755776

# Add so that tzdata don't ask for timezone info in a noninteractive installation.
RUN ln -sf /usr/share/zoneinfo/US/Pacific /etc/localtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.8-dev python3.8-distutils \
    python3.9-dev python3.9-distutils \
    python3.10-dev python3.10-distutils \
    python3.11-dev python3.11-distutils \
    && rm -rf /var/lib/apt/lists/*

# gcc-8 ======================================
# Ubuntu 22.04 doesn't come with gcc-8, must use the ones from 20.04.

RUN curl --fail-early -L \
    -O http://mirrors.kernel.org/ubuntu/pool/universe/g/gcc-8/gcc-8_8.4.0-3ubuntu2_amd64.deb \
    -O http://mirrors.kernel.org/ubuntu/pool/universe/g/gcc-8/gcc-8-base_8.4.0-3ubuntu2_amd64.deb \
    -O http://mirrors.kernel.org/ubuntu/pool/universe/g/gcc-8/libgcc-8-dev_8.4.0-3ubuntu2_amd64.deb \
    -O http://mirrors.kernel.org/ubuntu/pool/universe/g/gcc-8/cpp-8_8.4.0-3ubuntu2_amd64.deb \
    -O http://mirrors.kernel.org/ubuntu/pool/universe/g/gcc-8/libmpx2_8.4.0-3ubuntu2_amd64.deb \
    -O http://mirrors.kernel.org/ubuntu/pool/universe/g/gcc-8/g++-8_8.4.0-3ubuntu2_amd64.deb \
    -O http://mirrors.kernel.org/ubuntu/pool/universe/g/gcc-8/libstdc++-8-dev_8.4.0-3ubuntu2_amd64.deb \
    -O http://mirrors.kernel.org/ubuntu/pool/main/i/isl/libisl22_0.22.1-1_amd64.deb

RUN apt-get update && apt-get install -y --no-install-recommends \
    ./libmpx2_8.4.0-3ubuntu2_amd64.deb \
    ./cpp-8_8.4.0-3ubuntu2_amd64.deb \
    ./gcc-8-base_8.4.0-3ubuntu2_amd64.deb \
    ./libgcc-8-dev_8.4.0-3ubuntu2_amd64.deb \
    ./gcc-8_8.4.0-3ubuntu2_amd64.deb \
    ./g++-8_8.4.0-3ubuntu2_amd64.deb \
    ./libstdc++-8-dev_8.4.0-3ubuntu2_amd64.deb \
    ./libisl22_0.22.1-1_amd64.deb \
    && rm -rf /var/lib/apt/lists/*
