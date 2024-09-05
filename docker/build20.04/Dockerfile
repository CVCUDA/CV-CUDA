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

# Add so that tzdata don't ask for timezone info in a noninteractive installation.
RUN ln -sf /usr/share/zoneinfo/US/Pacific /etc/localtime

# need to update and install in one go, or else installation might use
# stale data from server stored in docker cache, with packages that don't exist anymore.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git git-lfs software-properties-common wget\
    && add-apt-repository ppa:ubuntu-toolchain-r/test \
    && apt-get update \
    && apt-get install -y --no-install-recommends gcc-11 g++-11 \
    && wget https://apt.llvm.org/llvm.sh \
    && chmod +x llvm.sh \
    && ./llvm.sh 11 && ./llvm.sh 14 && rm -f llvm.sh \
    && apt-get install -y --no-install-recommends ninja-build ccache libgtest-dev libgmock-dev shellcheck curl \
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

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip python3-pytest python3-dev doxygen \
    && rm -rf /var/lib/apt/lists/*

# python3 is python3.8 in ubuntu20.04
RUN python3 -m pip install pre-commit
# Needed for python documentation
RUN python3 -m pip install sphinx-rtd-theme sphinx==4.5.0
RUN python3 -m pip install breathe exhale recommonmark graphviz
# Needed for python sphinx docs and Python wheels
RUN python3 -m pip install numpy patchelf==0.17.2.1

# Python bindings ======================================

# Add deadsnakes apt repo to fetch older pythonv versions
ADD deadsnakes-ubuntu-ppa-focal.list /etc/apt/sources.list.d
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys BA6932366A755776

RUN for PYTHON_VERSION in 3.8 3.9 3.10 3.11; do \
    apt-get update \
    && apt-get install -y --no-install-recommends \
    python$PYTHON_VERSION-dev python$PYTHON_VERSION-distutils; \
    done && \
    rm -rf /var/lib/apt/lists/*

# gcc-8 ======================================
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    gcc-8 g++-8 \
    && rm -rf /var/lib/apt/lists/*

# Needed for OpenSSL
RUN apt-get update \
    && apt-get install -y --no-install-recommends libssl-dev \
    && rm -rf /var/lib/apt/lists/*
