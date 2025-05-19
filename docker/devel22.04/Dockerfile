# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

ARG VER_CUDA
ARG VER_UBUNTU=?

FROM nvidia/cuda:$VER_CUDA-devel-ubuntu$VER_UBUNTU

# The following ARG must be declared after the FROM statement above because
# FROM statement will make all existing arg values go null.
ARG VER_CUDA
ARG VER_NUMPY=?
ARG DEBIAN_FRONTEND=noninteractive

# Enable CUDA driver checks as this image will be used for running CUDA programs
ENV NVIDIA_DISABLE_REQUIRE=false

COPY requirements.txt /tmp/requirements.txt

# Config files we use
COPY vimrc /root/.vimrc
COPY gdbinit /root/.gdbinit

# Configure ccache
RUN mkdir -p /cache
COPY ccache.conf /etc/ccache.conf
ENV CCACHE_CONFIGPATH=/etc/ccache.conf
ENV PRE_COMMIT_HOME=/cache/pre-commit

# Add deadsnakes apt repo to fetch older python versions
ADD deadsnakes-ubuntu-ppa-jammy.list /etc/apt/sources.list.d
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys BA6932366A755776

# Add so that tzdata don't ask for timezone info in a noninteractive installation.
RUN ln -sf /usr/share/zoneinfo/US/Pacific /etc/localtime

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
    vim \
    gdb cgdb \
    less \
    apt-utils fonts-dejavu \
    python3 python3-dev python3-pytest doxygen graphviz \
    python3.8-dev python3.8-distutils \
    python3.9-dev python3.9-distutils \
    python3.10-dev python3.10-distutils \
    python3.11-dev python3.11-distutils \
    python3.12-dev \
    python3.13-dev \
    && rm -rf /var/lib/apt/lists/*

# Install/Update pip for all python versions in one layer
RUN for pyver in 3.8 3.9 3.10 3.11 3.12 3.13; do \
        echo "Installing pip for Python $pyver" && \
        if [ "$pyver" = "3.8" ]; then \
            curl -sS https://bootstrap.pypa.io/pip/3.8/get-pip.py | python$pyver ; \
        else \
            curl -sS https://bootstrap.pypa.io/get-pip.py | python$pyver ; \
        fi ; \
    done

# Determine architecture for CMake download
RUN dpkg_arch=$(dpkg --print-architecture) && \
    case "${dpkg_arch}" in \
        amd64) cmake_arch="x86_64";; \
        arm64) cmake_arch="aarch64";; \
        *) echo >&2 "error: unsupported architecture ('${dpkg_arch}')" && exit 1;; \
    esac && \
    echo "Downloading CMake for ${cmake_arch}" && \
    curl --fail -L https://cmake.org/files/v3.23/cmake-3.23.1-linux-${cmake_arch}.tar.gz --output /tmp/cmake-3.23.1.tar.gz \
    && tar -xzf /tmp/cmake-3.23.1.tar.gz -C /tmp/ && cd /tmp/cmake-3.23.1-linux-${cmake_arch}/ \
    && cp bin/ share/ doc/ /usr/local/ -r && rm -rf /tmp/cmake-3.23.1*

# gcc-8 ======================================
# Ubuntu 22.04 doesn't come with gcc-8, must use the ones from 20.04.
# Determine architecture for GCC8 deb package download
RUN dpkg_arch=$(dpkg --print-architecture) && \
    case "${dpkg_arch}" in \
        amd64) \
            pkg_arch="amd64"; \
            base_url="http://archive.ubuntu.com/ubuntu"; \
            include_libmpx=true;; \
        arm64) \
            pkg_arch="arm64"; \
            base_url="http://ports.ubuntu.com/ubuntu-ports"; \
            include_libmpx=false;; \
        *) echo >&2 "error: unsupported architecture ('${dpkg_arch}')" && exit 1;; \
    esac && \
    echo "Downloading GCC 8 debs for ${pkg_arch} from ${base_url}" && \
    cd /tmp && \
    curl --fail-early -L \
    -O ${base_url}/pool/universe/g/gcc-8/gcc-8_8.4.0-3ubuntu2_${pkg_arch}.deb \
    -O ${base_url}/pool/universe/g/gcc-8/gcc-8-base_8.4.0-3ubuntu2_${pkg_arch}.deb \
    -O ${base_url}/pool/universe/g/gcc-8/libgcc-8-dev_8.4.0-3ubuntu2_${pkg_arch}.deb \
    -O ${base_url}/pool/universe/g/gcc-8/cpp-8_8.4.0-3ubuntu2_${pkg_arch}.deb \
    $( $include_libmpx && echo "-O ${base_url}/pool/universe/g/gcc-8/libmpx2_8.4.0-3ubuntu2_${pkg_arch}.deb" ) \
    -O ${base_url}/pool/universe/g/gcc-8/g++-8_8.4.0-3ubuntu2_${pkg_arch}.deb \
    -O ${base_url}/pool/universe/g/gcc-8/libstdc++-8-dev_8.4.0-3ubuntu2_${pkg_arch}.deb \
    -O ${base_url}/pool/main/i/isl/libisl22_0.22.1-1_${pkg_arch}.deb

# Install downloaded GCC8 deb packages
RUN dpkg_arch=$(dpkg --print-architecture) && \
    case "${dpkg_arch}" in \
        amd64) \
            pkg_arch="amd64"; \
            install_libmpx=true;; \
        arm64) \
            pkg_arch="arm64"; \
            install_libmpx=false;; \
        *) echo >&2 "error: unsupported architecture ('${dpkg_arch}')" && exit 1;; \
    esac && \
    echo "Installing GCC 8 debs for ${pkg_arch}" && \
    apt-get update && apt-get install -y --no-install-recommends \
    $( $install_libmpx && echo "/tmp/libmpx2_8.4.0-3ubuntu2_${pkg_arch}.deb" ) \
    /tmp/cpp-8_8.4.0-3ubuntu2_${pkg_arch}.deb \
    /tmp/gcc-8-base_8.4.0-3ubuntu2_${pkg_arch}.deb \
    /tmp/libgcc-8-dev_8.4.0-3ubuntu2_${pkg_arch}.deb \
    /tmp/gcc-8_8.4.0-3ubuntu2_${pkg_arch}.deb \
    /tmp/g++-8_8.4.0-3ubuntu2_${pkg_arch}.deb \
    /tmp/libstdc++-8-dev_8.4.0-3ubuntu2_${pkg_arch}.deb \
    /tmp/libisl22_0.22.1-1_${pkg_arch}.deb \
    && rm -rf /var/lib/apt/lists/* /tmp/*.deb

# Install python packages separately for better caching and error isolation
# Set ENV vars within each RUN for pip substitution (CUDA_MAJOR_VERSION, VER_NUMPY)
RUN CUDA_MAJOR_VERSION=$(echo $VER_CUDA | cut -d. -f1) && \
    echo "Installing Python 3.8 requirements (CUDA $CUDA_MAJOR_VERSION, Numpy 1.24.4)" && \
    # VER_NUMPY not needed for py3.8 as version is fixed
    env CUDA_MAJOR_VERSION=$CUDA_MAJOR_VERSION python3.8 -m pip install -r /tmp/requirements.txt

RUN CUDA_MAJOR_VERSION=$(echo $VER_CUDA | cut -d. -f1) && \
    echo "Installing Python 3.9 requirements (CUDA $CUDA_MAJOR_VERSION, Numpy $VER_NUMPY)" && \
    env CUDA_MAJOR_VERSION=$CUDA_MAJOR_VERSION VER_NUMPY=$VER_NUMPY python3.9 -m pip install -r /tmp/requirements.txt

RUN CUDA_MAJOR_VERSION=$(echo $VER_CUDA | cut -d. -f1) && \
    echo "Installing Python 3.10 requirements (CUDA $CUDA_MAJOR_VERSION, Numpy $VER_NUMPY)" && \
    env CUDA_MAJOR_VERSION=$CUDA_MAJOR_VERSION VER_NUMPY=$VER_NUMPY python3.10 -m pip install -r /tmp/requirements.txt

RUN CUDA_MAJOR_VERSION=$(echo $VER_CUDA | cut -d. -f1) && \
    echo "Installing Python 3.11 requirements (CUDA $CUDA_MAJOR_VERSION, Numpy $VER_NUMPY)" && \
    env CUDA_MAJOR_VERSION=$CUDA_MAJOR_VERSION VER_NUMPY=$VER_NUMPY python3.11 -m pip install -r /tmp/requirements.txt

RUN CUDA_MAJOR_VERSION=$(echo $VER_CUDA | cut -d. -f1) && \
    echo "Installing Python 3.12 requirements (CUDA $CUDA_MAJOR_VERSION, Numpy $VER_NUMPY)" && \
    env CUDA_MAJOR_VERSION=$CUDA_MAJOR_VERSION VER_NUMPY=$VER_NUMPY python3.12 -m pip install -r /tmp/requirements.txt

RUN CUDA_MAJOR_VERSION=$(echo $VER_CUDA | cut -d. -f1) && \
    echo "Installing Python 3.13 requirements (CUDA $CUDA_MAJOR_VERSION, Numpy $VER_NUMPY)" && \
    env CUDA_MAJOR_VERSION=$CUDA_MAJOR_VERSION VER_NUMPY=$VER_NUMPY python3.13 -m pip install -r /tmp/requirements.txt

# Clean up pip cache
RUN rm -rf /root/.cache/pip
