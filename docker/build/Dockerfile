# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

FROM nvidia/cuda:$VER_CUDA-devel-ubuntu$VER_UBUNTU

# need to update and install in one go, or else installation might use
# stale data from server stored in docker cache, with packages that don't exist anymore.
RUN DEBIAN_FRONTEND="noninteractive" apt-get update \
    && apt-get install -y --no-install-recommends \
        git git-lfs \
        g++-11 \
        cmake ninja-build \
        ccache \
        libgtest-dev libgmock-dev \
        pre-commit shellcheck \
        curl \
    && rm -rf /var/lib/apt/lists/*

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

# Python bindings ======================================

# Add deadsnakes apt repo to fetch older pythonv versions
ADD deadsnakes-ubuntu-ppa-jammy.list /etc/apt/sources.list.d
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys BA6932366A755776

# Add so that tzdata don't ask for timezone info in a noninteractive installation.
RUN ln -sf /usr/share/zoneinfo/US/Pacific /etc/localtime

RUN DEBIAN_FRONTEND="noninteractive" apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.7-dev python3.7-distutils \
        python3.8-dev python3.8-distutils \
        python3.9-dev python3.9-distutils \
        python3.10-dev python3.10-distutils \
    && rm -rf /var/lib/apt/lists/*
