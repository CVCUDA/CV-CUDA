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

# Docker image used to test CV-CUDA on linux-x64

ARG VER_CUDA=?
ARG VER_UBUNTU=?

FROM nvidia/cuda:$VER_CUDA-runtime-ubuntu$VER_UBUNTU

ARG VER_CUDA=?

# For testing python bindings ======================================

ARG DEBIAN_FRONTEND=noninteractive

# Add deadsnakes apt repo to fetch older python versions
ADD deadsnakes-ubuntu-ppa-focal.list /etc/apt/sources.list.d
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys BA6932366A755776

# Add so that tzdata don't ask for timezone info in a noninteractive installation.
RUN ln -sf /usr/share/zoneinfo/US/Pacific /etc/localtime

# For running python tests
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.7 python3.7-distutils \
    python3.8 python3.8-distutils \
    python3.9 python3.9-distutils \
    python3.10 python3.10-distutils \
    python3.11 python3.11-distutils \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# It needs torch
RUN set -e \
    && for ver in 3.7 3.8 3.9 3.10 3.11; do \
    python$ver -m pip install torch numpy torchvision; \
    done \
    && rm -rf /root/.cache/pip

# Other dependencies of python tests
# binutils: for readelf
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    binutils \
    && rm -rf /var/lib/apt/lists/*
