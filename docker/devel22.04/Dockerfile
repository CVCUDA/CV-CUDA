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

# Docker image used for development of CV-CUDA on linux-x64

ARG BASE_IMAGE=?
ARG TAG_IMAGE=?
ARG VER_NUMPY=?

FROM $BASE_IMAGE:$TAG_IMAGE

ARG DEBIAN_FRONTEND=noninteractive

# need to update and install in one go, or else installation might use
# stale data from server stored in docker cache, with packages that don't exist anymore.

# HACK: need to pass 'sudo' as a variable to workaround Dockerfile linter, it says
#       we shouldn't install sudo in a container. But we know what we're doing!
RUN HACK_SUDO=sudo && apt-get update \
    && apt-get install -y --no-install-recommends \
    $HACK_SUDO \
    vim \
    gdb cgdb \
    less \
    wget curl \
    && rm -rf /var/lib/apt/lists/*

# Enable CUDA driver checks as this image will be used for running CUDA programs
ENV NVIDIA_DISABLE_REQUIRE=false

# Config files we use
COPY vimrc /root/.vimrc
COPY gdbinit /root/.gdbinit

# For running tests inside dev container
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    apt-utils \
    python3-typing-extensions \
    python3-pytest \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# For running tests inside dev container
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    apt-utils fonts-dejavu \
    && rm -rf /var/lib/apt/lists/*

# needed by tests (python3 is python3.10 in ubuntu22.04)
RUN python3 -m pip install torch==2.4.0 torchvision cupy-cuda11x \
    numpy==$VER_NUMPY \
    && rm -rf /root/.cache/pip
RUN python3.8 -m pip install torch==2.4.0 torchvision cupy-cuda11x \
    numpy sphinx-rtd-theme sphinx breathe exhale recommonmark \
    graphviz && rm -rf /root/.cache/pip
RUN python3.9 -m pip install pytest torch==2.4.0 torchvision cupy-cuda11x \
    numpy==$VER_NUMPY \
    && rm -rf /root/.cache/pip
RUN python3.11 -m pip install --upgrade pytest torch==2.4.0 cupy-cuda11x \
    numpy==$VER_NUMPY \
    && rm -rf /root/.cache/pip

WORKDIR /cvcuda
