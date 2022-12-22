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

# Docker image used for development of CV-CUDA on linux-x64

ARG BASE_IMAGE=?
ARG TAG_IMAGE=?

FROM $BASE_IMAGE:$TAG_IMAGE

# need to update and install in one go, or else installation might use
# stale data from server stored in docker cache, with packages that don't exist anymore.

# HACK: need to pass 'sudo' as a variable to workaround Dockerfile linter, it says
#       we shouldn't install sudo in a container. But we know what we're doing!
RUN HACK_SUDO=sudo && DEBIAN_FRONTEND="noninteractive" apt-get update \
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
RUN DEBIAN_FRONTEND="noninteractive" apt-get update \
    && apt-get install -y --no-install-recommends \
         python3-pytest \
         python3-pip \
    && rm -rf /var/lib/apt/lists/*

# needed by tests
RUN python3 -m pip install numba \
    && rm -rf /root/.cache/pip

RUN python3 -m pip install torch \
    && rm -rf /root/.cache/pip

WORKDIR /cvcuda
