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

# Docker image used to test CV-CUDA on linux-x64.
# Use update_samples_image.sh script present in the folder one level up to
# build this image.

ARG VER_TRT=?

FROM nvcr.io/nvidia/tensorrt:$VER_TRT-py3

# Copy the install_dependencies script and run it. All the dependencies will
# be installed by that script. This script is not present in the current dir
# by default. It is copied by the update_samples_image.sh script. Always
# use update_samples_image.sh script to build any samples docker image.
COPY install_dependencies.sh /workspace/
RUN sh /workspace/install_dependencies.sh
