#!/bin/bash -e

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

apt-get update && apt-get install -y --no-install-recommends build-essential \
    build-essential software-properties-common \
    && rm -rf /var/lib/apt/lists/*

add-apt-repository --yes ppa:ubuntu-toolchain-r/test
apt-get update && apt-get install -y --no-install-recommends gcc-11 g++-11 \
   && rm -rf /var/lib/apt/lists/*
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 11
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 11
update-alternatives --set gcc /usr/bin/gcc-11
update-alternatives --set g++ /usr/bin/g++-11
pip3 install torch==1.13.0 torchvision==0.14.0
rm -rf ./torchnvjpeg && git clone https://github.com/itsliupeng/torchnvjpeg.git
cd torchnvjpeg && python3 setup.py bdist_wheel && cd dist && pip3 install torchnvjpeg-*.whl
