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

# Docker image used to test CV-CUDA on linux-x64

ARG VER_TRT=?

FROM nvcr.io/nvidia/tensorrt:$VER_TRT-py3
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

RUN add-apt-repository ppa:ubuntu-toolchain-r/test
RUN apt-get update && apt-get install -y --no-install-recommends gcc-11 g++-11 \
    ninja-build && rm -rf /var/lib/apt/lists/*
RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 11
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 11
RUN update-alternatives --set gcc /usr/bin/gcc-11
RUN update-alternatives --set g++ /usr/bin/g++-11

RUN apt-get update && apt-get install -y --no-install-recommends libgtest-dev libgmock-dev python3-pip ninja-build ccache \
    mlocate && updatedb \
    && rm -rf /var/lib/apt/lists/*
RUN pip3 install --upgrade pip
RUN pip3 install torch==1.13.0 torchvision==0.14.0 av==10.0.0
RUN git clone https://github.com/itsliupeng/torchnvjpeg.git
RUN cd torchnvjpeg && python setup.py bdist_wheel && cd dist && pip3 install torchnvjpeg-0.1.0-cp38-cp38-linux_x86_64.whl
RUN echo "export PATH=$PATH:/opt/tensorrt/bin" >> ~/.bashrc

# Build VPF
ADD ffmpeg-master-latest-linux64-gpl-shared /workspace/ffmpeg-master-latest-linux64-gpl-shared
ADD Video_Codec_SDK /workspace/Video_Codec_SDK
ADD VideoProcessingFramework /workspace/VideoProcessingFramework
ENV PATH_TO_SDK="/workspace/Video_Codec_SDK"
ENV PATH_TO_FFMPEG="/workspace/ffmpeg-master-latest-linux64-gpl-shared"
ENV CUDACXX=/usr/local/cuda/bin/nvcc
ENV INSTALL_PREFIX="/workspace/VideoProcessingFramework/install"
ENV LD_LIBRARY_PATH="${PATH_TO_FFMPEG}/lib:$LD_LIBRARY_PATH"
ENV LD_LIBRARY_PATH="/workspace/VideoProcessingFramework/install/bin:${LD_LIBRARY_PATH}"
ENV PYTHONPATH="${PYTHONPATH}:/workspace/VideoProcessingFramework/install/bin/"
WORKDIR "/workspace/VideoProcessingFramework"
RUN rm -rf install
RUN rm -rf build
RUN mkdir -p install
RUN mkdir -p build
WORKDIR "/workspace/VideoProcessingFramework/build"

RUN cmake .. \
  -DFFMPEG_DIR:PATH="$PATH_TO_FFMPEG" \
  -DVIDEO_CODEC_SDK_DIR:PATH="$PATH_TO_SDK" \
  -DGENERATE_PYTHON_BINDINGS:BOOL="1" \
  -DGENERATE_PYTORCH_EXTENSION:BOOL="1" \
  -DCMAKE_INSTALL_PREFIX:PATH="$INSTALL_PREFIX"
RUN make && make install
WORKDIR /workspace
