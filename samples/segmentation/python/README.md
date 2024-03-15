
[//]: # "SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved."
[//]: # "SPDX-License-Identifier: Apache-2.0"
[//]: # ""
[//]: # "Licensed under the Apache License, Version 2.0 (the 'License');"
[//]: # "you may not use this file except in compliance with the License."
[//]: # "You may obtain a copy of the License at"
[//]: # "http://www.apache.org/licenses/LICENSE-2.0"
[//]: # ""
[//]: # "Unless required by applicable law or agreed to in writing, software"
[//]: # "distributed under the License is distributed on an 'AS IS' BASIS"
[//]: # "WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied."
[//]: # "See the License for the specific language governing permissions and"
[//]: # "limitations under the License."

# Semantic Segmentation : Locally and using Triton

## Pre-requisites

- Recommended Linux distros:
    - Ubuntu >= 20.04 (tested with 20.04 and 22.04)
    - WSL2 with Ubuntu >= 20.04 (tested with 20.04)
- CUDA driver >= 11.7
- Triton server and client docker >= 22.07
- Refer to the [Samples README](../README.md) for Pre-requisites to run the segmentation pipeline

# Instructions to run the sample without Triton

1. Launch the docker

      ```bash
      docker run -ti --gpus=all -v <local mount path>:/cvcuda -w /cvcuda nvcr.io/nvidia/tensorrt:22.09-py3
      ```

2. Install the dependencies

      ```bash
      ./samples/scripts/install_dependencies.sh
      ```

3. Run the segmentation sample for different data modalities

    a. Run segmentation on a single image

      ```bash
      python3 ./samples/segmentation/python/main.py -i ./samples/assets/images/tabby_tiger_cat.jpg -b 1
      ```

    b. Run segmentation on folder containing images with pytorch backend

      ```bash
      python3 ./samples/segmentation/python/main.py -i ./samples/assets/images -b 2 -bk pytorch
      ```

    c. Run segmentation on a video file with tensorrt backend

      ```bash
      python3 ./samples/segmentation/python/main.py -i ./samples/assets/videos/pexels-ilimdar-avgezer-7081456.mp4 -b 4 -bk tensorrt
      ```

4. To benchmark this run, we can use the benchmark.py in the following way. It should launch 1 process, ignore 1 batch from front and end as warmup batches, save per process and overall numbers as JSON files in /tmp directory. To understand more about performance benchmarking in CV-CUDA, please refer to [Performance Benchmarking README](../../scripts/README.md)

      ```bash
      python3 ./samples/scripts/benchmark.py -np 1 -w 1 -o /tmp ./samples/segmentation/python/main.py -b 4 -i ./samples/assets/videos/pexels-ilimdar-avgezer-7081456.mp4
      ```

# Instructions to run the sample with Triton

## Triton Server instructions

Triton has different public [Docker images](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver): `-py3-sdk` for Triton client libraries, `-py3` for Triton server libraries with TensorRT, ONNX, Pytorch, TensorFlow, `-pyt-python-py3` for Triton server libraries with PyTorch and Python backend only.

1. Launch the triton server

      ```bash
      docker run --shm-size=1g --ulimit memlock=-1 -p 8000:8000 -p 8001:8001 -p 8002:8002 --ulimit stack=67108864 -ti --gpus=all -v <local mount path>:/cvcuda -w /cvcuda nvcr.io/nvidia/tritonserver:22.12-py3
      ```
2. Install the dependencies

      ```bash
      ./samples/scripts/install_dependencies.sh
      pip3 install tensorrt
      ```
3. Install the CV-CUDA packages. Pre-built packages `.deb`, `.tar.xz`, `.whl` are only available on Github, so need to download from there. Otherwise, please build from source. Please note that since the above container comes with Python 3.8.10, we will install cvcuda-python3.8-0 package as mentioned below. If you have any other Python distributions, you would need to use the appropriate cvcuda-python packages below.

      ```bash
      wget https://github.com/CVCUDA/CV-CUDA/releases/download/v0.6.0-beta/cvcuda-lib-0.6.0_beta-cuda11-x86_64-linux.deb \
            https://github.com/CVCUDA/CV-CUDA/releases/download/v0.6.0-beta/cvcuda-python3.8-0.6.0_beta-cuda11-x86_64-linux.deb \
            https://github.com/CVCUDA/CV-CUDA/releases/download/v0.6.0-beta/cvcuda_cu11-0.6.0b0-cp310-cp310-linux_x86_64.whl \
            -P /tmp/cvcuda && \
      apt-get install -y /tmp/cvcuda/*.deb && \
      pip3 install /tmp/cvcuda/*.whl
      ```
4. Start the triton server.
   Update the `inference_backend` parameter in config.pbtxt to "pytorch" or "tensorrt". Default backend is "tensorrt"

      ```bash
      tritonserver --model-repository `pwd`/samples/segmentation/python/triton_models [--log-info=1]
      ```
## Triton Client instructions

1. Launch the triton client docker

      ```bash
      docker run -ti --net host --gpus=all -v <local_mount_path>:/cvcuda -w /cvcuda nvcr.io/nvidia/tritonserver:22.12-py3-sdk /bin/bash
      ```
In case the client and server are on the same machine in a local-server setup, we can simply reuse the server image (and even docker exec into the same container) by installing the Triton client utilities:

      ```bash
      pip3 install tritonclient[all]
      ```

Convert local video file to stream data, we need [PyAV](https://github.com/PyAV-Org/PyAV), the Pythonic bindings for FFmpeg libraries, which is already in the `install_dependencies.sh`:

      ```bash
      pip3 install av
      ```
2. Install the dependencies

      ```bash
      cd /cvcuda
      ./samples/scripts/install_dependencies.sh
      ```

3. Run client script for different data modalities

    a. Run segmentation on a single image

      ```bash
      python3 ./samples/segmentation/python/triton_client.py -i ./samples/assets/images/tabby_tiger_cat.jpg -b 1
      ```

    b. Run segmentation on folder containing images

      ```bash
      python3 ./samples/segmentation/python/triton_client.py -i ./samples/assets/images -b 2
      ```

    c. Run segmentation on a video file

      ```bash
      python3 ./samples/segmentation/python/triton_client.py -i ./samples/assets/videos/pexels-ilimdar-avgezer-7081456.mp4 -b 4
      ```
    d. Run segmentation on a video file with streamed encoding/decoding (highly recommended as performance is greatly improved in this mode), use --stream_video or -sv

      ```bash
      python3 ./samples/segmentation/python/triton_client.py -i ./samples/assets/videos/pexels-ilimdar-avgezer-7081456.mp4 -o /tmp -b 4 -sv [--log_level=debug]
      ```

4. To benchmark this client run, we can use the benchmark.py in the following way. It should launch 1 process, ignore 1 batch from front and end as warmup batches, save per process and overall numbers as JSON files in /tmp directory. To understand more about performance benchmarking in CV-CUDA, please refer to [Performance Benchmarking README](../../scripts/README.md)

      ```bash
      python3 ./samples/scripts/benchmark.py -np 1 -w 1 -o /tmp ./samples/segmentation/python/triton_client.py -i ./samples/assets/videos/pexels-ilimdar-avgezer-7081456.mp4 -b 4 -sv
      ```
