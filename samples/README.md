
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

# CV-CUDA Samples

## Description

CV-CUDA samples are written to showcase the use of various CV-CUDA APIs to construct fully functional end-to-end deep learning inference pipelines. Sample applications are available in C++ and Python.

## Pre-requisites

- Recommended Linux distributions:
    - Ubuntu >= 20.04 (tested with 20.04 and 22.04)
    - WSL2 with Ubuntu >= 20.04 (tested with 20.04)
- NVIDIA driver:
    - Linux: Driver version >= 535
- NVIDIA TensorRT >= 8.6.1
- NVIDIA nvImageCodec (https://github.com/NVIDIA/nvImageCodec)
- NVIDIA PyNvVideoCodec (https://catalog.ngc.nvidia.com/orgs/nvidia/resources/py_nvvideocodec)
- NVIDIA Video Processing Framework (only if running the Triton sample) (https://github.com/NVIDIA/VideoProcessingFramework)
    - Note: When installing VPF in a docker image like TensorRT, there is no need to install `libnvidia-encode` and `libnvidia-decode` as those already come pre-installed. Other docker images may require an installation of these libraries.
- NVIDIA TAO Converter >= 4.0.0
- NVIDIA NSIGHT == 2023.2.1 (only if you wish to run the benchmarking code)
- Additional Python packages requirements listed in the `requirements.txt` file under the `samples/scripts/` folder.



## Setting up the environment

1. We strongly recommend working in a docker container to set things up. This would greatly simplify the process of installing dependencies, compiling and running the samples. The following is required to work in a docker container with CV-CUDA samples:
   1. nvidia-docker >= 2.11.0
   2. A working NVIDIA NGC account (visit https://ngc.nvidia.com/setup to get started using NGC) and follow through the NGC documentation on https://docs.nvidia.com/ngc/ngc-catalog-user-guide/index.html#ngc-image-prerequisites
   3. docker CLI logged into nvcr.io (NGC's docker registry) to be able to pull docker image. (e.g. using `docker login nvcr.io`)

2. Clone this CV-CUDA git repository. We would call the location where it is stored as `CVCUDA_ROOT`.

3. Make sure your CUDA and TensorRT installations are ready. If you wish to install CUDA and TensorRT on your existing system, you may do so by downloading those packages from NVIDIA's website. If you are using docker, use the TensorRT container from NVIDIA NGC. It comes with CUDA and TensorRT pre-installed:
   1. Run the following command to start the container and continue rest of the steps in that container. Fill in the `CVCUDA_ROOT` with the location where you have cloned this CV-CUDA repository. This will make the samples available inside the container at the `/workspace/cvcuda_samples` path. Also fill in the `CVCUDA_INSTALL` with the location where CV-CUDA installation packages (.deb or .whl files) are stored. This container comes with Ubuntu v22.04, Python v3.10.12 and TensorRT v8.6.1.

      ```bash
      docker run -it --gpus=all -v <CVCUDA_ROOT>/samples:/workspace/cvcuda_samples -v <CVCUDA_INSTALL>:/workspace/cvcuda_install nvcr.io/nvidia/tensorrt:24.01-py3
      ```

3. Make sure the scripts present in the `/workspace/cvcuda_samples/scripts` directory is executable by executing following chmod commands:

   ```bash
   cd /workspace/cvcuda_samples/  # Assuming this is where the samples are
   chmod a+x ./scripts/*.sh
   chmod a+x ./scripts/*.py
   ```

4. Install all dependencies required to build and/or run the samples. These are mentioned above in the prerequisites section. A convenient script to install all the dependencies is available at `scripts/install_dependencies.sh`.

   ```bash
   cd /workspace/cvcuda_samples/  # Assuming this is where the samples are
   ./scripts/install_dependencies.sh
   ```

5. Install CV-CUDA packages. If you are only interested in running the Python samples, you would be fine installing just the Python wheel. If you are interested in building the non-Python samples from source, the Debian packages are required. Since our docker container has Ubuntu 22.04, CUDA 12 and Python 3.10.12, we will install the corresponding CV-CUDA package as shown below:
   1. Using the Python wheel (only works for the Python samples):
      ```bash
      cd /workspace/cvcuda_install/  # Assuming this is where the installation files are
      pip install cvcuda_cu12-0.6.0b0-cp310-cp310-linux_x86_64.whl
      ```

   2. OR using the Debian packages (required to build the non-Python samples from source, also works for the Python samples):

      ```bash
      cd /workspace/cvcuda_install/  # Assuming this is where the installation files are
      dpkg -i cvcuda-lib-0.6.0_beta-cuda12-x86_64-linux.deb
      dpkg -i cvcuda-dev-0.6.0_beta-cuda12-x86_64-linux.deb
      dpkg -i cvcuda-python3.10-0.6.0_beta-cuda12-x86_64-linux.deb
      ```

## Build the samples from source (Not required for Python samples)

1. After following the [Setting up the environment](#setting-up-the-environment) section, execute the following command to compile the samples from source. This only applies to C++ samples. Python samples do not require any compilation.

   ```bash
   cd /workspace/cvcuda_samples/  # Assuming this is where the samples are
   ./scripts/build_samples.sh  # Writes build files in /workspace/cvcuda_samples/build
   ```

## Run the samples

1. After following the [Setting up the environment](#setting-up-the-environment) section and compiling them from source, one can run the samples manually one by one or use the `scripts/run_samples.sh` script to run all samples in one shot. Some samples uses the TensorRT back-end to run the inference and it may require a serialization step to convert a PyTorch model into a TensorRT model. This step should take some time depending on the GPU used but usually it is only done once during the first run of the sample. The `scripts/run_samples.sh` script is supplied to serve only as a basic test case to test the samples under most frequently used command line parameters. It does not cover all the settings and command line parameters a sample may have to offer. Please explore and run the samples individually to explore all the capabilities of the samples.

   ```bash
   cd /workspace/cvcuda_samples/  # Assuming this is where the samples are and built samples are in /workspace/cvcuda_samples/build
   ./scripts/run_samples.sh
   ```

## Performance Benchmarking of the samples

See the [Performance Benchmarking](scripts/README.md) documentation to understand how to benchmark the samples.
