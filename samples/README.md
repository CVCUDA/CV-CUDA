# CV-CUDA Samples

## Description

These are some sample applications showcasing various CV-CUDA APIs. Sample applications are available in C++ and Python.

## Pre-requisites

- Recommended linux distros:
    - Ubuntu >= 20.04 (tested with 20.04 and 22.04)
    - WSL2 with Ubuntu >= 20.04 (tested with 20.04)
- NVIDIA driver
    - Linux: Driver version 520.56.06 or higher
- TensorRT == 8.5.2.2
- NVIDIA Video Processing Framework (https://github.com/NVIDIA/VideoProcessingFramework)
    - Follow the instructions from Github (https://github.com/NVIDIA/VideoProcessingFramework/blob/master/README.md) to install it via pip.
    - Note: ffmpeg is a VPF dependency. It can be built from source by following these steps (https://docs.nvidia.com/video-technologies/video-codec-sdk/12.0/ffmpeg-with-nvidia-gpu/index.html). The version of ffmpeg that comes/installs via apt-get in Ubuntu 20.04 may not be sufficient for VPF.
    - Note: When installing VPF in a docker image like TensorRT, there is no need to install `libnvidia-encode` and `libnvidia-decode` as those already come preinstalled. Other docker images may require an installation of these libraries.
- NVIDIA TAO Converter == 4.0.0
- NVIDIA NSIGHT == 2023.2.1 (only if you wish to run the benchmarking code)
- Python Packages:
    - torch == 1.13.0
    - torchvision == 0.14.0
    - torchnvjpeg (https://github.com/itsliupeng/torchnvjpeg)
    - av == 10.0.0
    - pycuda == 2022.1
    - nvtx == 0.2.5

Setting up the following is only required if you want to setup and run the samples in a docker container:
- nvidia-docker v2.11.0
- A working NVIDIA NGC account (visit https://ngc.nvidia.com/setup to get started using NGC) and follow through the NGC documentation here https://docs.nvidia.com/ngc/ngc-catalog-user-guide/index.html#ngc-image-prerequisites
- docker CLI logged into nvcr.io (NGC's docker registry) to be able to pull docker images.


## Steps to compile the samples from source

1. Get your CUDA and TensorRT installations ready. If you wish to install CUDA and TensorRT on your existing system you may do so by downloading those packages from NVIDIA's website. Or if you wish to work with in a docker container, you can use the TensorRT docker from NVIDIA NGC's catalog. It comes with CUDA and TensorRT pre-installed. Make sure you have setup NGC account properly and that your local docker installation has been logged into nvcr.io domain to be able to pull from that registry. Run the following command to start the container and continue rest of the installation steps in that container. Fill in the local_mount_path and docker_mount_path to reflect any paths on your system which you want to mount inside the container as well. This container comes with Ubuntu 20.04 with Python 3.8.10.

      ```bash
      docker run -it --gpus=all -v <local_mount_path>:<docker_mount_path> nvcr.io/nvidia/tensorrt:22.09-py3
      ```

2. Make sure that the other helper scripts present in the `samples/scripts` folder is executable by executing following chmod commands.

   ```bash
   cd samples
   chmod a+x scripts/*.sh
   chmod a+x scripts/*.py
   ```

3. Install all the dependencies required to run the samples. These are mentioned above in the prerequisites section. A convenient script to install all the dependencies is available at `scripts/install_dependencies.sh`. This script may require sudo privileges depending on your setup.

   ```bash
   ./scripts/install_dependencies.sh
   ```

4. Install the CV-CUDA packages. Please note that since the above container comes with Python 3.8.10, we will install nvcv-python3.8-0 package as mentioned below. If you have any other Python distributions, you would need to use the appropriate nvcv-python Debian package below.

   ```bash
   dpkg -i nvcv-lib-0.4.0_beta-cuda11-x86_64-linux.deb
   dpkg -i nvcv-dev-0.4.0_beta-cuda11-x86_64-linux.deb
   dpkg -i cvcuda-samples-0.4.0_beta-cuda11-x86_64-linux.deb
   dpkg -i nvcv-python3.8-0.4.0_beta-cuda11-x86_64-linux.deb
   ```
5. Copy the samples folder to the target directory.

   ```bash
   cp -rf /opt/nvidia/cvcuda*/samples ~/
   cd ~/samples
   ```

6. Build the samples (whichever sample requires a build)

   ```bash
   ./scripts/build_samples.sh
   ```

7. Run all the samples on by one. The `run_samples.sh` script conveniently runs all the samples in one shot. Some samples may use the TensorRT backend to run the inference and it may require a serialization step to convert a PyTorch model into a TensorRT model. This step should take some time depending on the GPUs used but usually it is only done once during the first run of the sample. The `run_samples.sh` script is supplied to serve only as a basic test case to test the samples under most frequently used command line parameters. It does not cover all the settings and command line parameters a sample may have to offer. Please explore and run the samples individually to explore all the capabilities of the samples.

   ```bash
   ./scripts/run_samples.sh
   ```

## Performance Benchmarking

See the [Performance Benchmarking](scripts/README.md) documentation.
