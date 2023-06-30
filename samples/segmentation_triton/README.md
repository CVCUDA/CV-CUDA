# This sample shows how to deploy the the Segmentation pipeline on Triton server using the Triton python backend

## Pre-requisites

- Recommended linux distros:
    - Ubuntu >= 20.04 (tested with 20.04 and 22.04)
    - WSL2 with Ubuntu >= 20.04 (tested with 20.04)
- CUDA driver >= 11.7
- Triton server and client docker >= 22.07
- Refer to the [Samples README](../README.md) for Pre-requisites to run the segmentation pipeline

# Instructions to run the triton sample

## Triton Server instructions

1. Launch the triton server

      ```
      docker run --shm-size=1g --ulimit memlock=-1 -p 8000:8000 -p 8001:8001 -p 8002:8002 --ulimit stack=67108864 -ti --gpus=all -v <local mount path>:/cvcuda -w /cvcuda nvcr.io/nvidia/tritonserver:22.12-py3
      ```
2. Install the dependencies

      ```
      ./samples/scripts/install_dependencies.sh
      ```
3. Start the triton server

      ```
      tritonserver --model-repository `pwd`/samples/segmentation_triton/python/models [--log-verbose=1]
      ```
## Triton Client instructions

1. Launch the triton client docker

      ```
      docker run -ti --net host --gpus=all -v <local_mount_path>:/cvcuda nvcr.io/nvidia/tritonserver:22.12-py3-sdk -w /cvcuda /bin/bash
      ```

2. Install the dependencies

      ```
      cd /cvcuda
      ./samples/scripts/install_dependencies.sh
      ```

3. Run client script for different data modalities

    a. Run segmentation on a single image

      ```
      python3 ./samples/segmentation_triton/python/triton_client.py -i ./samples/assets/images/tabby_tiger_cat.jpg -b 1
      ```

    b. Run segmentation on folder containing images

      ```
      python3 ./samples/segmentation_triton/python/triton_client.py -i ./samples/assets/images -b 2
      ```

    c. Run segmentation on a video file

      ```
      python3 ./samples/segmentation_triton/python/triton_client.py -i ./samples/assets/videos/pexels-ilimdar-avgezer-7081456.mp4 -b 4
      ```
