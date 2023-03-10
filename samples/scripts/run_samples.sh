#!/bin/bash -e

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

# Usage: run_samples.sh

# Crop and Resize Sample
# Batch size 2
LD_LIBRARY_PATH=./lib ./bin/nvcv_samples_cropandresize -i ./assets/images/ -b 2
export CUDA_MODULE_LOADING="LAZY"

mkdir -p models
# Export onnx model from torch
python3 ./scripts/export_resnet.py
# Serialize models
./scripts/serialize_models.sh

# Run classification sample for single image with batch size 1
python3 ./classification/python/inference.py -i ./assets/images/tabby_tiger_cat.jpg -l ./models/imagenet-classes.txt -b 1
# Run classification sample for single image with batch size 4, Uses Same image multiple times
python3 ./classification/python/inference.py -i ./assets/images/tabby_tiger_cat.jpg -l ./models/imagenet-classes.txt -b 4
# Run classification sample for image directory as input with batch size 2
python3 ./classification/python/inference.py -i ./assets/images/ -l ./models/imagenet-classes.txt -b 2
# Run the segmentation sample with default settings, without any command-line args.
python3 ./segmentation/python/inference.py
# Run it on a single image with high batch size for the background class writing to a specific directory
python3 segmentation/python/inference.py -i assets/images/tabby_tiger_cat.jpg -o /tmp -b 5 -c __background__
# Run it on a folder worth of images
python3 segmentation/python/inference.py -i assets/images/ -o /tmp -b 5 -c __background__
# Run on a single image with custom resized input given to the sample for the dog class
python3 segmentation/python/inference.py -i assets/images/Weimaraner.jpg -o /tmp -b 1 -c dog -th 224 -tw 224
# Run the segmentation sample with default settings for TensorRT backend.
python3 ./segmentation/python/inference.py -bk tensorrt
# Run it on a folder worth of images with TensorRT
python3 segmentation/python/inference.py -i assets/images/ -o /tmp -b 5 -c __background__ -bk tensorrt
# Run it on a video for class background.
python segmentation/python/inference.py -i assets/videos/pexels-ilimdar-avgezer-7081456.mp4 -b 5 -c __background__

# Classification sample
# Batch size 1
LD_LIBRARY_PATH=./lib ./bin/nvcv_samples_classification -e ./models/resnet50.engine -i ./assets/images/tabby_tiger_cat.jpg -l ./models/imagenet-classes.txt -b 1
# Batch size 2
LD_LIBRARY_PATH=./lib ./bin/nvcv_samples_classification -e ./models/resnet50.engine -i ./assets/images/tabby_tiger_cat.jpg -l ./models/imagenet-classes.txt -b 2
