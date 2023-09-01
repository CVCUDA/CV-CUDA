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

# Usage: benchmark_samples.sh

# Performs benchmarking of all Python samples.
# Since some samples may involve creation of a TensorRT model on the first run and since it takes
# a fair amount of time, we will need to run the benchmarking twice to get the correct results.
# Only the results of the second run will be used. The model artifacts from the first run will
# help us run the second run easily.

mkdir -p /tmp/benchmarking/classification
mkdir -p /tmp/benchmarking/segmentation
mkdir -p /tmp/benchmarking/detection

# 1. The Classification sample
# First dry run with 2 processes and 1 batch from start and end used as a warm-up batch.
python ./scripts/benchmark.py \
    -np 2 \
    -w 1 \
    -o /tmp/benchmarking/classification \
    ./classification/python/main.py \
    -b 4 \
    -i ./assets/videos/pexels-chiel-slotman-4423925-1920x1080-25fps.mp4
# Second run - the actual run.
python ./scripts/benchmark.py \
    -np 2 \
    -w 1 \
    -o /tmp/benchmarking/classification \
    ./classification/python/main.py \
    -b 4 \
    -i ./assets/videos/pexels-chiel-slotman-4423925-1920x1080-25fps.mp4

# 2. The Segmentation sample
# First dry run with 2 processes and 1 batch from start and end used as a warm-up batch.
python ./scripts/benchmark.py \
    -np 2 \
    -w 1 \
    -o /tmp/benchmarking/segmentation \
    ./segmentation/python/main.py \
    -b 4 \
    -i ./assets/videos/pexels-ilimdar-avgezer-7081456.mp4
# Second run - the actual run.
python ./scripts/benchmark.py \
    -np 2 \
    -w 1 \
    -o /tmp/benchmarking/segmentation \
    ./segmentation/python/main.py \
    -b 4 \
    -i ./assets/videos/pexels-ilimdar-avgezer-7081456.mp4

# 3. The Object Detection sample
# First dry run with 2 processes and 1 batch from start and end used as a warm-up batch.
python ./scripts/benchmark.py \
    -np 2 \
    -w 1 \
    -o /tmp/benchmarking/detection \
    ./object_detection/python/main.py \
    -b 4 \
    -i ./assets/videos/pexels-chiel-slotman-4423925-1920x1080-25fps.mp4
# Second run - the actual run.
python ./scripts/benchmark.py \
    -np 2 \
    -w 1 \
    -o /tmp/benchmarking/detection \
    ./object_detection/python/main.py \
    -b 4 \
    -i ./assets/videos/pexels-chiel-slotman-4423925-1920x1080-25fps.mp4

# Done.
