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

# Usage: benchmark_samples.sh <OUTPUT_DIR>

# Performs benchmarking of all Python samples.
# Since some samples may involve creation of a TensorRT model on the first run and since it takes
# a fair amount of time, we will need to run the benchmarking twice to get the correct results.
# Only the results of the second run will be used. The model artifacts from the first run will
# help us run the second run easily.


set -e  # Stops this script if any one command fails.

if [ "$#" -lt 1 ]; then
    echo "Usage: benchmark_samples.sh <OUTPUT_DIR> {USE_TENSORRT: True}"
    exit 1
fi

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
SAMPLES_ROOT="$(dirname "$SCRIPT_DIR")"  # removes the scripts dir
OUTPUT_DIR="$1"
USE_TRT=${2:-True}
CLASSIFICATION_OUT_DIR="$OUTPUT_DIR/classification"
SEGMENTATION_OUT_DIR="$OUTPUT_DIR/segmentation"
DETECTION_OUT_DIR="$OUTPUT_DIR/detection"

mkdir -p "$CLASSIFICATION_OUT_DIR"
mkdir -p "$SEGMENTATION_OUT_DIR"
mkdir -p "$DETECTION_OUT_DIR"

echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "CLASSIFICATION_OUT_DIR: $CLASSIFICATION_OUT_DIR"
echo "SEGMENTATION_OUT_DIR: $SEGMENTATION_OUT_DIR"
echo "DETECTION_OUT_DIR: $DETECTION_OUT_DIR"
if [ "$USE_TRT" = "True" ]; then
    echo "Using TensorRT as the inference back-end in all the runs."
    CLASSIFICATION_BACKEND="tensorrt"
    SEGMENTATION_BACKEND="tensorrt"
    DETECTION_BACKEND="tensorrt"
else
    echo "Not using TensorRT as the inference back-end in all the runs."
    CLASSIFICATION_BACKEND="pytorch"
    SEGMENTATION_BACKEND="pytorch"
    DETECTION_BACKEND="tensorflow"
fi

# 1. The Classification sample
# First dry run with 2 processes and 1 batch from start and end used as a warm-up batch.
echo "Running the classification sample (warm-up run)..."
python3 "$SCRIPT_DIR/benchmark.py" \
    -np 2 \
    -w 1 \
    -o "$CLASSIFICATION_OUT_DIR" \
    "$SAMPLES_ROOT/classification/python/main.py" \
    -b 4 \
    -bk $CLASSIFICATION_BACKEND \
    -i "$SAMPLES_ROOT/assets/videos/pexels-chiel-slotman-4423925-1920x1080-25fps.mp4"
# Second run - the actual run.
echo "Running the classification sample (actual run)..."
python3 "$SCRIPT_DIR/benchmark.py" \
    -np 2 \
    -w 1 \
    -o "$CLASSIFICATION_OUT_DIR" \
    "$SAMPLES_ROOT/classification/python/main.py" \
    -b 4 \
    -bk $CLASSIFICATION_BACKEND \
    -i "$SAMPLES_ROOT/assets/videos/pexels-chiel-slotman-4423925-1920x1080-25fps.mp4"

# 2. The Segmentation sample
# First dry run with 2 processes and 1 batch from start and end used as a warm-up batch.
echo "Running the segmentation sample (warm-up run)..."
python3 "$SCRIPT_DIR/benchmark.py" \
    -np 2 \
    -w 1 \
    -o "$SEGMENTATION_OUT_DIR" \
    "$SAMPLES_ROOT/segmentation/python/main.py" \
    -b 4 \
    -bk $SEGMENTATION_BACKEND \
    -i "$SAMPLES_ROOT/assets/videos/pexels-ilimdar-avgezer-7081456.mp4"
# Second run - the actual run.
echo "Running the segmentation sample (actual run)..."
python3 "$SCRIPT_DIR/benchmark.py" \
    -np 2 \
    -w 1 \
    -o "$SEGMENTATION_OUT_DIR" \
    "$SAMPLES_ROOT/segmentation/python/main.py" \
    -b 4 \
    -bk $SEGMENTATION_BACKEND \
    -i "$SAMPLES_ROOT/assets/videos/pexels-ilimdar-avgezer-7081456.mp4"

# 3. The Object Detection sample
# First dry run with 2 processes and 1 batch from start and end used as a warm-up batch.
echo "Running the detection sample (warm-up run)..."
python3 "$SCRIPT_DIR/benchmark.py" \
    -np 1 \
    -w 1 \
    -o "$DETECTION_OUT_DIR" \
    "$SAMPLES_ROOT/object_detection/python/main.py" \
    -b 4 \
    -bk $DETECTION_BACKEND \
    -i "$SAMPLES_ROOT/assets/videos/pexels-chiel-slotman-4423925-1920x1080-25fps.mp4"
# Second run - the actual run.
echo "Running the detection sample (actual run)..."
python3 "$SCRIPT_DIR/benchmark.py" \
    -np 1 \
    -w 1 \
    -o "$DETECTION_OUT_DIR" \
    "$SAMPLES_ROOT/object_detection/python/main.py" \
    -b 4 \
    -bk $DETECTION_BACKEND \
    -i "$SAMPLES_ROOT/assets/videos/pexels-chiel-slotman-4423925-1920x1080-25fps.mp4"

# Done.
