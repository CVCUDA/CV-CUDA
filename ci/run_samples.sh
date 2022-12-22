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

# Usage: run_samples.sh

# Crop and Resize
./build/bin/nvcv_samples_cropandresize -i ./samples/assets/ -b 2

# Classification
mkdir -p models
# Export onnx model from torch
if [ ! -f ./models/resnet50.onnx ]
then
        python ./samples/scripts/export_resnet.py
fi
# Serialize model . ONNX->TRT
./samples/scripts/serialize_models.sh
#batch size 2
./build/bin/nvcv_samples_classification -e ./models/resnet50.engine -i ./samples/assets/ -l models/imagenet-classes.txt -b 2
