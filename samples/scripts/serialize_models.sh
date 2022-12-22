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

# Classification
# resnet50

mkdir -p models

if [ ! -f ./models/imagenet-classes.txt ]
then
        wget https://raw.githubusercontent.com/xmartlabs/caffeflow/master/examples/imagenet/imagenet-classes.txt -O models/imagenet-classes.txt
fi

if [ ! -f ./models/resnet50.engine ]
then
        /opt/tensorrt/bin/trtexec --onnx=models/resnet50.onnx --saveEngine=models/resnet50.engine --minShapes=input:1x3x224x224 --maxShapes=input:32x3x224x224 --optShapes=input:32x3x224x224
fi
