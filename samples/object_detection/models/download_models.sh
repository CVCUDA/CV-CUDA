#!/bin/bash -e

# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Object Detection
# PeopleNet model
# This model is based on NVIDIA DetectNet_v2 detector with ResNet34 as feature extractor.

OUT_DIR='/tmp'

if [[ $# -ge 1 ]]; then
   OUT_DIR=$1
fi

# Download the etlt model and the labels files from NGC

if [ ! -f $OUT_DIR/resnet34_peoplenet_int8.etlt ]
then
	wget 'https://api.ngc.nvidia.com/v2/models/nvidia/tao/peoplenet/versions/deployable_quantized_v2.6.1/files/resnet34_peoplenet_int8.etlt' -P $OUT_DIR
fi

if [ ! -f $OUT_DIR/labels.txt ]
then
	wget 'https://api.ngc.nvidia.com/v2/models/nvidia/tao/peoplenet/versions/deployable_quantized_v2.6.1/files/labels.txt' -P $OUT_DIR
fi

# Use tao-converter which parses the .etlt model file, and generates an optimized TensorRT engine
# The model supports implicit batch dimension which requires the max batch size, input layer dimensions and ordering to be specified.
if [ ! -f ${OUT_DIR}/peoplenet.engine ]
then
	/tmp/tao_binaries/tao-converter -e $OUT_DIR/peoplenet.engine -k tlt_encode -d 3,544,960 -m 32 -i nchw $OUT_DIR/resnet34_peoplenet_int8.etlt
fi
