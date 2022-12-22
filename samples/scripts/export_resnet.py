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

import torch
from torchvision import models

# Get pretrained model
resnet50 = models.resnet50(pretrained=True)

# Export the model to ONNX
inputWidth = 224
inputHeight = 224
maxBatchSize = 32
x = torch.randn(maxBatchSize, 3, inputHeight, inputWidth, requires_grad=True)
torch.onnx.export(
    resnet50,
    x,
    "./models/resnet50.onnx",
    export_params=True,
    opset_version=12,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "maxBatchSize"}, "output": {0: "maxBatchSize"}},
)
