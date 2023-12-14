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

import cvcuda
import pytest as t
import numpy as np
import random

random.seed(1)


@t.mark.parametrize(
    "input, dtype, number",
    [
        (((5, 16, 23, 4), np.uint8, "NHWC"), np.int8, 2),
        (((1, 160, 221, 2), np.uint8, "NHWC"), np.int8, 3),
        (((1, 60, 1, 1), np.uint8, "NHWC"), np.int8, 1),
        (((6, 61, 12, 3), np.uint8, "NHWC"), np.int8, 5),
        (((5, 161, 23, 4), np.uint8, "NCHW"), np.int8, 2),
        (((1, 160, 221, 2), np.uint8, "NCHW"), np.int8, 3),
        (((1, 1, 2, 1), np.uint8, "NCHW"), np.int8, 1),
        (((6, 13, 1, 3), np.uint8, "NCHW"), np.int8, 5),
        (((16, 23, 4), np.uint8, "HWC"), np.int8, 2),
        (((160, 221, 2), np.uint8, "HWC"), np.int8, 3),
        (((60, 1, 1), np.uint8, "HWC"), np.int8, 1),
        (((61, 12, 3), np.uint8, "HWC"), np.int8, 5),
        (((161, 23, 4), np.uint8, "CHW"), np.int8, 2),
        (((160, 221, 2), np.uint8, "CHW"), np.int8, 3),
        (((1, 2, 1), np.uint8, "CHW"), np.int8, 1),
        (((13, 1, 3), np.uint8, "CHW"), np.int8, 5),
    ],
)
def test_op_stack(input, dtype, number):

    input_tensors = []

    numberOfTensors = 0

    updated_input = list(input)
    for _ in range(number):
        if updated_input[2] == "NHWC" or updated_input[2] == "NCHW":
            updated_input[0] = (random.randint(1, input[0][0]),) + input[0][
                1:
            ]  # Update the first value
            numberOfTensors += updated_input[0][0]
        else:
            numberOfTensors += 1
        input_tensor = cvcuda.Tensor(*updated_input)
        input_tensors.append(input_tensor)

    out = cvcuda.stack(input_tensors)

    assert out.shape[0] == numberOfTensors
    assert out.dtype == input_tensors[0].dtype

    if input_tensors[0].shape == 3:
        assert out.shape[1] == input_tensors[0].shape[0]
        assert out.shape[2] == input_tensors[0].shape[1]
        assert out.shape[3] == input_tensors[0].shape[2]
    if input_tensors[0].shape == 4:
        assert out.layout == input_tensors[0].layout
        assert out.shape[1] == input_tensors[0].shape[1]
        assert out.shape[2] == input_tensors[0].shape[2]
        assert out.shape[3] == input_tensors[0].shape[3]

    # check stack into
    outputTensorDef = list(updated_input)
    if updated_input[2] == "NHWC" or updated_input[2] == "NCHW":
        outputTensorDef[0] = (numberOfTensors,) + input[0][1:]
    else:
        outputTensorDef[0] = (numberOfTensors,) + input[0][0:]
        if updated_input[2] == "HWC":
            outputTensorDef[2] = "NHWC"
        else:
            outputTensorDef[2] = "NCHW"

    output_tensor = cvcuda.Tensor(*outputTensorDef)
    cvcuda.stack_into(output_tensor, input_tensors)

    assert output_tensor.shape[0] == numberOfTensors
    assert output_tensor.dtype == input_tensors[0].dtype

    if input_tensors[0].shape == 3:
        assert output_tensor.shape[1] == input_tensors[0].shape[0]
        assert output_tensor.shape[2] == input_tensors[0].shape[1]
        assert output_tensor.shape[3] == input_tensors[0].shape[2]
    if input_tensors[0].shape == 4:
        assert output_tensor.layout == input_tensors[0].layout
        assert output_tensor.shape[1] == input_tensors[0].shape[1]
        assert output_tensor.shape[2] == input_tensors[0].shape[2]
        assert output_tensor.shape[3] == input_tensors[0].shape[3]
