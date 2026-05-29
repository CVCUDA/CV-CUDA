# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
Additional test for float16 support in CV-CUDA operations
Add these test cases to existing test files after float16 support is merged
"""

import cvcuda
import pytest as t
import numpy as np

try:
    import cupy as cp

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


@t.mark.parametrize(
    "input_args,dtype,scale,offset",
    [
        # Float16 output test cases
        (((5, 16, 23, 4), np.uint8, "NHWC"), np.float16, 1.0 / 255.0, 0.0),
        (((16, 23, 3), np.uint8, "HWC"), np.float16, 1.0 / 255.0, 0.0),
        (((1, 224, 224, 3), np.uint8, "NHWC"), np.float16, 0.5, -0.5),
        # Float16 input to other types
        (((5, 16, 23, 4), np.float16, "NHWC"), np.float32, 1.0, 0.0),
        (((16, 23, 3), np.float16, "HWC"), np.uint8, 255.0, 0.0),
        # Float16 to float16 conversion
        (((1, 224, 224, 3), np.float16, "NHWC"), np.float16, 2.0, 0.5),
    ],
)
def test_op_convertto_float16(input_args, dtype, scale, offset):
    """Test convertto operator with float16 dtype"""
    input = cvcuda.Tensor(*input_args)
    out = cvcuda.convertto(input, dtype, scale, offset)
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == dtype

    out = cvcuda.Tensor(input.shape, dtype, input.layout)
    tmp = cvcuda.convertto_into(out, input, scale, offset)
    assert tmp is out
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == dtype

    stream = cvcuda.Stream()
    out = cvcuda.convertto(
        src=input, dtype=dtype, scale=scale, offset=offset, stream=stream
    )
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == dtype


@t.mark.parametrize(
    "input_args,base_args,scale_args",
    [
        # Float16 base/scale tensors with various input types
        (
            ((5, 16, 23, 4), np.float32, "NHWC"),
            ((1, 1), np.float16, "HW"),
            ((1, 1), np.float16, "HW"),
        ),
        (
            ((5, 16, 23, 4), np.float32, "NHWC"),
            ((1, 1, 4), np.float16, "HWC"),
            ((1, 1, 4), np.float16, "HWC"),
        ),
        (
            ((1, 224, 224, 3), np.float32, "NHWC"),
            ((1, 1, 3), np.float16, "HWC"),
            ((1, 1, 3), np.float16, "HWC"),
        ),
    ],
)
def test_op_normalize_float16(input_args, base_args, scale_args):
    """Test normalize operator with float16 base/scale tensors"""
    input = cvcuda.Tensor(*input_args)
    base = cvcuda.Tensor(*base_args)
    scale = cvcuda.Tensor(*scale_args)

    out = cvcuda.normalize(input, base, scale)
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == input.dtype

    out = cvcuda.Tensor(input.shape, input.dtype, input.layout)
    tmp = cvcuda.normalize_into(out, input, base, scale)
    assert tmp is out
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == input.dtype

    stream = cvcuda.Stream()
    out = cvcuda.normalize(
        src=input,
        base=base,
        scale=scale,
        flags=cvcuda.NormalizeFlags.SCALE_IS_STDDEV,
        stream=stream,
    )
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == input.dtype


@t.mark.skipif(not HAS_CUPY, reason="CuPy not available")
@t.mark.parametrize(
    "shape,layout",
    [
        ((224, 224, 3), "HWC"),
        ((1, 224, 224, 3), "NHWC"),
        ((16, 16, 4), "HWC"),
    ],
)
def test_as_tensor_float16(shape, layout):
    """Test as_tensor with float16 arrays (requires GPU memory)"""
    # Create float16 cupy array (CUDA-accessible memory)
    data = cp.random.randn(*shape).astype(np.float16)

    # Create tensor from float16 data
    tensor = cvcuda.as_tensor(data, layout=layout)

    assert tensor.shape == shape
    assert tensor.layout == layout
    assert tensor.dtype == np.float16
