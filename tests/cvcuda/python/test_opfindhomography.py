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

import nvcv
import cvcuda
import cvcuda_util
import pytest as t
import numpy as np


@t.mark.parametrize(
    "num_samples, num_points",
    [
        (16, 1024),
        (32, 1024),
        (64, 1024),
    ],
)
def test_op_findhomography(num_samples, num_points):
    tensor_args = ((num_samples, num_points * 2), nvcv.Type._2F32, "NW")
    src = cvcuda.Tensor(*tensor_args)
    dst = cvcuda.Tensor(*tensor_args)
    out = cvcuda.findhomography(src, dst)
    assert out.shape == (num_samples, 3, 3)
    assert out.dtype == np.float32

    create_tensor_args = ((num_samples, num_points, 2), np.float32, "NWC")
    src = cvcuda_util.create_tensor(*create_tensor_args)
    dst = cvcuda_util.create_tensor(*create_tensor_args)

    stream = cvcuda.Stream()
    out_tensor_args = ((num_samples, 3, 3), np.float32, "NHW")
    out = cvcuda.Tensor(*out_tensor_args)
    tmp = cvcuda.findhomography_into(
        models=out,
        srcPts=src,
        dstPts=dst,
        stream=stream,
    )
    assert tmp is out
    assert out.shape == (num_samples, 3, 3)
    assert out.dtype == nvcv.Type.F32


@t.mark.parametrize(
    "num_samples, num_points",
    [
        (16, 1024),
        (32, 1024),
        (64, 1024),
    ],
)
def test_op_findhomographyvarshape(num_samples, num_points):
    tensor_args = ((1, num_points * 2), nvcv.Type._2F32, "NW")
    srcBatch = cvcuda.TensorBatch(num_samples)
    dstBatch = cvcuda.TensorBatch(num_samples)
    for i in range(num_samples):
        src = cvcuda.Tensor(*tensor_args)
        dst = cvcuda.Tensor(*tensor_args)
        srcBatch.pushback(src)
        dstBatch.pushback(dst)

    outBatch = cvcuda.findhomography(srcPts=srcBatch, dstPts=dstBatch)
    assert outBatch.dtype == nvcv.Type.F32
    assert outBatch.layout == "NHW"

    stream = cvcuda.Stream()
    out_tensor_args = ((1, 3, 3), np.float32, "NHW")
    outBatch = cvcuda.TensorBatch(num_samples)
    for i in range(num_samples):
        out = cvcuda.Tensor(*(out_tensor_args))
        outBatch.pushback(out)

    tmpBatch = cvcuda.findhomography_into(
        models=outBatch,
        srcPts=srcBatch,
        dstPts=dstBatch,
        stream=stream,
    )
    assert tmpBatch is outBatch
    assert outBatch.ndim == 3
    assert outBatch.dtype == nvcv.Type.F32
    assert outBatch.capacity == srcBatch.capacity
