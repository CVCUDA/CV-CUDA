# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import torch
import pytest


def test_cache_limit_get_set():
    nvcv.clear_cache()

    # Verify initial cache limit (half of total gpu mem)
    total = torch.cuda.mem_get_info()[1]
    assert nvcv.get_cache_limit_inbytes() == total // 2

    # Verify we can also set the cache limit
    nvcv.set_cache_limit_inbytes(total)
    assert nvcv.get_cache_limit_inbytes() == total


def test_cache_current_byte_size():
    nvcv.clear_cache()

    nvcv_cache_size = 0
    assert nvcv.current_cache_size_inbytes() == nvcv_cache_size

    img_create = nvcv.Image.zeros((1, 1), nvcv.Format.F32)
    nvcv_cache_size += nvcv.internal.nbytes_in_cache(img_create)
    assert nvcv.current_cache_size_inbytes() == nvcv_cache_size

    image_batch_create = nvcv.ImageBatchVarShape(5)
    nvcv_cache_size += nvcv.internal.nbytes_in_cache(image_batch_create)
    assert nvcv.current_cache_size_inbytes() == nvcv_cache_size

    stream = nvcv.cuda.Stream()
    nvcv_cache_size += nvcv.internal.nbytes_in_cache(stream)
    assert nvcv.current_cache_size_inbytes() == nvcv_cache_size

    tensor_create = nvcv.Tensor(2, (37, 7), nvcv.Format.RGB8, rowalign=1)
    nvcv_cache_size += nvcv.internal.nbytes_in_cache(tensor_create)
    assert nvcv.current_cache_size_inbytes() == nvcv_cache_size

    tensor_batch_create = nvcv.TensorBatch(10)
    nvcv_cache_size += nvcv.internal.nbytes_in_cache(tensor_batch_create)
    assert nvcv.current_cache_size_inbytes() == nvcv_cache_size


def test_cache_external_cacheitem():
    nvcv.clear_cache()

    input_tensor = torch.rand(2, 30, 16, 1).cuda()
    input_tensor = input_tensor * 255
    input_tensor = input_tensor.to(dtype=torch.uint8)
    frames_cvcuda = cvcuda.as_tensor(input_tensor, "NHWC")
    assert nvcv.current_cache_size_inbytes() == 0

    frames_cvcuda_out = cvcuda.advcvtcolor(
        frames_cvcuda, cvcuda.ColorConversion.YUV2RGB_NV12, cvcuda.ColorSpec.BT2020
    )
    assert (
        nvcv.current_cache_size_inbytes()
        == nvcv.internal.nbytes_in_cache(frames_cvcuda_out)
    ) and (nvcv.internal.nbytes_in_cache(frames_cvcuda_out) > 0)


def test_cache_limit_clearing():
    nvcv.clear_cache()

    img_create = nvcv.Image.zeros((1, 1), nvcv.Format.F32)
    img_cache_size = nvcv.internal.nbytes_in_cache(img_create)

    # Cache should be emptied if new set limit is smaller than current cache size
    nvcv.set_cache_limit_inbytes(img_cache_size - 1)
    assert nvcv.current_cache_size_inbytes() == 0
    del img_create

    # Element should not be added to Cache, if its size exceeds cache limit
    nvcv.set_cache_limit_inbytes(img_cache_size - 1)
    img_create = nvcv.Image.zeros((1, 1), nvcv.Format.F32)
    assert nvcv.current_cache_size_inbytes() == 0
    del img_create

    # If cache grows too large, cache should be emptied and new element should be added
    nvcv.set_cache_limit_inbytes(img_cache_size)
    img_create = nvcv.Image.zeros((1, 1), nvcv.Format.F32)
    assert nvcv.current_cache_size_inbytes() == img_cache_size
    img_create2 = nvcv.Image.zeros((1, 1), nvcv.Format.F32)
    assert nvcv.current_cache_size_inbytes() == img_cache_size
    del img_create
    del img_create2


def test_cache_zero_cache_limit():
    nvcv.set_cache_limit_inbytes(0)

    assert nvcv.get_cache_limit_inbytes() == 0

    img_create = nvcv.Image.zeros((1, 1), nvcv.Format.F32)
    assert nvcv.internal.nbytes_in_cache(img_create) > 0
    assert nvcv.current_cache_size_inbytes() == 0


def test_cache_negative_cache_limit():
    with pytest.raises(ValueError):
        nvcv.set_cache_limit_inbytes(-1)
