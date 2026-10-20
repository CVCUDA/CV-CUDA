# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import numpy as np
import pytest
import warnings

import cvcuda
import cvcuda_util as util
import cupy
from cvcuda_test_utils import as_cupy_stream


RNG = np.random.default_rng(0)


@pytest.mark.parametrize("handle_kind", ["owned", "null", "legacy", "ptds"])
def test_as_cupy_stream_is_warning_free_and_non_owning(handle_kind):
    owner = cvcuda.Stream()
    handle = {"owned": owner.handle, "null": 0, "legacy": 1, "ptds": 2}[handle_kind]
    previous = cupy.cuda.get_current_stream().ptr

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        external = as_cupy_stream(handle)
        assert external.ptr == handle
        with external:
            assert cupy.cuda.get_current_stream().ptr == handle
            values = cupy.full((4,), 37, dtype=cupy.uint8)
        assert cupy.cuda.get_current_stream().ptr == previous
        external.synchronize()
        del external

    cupy.cuda.runtime.streamSynchronize(handle)
    np.testing.assert_array_equal(values.get(), np.full((4,), 37, dtype=np.uint8))


def test_create_tensor_uint16():
    tensor = util.create_tensor([10], np.uint16, None, max_random=(65535))
    assert tensor.dtype == np.uint16


def test_create_tensor_odd():
    tensor = util.create_tensor(
        [10], np.uint8, None, max_random=255, rng=RNG, transform_dist=util.dist_odd
    )
    h_data = cupy.asarray(tensor.cuda()).get()
    assert all([bool(val % 2 == 1) for val in h_data])


def test_create_image():
    image = util.create_image((11, 13), cvcuda.Format.RGB8, max_random=100, rng=RNG)
    assert image.size == (11, 13)
    assert image.format == cvcuda.Format.RGB8
    assert all([bool(val < 100) for val in np.array(image.cpu()).flatten()])


def test_create_image_batch():
    batch = util.create_image_batch(
        19, cvcuda.Format.RGBA8, max_size=(3, 3), max_random=33, rng=RNG
    )
    assert batch.capacity == 19
    assert len(batch) == 19
    assert batch.uniqueformat == cvcuda.Format.RGBA8
    assert batch.maxsize == (3, 3)
    assert all(
        [bool(val < 33) for img in batch for val in np.array(img.cpu()).flatten()]
    )


def test_clone_image_batch():
    batch1 = util.create_image_batch(17, cvcuda.Format.RGB8, size=(4, 4))
    batch2 = util.clone_image_batch(batch1)
    assert batch2.capacity == 17
    assert len(batch2) == 17
    assert batch2.uniqueformat == cvcuda.Format.RGB8
    assert batch2.maxsize == (4, 4)
