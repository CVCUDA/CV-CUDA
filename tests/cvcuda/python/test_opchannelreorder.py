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

import cvcuda
import numpy as np
import util


RNG = np.random.default_rng(0)


def test_op_channelreorder_varshape():

    input = util.create_image_batch(10, cvcuda.Format.RGB8, size=(123, 321), rng=RNG)
    order = util.create_tensor((10, 3), np.int32, "NC", max_random=(2, 2, 2), rng=RNG)

    out = cvcuda.channelreorder(input, order)
    assert len(out) == len(input)
    assert out.capacity == input.capacity
    assert out.uniqueformat == input.uniqueformat
    assert out.maxsize == input.maxsize

    order = util.create_tensor(
        (10, 4), np.int32, "NC", max_random=(3, 3, 3, 3), rng=RNG
    )
    out = cvcuda.channelreorder(input, order, format=cvcuda.Format.BGRA8)

    assert len(out) == len(input)
    assert out.capacity == input.capacity
    assert out.uniqueformat == cvcuda.Format.BGRA8
    assert out.maxsize == input.maxsize

    stream = cvcuda.Stream()
    out = util.clone_image_batch(input)
    tmp = cvcuda.channelreorder_into(src=input, dst=out, orders=order, stream=stream)
    assert tmp is out
    assert len(out) == len(input)
    assert out.capacity == input.capacity
    assert out.uniqueformat == input.uniqueformat
    assert out.maxsize == input.maxsize
