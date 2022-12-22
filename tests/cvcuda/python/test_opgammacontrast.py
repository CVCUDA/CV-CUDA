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
import pytest as t
import numpy as np
import util

RNG = np.random.default_rng(0)


@t.mark.parametrize(
    "nimages, format, max_size, max_pixel, max_gamma",
    [
        (
            5,
            cvcuda.Format.RGB8,
            (16, 23),
            128.0,
            12,
        ),
        (
            5,
            cvcuda.Format.RGB8,
            (10, 15),
            256.0,
            6,
        ),
        (
            5,
            cvcuda.Format.RGB8,
            (8, 8),
            256.0,
            7,
        ),
        (
            4,
            cvcuda.Format.RGB8,
            (11, 23),
            256.0,
            9,
        ),
    ],
)
def test_op_gamma_contrastvarshape(
    nimages,
    format,
    max_size,
    max_pixel,
    max_gamma,
):

    input = util.create_image_batch(
        nimages, format, max_size=max_size, max_random=max_pixel, rng=RNG
    )

    gamma = util.create_tensor((nimages), np.float32, "N", max_gamma, rng=RNG)

    out = cvcuda.gamma_contrast(input, gamma)

    assert len(out) == len(input)
    assert out.capacity == input.capacity
    assert out.uniqueformat == input.uniqueformat
    assert out.maxsize == input.maxsize

    stream = cvcuda.Stream()

    out = util.clone_image_batch(input)

    tmp = cvcuda.gamma_contrast_into(
        src=input,
        dst=out,
        gamma=gamma,
        stream=stream,
    )
    assert tmp is out
    assert len(out) == len(input)
    assert out.capacity == input.capacity
    assert out.uniqueformat == input.uniqueformat
    assert out.maxsize == input.maxsize
