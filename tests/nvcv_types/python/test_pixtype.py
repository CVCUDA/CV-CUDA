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

import nvcv
import pytest as t
import numpy as np


@t.mark.parametrize(
    "type,dt",
    [
        (nvcv.Type.U8, np.uint8),
        (nvcv.Type.U8, np.dtype(np.uint8)),
        (nvcv.Type.S8, np.int8),
        (nvcv.Type.U16, np.uint16),
        (nvcv.Type.S16, np.int16),
        (nvcv.Type.U32, np.uint32),
        (nvcv.Type.S32, np.int32),
        (nvcv.Type.U64, np.uint64),
        (nvcv.Type.S64, np.int64),
        (nvcv.Type.F32, np.float32),
        (nvcv.Type.F64, np.float64),
        (nvcv.Type._2F32, np.complex64),
        (nvcv.Type._2F64, np.complex128),
        (nvcv.Type._3S8, np.dtype("3i1")),
        (nvcv.Type._4S32, np.dtype("4i")),
    ],
)
def test_pixtype_dtype(type, dt):
    assert type == dt

    t = nvcv.Type(dt)
    assert type == t
    assert dt == t


@t.mark.parametrize("dt", [np.dtype([("f1", np.uint64), ("f2", np.int32)]), "invalid"])
def test_pixtype_dtype_conv_error(dt):
    with t.raises(TypeError):
        nvcv.Type(dt)
