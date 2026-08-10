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
import pytest as t

import cvcuda

import cvcuda_types as cv_types


def sub_check_type(cv_type: cvcuda.Type, dtype: np.dtype):
    assert dtype == cv_type
    t = cvcuda.Type(dtype)
    assert t == cv_type
    assert t == dtype


@t.mark.parametrize("cv_type", cv_types.TYPES)
def test_datatype_dtype(cv_type):
    dtype = cv_types.as_np_dtype(cv_type)
    sub_check_type(cv_type, dtype)


def test_datatype_repr_uses_public_type_name():
    assert repr(cvcuda.Type(np.uint8)) == "nvcv.Type.U8"


@t.mark.parametrize("cv_type1", cv_types.TYPES)
@t.mark.parametrize("cv_type2", cv_types.TYPES)
def test_datatype_dtype_conv(cv_type1, cv_type2):
    if cv_type1 == cv_type2:
        sub_check_type(cv_type1, cv_types.as_np_dtype(cv_type2))
        sub_check_type(cv_types.as_np_dtype(cv_type1), cv_type2)
    else:
        with t.raises(AssertionError):
            sub_check_type(cv_type1, cv_types.as_np_dtype(cv_type2))
        with t.raises(AssertionError):
            sub_check_type(cv_types.as_np_dtype(cv_type1), cv_type2)


@t.mark.parametrize("dt", [np.dtype([("f1", np.uint64), ("f2", np.int32)]), "invalid"])
def test_datatype_dtype_conv_error(dt):
    with t.raises(TypeError):
        cvcuda.Type(dt)


def test_datatype_is_hashable_and_value_consistent():
    # A Type constructed via Type(...) must hash identically to the equivalent
    # numpy.dtype form that Type.U8 (and friends) surface as, so that all three
    # representations are interchangeable equal-and-hash-equal dict keys.
    wrapper = cvcuda.Type(np.uint8)
    named = cvcuda.Type.U8
    npdt = np.dtype("uint8")

    assert wrapper == named == npdt
    assert hash(wrapper) == hash(named) == hash(npdt)

    d = {wrapper: "u8"}
    assert d[named] == "u8"
    assert d[npdt] == "u8"

    assert len({wrapper, named, npdt}) == 1
    assert len({cvcuda.Type.U8, cvcuda.Type.S8}) == 2
