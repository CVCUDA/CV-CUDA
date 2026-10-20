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

import operator

import cvcuda


def test_border_type():
    assert cvcuda.Border.CONSTANT != cvcuda.Border.REPLICATE
    assert cvcuda.Border.REPLICATE != cvcuda.Border.REFLECT
    assert cvcuda.Border.REFLECT != cvcuda.Border.WRAP
    assert cvcuda.Border.WRAP != cvcuda.Border.REFLECT101
    assert cvcuda.Border.REFLECT101 != cvcuda.Border.CONSTANT_COVERAGE
    assert int(cvcuda.Border.CONSTANT_COVERAGE) == 5
    assert cvcuda.Border.CONSTANT_COVERAGE.name == "CONSTANT_COVERAGE"
    assert cvcuda.Border(5) == cvcuda.Border.CONSTANT_COVERAGE


def test_border_type_integer_compatibility():
    value = cvcuda.Border.REPLICATE

    assert int(value) == 1
    assert operator.index(value) == 1
    assert value == 1
    assert hash(value) == hash(1)
    assert repr(value) == "<Border.REPLICATE: 1>"
