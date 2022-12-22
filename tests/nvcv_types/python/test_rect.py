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


def test_recti_default():
    r = nvcv.RectI()
    assert r.x == 0
    assert r.y == 0
    assert r.width == 0
    assert r.height == 0


@t.mark.parametrize("x,y,w,h", [(0, 0, 0, 0), (10, -12, -45, 14)])
def test_recti_ctor(x, y, w, h):
    r = nvcv.RectI(x, y, w, h)
    assert r.x == x
    assert r.y == y
    assert r.width == w
    assert r.height == h

    r = nvcv.RectI(y=y, width=w, height=h, x=x)
    assert r.x == x
    assert r.y == y
    assert r.width == w
    assert r.height == h
