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


@t.mark.parametrize(
    "format,gold_channels",
    [
        (nvcv.Format.RGBA8, 4),
        (nvcv.Format.RGB8, 3),
        (nvcv.Format._2S16, 2),
        (nvcv.Format.S8, 1),
        (nvcv.Format.NV12, 3),
    ],
)
def test_imgformat_numchannels(format, gold_channels):
    assert format.channels == gold_channels


@t.mark.parametrize(
    "format,gold_planes",
    [
        (nvcv.Format.RGBA8, 1),
        (nvcv.Format.RGB8, 1),
        (nvcv.Format._2S16, 1),
        (nvcv.Format.S8, 1),
        (nvcv.Format.NV12, 2),
    ],
)
def test_imgformat_planes(format, gold_planes):
    assert format.planes == gold_planes
