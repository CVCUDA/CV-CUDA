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


def test_imgbatchvarshape_creation_works():
    batch = nvcv.ImageBatchVarShape(15)
    assert batch.capacity == 15
    assert len(batch) == 0
    assert batch.uniqueformat is None
    assert batch.maxsize == (0, 0)

    # range must be empty
    cnt = 0
    for i in batch:
        cnt += 1
    assert cnt == 0


def test_imgbatchvarshape_one_image():
    batch = nvcv.ImageBatchVarShape(15)

    img = nvcv.Image((64, 32), nvcv.Format.RGBA8)
    batch.pushback(img)
    assert len(batch) == 1
    assert batch.uniqueformat == nvcv.Format.RGBA8
    assert batch.maxsize == (64, 32)

    # range must contain one
    cnt = 0
    for bimg in batch:
        assert bimg is img
        cnt += 1
    assert cnt == 1

    # remove added image
    batch.popback()

    # check if its indeed removed
    assert len(batch) == 0
    cnt = 0
    for bimg in batch:
        cnt += 1
    assert cnt == 0


def test_imgbatchvarshape_several_images():
    batch = nvcv.ImageBatchVarShape(15)

    # add 4 images with different dimensions
    imgs = [nvcv.Image((m * 2, m), nvcv.Format.RGBA8) for m in range(2, 10, 2)]
    batch.pushback(imgs)
    assert len(batch) == 4
    assert batch.maxsize == (16, 8)
    assert batch.uniqueformat == nvcv.Format.RGBA8

    # check if they were really added
    cnt = 0
    for bimg in batch:
        assert bimg is imgs[cnt]
        cnt += 1
    assert cnt == 4

    # now remove the last 2
    batch.popback(2)
    assert len(batch) == 2
    cnt = 0
    for bimg in batch:
        cnt += 1
    assert cnt == 2

    assert batch.maxsize == (8, 4)

    # add one with a different format
    batch.pushback(nvcv.Image((58, 26), nvcv.Format.NV12))
    assert batch.maxsize == (58, 26)
    assert batch.uniqueformat is None

    # clear everything
    batch.clear()
    assert len(batch) == 0
    cnt = 0
    for bimg in batch:
        cnt += 1
    assert cnt == 0

    assert batch.maxsize == (0, 0)
