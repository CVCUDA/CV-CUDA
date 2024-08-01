# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


DEF_OUT_DTYPE = np.int32
DEF_MAX_CAPACITY = 10000


def defaultNumStats(layout):
    return 9 if "D" in layout else 7


@t.mark.parametrize(
    "src_args",
    [
        (((2, 11, 26, 32, 1), np.uint8, "NDHWC")),
        (((3, 12, 29, 31), np.uint8, "NDHW")),
        (((10, 22, 33, 1), np.uint8, "DHWC")),
        (((14, 23, 34), np.uint8, "DHW")),
        (((2, 15, 25, 1), np.uint8, "NHWC")),
        (((3, 17, 24), np.uint8, "NHW")),
        (((28, 37, 1), np.uint8, "HWC")),
        (((18, 16), np.uint8, "HW")),
    ],
)
def test_op_label_api(src_args):
    src = cvcuda.Tensor(*src_args)

    if "D" not in src_args[2]:
        dst, count, stats = cvcuda.label(src)
        assert count is None and stats is None
        assert dst.layout == src.layout
        assert dst.shape == src.shape
        assert dst.dtype == DEF_OUT_DTYPE
        connectivity = cvcuda.CONNECTIVITY_4_2D
    else:
        connectivity = cvcuda.CONNECTIVITY_6_3D
        dst, count, stats = cvcuda.label(src, connectivity)
        assert count is None and stats is None
        assert dst.layout == src.layout
        assert dst.shape == src.shape
        assert dst.dtype == DEF_OUT_DTYPE

    out = cvcuda.Tensor(src.shape, DEF_OUT_DTYPE, src.layout)
    tmp, count, stats = cvcuda.label_into(out, src=src, connectivity=connectivity)
    assert tmp is out and count is None and stats is None
    assert out.layout == src.layout
    assert out.shape == src.shape
    assert out.dtype == DEF_OUT_DTYPE

    num_samples = src_args[0][0] if "N" in src_args[2] else 1
    bg_label = cvcuda.Tensor((num_samples,), src.dtype, "N")
    min_thresh = cvcuda.Tensor((num_samples,), src.dtype, "N")
    max_thresh = cvcuda.Tensor((num_samples,), src.dtype, "N")

    out, count, stats = cvcuda.label(
        src,
        connectivity,
        bg_label=bg_label,
        min_thresh=min_thresh,
        max_thresh=max_thresh,
    )
    assert count is None and stats is None
    assert out.layout == src.layout
    assert out.shape == src.shape
    assert out.dtype == DEF_OUT_DTYPE

    out, count, stats = cvcuda.label(src, connectivity, count=True, stats=False)
    assert count is not None and stats is None

    out, count, stats = cvcuda.label(src, connectivity, count=True, stats=True)
    assert count is not None and stats is not None

    min_size = cvcuda.Tensor((num_samples,), DEF_OUT_DTYPE, "N")

    out, count, stats = cvcuda.label(
        src,
        connectivity,
        assign_labels=cvcuda.LABEL.SEQUENTIAL,
        count=True,
        stats=True,
        bg_label=bg_label,
        min_size=min_size,
    )
    assert count is not None and stats is not None
    assert out.layout == src.layout
    assert out.shape == src.shape
    assert out.dtype == DEF_OUT_DTYPE

    mask_layout = "".join([lc for lc in src_args[2] if lc != "N"])
    mask_shape = tuple([sv for sv, lc in zip(src_args[0], src_args[2]) if lc != "N"])
    mask = cvcuda.Tensor(mask_shape, np.int8, mask_layout)

    out, count, stats = cvcuda.label(
        src,
        connectivity,
        cvcuda.LABEL.FAST,
        mask_type=cvcuda.REMOVE_ISLANDS_OUTSIDE_MASK_ONLY,
        count=True,
        stats=True,
        bg_label=bg_label,
        min_size=min_size,
        mask=mask,
    )
    assert count is not None and stats is not None
    assert out.layout == src.layout
    assert out.shape == src.shape
    assert out.dtype == DEF_OUT_DTYPE

    mask = cvcuda.Tensor(src.shape, np.uint8, src.layout)

    t_out, _, _ = cvcuda.label_into(
        out,
        count,
        stats,
        src,
        connectivity,
        bg_label=bg_label,
        min_size=min_size,
        mask=mask,
    )
    assert t_out is out

    t_out, t_count, t_stats = cvcuda.label_into(out, count, stats, src, connectivity)
    assert t_out is out and t_count is count and t_stats is stats
    assert out.layout == src.layout
    assert out.shape == src.shape
    assert out.dtype == DEF_OUT_DTYPE
    assert count.layout == "N"
    assert count.shape[0] == num_samples
    assert count.dtype == DEF_OUT_DTYPE
    assert stats.layout == "NMA"
    assert stats.shape == (num_samples, DEF_MAX_CAPACITY, defaultNumStats(src_args[2]))
    assert stats.dtype == DEF_OUT_DTYPE

    out, count, stats = cvcuda.label(
        src, connectivity, count=True, stats=True, max_labels=12345
    )
    assert stats.shape == (num_samples, 12345, defaultNumStats(src_args[2]))

    stream = cvcuda.Stream()
    out, _, _ = cvcuda.label(src=src, connectivity=connectivity, stream=stream)
    assert out.layout == src.layout
    assert out.shape == src.shape
    assert out.dtype == DEF_OUT_DTYPE

    out = cvcuda.Tensor(src.shape, np.uint32, src.layout)
    tmp, _, _ = cvcuda.label_into(
        dst=out, src=src, connectivity=connectivity, stream=stream
    )
    assert tmp is out
    assert out.layout == src.layout
    assert out.shape == src.shape
    assert out.dtype == np.uint32
