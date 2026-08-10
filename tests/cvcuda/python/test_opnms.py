# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import cvcuda_util as util
import pytest
import numpy as np
import cvcuda_types as cv_types
import cvcuda_tools as cv_tools
import cupy

RNG = np.random.default_rng(0)


@pytest.mark.parametrize(
    "in_shape,in_dtype,in_layout,sc_shape,sc_dtype,sc_layout",
    [
        ((2, 10, 4), cvcuda.Type.S16, "NWC", (2, 10, 1), cvcuda.Type.F32, "NWC"),
        ((2, 10, 1), cvcuda.Type._4S16, "NWC", (2, 10), cvcuda.Type.F32, "NW"),
        ((2, 10), cvcuda.Type._4S16, "NW", (2, 10, 1), cvcuda.Type.F32, "NWC"),
    ],
)
def test_op_nms_api(in_shape, in_dtype, in_layout, sc_shape, sc_dtype, sc_layout):
    t_in = cvcuda.Tensor(in_shape, in_dtype, in_layout)
    t_sc = cvcuda.Tensor(sc_shape, sc_dtype, sc_layout)

    t_out = cvcuda.nms(t_in, t_sc)
    assert t_out.shape[0:2] == t_in.shape[0:2]
    assert t_out.dtype == cvcuda.Type.U8

    t_tmp = cvcuda.nms_into(t_out, t_in, t_sc)
    assert t_tmp is t_out

    stream = cvcuda.Stream()
    t_out = cvcuda.nms(
        src=t_in, scores=t_sc, score_threshold=0.33, iou_threshold=0.66, stream=stream
    )
    assert t_out.shape[0:2] == t_in.shape[0:2]
    assert t_out.dtype == cvcuda.Type.U8

    t_tmp = cvcuda.nms_into(
        dst=t_out,
        src=t_in,
        scores=t_sc,
        score_threshold=0.44,
        iou_threshold=0.22,
        stream=stream,
    )
    assert t_tmp is t_out


def gold_area(bbox):
    return bbox[2] * bbox[3]


def gold_iou(bbox1, bbox2):
    x0 = max(bbox1[0], bbox2[0])
    y0 = max(bbox1[1], bbox2[1])
    x1 = min(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2])
    y1 = min(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3])
    w = x1 - x0
    h = y1 - y0
    iou = 0
    if w > 0 and h > 0:
        inter_area = w * h
        union_area = gold_area(bbox1) + gold_area(bbox2) - inter_area
        if union_area > 0:
            iou = inter_area / union_area
    return iou


def gold_nms(in_bboxes, in_scores, score_threshold, iou_threshold):
    out_bboxes = np.zeros(in_bboxes.shape[0:2], dtype=np.uint8)
    for i, (bboxes, scores) in enumerate(zip(in_bboxes, in_scores)):
        for j, (bbox1, score1) in enumerate(zip(bboxes, scores)):
            if score1 < score_threshold:
                continue
            discard = False
            for k, (bbox2, score2) in enumerate(zip(bboxes, scores)):
                if k == j:
                    continue
                if gold_iou(bbox1, bbox2) > iou_threshold:
                    if (score1 < score2) or (
                        score1 == score2 and gold_area(bbox1) < gold_area(bbox2)
                    ):
                        discard = True
                        break
            if not discard:
                out_bboxes[i, j] = 1
    return out_bboxes


@pytest.mark.parametrize("num_samples,num_bboxes", [(1, 10), (3, 33), (7, 88)])
def test_op_nms_content(num_samples, num_bboxes):
    a_src = RNG.integers(
        (0, 0, 10, 10),
        high=(100, 100, 30, 30),
        size=(num_samples, num_bboxes, 4),
        dtype=np.int16,
    )
    t_src = util.to_cvcuda_tensor(a_src, "NWC")

    a_scores = RNG.random(size=(num_samples, num_bboxes, 1), dtype=np.float32)
    t_scores = util.to_cvcuda_tensor(a_scores, "NWC")

    score_threshold = 0.22
    iou_threshold = 0.51

    t_dst = cvcuda.nms(t_src, t_scores, score_threshold, iou_threshold)

    a_dst_test = cupy.asarray(t_dst.cuda()).get()
    a_dst_gold = gold_nms(a_src, a_scores, score_threshold, iou_threshold)

    np.testing.assert_array_equal(a_dst_test, a_dst_gold)


def _op(src: cvcuda.Tensor) -> cvcuda.Tensor:
    if src.layout == "NWC":
        sc_shape = (src.shape[0], src.shape[1], 1)
    else:
        sc_shape = (src.shape[0], src.shape[1], 1)
    t_sc = cvcuda.Tensor(sc_shape, cvcuda.Type.F32, "NWC")
    return cvcuda.nms(src, t_sc)


_num_samples = 2
_num_bboxes = 10

_supported_input_configs = [
    (cvcuda.Type.S16, "NWC", 4),  # S16 with 4 channels via NWC layout
    (cvcuda.Type._4S16, "NWC", 1),  # _4S16 packed with NWC layout (channel dim = 1)
    (cvcuda.Type._4S16, "NW", 1),  # _4S16 packed with NW layout
]


@pytest.mark.parametrize("dtype,layout,channels", _supported_input_configs)
def test_op_nms_input(dtype, layout, channels):
    cv_tools.assert_layouts(
        _op, layout, dtype=dtype, wrapper="tensor", channels=channels
    )


@pytest.mark.parametrize("dtype", cv_types.SCALAR_TYPES_SET - {cvcuda.Type.S16})
def test_op_nms_dtype_negative(dtype):
    cv_tools.assert_layouts(
        _op, "NWC", dtype=dtype, wrapper="tensor", channels=4, negative=True
    )


@pytest.mark.parametrize("dtype", cv_types.SCALAR_TYPES_SET - {cvcuda.Type.F32})
def test_op_nms_scores_dtype_negative(dtype):
    t_in = cvcuda.Tensor((_num_samples, _num_bboxes, 4), cvcuda.Type.S16, "NWC")
    t_sc = cvcuda.Tensor((_num_samples, _num_bboxes, 1), dtype, "NWC")
    with pytest.raises(RuntimeError):
        cvcuda.nms(t_in, t_sc)


@pytest.mark.parametrize("channels", {1, 2, 3, 5})
def test_op_nms_channels_negative(channels):
    cv_tools.assert_layouts(
        _op,
        "NWC",
        dtype=cvcuda.Type.S16,
        wrapper="tensor",
        channels=channels,
        negative=True,
    )
