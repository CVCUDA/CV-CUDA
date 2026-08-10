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

import numpy as np
import cvcuda

import pytest
import cvcuda_tools as cv_tools
import cupy


def pad_sequence(sequences, batch_first=False, padding_value=0.0, padding_side="right"):
    """
    Pad a list of variable length arrays with padding_value.

    This is a numpy equivalent of PyTorch's nn.utils.rnn.pad_sequence.

    Parameters:
    -----------
    sequences : list of array-like
        List of variable length sequences.
    batch_first : bool, optional
        If True, output will be B x T x * format, T x B x * otherwise.
    padding_value : float, optional
        Value for padded elements. Default: 0.0.
    padding_side : str, optional
        The side to pad sequences on ('right' or 'left'). Default: 'right'.

    Returns:
    --------
    numpy.ndarray
        Padded array of shape T x B x * if batch_first is False,
        B x T x * otherwise, where B is batch size and T is the length
        of the longest sequence.
    """
    sequences = [np.asarray(seq) for seq in sequences]

    max_len = max(len(seq) for seq in sequences)

    batch_size = len(sequences)

    trailing_dims = sequences[0].shape[1:] if sequences[0].ndim > 1 else ()

    dtype = sequences[0].dtype

    if batch_first:
        out_shape = (batch_size, max_len) + trailing_dims
    else:
        out_shape = (max_len, batch_size) + trailing_dims

    out = np.full(out_shape, padding_value, dtype=dtype)

    for i, seq in enumerate(sequences):
        length = len(seq)
        if batch_first:
            if padding_side == "right":
                out[i, :length] = seq
            else:  # left
                out[i, -length:] = seq
        else:
            if padding_side == "right":
                out[:length, i] = seq
            else:  # left
                out[-length:, i] = seq

    return out


RNG = np.random.default_rng(0)


@pytest.mark.parametrize(
    "contourData, numPointsInContour, openCvRes",
    [
        (
            [
                [
                    845,
                    600,
                    845,
                    601,
                    847,
                    603,
                    859,
                    603,
                    860,
                    604,
                    865,
                    604,
                    866,
                    603,
                    867,
                    603,
                    868,
                    602,
                    868,
                    601,
                    867,
                    600,
                ],
                [
                    965,
                    489,
                    964,
                    490,
                    963,
                    490,
                    962,
                    491,
                    962,
                    494,
                    963,
                    495,
                    963,
                    499,
                    964,
                    500,
                    964,
                    501,
                    966,
                    503,
                    1011,
                    503,
                    1012,
                    504,
                    1013,
                    503,
                    1027,
                    503,
                    1027,
                    502,
                    1028,
                    501,
                    1028,
                    490,
                    1027,
                    489,
                ],
                [
                    1050,
                    198,
                    1049,
                    199,
                    1040,
                    199,
                    1040,
                    210,
                    1041,
                    211,
                    1040,
                    212,
                    1040,
                    214,
                    1045,
                    214,
                    1046,
                    213,
                    1049,
                    213,
                    1050,
                    212,
                    1051,
                    212,
                    1052,
                    211,
                    1053,
                    211,
                    1054,
                    210,
                    1055,
                    210,
                    1056,
                    209,
                    1058,
                    209,
                    1059,
                    208,
                    1059,
                    200,
                    1058,
                    200,
                    1057,
                    199,
                    1051,
                    199,
                ],
            ],
            [11, 18, 23],
            [
                [868.0, 604.0, 845.0, 604.0, 845.0, 600.0, 868.0, 600.0],
                [962.0, 504.0, 962.0, 489.0, 1028.0, 489.0, 1028.0, 504.0],
                [1040.0, 214.0, 1040.0, 198.0, 1059.0, 198.0, 1059.0, 214.0],
            ],
        ),
    ],
)
def test_op_minarearect(contourData, numPointsInContour, openCvRes):

    batchSize = len(contourData)
    numPointsInContour_np = np.asarray(numPointsInContour, dtype=np.int32)[None, :]
    src_np = pad_sequence(
        [np.asarray(t, dtype=np.int16) for t in contourData], batch_first=True
    ).reshape(batchSize, -1, 2)
    gold_np = np.asarray(openCvRes, dtype=np.float32)

    src_dev = cupy.asarray(src_np)
    pointNumInContour_dev = cupy.asarray(numPointsInContour_np)
    gold_dev = cupy.asarray(gold_np)

    src_cvcuda = cvcuda.as_tensor(src_dev, "NWC")
    pointNumInContour_cvcuda = cvcuda.as_tensor(pointNumInContour_dev, "NW")
    gold_cvcuda = cvcuda.as_tensor(gold_dev, "NW")

    result_cvcuda = cvcuda.minarearect(
        src_cvcuda, pointNumInContour_cvcuda, src_np.shape[0]
    )
    assert result_cvcuda.layout == gold_cvcuda.layout
    assert result_cvcuda.shape == gold_cvcuda.shape
    assert result_cvcuda.dtype == gold_cvcuda.dtype
    result_host = cupy.asarray(result_cvcuda.cuda()).get().reshape(batchSize, -1, 2)
    result_sorted = np.sort(result_host, axis=1)
    gold_sorted = np.sort(gold_np.reshape(batchSize, -1, 2), axis=1)
    assert np.all(np.abs(gold_sorted - result_sorted) < 5.0)

    stream = cvcuda.Stream()
    out = cvcuda.Tensor(gold_cvcuda.shape, gold_cvcuda.dtype, gold_cvcuda.layout)
    tmp = cvcuda.minarearect_into(
        out, src_cvcuda, pointNumInContour_cvcuda, src_np.shape[0]
    )
    assert tmp is out

    stream = cvcuda.Stream()
    out = cvcuda.minarearect(
        src=src_cvcuda,
        numPointsInContour=pointNumInContour_cvcuda,
        totalContours=src_np.shape[0],
        stream=stream,
    )
    assert out.layout == gold_cvcuda.layout
    assert out.shape == gold_cvcuda.shape
    assert out.dtype == gold_cvcuda.dtype

    tmp = cvcuda.minarearect_into(
        src=src_cvcuda,
        dst=out,
        numPointsInContour=pointNumInContour_cvcuda,
        totalContours=src_np.shape[0],
        stream=stream,
    )
    assert tmp is out


def _minarearect_params(dtype, layout, channels):
    num_contours = 1
    points_per_contour = 5
    num_points = np.array([[points_per_contour] * num_contours], dtype=np.int32)
    num_points_tensor = cvcuda.as_tensor(cupy.asarray(num_points), "NW")
    return {
        "numPointsInContour": num_points_tensor,
        "totalContours": num_contours,
    }


globals().update(
    cv_tools.make_op_tests(
        name="minarearect",
        runner_info=[("tensor", cvcuda.minarearect, _minarearect_params)],
        keystone_dlc=(cvcuda.Type.U16, "NWC", 2),
        supported_dtypes={cvcuda.Type.U16, cvcuda.Type.S16, cvcuda.Type.S32},
        supported_layouts={"NWC"},
        supported_channels={2},
    )
)
