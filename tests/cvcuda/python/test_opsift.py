# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


def gold_max_features(in_tensor):
    w = in_tensor.shape[str(in_tensor.layout).index("W")]
    h = in_tensor.shape[str(in_tensor.layout).index("H")]
    return max(w * h // 20, 1)


@t.mark.parametrize(
    "in_args",
    [
        (((3, 13, 24, 1), np.uint8, "NHWC")),
        (((23, 34, 1), np.uint8, "HWC")),
    ],
)
def test_op_sift_api(in_args):
    input = cvcuda.Tensor(*in_args)
    num_samples = input.shape[0] if input.ndim == 4 else 1

    outs = cvcuda.sift(input)
    feat_coords, feat_metadata, feat_descriptors, num_features = outs
    assert feat_coords.shape[0] == num_samples
    assert feat_coords.shape[1] == gold_max_features(input)
    assert feat_coords.shape[2] == 4
    assert feat_coords.dtype == cvcuda.Type.F32
    assert feat_metadata.shape[0] == num_samples
    assert feat_metadata.shape[1] == gold_max_features(input)
    assert feat_metadata.shape[2] == 3
    assert feat_metadata.dtype == cvcuda.Type.F32
    assert feat_descriptors.shape[0] == num_samples
    assert feat_descriptors.shape[1] == gold_max_features(input)
    assert feat_descriptors.shape[2] == 128
    assert feat_descriptors.dtype == cvcuda.Type.U8
    assert num_features.shape[0] == num_samples
    assert num_features.dtype == cvcuda.Type.S32

    rets = cvcuda.sift_into(
        feat_coords, feat_metadata, feat_descriptors, num_features, input
    )
    for ret, out in zip(rets, outs):
        assert ret is out

    stream = cvcuda.Stream()

    outs = cvcuda.sift(
        src=input,
        max_features=123,
        num_octave_layers=2,
        contrast_threshold=0.25,
        edge_threshold=25.0,
        init_sigma=1.2,
        flags=cvcuda.SIFT.USE_ORIGINAL_INPUT,
        stream=stream,
    )
    feat_coords, feat_metadata, feat_descriptors, num_features = outs
    assert feat_coords.shape[0] == num_samples
    assert feat_coords.shape[1] == 123
    assert feat_coords.shape[2] == 4
    assert feat_coords.dtype == cvcuda.Type.F32
    assert feat_metadata.shape[0] == num_samples
    assert feat_metadata.shape[1] == 123
    assert feat_metadata.shape[2] == 3
    assert feat_metadata.dtype == cvcuda.Type.F32
    assert feat_descriptors.shape[0] == num_samples
    assert feat_descriptors.shape[1] == 123
    assert feat_descriptors.shape[2] == 128
    assert feat_descriptors.dtype == cvcuda.Type.U8
    assert num_features.shape[0] == num_samples
    assert num_features.dtype == cvcuda.Type.S32

    rets = cvcuda.sift_into(
        feat_coords=feat_coords,
        feat_metadata=feat_metadata,
        feat_descriptors=feat_descriptors,
        num_features=num_features,
        src=input,
        num_octave_layers=1,
        contrast_threshold=0.13,
        edge_threshold=23.0,
        init_sigma=0.7,
        flags=cvcuda.SIFT.USE_EXPANDED_INPUT,
        stream=stream,
    )
    for ret, out in zip(rets, outs):
        assert ret is out
