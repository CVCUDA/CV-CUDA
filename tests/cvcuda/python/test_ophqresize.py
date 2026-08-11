# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest
import cvcuda_util as util
import numpy as np
import cvcuda_types as cv_types
import cvcuda_tools as cv_tools

RNG = np.random.default_rng(12345)


def get_shape(in_shape, layout, out_size):
    assert len(out_size) in (2, 3)
    out_size_layout = "HW" if len(out_size) == 2 else "DHW"
    assert len(out_size) == len(out_size_layout)
    out_size_map = dict(zip(out_size_layout, out_size))
    assert len(in_shape) == len(layout)
    return tuple(
        out_size_map.get(name, extent) for name, extent in zip(layout, in_shape)
    )


@pytest.mark.parametrize(
    "src_args, dst_args, interpolation_args, roi",
    [
        (
            ((2, 244, 244, 3), cvcuda.Type.U8, "NHWC"),
            ((122, 122), cvcuda.Type.U8),
            (cvcuda.Interp.NEAREST, cvcuda.Interp.NEAREST, False),
            None,
        ),
        (
            ((2, 244, 244, 3), cvcuda.Type.U8, "NHWC"),
            ((122, 244), cvcuda.Type.U8),
            (cvcuda.Interp.LINEAR, cvcuda.Interp.LINEAR, False),
            None,
        ),
        (
            ((1, 244, 244, 2), cvcuda.Type.U8, "NHWC"),
            ((122, 122), cvcuda.Type.F32),
            (cvcuda.Interp.LINEAR, cvcuda.Interp.CUBIC, True),
            (50, 10, 230, 220),
        ),
        (
            ((3, 101, 244, 301, 3), cvcuda.Type.U16, "NDHWC"),
            ((122, 54, 101), cvcuda.Type.U16),
            (cvcuda.Interp.GAUSSIAN, cvcuda.Interp.CUBIC, True),
            None,
        ),
        (
            ((54, 54, 54, 4), cvcuda.Type.U8, "DHWC"),
            ((100, 100, 100), cvcuda.Type.U8),
            (cvcuda.Interp.LANCZOS, cvcuda.Interp.LINEAR, True),
            (54, 0, 0, 0, 54, 54),
        ),
        (
            ((101, 102, 103), cvcuda.Type.U8, "DHW"),
            ((41, 45, 49), cvcuda.Type.F32),
            (cvcuda.Interp.NEAREST, cvcuda.Interp.LINEAR, False),
            None,
        ),
        (
            ((101, 102, 103), cvcuda.Type.U8, "DHW"),
            ((101, 45, 49), cvcuda.Type.F32),
            (cvcuda.Interp.NEAREST, cvcuda.Interp.LINEAR, False),
            None,
        ),
    ],
)
def test_op_hq_resize_api(src_args, dst_args, interpolation_args, roi):
    stream = cvcuda.Stream()
    src_shape, src_type, layout = src_args
    assert len(layout) == len(src_shape)
    dst_size, dst_type = dst_args
    min_interpolation, mag_interpolation, antialias = interpolation_args
    out_shape = get_shape(src_shape, layout, dst_size)

    t_src = util.create_tensor(*src_args)

    if src_type != dst_type:
        t_dst = util.create_tensor(out_shape, dst_type, layout)
        t_tmp = cvcuda.hq_resize_into(
            t_dst,
            t_src,
            min_interpolation=min_interpolation,
            mag_interpolation=mag_interpolation,
            antialias=antialias,
            stream=stream,
            roi=roi,
        )
        assert t_tmp is t_dst
    else:
        t_dst = cvcuda.hq_resize(
            t_src,
            dst_size,
            min_interpolation=min_interpolation,
            mag_interpolation=mag_interpolation,
            antialias=antialias,
            stream=stream,
            roi=roi,
        )
        assert t_dst.layout == t_src.layout
        assert t_dst.dtype == dst_type
        assert t_dst.shape == out_shape


@pytest.mark.parametrize(
    "num_samples, src_args, dst_type, interpolation_args, roi",
    [
        (
            1,
            ((512, 1024, 3), np.uint8, "HWC"),
            np.uint8,
            (cvcuda.Interp.LINEAR, cvcuda.Interp.LINEAR, True),
            None,
        ),
        (
            5,
            ((122, 244, 4), np.float32, "HWC"),
            np.float32,
            (cvcuda.Interp.CUBIC, cvcuda.Interp.CUBIC, False),
            [(100, 200, 10, 10)],
        ),
        (
            3,
            ((244, 122), np.uint8, "HW"),
            np.float32,
            (cvcuda.Interp.NEAREST, cvcuda.Interp.NEAREST, False),
            [(200, 100, 10, 10)],
        ),
    ],
)
def test_op_hq_resize_var_shape_api(
    num_samples, src_args, dst_type, interpolation_args, roi
):
    stream = cvcuda.Stream()

    src_shape, src_type, layout = src_args
    assert len(layout) == len(src_shape)
    min_interpolation, mag_interpolation, antialias = interpolation_args

    b_src = cvcuda.ImageBatchVarShape(num_samples)
    out_sizes = []
    for _ in range(num_samples):
        sample_size = tuple(
            RNG.integers(1, extent + 1)
            for name, extent in zip(layout, src_shape)
            if name in "HW"
        )
        sample_shape = get_shape(src_shape, layout, sample_size)
        h_data = util.generate_data(sample_shape, src_type, rng=RNG)
        image = util.to_cvcuda_image(h_data)
        b_src.pushback(image)
        out_sizes.append(
            tuple(RNG.integers(1, 2 * extent + 1) for extent in sample_size)
        )

    if src_type != dst_type:
        b_dst = cvcuda.ImageBatchVarShape(num_samples)
        assert len(out_sizes) == num_samples
        for out_size in out_sizes:
            out_shape = get_shape(src_shape, layout, out_size)
            h_data = util.generate_data(out_shape, dst_type, rng=RNG)
            image = util.to_cvcuda_image(h_data)
            b_dst.pushback(image)

        b_tmp = cvcuda.hq_resize_into(
            b_dst,
            b_src,
            min_interpolation=min_interpolation,
            mag_interpolation=mag_interpolation,
            antialias=antialias,
            stream=stream,
            roi=roi,
        )
        assert b_tmp is b_dst
    else:
        b_dst = cvcuda.hq_resize(
            b_src,
            out_sizes,
            min_interpolation=min_interpolation,
            mag_interpolation=mag_interpolation,
            antialias=antialias,
            stream=stream,
            roi=roi,
        )

        assert len(b_dst) == len(b_src)
        assert b_dst.capacity == b_src.capacity
        assert b_dst.uniqueformat == b_src.uniqueformat
        assert b_dst.maxsize == tuple(
            max(extent) for extent in reversed(list(zip(*out_sizes)))
        )


@pytest.mark.parametrize(
    "num_samples, src_args, dst_type, interpolation_args, use_roi",
    [
        (
            7,
            ((244, 244, 3), cvcuda.Type.U8, "HWC"),
            cvcuda.Type.U8,
            (cvcuda.Interp.NEAREST, cvcuda.Interp.NEAREST, False),
            False,
        ),
        (
            5,
            ((244, 244), cvcuda.Type.U8, "HW"),
            cvcuda.Type.F32,
            (cvcuda.Interp.LINEAR, cvcuda.Interp.CUBIC, True),
            True,
        ),
        (
            3,
            ((101, 244, 301, 3), cvcuda.Type.U16, "DHWC"),
            cvcuda.Type.U16,
            (cvcuda.Interp.GAUSSIAN, cvcuda.Interp.CUBIC, True),
            True,
        ),
        (
            1,
            ((101, 102, 103), cvcuda.Type.U8, "DHW"),
            cvcuda.Type.F32,
            (cvcuda.Interp.NEAREST, cvcuda.Interp.LINEAR, False),
            False,
        ),
    ],
)
def test_op_hq_resize_tensor_batch_api(
    num_samples, src_args, dst_type, interpolation_args, use_roi
):
    stream = cvcuda.Stream()

    src_shape, src_type, layout = src_args
    assert len(layout) == len(src_shape)
    min_interpolation, mag_interpolation, antialias = interpolation_args

    b_src = cvcuda.TensorBatch(num_samples)
    out_sizes = []
    rois = []
    for _ in range(num_samples):
        sample_size = tuple(
            RNG.integers(1, extent + 1)
            for name, extent in zip(layout, src_shape)
            if name in "DHW"
        )
        sample_shape = get_shape(src_shape, layout, sample_size)
        t_src = util.create_tensor(sample_shape, src_type, layout)
        b_src.pushback(t_src)
        out_sizes.append(
            tuple(RNG.integers(1, 2 * extent + 1) for extent in sample_size)
        )
        if use_roi:
            roi = tuple(
                RNG.integers(1, extent + 1) for _ in range(2) for extent in sample_size
            )
            rois.append(roi)

    if src_type != dst_type:
        b_dst = cvcuda.TensorBatch(num_samples)
        assert len(out_sizes) == num_samples
        for out_size in out_sizes:
            out_shape = get_shape(src_shape, layout, out_size)
            t_dst = util.create_tensor(out_shape, dst_type, layout)
            b_dst.pushback(t_dst)

        b_tmp = cvcuda.hq_resize_into(
            b_dst,
            b_src,
            min_interpolation=min_interpolation,
            mag_interpolation=mag_interpolation,
            antialias=antialias,
            stream=stream,
            roi=None if not use_roi else rois,
        )
        assert b_dst is b_tmp
    else:
        b_dst = cvcuda.hq_resize(
            b_src,
            out_sizes,
            min_interpolation=min_interpolation,
            mag_interpolation=mag_interpolation,
            antialias=antialias,
            stream=stream,
            roi=None if not use_roi else rois,
        )
        assert len(b_dst) == len(b_src)
        assert b_dst.capacity == b_src.capacity
        assert b_dst.layout == b_src.layout
        assert b_dst.ndim == b_src.ndim
        assert b_dst.dtype == dst_type
        for i in range(num_samples):
            assert b_dst[i].shape == get_shape(src_shape, layout, out_sizes[i])


def _hqresize_params(dtype, layout, channels):
    return {
        "out_size": (12, 12),
        "interpolation": cvcuda.Interp.LINEAR,
    }


def _hqresize_varshape_params(dtype, layout, channels):
    return {
        "out_size": [(12, 12), (12, 12)],
        "interpolation": cvcuda.Interp.LINEAR,
    }


def _hqresize_tensorbatch_params(dtype, layout, channels):
    if "D" in layout:
        return {
            "out_size": [(1, 12, 12), (1, 12, 12)],
            "interpolation": cvcuda.Interp.LINEAR,
        }
    return {
        "out_size": [(12, 12), (12, 12)],
        "interpolation": cvcuda.Interp.LINEAR,
    }


_interleaved_layouts = {"NHWC", "NHW", "HWC", "HW"}
_planar_layouts = {"NCHW", "CHW", "NFHW", "FHW", "FHWC"}
# 2D channel-first layouts support multi-channel planar resize (processed plane-by-plane).
_planar_2d_layouts = {"NCHW", "CHW"}


globals().update(
    cv_tools.make_op_tests(
        name="hq_resize",
        runner_info=[
            ("tensor", cvcuda.hq_resize, _hqresize_params),
            ("image_batch", cvcuda.hq_resize, _hqresize_varshape_params),
        ],
        keystone_dlc=(cvcuda.Type.U8, "NHWC", 3),
        supported_dtypes={
            cvcuda.Type.U8,
            cvcuda.Type.U16,
            cvcuda.Type.S16,
            cvcuda.Type.F32,
        },
        supported_layouts=_interleaved_layouts | _planar_layouts,
        supported_channels=cv_types.CHANNELS,
        # Only 2D channel-first planar layouts support multi-channel; frame/3D planar layouts
        # (NFHW/FHW/FHWC) remain single-channel.
        exclude_dlc=[
            (None, layout, ch)
            for layout in _planar_layouts - _planar_2d_layouts
            for ch in cv_types.CHANNELS - {1}
        ],
    )
)


globals().update(
    cv_tools.make_op_tests(
        name="hq_resize_tensor_batch",
        runner_info=[
            ("tensor_batch", cvcuda.hq_resize, _hqresize_tensorbatch_params),
        ],
        keystone_dlc=(cvcuda.Type.U8, "NCHW", 1),
        supported_dtypes={
            cvcuda.Type.U8,
            cvcuda.Type.U16,
            cvcuda.Type.S16,
            cvcuda.Type.F32,
        },
        # NCHW/CHW (channel-first 2D) planar tensor batches are expanded plane-by-plane and accept
        # any channel count; the interleaved path supports dynamic channels too, so there is no
        # channel-count upper limit to assert as negative.
        supported_layouts={"HW", "HWC", "DHW", "DHWC", "NCHW", "CHW"},
        supported_channels=cv_types.CHANNELS,
    )
)
