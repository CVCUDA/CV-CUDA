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
import pytest
import numpy as np

import cvcuda_util as util
import cvcuda_types as cv_types
import cvcuda_tools as cv_tools
import cupy


RNG = np.random.default_rng(0)
MAPS = {
    cvcuda.Remap.ABSOLUTE: {
        "ident": lambda w, h: np.stack(
            np.meshgrid(np.arange(w), np.arange(h)), axis=2
        ).astype(np.float32),
        "flipH": lambda w, h: np.stack(
            np.meshgrid(np.arange(w)[::-1], np.arange(h)), axis=2
        ).astype(np.float32),
        "flipV": lambda w, h: np.stack(
            np.meshgrid(np.arange(w), np.arange(h)[::-1]), axis=2
        ).astype(np.float32),
        "flipB": lambda w, h: np.stack(
            np.meshgrid(np.arange(w)[::-1], np.arange(h)[::-1]), axis=2
        ).astype(np.float32),
    },
    cvcuda.Remap.RELATIVE_NORMALIZED: {
        "ident": lambda w, h: np.array(
            [[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]], dtype=np.float32
        ),
        "flipH": lambda w, h: np.array(
            [[[1.0, 0.0], [-1.0, 0.0]], [[1.0, 0.0], [-1.0, 0.0]]], dtype=np.float32
        ),
        "flipV": lambda w, h: np.array(
            [[[0.0, 1.0], [0.0, 1.0]], [[0.0, -1.0], [0.0, -1.0]]], dtype=np.float32
        ),
        "flipB": lambda w, h: np.array(
            [[[1.0, 1.0], [-1.0, 1.0]], [[1.0, -1.0], [-1.0, -1.0]]], dtype=np.float32
        ),
    },
}
CALC_REF = {
    "ident": lambda a: a,
    "flipH": lambda a: a[:, :, ::-1, :] if len(a.shape) == 4 else a[:, ::-1, :],
    "flipV": lambda a: a[:, ::-1, :, :] if len(a.shape) == 4 else a[::-1, :, :],
    "flipB": lambda a: a[:, ::-1, ::-1, :] if len(a.shape) == 4 else a[::-1, ::-1, :],
}
REF_SHAPE = {
    "absolute": lambda s, m: s.shape[0:1] + m.shape[1:3] + s.shape[3:]
    if len(s.shape) == 4
    else m.shape[0:2] + s.shape[2:],
    "relative": lambda s, m: s.shape,
}
REF_SHAPE["default"] = REF_SHAPE["absolute"]


@pytest.mark.parametrize(
    "src_args, map_args",
    [
        (
            ((5, 16, 23, 3), cvcuda.Type.U8, "NHWC"),
            ((5, 17, 13, 2), cvcuda.Type.F32, "NHWC"),
        ),
        (
            ((5, 33, 28, 4), cvcuda.Type.U8, "NHWC"),
            ((1, 22, 18, 1), cvcuda.Type._2F32, "NHWC"),
        ),
        (
            ((13, 21, 1), cvcuda.Type._3U8, "HWC"),
            ((22, 11, 2), cvcuda.Type.F32, "HWC"),
        ),
        (
            ((13, 21, 1), cvcuda.Type._4U8, "HWC"),
            ((11, 22, 1), cvcuda.Type._2F32, "HWC"),
        ),
    ],
)
def test_op_remap_api(src_args, map_args):
    t_src = cvcuda.Tensor(*src_args)
    t_map = cvcuda.Tensor(*map_args)

    t_dst = cvcuda.remap(t_src, t_map)
    assert t_dst.layout == t_src.layout
    assert t_dst.dtype == t_src.dtype
    assert t_dst.shape == REF_SHAPE["default"](t_src, t_map)

    t_dst = cvcuda.Tensor(t_src.shape, t_src.dtype, t_src.layout)
    t_tmp = cvcuda.remap_into(t_dst, t_src, t_map)
    assert t_tmp is t_dst

    stream = cvcuda.Stream()
    t_dst = cvcuda.remap(
        src=t_src,
        map=t_map,
        src_interp=cvcuda.Interp.CUBIC,
        map_interp=cvcuda.Interp.LINEAR,
        map_type=cvcuda.Remap.RELATIVE_NORMALIZED,
        align_corners=True,
        border=cvcuda.Border.REFLECT101,
        border_value=1.0,
        stream=stream,
    )
    assert t_dst.layout == t_src.layout
    assert t_dst.dtype == t_src.dtype
    assert t_dst.shape == REF_SHAPE["relative"](t_src, t_map)

    t_tmp = cvcuda.remap_into(
        dst=t_dst,
        src=t_src,
        map=t_map,
        src_interp=cvcuda.Interp.LINEAR,
        map_interp=cvcuda.Interp.CUBIC,
        map_type=cvcuda.Remap.ABSOLUTE_NORMALIZED,
        align_corners=False,
        border=cvcuda.Border.REPLICATE,
        border_value=[0.6, 0.7, 0.8, 0.9],
        stream=stream,
    )
    assert t_tmp is t_dst


def test_op_remap_rejects_border_value_longer_than_float4():
    t_src = cvcuda.Tensor((1, 4, 4, 3), cvcuda.Type.U8, "NHWC")
    t_map = cvcuda.Tensor((1, 4, 4, 1), cvcuda.Type._2F32, "NHWC")

    with pytest.raises(RuntimeError):
        cvcuda.remap(
            t_src,
            t_map,
            border=cvcuda.Border.CONSTANT,
            border_value=np.arange(5, dtype=np.float32),
        )


@pytest.mark.parametrize(
    "map_type, map_kind, num_maps, num_imgs, img_size, img_format",
    [
        (
            cvcuda.Remap.RELATIVE_NORMALIZED,
            "ident",
            1,
            7,
            (12, 32),
            cvcuda.Format.RGBA8,
        ),
        (
            cvcuda.Remap.RELATIVE_NORMALIZED,
            "ident",
            4,
            4,
            (13, 21),
            cvcuda.Format.BGRA8,
        ),
        (cvcuda.Remap.RELATIVE_NORMALIZED, "ident", 1, 3, (27, 42), cvcuda.Format.HSV8),
        (cvcuda.Remap.RELATIVE_NORMALIZED, "ident", 6, 6, (18, 39), cvcuda.Format.BGR8),
        (cvcuda.Remap.RELATIVE_NORMALIZED, "flipH", 1, 4, (2, 2), cvcuda.Format.RGB8),
        (cvcuda.Remap.RELATIVE_NORMALIZED, "flipV", 1, 3, (2, 2), cvcuda.Format.RGB8),
        (cvcuda.Remap.RELATIVE_NORMALIZED, "flipB", 1, 2, (2, 2), cvcuda.Format.RGB8),
        (cvcuda.Remap.ABSOLUTE, "ident", 1, 3, (11, 33), cvcuda.Format.RGBA8),
        (cvcuda.Remap.ABSOLUTE, "ident", 2, 2, (13, 22), cvcuda.Format.BGRA8),
        (cvcuda.Remap.ABSOLUTE, "ident", 1, 4, (26, 41), cvcuda.Format.HSV8),
        (cvcuda.Remap.ABSOLUTE, "ident", 1, 1, (99, 88), cvcuda.Format.RGB8),
        (cvcuda.Remap.ABSOLUTE, "flipH", 1, 5, (16, 33), cvcuda.Format.RGB8),
        (cvcuda.Remap.ABSOLUTE, "flipV", 2, 2, (15, 32), cvcuda.Format.RGB8),
        (cvcuda.Remap.ABSOLUTE, "flipB", 1, 8, (13, 13), cvcuda.Format.RGB8),
    ],
)
def test_op_remap_content(map_type, map_kind, num_maps, num_imgs, img_size, img_format):
    a_src = np.stack(
        [util.create_image_pattern(img_size, img_format) for _ in range(num_imgs)]
    )
    a_map = np.stack(
        [MAPS[map_type][map_kind](img_size[0], img_size[1]) for _ in range(num_maps)]
    )

    t_map = util.to_cvcuda_tensor(a_map, "NHWC")
    t_src = util.to_cvcuda_tensor(a_src, "NHWC")

    t_dst = cvcuda.remap(t_src, t_map, map_type=map_type)

    a_dst = cupy.asarray(t_dst.cuda()).get()

    a_ref = CALC_REF[map_kind](a_src)

    np.testing.assert_array_equal(a_dst, a_ref)


@pytest.mark.parametrize(
    "num_images, img_format, max_size",
    [
        (4, cvcuda.Format.Y8, (73, 98)),
        (11, cvcuda.Format.RGB8, (33, 32)),
        (8, cvcuda.Format.RGBA8, (13, 42)),
        (3, cvcuda.Format.F32, (53, 68)),
    ],
)
def test_op_remapvarshape_api(num_images, img_format, max_size):
    b_src = util.create_image_batch(num_images, img_format, max_size=max_size, rng=RNG)
    t_map = cvcuda.Tensor((1, max_size[1], max_size[0], 1), cvcuda.Type._2F32, "NHWC")

    b_dst = cvcuda.remap(b_src, t_map)
    assert len(b_dst) == len(b_src)
    assert b_dst.capacity == b_src.capacity
    assert b_dst.uniqueformat == b_src.uniqueformat
    assert b_dst.maxsize == max_size

    stream = cvcuda.Stream()
    b_dst = util.clone_image_batch(b_src)
    b_tmp = cvcuda.remap_into(
        src=b_src,
        dst=b_dst,
        map=t_map,
        src_interp=cvcuda.Interp.LINEAR,
        map_interp=cvcuda.Interp.NEAREST,
        map_type=cvcuda.Remap.ABSOLUTE_NORMALIZED,
        align_corners=False,
        border=cvcuda.Border.WRAP,
        border_value=1.0,
        stream=stream,
    )
    assert b_tmp is b_dst


@pytest.mark.parametrize(
    "map_type, img_size, img_format",
    [
        (cvcuda.Remap.ABSOLUTE, (33, 65), cvcuda.Format.RGB8),
        (cvcuda.Remap.RELATIVE_NORMALIZED, (2, 2), cvcuda.Format.RGB8),
    ],
)
def test_op_remapvarshape_content(map_type, img_size, img_format):
    num_imgs = len(MAPS[map_type].keys())

    a_img = util.create_image_pattern(img_size, img_format)

    b_src = cvcuda.ImageBatchVarShape(num_imgs)

    for _ in range(num_imgs):
        b_src.pushback(util.to_cvcuda_image(a_img))

    a_map = np.stack(
        [
            MAPS[map_type][map_kind](img_size[0], img_size[1])
            for map_kind in MAPS[map_type].keys()
        ]
    )

    t_map = util.to_cvcuda_tensor(a_map, "NHWC")

    b_dst = cvcuda.remap(b_src, t_map, map_type=map_type)

    for src, dst, map_kind in zip(b_src, b_dst, MAPS[map_type].keys()):
        a_src = src.cpu()
        a_dst = dst.cpu()
        a_ref = CALC_REF[map_kind](a_src)

        np.testing.assert_array_equal(a_dst, a_ref)


def _remap_params(dtype, layout, channels):
    map_layout = "NHWC" if "N" in layout else "HWC"
    map_shape = cv_types.resolve_shape(
        map_layout, channels=1, size=(24, 24), batch_size=1
    )
    map_tensor = cvcuda.Tensor(map_shape, cvcuda.Type._2F32, map_layout)
    return {"map": map_tensor}


def _remap_varshape_params(dtype, layout, channels):
    map_tensor = cvcuda.Tensor((1, 24, 24, 1), cvcuda.Type._2F32, "NHWC")
    return {"map": map_tensor}


globals().update(
    cv_tools.make_op_tests(
        name="remap",
        runner_info=[
            ("tensor", cvcuda.remap, _remap_params),
            ("image_batch", cvcuda.remap, _remap_varshape_params),
        ],
        keystone_dlc=(cvcuda.Type.U8, "NHWC", 3),
        supported_dtypes={cvcuda.Type.U8, cvcuda.Type.F32},
        supported_layouts={"NHWC", "HWC", "NCHW", "CHW"},
        supported_channels={1, 3, 4},
        # F32 only supports channel 1
        exclude_dlc=[
            (cvcuda.Type.F32, None, 3),
            (cvcuda.Type.F32, None, 4),
        ],
    )
)
