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
import cvcuda_util as util


RNG = np.random.default_rng(0)


def gold_val_dtype(in_dtype):
    if in_dtype in {cvcuda.Type.U8, cvcuda.Type.U16, cvcuda.Type.U32}:
        return cvcuda.Type.U32
    elif in_dtype in {cvcuda.Type.S8, cvcuda.Type.S16, cvcuda.Type.S32}:
        return cvcuda.Type.S32
    else:
        return in_dtype


def gold_num_dtype():
    return cvcuda.Type.S32


@t.mark.parametrize(
    "operator",
    [
        cvcuda.min_loc_into,
        cvcuda.max_loc_into,
    ],
)
@t.mark.parametrize(
    "val_args,loc_args,num_args",
    [
        (
            ((1,), cvcuda.Type.U32, "N"),
            ((1, 10), cvcuda.Type._2S32, "NM"),
            ((1,), cvcuda.Type.S32, "N"),
        ),
        (
            ((2, 1), cvcuda.Type.U32, "NC"),
            ((2, 10, 2), cvcuda.Type.S32, "NMC"),
            ((2, 1), cvcuda.Type.S32, "NC"),
        ),
    ],
)
def test_opminmaxloc_output_api(operator, val_args, loc_args, num_args):
    t_val = cvcuda.Tensor(*val_args)
    t_loc = cvcuda.Tensor(*loc_args)
    t_min = cvcuda.Tensor(*num_args)
    t_src = cvcuda.Tensor((t_val.shape[0], 11, 11), t_val.dtype, "NHW")

    rets = operator(t_val, t_loc, t_min, t_src)
    assert rets[0] is t_val
    assert rets[1] is t_loc
    assert rets[2] is t_min


@t.mark.parametrize(
    "src_args",
    [
        ((2, 16, 23, 1), np.uint8, "NHWC"),
        ((3, 1, 17, 22), np.uint16, "NCHW"),
        ((1, 21, 16, 1), np.uint32, "NHWC"),
        ((2, 1, 15, 24), np.int8, "NCHW"),
        ((3, 16, 23, 1), np.int16, "NHWC"),
        ((1, 1, 22, 15), np.int32, "NCHW"),
        ((2, 21, 14, 1), np.float32, "NHWC"),
        ((3, 1, 10, 14), np.float64, "NCHW"),
        ((18, 21, 1), np.int16, "HWC"),
        ((1, 19, 20), np.int32, "CHW"),
    ],
)
def test_opminmaxloc_tensor_api(src_args):
    src = cvcuda.Tensor(*src_args)
    num_samples = src.shape[0] if src.ndim == 4 else 1

    outs = cvcuda.min_loc(src, 100)
    min_val, min_loc, num_min = outs
    assert min_val.shape == (num_samples, 1)
    assert min_loc.shape == (num_samples, 100, 2)
    assert num_min.shape == (num_samples, 1)
    assert min_val.dtype == gold_val_dtype(src.dtype)
    assert num_min.dtype == gold_num_dtype()

    rets = cvcuda.min_loc_into(min_val, min_loc, num_min, src)
    for ret, out in zip(rets, outs):
        assert ret is out

    outs = cvcuda.max_loc(src, 100)
    max_val, max_loc, num_max = outs
    assert max_val.shape == (num_samples, 1)
    assert max_loc.shape == (num_samples, 100, 2)
    assert num_max.shape == (num_samples, 1)
    assert max_val.dtype == gold_val_dtype(src.dtype)
    assert num_max.dtype == gold_num_dtype()

    rets = cvcuda.max_loc_into(max_val, max_loc, num_max, src)
    for ret, out in zip(rets, outs):
        assert ret is out

    outs = cvcuda.min_max_loc(src, 100)
    min_val, min_loc, num_min, max_val, max_loc, num_max = outs
    assert min_val.shape == (num_samples, 1)
    assert min_loc.shape == (num_samples, 100, 2)
    assert num_min.shape == (num_samples, 1)
    assert max_val.shape == (num_samples, 1)
    assert max_loc.shape == (num_samples, 100, 2)
    assert num_max.shape == (num_samples, 1)
    assert min_val.dtype == gold_val_dtype(src.dtype)
    assert max_val.dtype == gold_val_dtype(src.dtype)
    assert num_min.dtype == gold_num_dtype()
    assert num_max.dtype == gold_num_dtype()

    rets = cvcuda.min_max_loc_into(
        min_val, min_loc, num_min, max_val, max_loc, num_max, src
    )
    for ret, out in zip(rets, outs):
        assert ret is out

    stream = cvcuda.Stream()

    outs = cvcuda.min_loc(src=src, max_locations=100, stream=stream)
    min_val, min_loc, num_min = outs
    assert min_val.shape == (num_samples, 1)
    assert min_loc.shape == (num_samples, 100, 2)
    assert num_min.shape == (num_samples, 1)
    assert min_val.dtype == gold_val_dtype(src.dtype)
    assert num_min.dtype == gold_num_dtype()

    rets = cvcuda.min_loc_into(
        min_val=min_val, min_loc=min_loc, num_min=num_min, src=src, stream=stream
    )
    for ret, out in zip(rets, outs):
        assert ret is out

    outs = cvcuda.max_loc(src=src, max_locations=100, stream=stream)
    max_val, max_loc, num_max = outs
    assert max_val.shape == (num_samples, 1)
    assert max_loc.shape == (num_samples, 100, 2)
    assert num_max.shape == (num_samples, 1)
    assert max_val.dtype == gold_val_dtype(src.dtype)
    assert num_max.dtype == gold_num_dtype()

    rets = cvcuda.max_loc_into(
        max_val=max_val, max_loc=max_loc, num_max=num_max, src=src, stream=stream
    )
    for ret, out in zip(rets, outs):
        assert ret is out

    outs = cvcuda.min_max_loc(src=src, max_locations=100, stream=stream)
    min_val, min_loc, num_min, max_val, max_loc, num_max = outs
    assert min_val.shape == (num_samples, 1)
    assert min_loc.shape == (num_samples, 100, 2)
    assert num_min.shape == (num_samples, 1)
    assert max_val.shape == (num_samples, 1)
    assert max_loc.shape == (num_samples, 100, 2)
    assert num_max.shape == (num_samples, 1)
    assert min_val.dtype == gold_val_dtype(src.dtype)
    assert max_val.dtype == gold_val_dtype(src.dtype)
    assert num_min.dtype == gold_num_dtype()
    assert num_max.dtype == gold_num_dtype()

    rets = cvcuda.min_max_loc_into(
        min_val=min_val,
        min_loc=min_loc,
        num_min=num_min,
        max_val=max_val,
        max_loc=max_loc,
        num_max=num_max,
        src=src,
        stream=stream,
    )
    for ret, out in zip(rets, outs):
        assert ret is out


@t.mark.parametrize(
    "num_images, img_format, max_size",
    [
        (1, cvcuda.Format.U8, (73, 98)),
        (2, cvcuda.Format.U16, (82, 37)),
        (3, cvcuda.Format.U32, (13, 18)),
        (4, cvcuda.Format.S8, (12, 13)),
        (5, cvcuda.Format.S16, (11, 23)),
        (6, cvcuda.Format.S32, (20, 11)),
        (7, cvcuda.Format.F32, (14, 28)),
        (8, cvcuda.Format.F64, (29, 19)),
    ],
)
def test_opminmaxloc_varshape_api(num_images, img_format, max_size):
    src = util.create_image_batch(num_images, img_format, max_size=max_size, rng=RNG)
    src_dtype = [util.IMG_FORMAT_TO_TYPE[img.format] for img in src]

    outs = cvcuda.min_loc(src, 10)
    for out in outs:
        assert out.shape[0] == num_images
    min_val, min_loc, num_min = outs
    assert min_val.dtype == gold_val_dtype(src_dtype[0])
    assert num_min.dtype == gold_num_dtype()

    rets = cvcuda.min_loc_into(min_val, min_loc, num_min, src)
    for ret, out in zip(rets, outs):
        assert ret is out

    outs = cvcuda.max_loc(src, 10)
    for out in outs:
        assert out.shape[0] == num_images
    max_val, max_loc, num_max = outs
    assert max_val.dtype == gold_val_dtype(src_dtype[0])
    assert num_max.dtype == gold_num_dtype()

    rets = cvcuda.max_loc_into(max_val, max_loc, num_max, src)
    for ret, out in zip(rets, outs):
        assert ret is out

    outs = cvcuda.min_max_loc(src, 10)
    for out in outs:
        assert out.shape[0] == num_images
    min_val, min_loc, num_min, max_val, max_loc, num_max = outs
    assert min_val.dtype == gold_val_dtype(src_dtype[0])
    assert max_val.dtype == gold_val_dtype(src_dtype[0])
    assert num_min.dtype == gold_num_dtype()
    assert num_max.dtype == gold_num_dtype()

    rets = cvcuda.min_max_loc_into(
        min_val, min_loc, num_min, max_val, max_loc, num_max, src
    )
    for ret, out in zip(rets, outs):
        assert ret is out

    stream = cvcuda.cuda.Stream()

    outs = cvcuda.min_loc(src=src, max_locations=10, stream=stream)
    for out in outs:
        assert out.shape[0] == num_images
    min_val, min_loc, num_min = outs
    assert min_val.dtype == gold_val_dtype(src_dtype[0])
    assert num_min.dtype == gold_num_dtype()

    rets = cvcuda.min_loc_into(
        min_val=min_val, min_loc=min_loc, num_min=num_min, src=src, stream=stream
    )
    for ret, out in zip(rets, outs):
        assert ret is out

    outs = cvcuda.max_loc(src=src, max_locations=10, stream=stream)
    for out in outs:
        assert out.shape[0] == num_images
    max_val, max_loc, num_max = outs
    assert max_val.dtype == gold_val_dtype(src_dtype[0])
    assert num_max.dtype == gold_num_dtype()

    rets = cvcuda.max_loc_into(
        max_val=max_val, max_loc=max_loc, num_max=num_max, src=src, stream=stream
    )
    for ret, out in zip(rets, outs):
        assert ret is out

    outs = cvcuda.min_max_loc(src=src, max_locations=10, stream=stream)
    for out in outs:
        assert out.shape[0] == num_images
    min_val, min_loc, num_min, max_val, max_loc, num_max = outs
    assert min_val.dtype == gold_val_dtype(src_dtype[0])
    assert max_val.dtype == gold_val_dtype(src_dtype[0])
    assert num_min.dtype == gold_num_dtype()
    assert num_max.dtype == gold_num_dtype()

    rets = cvcuda.min_max_loc_into(
        min_val=min_val,
        min_loc=min_loc,
        num_min=num_min,
        max_val=max_val,
        max_loc=max_loc,
        num_max=num_max,
        src=src,
        stream=stream,
    )
    for ret, out in zip(rets, outs):
        assert ret is out


@t.mark.parametrize("input_type", ["tensor", "image_batch"])
def test_opminmaxloc_content(input_type):
    # Test with fixed number of images and lists of minimum and maximum locations,
    # the lists must be in ascending order in x dimension for comparisons
    n_img = 5
    l_min_loc = [[123, 456], [222, 333], [777, 444]]
    l_max_loc = [[100, 789], [111, 555], [888, 333]]

    src = None
    if input_type == "tensor":
        shape = (n_img, 1080, 1920, 1)
        a_src = RNG.integers(1, high=255, size=shape, dtype=np.uint8)
        for i in range(n_img):
            for min_loc in l_min_loc:
                a_src[i, min_loc[1] + i, min_loc[0] + i, 0] = 0
            for max_loc in l_max_loc:
                a_src[i, max_loc[1] - i, max_loc[0] - i, 0] = 255

        src = util.to_nvcv_tensor(a_src, "NHWC")

    elif input_type == "image_batch":
        src = cvcuda.ImageBatchVarShape(n_img)
        for i in range(n_img):
            shape = (1080 - i, 1920 - i, 1)
            a_img = RNG.integers(1, high=255, size=shape, dtype=np.uint8)
            for min_loc in l_min_loc:
                a_img[min_loc[1] + i, min_loc[0] + i, 0] = 0
            for max_loc in l_max_loc:
                a_img[max_loc[1] - i, max_loc[0] - i, 0] = 255

            src.pushback(util.to_nvcv_image(a_img))

    outputs = cvcuda.min_max_loc(src, max_locations=len(l_min_loc))

    a_test_min_val = util.to_cpu_numpy_buffer(outputs[0].cuda())
    a_test_min_loc = util.to_cpu_numpy_buffer(outputs[1].cuda())
    a_test_num_min = util.to_cpu_numpy_buffer(outputs[2].cuda())
    a_test_max_val = util.to_cpu_numpy_buffer(outputs[3].cuda())
    a_test_max_loc = util.to_cpu_numpy_buffer(outputs[4].cuda())
    a_test_num_max = util.to_cpu_numpy_buffer(outputs[5].cuda())

    # Locations are found in no particular order, so they must be sorted for comparison

    for i in range(n_img):
        a_test_min_loc[i] = a_test_min_loc[i, np.argsort(a_test_min_loc[i, :, 0]), :]
        a_test_max_loc[i] = a_test_max_loc[i, np.argsort(a_test_max_loc[i, :, 0]), :]

    a_gold_min_loc = np.zeros(a_test_min_loc.shape, dtype=a_test_min_loc.dtype)
    a_gold_max_loc = np.zeros(a_test_max_loc.shape, dtype=a_test_max_loc.dtype)

    for i in range(n_img):
        for j, min_loc in enumerate(l_min_loc):
            a_gold_min_loc[i, j, 0:2] = np.array(min_loc) + i
        for j, max_loc in enumerate(l_max_loc):
            a_gold_max_loc[i, j, 0:2] = np.array(max_loc) - i

    np.testing.assert_array_equal(a_test_min_val, np.full([n_img, 1], 0))
    np.testing.assert_array_equal(a_test_min_loc, a_gold_min_loc)
    np.testing.assert_array_equal(a_test_num_min, np.full([n_img, 1], len(l_min_loc)))

    np.testing.assert_array_equal(a_test_max_val, np.full([n_img, 1], 255))
    np.testing.assert_array_equal(a_test_max_loc, a_gold_max_loc)
    np.testing.assert_array_equal(a_test_num_max, np.full([n_img, 1], len(l_max_loc)))
