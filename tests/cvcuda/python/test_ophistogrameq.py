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
import cvcuda_tools as cv_tools
import cvcuda_util as util

RNG = np.random.default_rng(0)


def pillow_reference(host):
    from PIL import Image, ImageOps

    return np.stack(
        [
            np.asarray(ImageOps.equalize(Image.fromarray(host[..., c])))
            for c in range(host.shape[-1])
        ],
        axis=-1,
    )


def pillow_equalized_input(height, width, channels, sample=0):
    rng = np.random.default_rng(sample)
    host = rng.integers(0, 64, (height, width, channels), dtype=np.uint8)
    host[-1, -1, :] = 255
    gold = pillow_reference(host)
    if height * width > 256:
        assert not np.array_equal(host, gold)
    return host, gold


@pytest.mark.parametrize("size", [(8, 8), (16, 16), (32, 32), (64, 64), (128, 128)])
@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("layout", ["HWC", "CHW", "NHWC", "NCHW"])
def test_op_histogrameq_pillow_reference(size, channels, layout):
    pairs = [pillow_equalized_input(*size, channels, sample) for sample in range(2)]
    host, gold = (np.stack(items) for items in zip(*pairs))
    if layout.endswith("CHW"):
        host, gold = (
            np.ascontiguousarray(x.transpose(0, 3, 1, 2)) for x in (host, gold)
        )
    if not layout.startswith("N"):
        host, gold = host[0], gold[0]
    src = util.to_cvcuda_tensor(host, layout)
    out = cvcuda.histogrameq(src, src.dtype, method=cvcuda.HistogramEq.PILLOW)
    np.testing.assert_array_equal(util.to_cpu_numpy_buffer(out.cuda()), gold)


@pytest.mark.parametrize(
    "channels,planar", [(1, False), (3, False), (4, False), (3, True)]
)
def test_op_histogrameq_pillow_varshape_reference(channels, planar):
    pairs = [
        pillow_equalized_input(h, w, channels, i)
        for i, (h, w) in enumerate([(32, 32), (65, 63)])
    ]
    formats = {
        (1, False): cvcuda.Format.U8,
        (3, False): cvcuda.Format.RGB8,
        (3, True): cvcuda.Format.RGB8p,
        (4, False): cvcuda.Format.RGBA8,
    }
    fmt = formats[channels, planar]
    src = cvcuda.ImageBatchVarShape(len(pairs))
    for host, _ in pairs:
        data = (
            [util.to_cuda_buffer(host[..., c]) for c in range(channels)]
            if fmt.planes > 1
            else util.to_cuda_buffer(host)
        )
        src.pushback(cvcuda.as_image(data, fmt))
    out = cvcuda.histogrameq(src, method=cvcuda.HistogramEq.PILLOW)
    for image, (_, gold) in zip(out, pairs):
        got = util.to_cpu_numpy_buffer(image.cuda())
        if fmt.planes > 1:
            got = got.transpose(1, 2, 0)
        elif channels == 1:
            got = got[..., None]
        np.testing.assert_array_equal(got, gold)


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
@pytest.mark.parametrize("layout", ["NHWC", "NCHW"])
def test_op_histogrameq_pillow_float_reference(dtype, layout):
    host, _ = pillow_equalized_input(32, 32, 3)
    host[0, 0, :] = 0
    gold = pillow_reference(host)
    host = (host.astype(dtype) / dtype(255))[None]
    gold = (gold.astype(np.float32) * np.float32(1.0 / 255.0)).astype(dtype)[None]
    if layout == "NCHW":
        host, gold = (
            np.ascontiguousarray(x.transpose(0, 3, 1, 2)) for x in (host, gold)
        )
    src = util.to_cvcuda_tensor(host, layout)
    out = cvcuda.histogrameq(src, src.dtype, method=cvcuda.HistogramEq.PILLOW)
    np.testing.assert_array_equal(util.to_cpu_numpy_buffer(out.cuda()), gold)


@pytest.mark.parametrize(
    "values,counts",
    [
        ([0], [1024]),
        ([31], [1024]),
        ([255], [1024]),
        ([0, 63], [254, 770]),
        ([0, 63], [766, 258]),
        ([7, 63, 255], [800, 223, 1]),
        ([7, 63, 127], [100, 123, 801]),
        ([7, 63, 127], [100, 700, 224]),
    ],
)
def test_op_histogrameq_pillow_edges(values, counts):
    host = np.repeat(np.array(values, dtype=np.uint8), counts).reshape(32, 32, 1)
    gold = pillow_reference(host)
    histogram = np.bincount(host.ravel(), minlength=256)
    nonempty = np.flatnonzero(histogram)
    step = (host.size - histogram[nonempty[-1]]) // 255
    running = step // 2
    lut = []
    for value, count in enumerate(histogram):
        lut.append(value if step == 0 else min(running // step, 255))
        running += count
    np.testing.assert_array_equal(np.array(lut, dtype=np.uint8)[host], gold)
    src = util.to_cvcuda_tensor(host, "HWC")
    out = cvcuda.histogrameq(src, src.dtype, method=cvcuda.HistogramEq.PILLOW)
    np.testing.assert_array_equal(util.to_cpu_numpy_buffer(out.cuda()), gold)


@pytest.mark.parametrize("seed", range(20))
def test_op_histogrameq_pillow_seeded_properties(seed):
    rng = np.random.default_rng(seed)
    h, w = int(rng.integers(17, 97)), int(rng.integers(1, 100)) * 2 + 1
    channels = [1, 3, 4][seed % 3]
    host, gold = pillow_equalized_input(h, w, channels, seed)
    src = util.to_cvcuda_tensor(host, "HWC")
    out = cvcuda.histogrameq(src, src.dtype, method=cvcuda.HistogramEq.PILLOW)
    got = util.to_cpu_numpy_buffer(out.cuda())
    np.testing.assert_array_equal(got, gold)
    for c in range(channels):
        order = np.argsort(host[..., c], axis=None)
        x = host[..., c].ravel()[order]
        y = got[..., c].ravel()[order].astype(np.int16)
        assert np.all(np.diff(y) >= 0)
        assert np.all(np.diff(y)[np.diff(x) == 0] == 0)
    floating = host.astype(np.float32) / np.float32(255)
    src = util.to_cvcuda_tensor(floating, "HWC")
    out = cvcuda.histogrameq(src, src.dtype, method=cvcuda.HistogramEq.PILLOW)
    np.testing.assert_array_equal(
        util.to_cpu_numpy_buffer(out.cuda()),
        gold.astype(np.float32) * np.float32(1 / 255),
    )


def test_op_histogrameq_pillow_large_layout_batch_stream_equivalence():
    hosts, golds = zip(*(pillow_equalized_input(1080, 1920, 3, i) for i in range(2)))
    host, gold = np.stack(hosts), np.stack(golds)
    src = util.to_cvcuda_tensor(host, "NHWC")
    default = cvcuda.histogrameq(src, src.dtype, method=cvcuda.HistogramEq.PILLOW)
    np.testing.assert_array_equal(util.to_cpu_numpy_buffer(default.cuda()), gold)
    stream = cvcuda.Stream()
    out = cvcuda.histogrameq(
        src, src.dtype, method=cvcuda.HistogramEq.PILLOW, stream=stream
    )
    stream.sync()
    np.testing.assert_array_equal(util.to_cpu_numpy_buffer(out.cuda()), gold)
    planar = util.to_cvcuda_tensor(
        np.ascontiguousarray(host.transpose(0, 3, 1, 2)), "NCHW"
    )
    out = cvcuda.histogrameq(planar, planar.dtype, method=cvcuda.HistogramEq.PILLOW)
    np.testing.assert_array_equal(
        util.to_cpu_numpy_buffer(out.cuda()).transpose(0, 2, 3, 1), gold
    )
    for fmt in (cvcuda.Format.RGB8, cvcuda.Format.RGB8p):
        batch = cvcuda.ImageBatchVarShape(2)
        for image in hosts:
            data = (
                [util.to_cuda_buffer(image[..., c]) for c in range(3)]
                if fmt.planes > 1
                else util.to_cuda_buffer(image)
            )
            batch.pushback(cvcuda.as_image(data, fmt))
        out = cvcuda.histogrameq(batch, method=cvcuda.HistogramEq.PILLOW)
        for image, expected in zip(out, golds):
            got = util.to_cpu_numpy_buffer(image.cuda())
            if fmt.planes > 1:
                got = got.transpose(1, 2, 0)
            np.testing.assert_array_equal(got, expected)


def opencv_reference(host):
    gold = np.empty_like(host)
    for c in range(host.shape[-1]):
        hist = np.bincount(host[..., c].ravel(), minlength=256)
        first = np.flatnonzero(hist)[0]
        if hist[first] == host.shape[0] * host.shape[1]:
            gold[..., c] = first
        else:
            hist[first] = 0
            cdf = np.cumsum(hist)
            lut = np.rint(cdf.astype(np.float64) / cdf[-1] * 255).astype(np.uint8)
            gold[..., c] = lut[host[..., c]]
    return gold


@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("minimum", [0, 17, 255])
@pytest.mark.parametrize("constant", [False, True])
def test_op_histogrameq_opencv_seeded_gold(channels, minimum, constant):
    rng = np.random.default_rng(123)
    hosts = []
    for h, w in [(33, 129), (35, 131), (37, 133)]:
        host = rng.integers(minimum, 256, (h, w, channels), dtype=np.uint8)
        host[: h // 2] = minimum
        if constant:
            host.fill(minimum)
        hosts.append(host)
    golds = [opencv_reference(host) for host in hosts]
    for layout in ("NHWC", "NCHW"):
        host = np.stack([image[:33, :129] for image in hosts])
        gold = np.stack([opencv_reference(image) for image in host])
        if layout == "NCHW":
            host, gold = (
                np.ascontiguousarray(x.transpose(0, 3, 1, 2)) for x in (host, gold)
            )
        src = util.to_cvcuda_tensor(host, layout)
        out = cvcuda.histogrameq(src, src.dtype, method=cvcuda.HistogramEq.OPENCV)
        np.testing.assert_array_equal(util.to_cpu_numpy_buffer(out.cuda()), gold)
    src = cvcuda.ImageBatchVarShape(3)
    for host in hosts:
        src.pushback(util.to_cvcuda_image(host))
    out = cvcuda.histogrameq(src, method=cvcuda.HistogramEq.OPENCV)
    for image, gold in zip(out, golds):
        got = util.to_cpu_numpy_buffer(image.cuda())
        np.testing.assert_array_equal(got.reshape(gold.shape), gold)


def test_op_histogrameq_opencv_cdf_reuse_stress():
    rng = np.random.default_rng(987)
    host = rng.integers(17, 256, (64, 33, 129, 3), dtype=np.uint8)
    host[:, :25] = 17
    gold = np.stack([opencv_reference(image) for image in host])
    src = util.to_cvcuda_tensor(host, "NHWC")
    for _ in range(100):
        out = cvcuda.histogrameq(src, src.dtype, method=cvcuda.HistogramEq.OPENCV)
        np.testing.assert_array_equal(util.to_cpu_numpy_buffer(out.cuda()), gold)


@pytest.mark.parametrize(
    "input",
    [
        (((1, 460, 640, 1), cvcuda.Type.U8, "NHWC")),
        (((2, 640, 460, 3), cvcuda.Type.U8, "NHWC")),
        (((1, 1920, 1080, 3), cvcuda.Type.U8, "NHWC")),
        (((3, 1000, 1000, 1), cvcuda.Type.U8, "NHWC")),
        (((2, 100, 100, 1), cvcuda.Type.U8, "NHWC")),
        (((2, 460, 640, 1), cvcuda.Type.U8, "NHWC")),
        (((100, 100, 3), cvcuda.Type.U8, "HWC")),
        (((460, 640, 1), cvcuda.Type.U8, "HWC")),
    ],
)
def test_op_histogrameq(input):

    inputTensor = cvcuda.Tensor(*input)

    out = cvcuda.histogrameq(inputTensor, inputTensor.dtype)
    assert out.layout == inputTensor.layout
    assert out.shape == inputTensor.shape
    assert out.dtype == inputTensor.dtype

    out = cvcuda.Tensor(inputTensor.shape, inputTensor.dtype, inputTensor.layout)
    tmp = cvcuda.histogrameq_into(out, inputTensor)

    assert tmp is out
    assert out.layout == inputTensor.layout
    assert out.shape == inputTensor.shape
    assert out.dtype == inputTensor.dtype

    stream = cvcuda.Stream()
    out = cvcuda.histogrameq(src=inputTensor, dtype=inputTensor.dtype, stream=stream)
    assert out.layout == inputTensor.layout
    assert out.shape == inputTensor.shape
    assert out.dtype == inputTensor.dtype

    tmp = cvcuda.histogrameq_into(dst=out, src=inputTensor, stream=stream)
    assert tmp is out
    assert out.layout == inputTensor.layout
    assert out.shape == inputTensor.shape
    assert out.dtype == inputTensor.dtype


@pytest.mark.parametrize(
    "num_images, format, max_size",
    [
        (
            1,
            cvcuda.Format.RGB8,
            (480, 720),
        ),
        (
            5,
            cvcuda.Format.RGB8,
            (720, 480),
        ),
        (
            4,
            cvcuda.Format.Y8,
            (1920, 1080),
        ),
        (
            2,
            cvcuda.Format.BGR8,
            (1000, 1000),
        ),
        (
            3,
            cvcuda.Format.Y8,
            (100, 100),
        ),
        (
            5,
            cvcuda.Format.Y8,
            (460, 640),
        ),
    ],
)
def test_op_histogrameq_varshape(num_images, format, max_size):

    b_src = util.create_image_batch(num_images, format, max_size=max_size, rng=RNG)

    out = cvcuda.histogrameq(b_src)
    assert out.uniqueformat is not None
    assert out.uniqueformat == b_src.uniqueformat
    assert len(out) == len(b_src)
    assert out.capacity == b_src.capacity
    assert out.uniqueformat == b_src.uniqueformat
    assert out.maxsize <= max_size

    tmp = cvcuda.histogrameq_into(out, b_src)
    assert tmp is out
    assert out.uniqueformat is not None
    assert out.uniqueformat == b_src.uniqueformat
    assert len(out) == len(b_src)
    assert out.capacity == b_src.capacity
    assert out.uniqueformat == b_src.uniqueformat
    assert out.maxsize <= max_size

    stream = cvcuda.Stream()
    out = cvcuda.histogrameq(
        src=b_src,
        stream=stream,
    )
    assert out.uniqueformat is not None
    assert out.uniqueformat == b_src.uniqueformat
    assert len(out) == len(b_src)
    assert out.capacity == b_src.capacity
    assert out.uniqueformat == b_src.uniqueformat
    assert out.maxsize <= max_size

    tmp = cvcuda.histogrameq_into(
        src=b_src,
        dst=out,
        stream=stream,
    )
    assert tmp is out
    assert out.uniqueformat is not None
    assert out.uniqueformat == b_src.uniqueformat
    assert len(out) == len(b_src)
    assert out.capacity == b_src.capacity
    assert out.uniqueformat == b_src.uniqueformat
    assert out.maxsize <= max_size


def _histogrameq_params(dtype, layout, channels):

    return {"dtype": dtype}


globals().update(
    cv_tools.make_op_tests(
        name="histogrameq",
        runner_info=[
            ("tensor", cvcuda.histogrameq, _histogrameq_params),
            ("image_batch", cvcuda.histogrameq, None),
        ],
        keystone_dlc=(cvcuda.Type.U8, "NHWC", 3),
        supported_dtypes={
            cvcuda.Type.U8,
            cvcuda.Type.F16,
            cvcuda.Type.F32,
            cvcuda.Type.F64,
        },
        supported_layouts={"NHWC", "HWC", "NCHW", "CHW"},
        supported_channels={1, 2, 3, 4},
        exclude_dlc=[(None, "NCHW", 2), (None, "CHW", 2)],
    )
)
