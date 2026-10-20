# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import cupy
import cvcuda
import numpy as np

DTYPES = [
    np.uint8,
    np.int8,
    np.uint16,
    np.int16,
    np.uint32,
    np.int32,
    np.float16,
    np.float32,
    np.float64,
]
DTYPE_CHANNELS = [(dtype, 3) for dtype in DTYPES] + [
    (np.uint8, channels) for channels in (1, 2, 4)
]
LAYOUTS = ["HWC", "NHWC", "CHW", "NCHW"]
OTHER_LAYOUT = {"HWC": "CHW", "CHW": "HWC", "NHWC": "NCHW", "NCHW": "NHWC"}


def pillow_fixtures():
    reference = json.loads(
        Path(__file__).with_name("split_merge_pillow.json").read_text()
    )
    assert reference["pillow"] == "12.3.0"
    return reference["fixtures"]


def pattern(dtype, layout, channels, seed=17):
    sizes = {"N": 2, "H": 3, "W": 17, "C": channels}
    shape = tuple(sizes[axis] for axis in layout)
    rng = np.random.default_rng(seed)
    data = rng.integers(
        0, 256, np.prod(shape) * np.dtype(dtype).itemsize, dtype=np.uint8
    )
    return data.view(dtype).reshape(shape)


def upload(values, layout):
    return cvcuda.as_tensor(cupy.asarray(values), layout)


def read(tensor):
    return cupy.asnumpy(cupy.asarray(tensor.cuda()))


def assert_bits_equal(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(
        np.ascontiguousarray(actual).view(np.uint8),
        np.ascontiguousarray(expected).view(np.uint8),
    )


def transpose(values, source_layout, destination_layout):
    return values.transpose(
        tuple(source_layout.index(axis) for axis in destination_layout)
    )


VARSHAPE_MATRIX = [
    (dtype, channels, planar)
    for dtype in DTYPES
    for channels in (1, 2, 3, 4)
    for planar in (False, True)
]
RAGGED_SIZES = [
    (1, 1),
    (7, 5),
    (8, 3),
    (9, 7),
    (17, 3),
    (257, 5),
    (513, 1),
    (33, 259),
]


def image_pattern(dtype, channels, width, height, seed=17):
    dtype = np.dtype(dtype)
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 256, width * height * channels * dtype.itemsize, np.uint8)
    payloads = {
        1: [0, 0xFF, 0x80, 0x7F],
        2: [0, 0x8000, 0x7C00, 0xFC00, 0x7E42, 0x7C01, 1, 0xFFFF],
        4: [0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC01234, 0x7F800042, 1],
        8: [
            0,
            0x8000000000000000,
            0x7FF0000000000000,
            0xFFF0000000000000,
            0x7FF8000000001234,
            0x7FF0000000000042,
            1,
        ],
    }[dtype.itemsize]
    words = raw.view(f"u{dtype.itemsize}")
    count = min(words.size, len(payloads))
    words[:count] = payloads[:count]
    return raw.view(dtype).reshape(height, width, channels)


def image_batch(images):
    images = list(images)
    batch = cvcuda.ImageBatchVarShape(max(1, len(images)))
    if images:
        batch.pushback(images)
    return batch


def read_image(image):
    planes = image.cuda()
    if not isinstance(planes, (tuple, list)):
        array = cupy.asnumpy(cupy.from_dlpack(planes))
        if image.format.planes > 1:
            return array.reshape(
                image.format.planes, image.height, image.width
            ).transpose(1, 2, 0)
        return array.reshape(image.height, image.width, -1)
    arrays = [cupy.asnumpy(cupy.from_dlpack(plane)) for plane in planes]
    arrays = [array.reshape(image.height, image.width, -1) for array in arrays]
    return np.concatenate(arrays, axis=2)


class ImageBuffer:
    def __init__(
        self, values, planar=False, salt=0, poison=0xA5, initialize=True, format=None
    ):
        self.values = values
        self.host = []
        self.storage = []
        self.offsets = []
        self.views = []
        height, width, channels = values.shape
        itemsize = values.dtype.itemsize
        native_planar = planar and channels == 4 and itemsize == 1
        if native_planar:
            if format is None:
                kind = (
                    cvcuda.Format.U8.value ^ cvcuda.Format.S8.value
                    if values.dtype.kind == "i"
                    else 0
                )
                format = cvcuda.Format(cvcuda.Format.RGBA8p.value ^ kind)
            self.image = cvcuda.Image(
                (width, height), format, rowalign=1 << (salt % 5 + 3)
            )
            pixels = cupy.asarray(self.image.cuda())
        for plane in range(channels if planar else 1):
            plane_channels = 1 if planar else channels
            pitch = (
                pixels.strides[0] // height
                if native_planar
                else (width * plane_channels + 1 + salt + plane) * itemsize
            )
            offset = 0 if native_planar else (3 + salt + plane) * itemsize
            size = pitch * height if native_planar else offset + pitch * height + 64
            host = np.full(size, poison, np.uint8)
            self.host.append(host)
            self.offsets.append(offset)
            storage = (
                cupy.ndarray(
                    (size,),
                    dtype=np.uint8,
                    memptr=pixels.data + plane * pixels.strides[0],
                )
                if native_planar
                else cupy.empty_like(host)
            )
            self.storage.append(storage)
            self.views.append(
                cupy.ndarray(
                    (height, width, plane_channels),
                    dtype=values.dtype,
                    memptr=storage.data + offset,
                    strides=(pitch, plane_channels * itemsize, itemsize),
                )
            )
        if initialize:
            self.put(values)
        for storage, host in zip(self.storage, self.host):
            storage.set(host)
        if not native_planar:
            self.image = (
                cvcuda.as_image(self.views, format)
                if format is not None
                else cvcuda.as_image(self.views)
            )

    def put(self, values):
        for plane, (host, offset, view) in enumerate(
            zip(self.host, self.offsets, self.views)
        ):
            values_plane = (
                values if len(self.views) == 1 else np.take(values, [plane], axis=2)
            )
            raw = (
                np.ascontiguousarray(values_plane)
                .view(np.uint8)
                .reshape(values.shape[0], -1)
            )
            for row in range(values.shape[0]):
                start = offset + row * view.strides[0]
                stop = start + raw.shape[1]
                host[start:stop] = raw[row]

    def check(self):
        for storage, host in zip(self.storage, self.host):
            np.testing.assert_array_equal(cupy.asnumpy(storage), host)

    def view(self, byte_offset=0, row_delta=0):
        planes = [
            cupy.ndarray(
                view.shape,
                dtype=view.dtype,
                memptr=view.data + byte_offset,
                strides=(view.strides[0] + row_delta, *view.strides[1:]),
            )
            for view in self.views
        ]
        return cvcuda.as_image(planes, self.image.format)
