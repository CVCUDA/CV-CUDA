# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cupy
import cvcuda
import numpy as np
import pytest

from channel_copy_test_utils import (
    DTYPE_CHANNELS,
    ImageBuffer,
    LAYOUTS,
    OTHER_LAYOUT,
    RAGGED_SIZES,
    VARSHAPE_MATRIX,
    assert_bits_equal,
    image_batch,
    image_pattern,
    pattern,
    pillow_fixtures,
    read,
    read_image,
    transpose,
    upload,
)


@pytest.mark.parametrize("dtype,channels", DTYPE_CHANNELS)
@pytest.mark.parametrize("layout", LAYOUTS)
def test_split_bit_exact(dtype, layout, channels):
    values = pattern(dtype, layout, channels)
    source = upload(values, layout)
    stream = cvcuda.Stream()
    bands = cvcuda.split(source, stream=stream)
    stream.sync()
    assert isinstance(bands, tuple)
    assert len(bands) == channels
    axis = layout.index("C")
    for channel, band in enumerate(bands):
        assert band.layout == layout
        assert_bits_equal(read(band), np.take(values, [channel], axis=axis))
    destination_layout = OTHER_LAYOUT[layout]
    expected = [
        transpose(np.take(values, [channel], axis=axis), layout, destination_layout)
        for channel in range(channels)
    ]
    destinations = [
        cvcuda.Tensor(value.shape, dtype, destination_layout) for value in expected
    ]
    assert cvcuda.split_into(destinations, source, stream=stream) is destinations
    stream.sync()
    for destination, value in zip(destinations, expected):
        assert_bits_equal(read(destination), value)
    cupy.asarray(source.cuda()).fill(0)
    cupy.cuda.get_current_stream().synchronize()
    for channel, band in enumerate(bands):
        assert_bits_equal(read(band), np.take(values, [channel], axis=axis))


def test_split_rejects_invalid_inputs():
    source = cvcuda.Tensor((3, 7, 3), np.uint8, "HWC")
    band = cvcuda.Tensor((3, 7, 1), np.uint8, "HWC")
    for destinations in ([], [band], [band] * 3, [band] * 4):
        with pytest.raises((ValueError, RuntimeError)):
            cvcuda.split_into(destinations, source)
    with pytest.raises((ValueError, RuntimeError)):
        cvcuda.split_into([band], band)
    for shape, dtype, layout in [
        ((3, 7, 5), np.uint8, "HWC"),
        ((3, 7, 1), np.int64, "HWC"),
        ((3, 7), np.uint8, "HW"),
    ]:
        source = cvcuda.Tensor(shape, dtype, layout)
        with pytest.raises((ValueError, RuntimeError)):
            cvcuda.split(source)


def test_split_default_and_current_stream():
    values = pattern(np.uint8, "NHWC", 2)
    source = upload(values, "NHWC")
    default_bands = cvcuda.split(source)
    cvcuda.Stream.current.sync()
    stream = cvcuda.Stream()
    with stream:
        bands = cvcuda.split(source)
    for band in bands:
        assert band.cuda().__cuda_array_interface__["stream"] == stream.handle
    stream.sync()
    for channel, (actual, default) in enumerate(zip(bands, default_bands)):
        expected = np.take(values, [channel], axis=-1)
        assert_bits_equal(read(actual), expected)
        assert_bits_equal(read(default), expected)


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("sequence_type", [tuple, list])
def test_split_into_sequence_identity_and_current_stream(layout, sequence_type):
    values = pattern(np.uint8, layout, 3)
    source = upload(values, layout)
    axis = layout.index("C")
    expected = [np.take(values, [channel], axis=axis) for channel in range(3)]
    original = tuple(cvcuda.Tensor(value.shape, np.uint8, layout) for value in expected)
    destinations = sequence_type(original)
    stream = cvcuda.Stream()
    with stream:
        assert cvcuda.split_into(destinations, source) is destinations
    assert len(destinations) == len(original)
    for actual, band in zip(destinations, original):
        assert actual is band
        assert actual.cuda().__cuda_array_interface__["stream"] == stream.handle
    stream.sync()
    for actual, value in zip(destinations, expected):
        assert_bits_equal(read(actual), value)


@pytest.mark.parametrize(
    "fixture", pillow_fixtures(), ids=lambda fixture: fixture["mode"]
)
def test_split_pillow_oracle(fixture):
    source = np.array(fixture["source"], dtype=np.uint8).reshape(fixture["shape"])
    bands = cvcuda.split(upload(source, "HWC"))
    cvcuda.Stream.current.sync()
    for band, values in zip(bands, fixture["bands"]):
        expected = np.array(values, dtype=np.uint8).reshape(2, 3, 1)
        assert_bits_equal(read(band), expected)


@pytest.mark.parametrize("dtype,channels,planar", VARSHAPE_MATRIX)
def test_split_varshape_bit_exact_cpu_gold_and_tensor_parity(dtype, channels, planar):
    values = [
        image_pattern(dtype, channels, width, height, index + 17)
        for index, (width, height) in enumerate(RAGGED_SIZES)
    ]
    sources = [ImageBuffer(value, planar, index) for index, value in enumerate(values)]
    source = image_batch(buffer.image for buffer in sources)
    destinations = [
        [
            ImageBuffer(
                np.take(value, [channel], axis=2),
                salt=index + channel + 7,
                initialize=False,
            )
            for index, value in enumerate(values)
        ]
        for channel in range(channels)
    ]
    output = tuple(
        image_batch(buffer.image for buffer in band) for band in destinations
    )
    cupy.cuda.get_current_stream().synchronize()
    stream = cvcuda.Stream()
    allocated = cvcuda.split(source, stream=stream)
    assert isinstance(allocated, tuple)
    assert len(allocated) == channels
    assert cvcuda.split_into(output, source, stream=stream) is output
    stream.sync()
    allocated_images = [list(band) for band in allocated]
    for channel, (band, buffers) in enumerate(zip(allocated, destinations)):
        assert isinstance(band, cvcuda.ImageBatchVarShape)
        assert len(band) == len(values)
        for index, (image, buffer) in enumerate(zip(band, buffers)):
            gold = np.take(values[index], [channel], axis=2)
            assert image.size == (gold.shape[1], gold.shape[0])
            assert image.format.channels == image.format.planes == 1
            assert_bits_equal(read_image(image), gold)
            assert_bits_equal(read_image(buffer.image), gold)
            buffer.put(gold)
            buffer.check()
    for index, value in enumerate(values):
        sources[index].check()
        if not planar:
            for layout in LAYOUTS:
                tensor_values = value if len(layout) == 3 else value[None]
                source_layout = "HWC" if len(layout) == 3 else "NHWC"
                tensor_values = transpose(tensor_values, source_layout, layout)
                tensor_bands = cvcuda.split(
                    upload(np.ascontiguousarray(tensor_values), layout), stream=stream
                )
                stream.sync()
                for channel, tensor in enumerate(tensor_bands):
                    actual = transpose(read(tensor), layout, source_layout).reshape(
                        value.shape[:2] + (1,)
                    )
                    assert_bits_equal(
                        actual, read_image(allocated_images[channel][index])
                    )
    for buffer in sources:
        for storage in buffer.storage:
            storage.fill(0)
    cupy.cuda.get_current_stream().synchronize()
    for channel, band in enumerate(allocated):
        for index, image in enumerate(band):
            assert_bits_equal(
                read_image(image), np.take(values[index], [channel], axis=2)
            )


@pytest.mark.parametrize("sequence_type", [list, tuple])
@pytest.mark.parametrize(
    "format", [cvcuda.Format.BGR8, cvcuda.Format.BGR8p, cvcuda.Format.BGRA8]
)
def test_split_varshape_physical_order_repeated_sources_and_streams(
    sequence_type, format
):
    values = image_pattern(np.uint8, format.channels, 17, 3)
    source_buffer = ImageBuffer(values, format.planes > 1, format=format)
    source = image_batch([source_buffer.image, source_buffer.view()])
    destinations = [
        [
            ImageBuffer(values[..., :1], salt=channel + index, initialize=False)
            for index in range(2)
        ]
        for channel in range(format.channels)
    ]
    output = sequence_type(
        image_batch(buffer.image for buffer in band) for band in destinations
    )
    cupy.cuda.get_current_stream().synchronize()
    default = cvcuda.split(source)
    cvcuda.Stream.current.sync()
    stream = cvcuda.Stream()
    with stream:
        current = cvcuda.split(source)
        assert cvcuda.split_into(output, source) is output
    stream.sync()
    for channel in range(format.channels):
        gold = np.take(values, [channel], axis=2)
        for band in (default[channel], current[channel], output[channel]):
            for image in band:
                assert_bits_equal(read_image(image), gold)
        for buffer in destinations[channel]:
            buffer.put(gold)
            buffer.check()
    source_buffer.check()


def test_split_varshape_invalid_metadata_leaves_all_destinations_untouched():
    source_buffer = ImageBuffer(image_pattern(np.uint8, 2, 7, 3))
    source = image_batch([source_buffer.image, source_buffer.image])
    buffers = [
        ImageBuffer(image_pattern(np.uint8, 1, 7, 3), salt=index, initialize=False)
        for index in range(4)
    ]
    good = [
        image_batch(buffer.image for buffer in buffers[:2]),
        image_batch(buffer.image for buffer in buffers[2:]),
    ]
    wrong_size = ImageBuffer(image_pattern(np.uint8, 1, 9, 3))
    wrong_type = ImageBuffer(image_pattern(np.int8, 1, 7, 3))
    wrong_format = ImageBuffer(
        image_pattern(np.uint8, 1, 7, 3), format=cvcuda.Format.Y8
    )
    empty = image_batch([])
    invalid = [[], good[:1], good + good, [good[0]] * 2, [good[0], empty]]
    for image in (
        wrong_size.image,
        wrong_type.image,
        wrong_format.image,
        source_buffer.image,
    ):
        invalid.append([good[0], image_batch([buffers[2].image, image])])
    invalid.append([good[0], image_batch([buffers[2].image])])
    cupy.cuda.get_current_stream().synchronize()
    for output in invalid:
        with pytest.raises((ValueError, RuntimeError)):
            cvcuda.split_into(output, source)
        cvcuda.Stream.current.sync()
        for buffer in buffers + [source_buffer, wrong_size, wrong_type, wrong_format]:
            buffer.check()
    for batch in (empty, image_batch([source_buffer.image, buffers[0].image])):
        with pytest.raises((ValueError, RuntimeError)):
            cvcuda.split(batch)
    with pytest.raises((ValueError, RuntimeError)):
        cvcuda.split(image_batch([cvcuda.Image((8, 6), cvcuda.Format.NV12)]))


def test_split_varshape_propagates_native_overlap_error():
    source_buffer = ImageBuffer(image_pattern(np.uint8, 1, 7, 3))
    other_source = ImageBuffer(image_pattern(np.uint8, 1, 7, 3, seed=29))
    destination = ImageBuffer(image_pattern(np.uint8, 1, 7, 3), initialize=False)
    source = image_batch([source_buffer.image, other_source.image])
    output = image_batch([other_source.view(1), destination.image])
    cupy.cuda.get_current_stream().synchronize()
    with pytest.raises(
        (ValueError, RuntimeError), match="Destination spans must not overlap"
    ):
        cvcuda.split_into([output], source)
    cvcuda.Stream.current.sync()
    for buffer in (source_buffer, other_source, destination):
        buffer.check()


@pytest.mark.parametrize("dtype", [np.uint64, np.int64, np.complex64])
def test_split_varshape_rejects_unsupported_scalar_dtype(dtype):
    buffer = ImageBuffer(np.zeros((3, 7, 1), dtype=dtype))
    with pytest.raises((ValueError, RuntimeError)):
        cvcuda.split(image_batch([buffer.image]))
    cvcuda.Stream.current.sync()
    buffer.check()
