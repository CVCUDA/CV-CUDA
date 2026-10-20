# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import threading
import weakref
from contextlib import contextmanager

import cupy
import cvcuda
import numpy as np
import pytest

from cvcuda_test_utils import as_cupy_stream
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
from test_interop_cai_stream import _enqueue_gpu_delay as enqueue_gpu_delay


@contextmanager
def _hold_stream(stream):
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def wait_for_release(_argument):
        entered.set()
        release.wait()
        finished.set()

    # Finalizing a CV-CUDA allocation calls cudaFree, which may synchronize with
    # the held stream. A collection that reaches earlier tests' garbage while the
    # gate is closed would then never return, so collect it up front and keep
    # automatic collection off until the gate opens.
    gc.collect()
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        as_cupy_stream(stream.handle).launch_host_func(wait_for_release, None)
        try:
            assert entered.wait(10), "Stream gate callback did not start"
            yield
        finally:
            release.set()
            assert finished.wait(10), "Stream gate callback did not finish"
    finally:
        if gc_was_enabled:
            gc.enable()


def test_hold_stream_collects_prior_garbage_before_gating():
    class Cycle:
        pass

    finalized = threading.Event()
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        cycle = Cycle()
        cycle.self = cycle
        weakref.finalize(cycle, finalized.set)
        del cycle
        with _hold_stream(cvcuda.Stream()):
            assert finalized.is_set()
    finally:
        if gc_was_enabled:
            gc.enable()


def test_hold_stream_suspends_automatic_collection():
    gc_was_enabled = gc.isenabled()
    gc.enable()
    try:
        with _hold_stream(cvcuda.Stream()):
            assert not gc.isenabled()
        assert gc.isenabled()
    finally:
        if not gc_was_enabled:
            gc.disable()


@pytest.mark.parametrize("dtype,channels", DTYPE_CHANNELS)
@pytest.mark.parametrize("layout", LAYOUTS)
def test_merge_bit_exact(dtype, layout, channels):
    values = [pattern(dtype, layout, 1, channel + 17) for channel in range(channels)]
    bands = [upload(value, layout) for value in values]
    expected = np.concatenate(values, axis=layout.index("C"))
    stream = cvcuda.Stream()
    output = cvcuda.merge(bands, stream=stream)
    stream.sync()
    assert output.layout == layout
    assert_bits_equal(read(output), expected)
    destination_layout = OTHER_LAYOUT[layout]
    expected_planar = transpose(expected, layout, destination_layout)
    destination = cvcuda.Tensor(expected_planar.shape, dtype, destination_layout)
    assert cvcuda.merge_into(destination, bands, stream=stream) is destination
    allocated = cvcuda.merge(bands, layout=destination_layout, stream=stream)
    stream.sync()
    assert_bits_equal(read(destination), expected_planar)
    assert_bits_equal(read(allocated), expected_planar)
    for band, value in zip(bands, values):
        assert_bits_equal(read(band), value)


def test_merge_round_trip_and_repeated_bands():
    values = pattern(np.float32, "NHWC", 4)
    source = upload(values, "NHWC")
    bands = cvcuda.split(source)
    result = cvcuda.merge(bands)
    reordered = cvcuda.merge([bands[2], bands[0], bands[2]])
    cvcuda.Stream.current.sync()
    assert_bits_equal(read(result), values)
    assert_bits_equal(read(reordered), values[..., [2, 0, 2]])


def _wrapped_packed_u8(
    shape,
    channels,
    poison,
    offset=0,
    row_padding=0,
    sample_padding=0,
    column_padding=0,
    channel_stride=1,
):
    samples, rows, columns = shape
    column_stride = channels + column_padding
    row_stride = columns * column_stride + row_padding
    sample_stride = rows * row_stride + sample_padding
    strides = (sample_stride, row_stride, column_stride, channel_stride)
    expected = np.full(samples * sample_stride + offset + 128, poison, dtype=np.uint8)
    storage = cupy.asarray(expected)
    view = cupy.ndarray(
        shape + (channels,),
        dtype=np.uint8,
        memptr=storage.data + 64 + offset,
        strides=strides,
    )
    host = np.ndarray(
        shape + (channels,),
        dtype=np.uint8,
        buffer=expected,
        offset=64 + offset,
        strides=strides,
    )
    return storage, view, cvcuda.as_tensor(view, "NHWC"), expected, host


def _check_packed_split_merge(source, bands, output):
    values = np.arange(source[4].size, dtype=np.uint8).reshape(source[4].shape)
    source[4][...] = values
    source[0].set(source[3])
    cupy.cuda.get_current_stream().synchronize()
    stream = cvcuda.Stream()
    destinations = [band[2] for band in bands]
    assert cvcuda.split_into(destinations, source[2], stream=stream) is destinations
    for channel, band in enumerate(bands):
        band[4][...] = values[..., channel, None]
    for order in ([0, 1, 2], [2, 0, 2]):
        assert (
            cvcuda.merge_into(
                output[2], [bands[channel][2] for channel in order], stream=stream
            )
            is output[2]
        )
        stream.sync()
        output[4][...] = values[..., order]
        for storage, _view, _tensor, expected, _host in [source, output, *bands]:
            np.testing.assert_array_equal(cupy.asnumpy(storage), expected)


@pytest.mark.parametrize(
    "shape", [(2, 3, 257), (2, 3, 512), (1, 1024, 1024), (1, 2000, 3001)]
)
@pytest.mark.parametrize(
    "image_offset,band_offset,image_padding,band_padding",
    [
        (0, 0, 0, 0),
        (1, 0, 0, 0),
        (0, 1, 0, 0),
        (8, 8, 0, 0),
        (0, 0, 3, 0),
        (0, 0, 0, 3),
    ],
)
def test_split_merge_packed_boundaries_and_canaries(
    shape, image_offset, band_offset, image_padding, band_padding
):
    source = _wrapped_packed_u8(
        shape, 3, 0xC3, image_offset, image_padding, image_padding
    )
    bands = [
        _wrapped_packed_u8(
            shape,
            1,
            0x31 + channel * 0x11,
            (channel + 2) * band_offset,
            (channel + 2) * band_padding,
            (channel + 2) * band_padding,
        )
        for channel in range(3)
    ]
    output = _wrapped_packed_u8(
        shape, 3, 0xE7, image_offset, image_padding, image_padding
    )
    _check_packed_split_merge(source, bands, output)


@pytest.mark.parametrize("bad_tensor", range(5))
@pytest.mark.parametrize("guard", ["sample_padding", "row_padding", "offset"])
def test_split_merge_packed_independent_guards(bad_tensor, guard):
    shape = (
        2 if guard == "sample_padding" else 1,
        3 if guard == "row_padding" else 1,
        129,
    )
    buffers = [
        _wrapped_packed_u8(
            shape,
            3 if index in (0, 4) else 1,
            poison,
            **({guard: 1} if index == bad_tensor else {}),
        )
        for index, poison in enumerate((0xC3, 0x31, 0x42, 0x53, 0xE7))
    ]
    _check_packed_split_merge(buffers[0], buffers[1:4], buffers[4])


@pytest.mark.parametrize("bad_tensor", range(5))
def test_split_merge_rejects_nonpacked_columns(bad_tensor):
    with pytest.raises(RuntimeError, match="Pitch of dimension 2.*packed"):
        _wrapped_packed_u8(
            (1, 1, 129), 3 if bad_tensor in (0, 4) else 1, 0xC3, column_padding=1
        )


@pytest.mark.parametrize("bad_band", range(3))
def test_split_merge_independently_spaced_single_channel_band(bad_band):
    shape = (1, 1, 129)
    source = _wrapped_packed_u8(shape, 3, 0xC3)
    bands = [
        _wrapped_packed_u8(
            shape,
            1,
            0x31 + channel * 0x11,
            column_padding=1 if channel == bad_band else 0,
            channel_stride=2 if channel == bad_band else 1,
        )
        for channel in range(3)
    ]
    output = _wrapped_packed_u8(shape, 3, 0xE7)
    _check_packed_split_merge(source, bands, output)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_merge_one_band_storage_independence_and_current_stream(layout):
    values = pattern(np.uint8, layout, 1)
    band = upload(values, layout)
    other_layout = OTHER_LAYOUT[layout]
    stream = cvcuda.Stream()
    with stream:
        output = cvcuda.merge((band,))
        other = cvcuda.merge([band], layout=other_layout)
    for result in (output, other):
        assert result is not band
        assert result.cuda().__cuda_array_interface__["stream"] == stream.handle
    stream.sync()
    cupy.asarray(band.cuda()).fill(0)
    cupy.cuda.get_current_stream().synchronize()
    assert_bits_equal(read(output), values)
    assert_bits_equal(read(other), transpose(values, layout, other_layout))
    cupy.asarray(output.cuda()).fill(123)
    cupy.cuda.get_current_stream().synchronize()
    assert_bits_equal(read(band), np.zeros_like(values))
    assert_bits_equal(read(other), transpose(values, layout, other_layout))


@pytest.mark.parametrize("use_current_stream", [False, True])
def test_split_merge_into_orders_cross_stream_work(use_current_stream):
    values = pattern(np.uint8, "NHWC", 3)
    producer = cvcuda.Stream()
    splitter = cvcuda.Stream()
    consumer = cvcuda.Stream()
    producer_cuda = as_cupy_stream(producer.handle)
    consumer_cuda = as_cupy_stream(consumer.handle)
    source = upload(np.zeros_like(values), "NHWC")
    source_view = cupy.asarray(source.cuda())
    new_values = cupy.asarray(values)
    bands = [
        cvcuda.Tensor(values.shape[:-1] + (1,), np.uint8, "NHWC") for _ in range(3)
    ]
    ordered = [bands[2], bands[0], bands[2]]
    output = cvcuda.Tensor(values.shape, np.uint8, "NHWC")
    output_view = cupy.asarray(output.cuda())
    snapshot = cupy.empty_like(new_values)
    cupy.cuda.get_current_stream().synchronize()

    for _attempt in range(3):
        try:
            enqueue_gpu_delay(producer.handle, cycles=1)
            source.submitStreamSync(producer)
            with producer_cuda:
                source_view[...] = new_values
                source_view.fill(0)
            cvcuda.split_into(bands, source, stream=splitter)
            cvcuda.merge_into(output, ordered, stream=consumer)
            with consumer_cuda:
                snapshot[...] = output_view
            consumer.sync()
            cvcuda.internal.syncAuxStream()
            assert_bits_equal(cupy.asnumpy(snapshot), np.zeros_like(values))

            source.submitStreamSync(producer)
            enqueue_gpu_delay(producer.handle, cycles=500_000_000)
            with producer_cuda:
                source_view[...] = new_values
            producer_pending = not producer_cuda.done
            if use_current_stream:
                with splitter:
                    assert cvcuda.split_into(bands, source) is bands
                with consumer:
                    assert cvcuda.merge_into(output, ordered) is output
            else:
                assert cvcuda.split_into(bands, source, stream=splitter) is bands
                assert cvcuda.merge_into(output, ordered, stream=consumer) is output
            with consumer_cuda:
                snapshot[...] = output_view
            observed_pending = producer_pending and not producer_cuda.done
            consumer.sync()
            assert_bits_equal(cupy.asnumpy(snapshot), values[..., [2, 0, 2]])
            if observed_pending:
                return
        finally:
            producer.sync()
            splitter.sync()
            consumer.sync()
    pytest.skip("Producer observation window expired in all three attempts")


def test_merge_rejects_invalid_inputs():
    band = cvcuda.Tensor((3, 7, 1), np.uint8, "HWC")
    invalid = [
        [],
        [band] * 5,
        [band, cvcuda.Tensor((3, 7, 2), np.uint8, "HWC")],
        [band, cvcuda.Tensor((4, 7, 1), np.uint8, "HWC")],
        [band, cvcuda.Tensor((3, 7, 1), np.float32, "HWC")],
        [band, cvcuda.Tensor((1, 3, 7), np.uint8, "CHW")],
    ]
    for bands in invalid:
        with pytest.raises((ValueError, RuntimeError)):
            cvcuda.merge(bands)
    with pytest.raises((ValueError, RuntimeError)):
        cvcuda.merge_into(band, [band])
    with pytest.raises((ValueError, RuntimeError)):
        cvcuda.merge([band], layout="NHWC")
    with pytest.raises((ValueError, RuntimeError)):
        cvcuda.merge_into(cvcuda.Tensor((3, 7, 2), np.uint8, "HWC"), [band])


@pytest.mark.parametrize(
    "fixture", pillow_fixtures(), ids=lambda fixture: fixture["mode"]
)
def test_merge_pillow_oracle(fixture):
    bands = [
        upload(np.array(values, dtype=np.uint8).reshape(2, 3, 1), "HWC")
        for values in fixture["bands"]
    ]
    merged = cvcuda.merge(bands)
    cvcuda.Stream.current.sync()
    expected = np.array(fixture["merged"], dtype=np.uint8).reshape(fixture["shape"])
    assert_bits_equal(read(merged), expected)


@pytest.mark.parametrize("dtype,channels,planar", VARSHAPE_MATRIX)
def test_merge_varshape_bit_exact_cpu_gold_and_tensor_parity(dtype, channels, planar):
    values = [
        [
            image_pattern(dtype, 1, width, height, index * 29 + channel * 17)
            for index, (width, height) in enumerate(RAGGED_SIZES)
        ]
        for channel in range(channels)
    ]
    buffers = [
        [ImageBuffer(value, salt=index + channel) for index, value in enumerate(band)]
        for channel, band in enumerate(values)
    ]
    bands = tuple(image_batch(buffer.image for buffer in band) for band in buffers)
    expected = [
        np.concatenate([band[index] for band in values], axis=2)
        for index in range(len(RAGGED_SIZES))
    ]
    destinations = [
        ImageBuffer(value, planar, index + 7, initialize=False)
        for index, value in enumerate(expected)
    ]
    parity_destinations = [
        ImageBuffer(value, not planar, index + 11, initialize=False)
        for index, value in enumerate(expected)
    ]
    output = image_batch(buffer.image for buffer in destinations)
    parity_output = image_batch(buffer.image for buffer in parity_destinations)
    cupy.cuda.get_current_stream().synchronize()
    stream = cvcuda.Stream()
    allocated = cvcuda.merge(bands, format=output.uniqueformat, stream=stream)
    assert cvcuda.merge_into(output, bands, stream=stream) is output
    assert cvcuda.merge_into(parity_output, list(bands), stream=stream) is parity_output
    stream.sync()
    assert isinstance(allocated, cvcuda.ImageBatchVarShape)
    assert allocated.uniqueformat == output.uniqueformat
    assert len(allocated) == len(expected)
    allocated_images = list(allocated)
    output_images = list(output)
    parity_images = list(parity_output)
    for index, gold in enumerate(expected):
        assert allocated_images[index].size == (gold.shape[1], gold.shape[0])
        for image in (
            allocated_images[index],
            output_images[index],
            parity_images[index],
        ):
            assert_bits_equal(read_image(image), gold)
        for buffer in (destinations[index], parity_destinations[index]):
            buffer.put(gold)
            buffer.check()
        if not planar:
            for layout in LAYOUTS:
                source_layout = "HWC" if len(layout) == 3 else "NHWC"
                tensors = []
                for band in values:
                    value = band[index] if len(layout) == 3 else band[index][None]
                    tensors.append(
                        upload(transpose(value, source_layout, layout), layout)
                    )
                tensor = cvcuda.merge(tensors, layout=layout, stream=stream)
                stream.sync()
                actual = transpose(read(tensor), layout, source_layout).reshape(
                    gold.shape
                )
                assert_bits_equal(actual, read_image(allocated_images[index]))
    for band in buffers:
        for buffer in band:
            buffer.check()
            for storage in buffer.storage:
                storage.fill(0)
    cupy.cuda.get_current_stream().synchronize()
    for index, gold in enumerate(expected):
        assert_bits_equal(read_image(allocated_images[index]), gold)


@pytest.mark.parametrize("sequence_type", [list, tuple])
@pytest.mark.parametrize(
    "format", [cvcuda.Format.BGR8, cvcuda.Format.BGR8p, cvcuda.Format.BGRA8]
)
def test_merge_varshape_repeated_sources_physical_order_and_streams(
    sequence_type, format
):
    values = [
        image_pattern(np.uint8, 1, 17, 3, channel + 17)
        for channel in range(format.channels)
    ]
    buffers = [ImageBuffer(value, salt=channel) for channel, value in enumerate(values)]
    inputs = [image_batch([buffer.image, buffer.view()]) for buffer in buffers]
    for repeated in (False, True):
        bands = sequence_type([inputs[0]] * format.channels if repeated else inputs)
        gold = np.concatenate(
            [values[0]] * format.channels if repeated else values, axis=2
        )
        destinations = [
            ImageBuffer(gold, format.planes > 1, index, initialize=False, format=format)
            for index in range(2)
        ]
        output = image_batch(buffer.image for buffer in destinations)
        cupy.cuda.get_current_stream().synchronize()
        default = cvcuda.merge(bands, format=format)
        cvcuda.Stream.current.sync()
        stream = cvcuda.Stream()
        with stream:
            current = cvcuda.merge(bands, format=format)
            assert cvcuda.merge_into(output, bands) is output
        stream.sync()
        for batch in (default, current, output):
            assert batch.uniqueformat == format
            for image in batch:
                assert_bits_equal(read_image(image), gold)
        for buffer in destinations:
            buffer.put(gold)
            buffer.check()
    for buffer in buffers:
        buffer.check()


def test_merge_varshape_invalid_metadata_leaves_all_destinations_untouched():
    destinations = [
        ImageBuffer(image_pattern(np.uint8, 2, 7, 3), salt=index, initialize=False)
        for index in range(2)
    ]
    output = image_batch(buffer.image for buffer in destinations)
    band = ImageBuffer(image_pattern(np.uint8, 1, 7, 3))
    good = image_batch([band.image, band.image])
    wrong_size = ImageBuffer(image_pattern(np.uint8, 1, 9, 3))
    wrong_type = ImageBuffer(image_pattern(np.int8, 1, 7, 3))
    wrong_format = ImageBuffer(
        image_pattern(np.uint8, 1, 7, 3), format=cvcuda.Format.Y8
    )
    empty = image_batch([])
    invalid = [
        [],
        [good],
        [good] * 3,
        [good] * 5,
        [good, empty],
        [good, image_batch([band.image])],
    ]
    for image in (
        wrong_size.image,
        wrong_type.image,
        wrong_format.image,
        destinations[0].image,
    ):
        invalid.append([good, image_batch([band.image, image])])
    cupy.cuda.get_current_stream().synchronize()
    for inputs in invalid:
        with pytest.raises((ValueError, RuntimeError)):
            cvcuda.merge_into(output, inputs)
        cvcuda.Stream.current.sync()
        for buffer in destinations + [band, wrong_size, wrong_type, wrong_format]:
            buffer.check()
    for inputs in ([], [empty], [good] * 5):
        with pytest.raises((ValueError, RuntimeError)):
            cvcuda.merge(inputs, format=output.uniqueformat)
    with pytest.raises(TypeError):
        cvcuda.merge([good, good])
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        cvcuda.merge([good, good], format=output.uniqueformat, layout="HWC")
    with pytest.raises((ValueError, RuntimeError)):
        cvcuda.merge([good, good], format=cvcuda.Format.U8)
    with pytest.raises((ValueError, RuntimeError)):
        cvcuda.merge([good] * 3, format=cvcuda.Format.NV12)
    for destination in (empty, image_batch([destinations[0].image, band.image])):
        with pytest.raises((ValueError, RuntimeError)):
            cvcuda.merge_into(destination, [good, good])
    cvcuda.Stream.current.sync()
    for buffer in destinations + [band]:
        buffer.check()


def test_merge_varshape_propagates_native_overlap_error():
    source = ImageBuffer(image_pattern(np.uint8, 1, 7, 3))
    other_source = ImageBuffer(image_pattern(np.uint8, 1, 7, 3, seed=29))
    destination = ImageBuffer(image_pattern(np.uint8, 1, 7, 3), initialize=False)
    inputs = [image_batch([source.image, other_source.image])]
    output = image_batch([other_source.view(1), destination.image])
    cupy.cuda.get_current_stream().synchronize()
    with pytest.raises(
        (ValueError, RuntimeError), match="Destination spans must not overlap"
    ):
        cvcuda.merge_into(output, inputs)
    cvcuda.Stream.current.sync()
    for buffer in (source, other_source, destination):
        buffer.check()


@pytest.mark.parametrize("use_current_stream", [False, True])
def test_split_merge_varshape_into_orders_cross_stream_work(use_current_stream):
    values = image_pattern(np.uint8, 3, 257, 5)
    source_buffer = ImageBuffer(np.zeros_like(values))
    source = image_batch([source_buffer.image])
    band_buffers = [ImageBuffer(values[..., :1], initialize=False) for _ in range(3)]
    bands = [image_batch([buffer.image]) for buffer in band_buffers]
    destination = ImageBuffer(values, initialize=False)
    output = image_batch([destination.image])
    producer, splitter, consumer = cvcuda.Stream(), cvcuda.Stream(), cvcuda.Stream()
    producer_cuda = as_cupy_stream(producer.handle)
    consumer_cuda = as_cupy_stream(consumer.handle)
    new_values = cupy.asarray(values)
    snapshot = cupy.empty_like(new_values)
    cupy.cuda.get_current_stream().synchronize()
    cvcuda.split_into(bands, source, stream=splitter)
    cvcuda.merge_into(output, [bands[2], bands[0], bands[2]], stream=consumer)
    with producer_cuda:
        source_buffer.views[0][...] = new_values
        source_buffer.views[0].fill(0)
    producer.sync()
    with consumer_cuda:
        snapshot[...] = destination.views[0]
    consumer.sync()
    source_buffer.image.submitStreamSync(producer)
    try:
        with _hold_stream(producer):
            with producer_cuda:
                source_buffer.views[0][...] = new_values
            if use_current_stream:
                with splitter:
                    assert cvcuda.split_into(bands, source) is bands
                with consumer:
                    assert (
                        cvcuda.merge_into(output, [bands[2], bands[0], bands[2]])
                        is output
                    )
            else:
                assert cvcuda.split_into(bands, source, stream=splitter) is bands
                assert (
                    cvcuda.merge_into(
                        output, [bands[2], bands[0], bands[2]], stream=consumer
                    )
                    is output
                )
            with consumer_cuda:
                snapshot[...] = destination.views[0]
            assert not producer_cuda.done
        consumer.sync()
        assert_bits_equal(cupy.asnumpy(snapshot), values[..., [2, 0, 2]])
    finally:
        producer.sync()
        splitter.sync()
        consumer.sync()


@pytest.mark.parametrize("split", [False, True])
def test_split_merge_varshape_into_retains_pending_input_owners(split):
    cvcuda.clear_cache()
    values = image_pattern(np.uint8, 3, 257, 5)
    buffers = (
        [ImageBuffer(values)]
        if split
        else [ImageBuffer(np.take(values, [channel], axis=2)) for channel in range(3)]
    )
    owners = [weakref.ref(view) for buffer in buffers for view in buffer.views]
    inputs = [image_batch([buffer.image]) for buffer in buffers]
    stream = cvcuda.Stream()
    external = as_cupy_stream(stream.handle)
    cupy.cuda.get_current_stream().synchronize()
    result = (
        cvcuda.split(inputs[0], stream=stream)
        if split
        else cvcuda.merge(inputs, format=cvcuda.Format.RGB8, stream=stream)
    )
    stream.sync()
    try:
        with _hold_stream(stream):
            if split:
                cvcuda.split_into(result, inputs[0], stream=stream)
            else:
                cvcuda.merge_into(result, inputs, stream=stream)
            del inputs, buffers
            gc.collect()
            assert not external.done
            assert all(owner() is not None for owner in owners)
    finally:
        stream.sync()
    if split:
        for channel, band in enumerate(result):
            assert_bits_equal(
                read_image(next(iter(band))), np.take(values, [channel], axis=2)
            )
    else:
        assert_bits_equal(read_image(next(iter(result))), values)


@pytest.mark.parametrize("dtype", [np.uint64, np.int64, np.complex64])
def test_merge_varshape_rejects_unsupported_scalar_dtype(dtype):
    buffer = ImageBuffer(np.zeros((3, 7, 1), dtype=dtype))
    with pytest.raises((ValueError, RuntimeError)):
        cvcuda.merge([image_batch([buffer.image])], format=buffer.image.format)
    cvcuda.Stream.current.sync()
    buffer.check()
