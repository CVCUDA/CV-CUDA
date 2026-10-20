# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cvcuda

from python_bench_utils import (
    create_image_batch_varshape,
    create_stream_cache,
    create_tensor,
    get_dtype,
    get_dtype_size,
    get_format_from_dtype,
    get_num_channels,
    parse_shape,
)


def _image_format(dtype_name, channels, planar):
    if planar and channels == 4 and get_dtype(dtype_name) == cvcuda.Type.S8:
        signed_kind = cvcuda.Format.U8.value ^ cvcuda.Format.S8.value
        return cvcuda.Format(cvcuda.Format.RGBA8p.value ^ signed_kind)
    try:
        image_format = get_format_from_dtype(dtype_name, channels, planar=planar)
        if image_format.planes == (channels if planar else 1):
            return image_format
    except ValueError:
        pass
    import cupy as cp

    dtype = {
        cvcuda.Type.U8: cp.uint8,
        cvcuda.Type.S8: cp.int8,
        cvcuda.Type.U16: cp.uint16,
        cvcuda.Type.S16: cp.int16,
        cvcuda.Type.U32: cp.uint32,
        cvcuda.Type.S32: cp.int32,
        cvcuda.Type.F16: cp.float16,
        cvcuda.Type.F32: cp.float32,
        cvcuda.Type.F64: cp.float64,
    }[get_dtype(dtype_name)]
    pixels = (
        [cp.empty((1, 1, 1), dtype=dtype) for _ in range(channels)]
        if planar and channels > 1
        else cp.empty((1, 1, channels), dtype=dtype)
    )
    return cvcuda.as_image(pixels).format


def _varshape(state, split, shape, dtype_name, channels, planar, variation):
    samples, height, width = shape
    if min(shape) <= 0 or channels not in range(1, 5) or variation < 0:
        state.skip(
            "VarShape channel-copy benchmarks require positive shapes, C1..4 and nonnegative sizeVariation"
        )
        return None
    import cupy as cp

    with cp.cuda.Device(state.get_device()):
        image_format = _image_format(dtype_name, channels, planar)
        band_format = _image_format(dtype_name, 1, False)
    image = create_image_batch_varshape(
        (samples, height, width, channels),
        variation,
        image_format,
        dtype=get_dtype(dtype_name),
        device=state.get_device(),
        fill_mode="checkerboard" if split else 0,
    )
    bands = [
        create_image_batch_varshape(
            (samples, height, width, 1),
            variation,
            band_format,
            dtype=get_dtype(dtype_name),
            device=state.get_device(),
            fill_mode=0 if split else "checkerboard",
        )
        for _ in range(channels)
    ]
    scalar_bytes = get_dtype_size(dtype_name) // get_num_channels(dtype_name)
    size = (
        sum(sample.size[0] * sample.size[1] for sample in image)
        * channels
        * scalar_bytes
    )
    state.add_global_memory_reads(size)
    state.add_global_memory_writes(size)
    get_stream = create_stream_cache()

    def run(launch):
        stream = get_stream(launch)
        if split:
            cvcuda.split_into(bands, image, stream=stream)
        else:
            cvcuda.merge_into(image, bands, stream=stream)

    return run


def channel_copy(state, split):
    """VarShape shapes accept an @sizeVariation suffix (per-image pixel step)."""
    shape_text, separator, variation = state.get_string("shape").partition("@")
    samples, height, width = parse_shape(shape_text)
    dtype_name = state.get_string("InOutDataType")
    dtype = get_dtype(dtype_name)
    channels = state.get_int64("channels")
    layout = state.get_string("layout")
    input_kind = state.get_string("inputKind")
    if input_kind not in ("Tensor", "VarShape") or layout not in (
        "NHWC",
        "NCHW",
        "NCHW_FAKE",
    ):
        state.skip(
            "Channel-copy benchmarks require Tensor or VarShape and NHWC, NCHW or NCHW_FAKE"
        )
        return None
    fake = layout == "NCHW_FAKE"
    if input_kind == "VarShape":
        if fake:
            state.skip(
                "Fake-planar (NCHW_FAKE) channel-copy benchmarks are Tensor-only"
            )
            return None
        return _varshape(
            state,
            split,
            (samples, height, width),
            dtype_name,
            channels,
            layout == "NCHW",
            int(variation) if separator else 0,
        )
    if separator:
        state.skip("Ragged shape suffixes are VarShape-only")
        return None
    if fake and dtype == cvcuda.Type.U32:
        state.skip(
            "Reformat does not support U32; benchmark native layouts for this dtype"
        )
        return None
    planar = layout != "NHWC"
    scalar_bytes = get_dtype_size(dtype_name) // get_num_channels(dtype_name)
    size = samples * height * width * channels * scalar_bytes * (3 if fake else 1)
    state.add_global_memory_reads(size)
    state.add_global_memory_writes(size)

    def tensor(channel_count, is_planar, fill):
        shape = (
            (samples, channel_count, height, width)
            if is_planar
            else (samples, height, width, channel_count)
        )
        return create_tensor(
            shape,
            dtype,
            state.get_device(),
            layout="NCHW" if is_planar else "NHWC",
            fill_mode=fill,
        )

    image = tensor(channels, planar, "checkerboard" if split else 0)
    interleaved = tensor(channels, False, 0) if fake else image
    bands = [tensor(1, planar, 0 if split else "checkerboard") for _ in range(channels)]
    inter_bands = [tensor(1, False, 0) for _ in range(channels)] if fake else bands
    get_stream = create_stream_cache()

    def run(launch):
        stream = get_stream(launch)
        if split:
            if fake:
                cvcuda.reformat_into(interleaved, image, stream=stream)
            cvcuda.split_into(inter_bands, interleaved, stream=stream)
            if fake:
                for destination, source in zip(bands, inter_bands):
                    cvcuda.reformat_into(destination, source, stream=stream)
        else:
            if fake:
                for destination, source in zip(inter_bands, bands):
                    cvcuda.reformat_into(destination, source, stream=stream)
            cvcuda.merge_into(interleaved, inter_bands, stream=stream)
            if fake:
                cvcuda.reformat_into(image, interleaved, stream=stream)

    return run
