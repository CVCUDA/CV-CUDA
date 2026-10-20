# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CAI metadata must retain NumPy's spelling, including vector-type fallback."""

import numpy as np
import pytest

import cvcuda


@pytest.mark.parametrize(
    "dtype",
    [
        "u1",
        "u2",
        "u4",
        "u8",
        "i1",
        "i2",
        "i4",
        "i8",
        "f2",
        "f4",
        "f8",
        "c8",
        "c16",
        "2u1",
        "3f4",
        "4i2",
    ],
)
@pytest.mark.parametrize("rowalign", [1, 32])
def test_export_dtype_and_strides(dtype, rowalign):
    cvcuda.clear_cache()
    dtype = np.dtype(dtype)
    # A 12-byte vector needs row bytes divisible by both 12 and the alignment.
    width = 16 if dtype.itemsize == 12 else 13
    tensor = cvcuda.Tensor((2, 7, width), dtype, "NHW", rowalign=rowalign)
    exported = tensor.cuda()
    interface = exported.__cuda_array_interface__
    assert interface["typestr"] == dtype.str
    assert interface["shape"] == tensor.shape
    pitch = ((width * dtype.itemsize + rowalign - 1) // rowalign) * rowalign
    expected_strides = (7 * pitch, pitch, dtype.itemsize)
    assert interface["strides"] == expected_strides
    assert exported.strides == tuple(s // dtype.itemsize for s in expected_strides)
    assert interface["version"] == 3
    assert interface["data"][0] != 0 and not interface["data"][1]
    assert exported.__cuda_array_interface__ == interface


def test_export_rejects_stride_not_divisible_by_element_size():
    cvcuda.clear_cache()
    tensor = cvcuda.Tensor((2, 7, 13), np.dtype("3f4"), "NHW", rowalign=32)
    with pytest.raises(
        RuntimeError, match="Stride must be a multiple of the element size"
    ):
        tensor.cuda()


@pytest.mark.parametrize(
    "dtype",
    ["uint8", "int16", "float16", "float32", "float64", "complex64", "complex128"],
)
@pytest.mark.parametrize("nondefault", [False, True])
def test_export_retains_values_strides_stream_and_lifetime(dtype, nondefault):
    import gc

    torch = pytest.importorskip("torch")
    stream = torch.cuda.Stream() if nondefault else torch.cuda.default_stream()
    with torch.cuda.stream(stream), cvcuda.as_stream(stream):
        source = (
            torch.arange(2 * 3 * 14 * 13, device="cuda")
            .to(getattr(torch, dtype))
            .reshape(2, 3, 14, 13)[:, :, ::2, :]
        )
        source_stride = source.stride()
        native = cvcuda.as_tensor(source, "NCHW")
        buffer = native.cuda()
        exported = torch.from_dlpack(buffer)
        assert exported.data_ptr() == buffer.__cuda_array_interface__["data"][0]
        assert exported.stride() == source_stride
        source.add_(1)
        torch.testing.assert_close(exported, source, rtol=0, atol=0)
        expected = source.clone()
        del source, native, buffer
        cvcuda.clear_cache()
        gc.collect()
        actual = exported.clone()
    stream.synchronize()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "format_name,dtype",
    [
        ("U8", "u1"),
        ("S16", "i2"),
        ("F16", "f2"),
        ("F32", "f4"),
        ("F64", "f8"),
        ("C64", "c8"),
        ("C128", "c16"),
        ("RGB8", "u1"),
        ("RGB8p", "u1"),
        ("NV12", "u1"),
    ],
)
def test_image_export_dtype(format_name, dtype):
    image = cvcuda.Image((14, 8), getattr(cvcuda.Format, format_name))
    buffers = image.cuda()
    if not isinstance(buffers, (tuple, list)):
        buffers = [buffers]
    for buffer in buffers:
        assert buffer.__cuda_array_interface__["typestr"] == np.dtype(dtype).str
