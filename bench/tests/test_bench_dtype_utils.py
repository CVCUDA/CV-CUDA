# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared C++ and Python benchmark dtype utility contracts."""

import importlib.util
import sys
import types
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytest

BENCH = Path(__file__).resolve().parents[1]


@pytest.fixture
def channel_copy_module(monkeypatch):
    fake = types.ModuleType("cvcuda")
    fake.Tensor = object
    fake.Type = types.SimpleNamespace(
        **{
            name: name
            for name in (
                "U8",
                "S8",
                "U16",
                "S16",
                "U32",
                "S32",
                "U64",
                "S64",
                "F16",
                "F32",
                "F64",
            )
        }
    )

    class Formats(types.SimpleNamespace):
        def __call__(self, value):
            return types.SimpleNamespace(planes=4, value=value)

    fake.Format = Formats(
        **{
            name: types.SimpleNamespace(planes=planes)
            for name, planes in (
                ("U8", 1),
                ("S8", 1),
                ("U16", 1),
                ("S16", 1),
                ("S32", 1),
                ("F16", 1),
                ("F32", 1),
                ("F64", 1),
                ("_2F16", 1),
                ("_2F32", 1),
                ("_2S16", 1),
                ("RGB8", 1),
                ("RGBA8", 1),
                ("RGBf16", 1),
                ("RGBAf16", 1),
                ("RGBf32", 1),
                ("RGBAf32", 1),
                ("RGB8p", 3),
                ("RGBA8p", 4),
                ("RGBf16p", 3),
                ("RGBAf16p", 4),
                ("RGBf32p", 3),
                ("RGBAf32p", 4),
            )
        }
    )
    fake.Format.U8.value = 0x11
    fake.Format.S8.value = 0x8011
    fake.Format.RGBA8p.value = 0x22004

    def as_image(pixels):
        if (
            isinstance(pixels, list)
            and len(pixels) == 4
            and pixels[0].dtype.itemsize == 1
        ):
            raise ValueError("Code of 4th plane packing must be 0")
        return types.SimpleNamespace(
            format=types.SimpleNamespace(
                planes=len(pixels) if isinstance(pixels, list) else 1
            )
        )

    fake.as_image = as_image
    monkeypatch.setitem(sys.modules, "cvcuda", fake)
    fake_cupy = types.ModuleType("cupy")
    for name in (
        "uint8",
        "int8",
        "uint16",
        "int16",
        "uint32",
        "int32",
        "int64",
        "float16",
        "float32",
        "float64",
        "empty",
    ):
        setattr(fake_cupy, name, getattr(np, name))
    fake_cupy.cuda = types.SimpleNamespace(
        Device=lambda _device: nullcontext(), set_allocator=lambda _allocator: None
    )
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)
    spec = importlib.util.spec_from_file_location(
        "channel_copy_utils_test", BENCH / "python/python_bench_utils.py"
    )
    utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils)
    monkeypatch.setitem(sys.modules, "python_bench_utils", utils)
    spec = importlib.util.spec_from_file_location(
        "channel_copy_bench_test", BENCH / "python/channel_copy_bench.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "dtype",
    [
        "uint8",
        "int8",
        "uint16",
        "int16",
        "uint32",
        "int32",
        "float16",
        "float32",
        "float64",
    ],
)
@pytest.mark.parametrize("channels", [1, 2, 3, 4])
@pytest.mark.parametrize("planar", [False, True])
def test_channel_copy_varshape_formats(channel_copy_module, dtype, channels, planar):
    image_format = channel_copy_module._image_format(dtype, channels, planar)
    assert image_format.planes == (channels if planar else 1)
    if dtype in ("uint8", "float32") and channels == 3:
        name = "RGB8" if dtype == "uint8" else "RGBf32"
        assert image_format is getattr(
            channel_copy_module.cvcuda.Format, name + ("p" if planar else "")
        )


def test_channel_copy_signed_four_plane_format_preserves_kind(channel_copy_module):
    module = channel_copy_module
    image_format = module._image_format("int8", 4, True)
    formats = module.cvcuda.Format
    assert (
        image_format.value == formats.RGBA8p.value ^ formats.U8.value ^ formats.S8.value
    )
    assert module._image_format("uint8", 4, True) is formats.RGBA8p


@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("layout", ["NHWC", "NCHW", "NCHW_FAKE"])
@pytest.mark.parametrize("variation", [0, 17])
@pytest.mark.parametrize(
    "dtype,channels",
    [("uchar3", 3), ("float3", 3), ("uint32", 2), ("float64", 4), ("int8", 4)],
)
def test_channel_copy_varshape_submits_one_batch(
    channel_copy_module, split, layout, variation, dtype, channels
):
    module = channel_copy_module
    batches, fills, calls, reads, writes, skips = [], [], [], [], [], []
    stream = object()
    module.create_stream_cache = lambda: lambda _launch: stream

    def create_batch(shape, size_variation, image_format, **kwargs):
        assert shape == (4, 64, 96, channels if not batches else 1)
        assert size_variation == variation
        assert image_format.planes == (
            channels if not batches and layout == "NCHW" else 1
        )
        assert kwargs["dtype"] == module.get_dtype(dtype)
        assert kwargs["device"] == 0
        batch = [
            types.SimpleNamespace(size=(96 + offset, 64 + offset))
            for offset in (0, variation, 0, -variation)
        ]
        batches.append(batch)
        fills.append(kwargs["fill_mode"])
        return batch

    module.create_image_batch_varshape = create_batch
    module.cvcuda.split_into = lambda *args, **kwargs: calls.append(
        ("split", args, kwargs)
    )
    module.cvcuda.merge_into = lambda *args, **kwargs: calls.append(
        ("merge", args, kwargs)
    )
    state = types.SimpleNamespace(
        get_string=lambda name: {
            "shape": "4x64x96" + (f"@{variation}" if variation else ""),
            "InOutDataType": dtype,
            "layout": layout,
            "inputKind": "VarShape",
        }[name],
        get_int64=lambda name: {"channels": channels}[name],
        get_device=lambda: 0,
        skip=skips.append,
        add_global_memory_reads=reads.append,
        add_global_memory_writes=writes.append,
    )
    run = module.channel_copy(state, split)
    if layout == "NCHW_FAKE":
        assert run is None
        assert skips == [
            "Fake-planar (NCHW_FAKE) channel-copy benchmarks are Tensor-only"
        ]
        assert not batches and not calls and not reads and not writes
        return
    assert not skips and not calls
    assert len(batches) == channels + 1
    assert fills == (
        ["checkerboard"] + [0] * channels
        if split
        else [0] + ["checkerboard"] * channels
    )
    scalar_bytes = module.get_dtype_size(dtype) // module.get_num_channels(dtype)
    expected_bytes = (4 * 64 * 96 + 2 * variation**2) * channels * scalar_bytes
    assert reads == writes == [expected_bytes]
    run(object())
    assert calls == [
        (
            "split" if split else "merge",
            (batches[1:], batches[0]) if split else (batches[0], batches[1:]),
            {"stream": stream},
        )
    ]


def test_cuda_compatible_wide_vector_types():
    generator = (BENCH / "cpp/GenerateBenchConfig.cmake").read_text()
    helpers = (BENCH / "cpp/CppBenchUtils.hpp").read_text()
    assert '"${DTYPE}_16a"' in generator
    for base in ("longlong", "ulonglong", "double"):
        assert f'NVBENCH_DECLARE_TYPE_STRINGS({base}4_16a, "{base}4"' in helpers


@pytest.mark.parametrize(
    "dtype,expected", [("longlong3", "S64"), ("ulonglong4", "U64"), ("double4", "F64")]
)
def test_python_wide_dtype_and_size(monkeypatch, dtype, expected):
    import types

    fake = types.ModuleType("cvcuda")
    fake.Tensor = object
    fake.Type = types.SimpleNamespace(
        **{
            n: n
            for n in (
                "U8",
                "S8",
                "U16",
                "S16",
                "U32",
                "S32",
                "U64",
                "S64",
                "F16",
                "F32",
                "F64",
            )
        }
    )
    monkeypatch.setitem(sys.modules, "cvcuda", fake)
    spec = importlib.util.spec_from_file_location(
        "wide_bench_utils", BENCH / "python/python_bench_utils.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.get_dtype(dtype) == expected
    assert module.get_dtype_size(dtype) == 8 * int(dtype[-1])


def test_python_constant_coverage_border(monkeypatch):
    import types

    fake = types.ModuleType("cvcuda")
    fake.Tensor = object
    fake.Border = types.SimpleNamespace(
        CONSTANT="constant",
        REPLICATE="replicate",
        REFLECT="reflect",
        WRAP="wrap",
        REFLECT101="reflect101",
        CONSTANT_COVERAGE="constant_coverage",
    )
    monkeypatch.setitem(sys.modules, "cvcuda", fake)
    spec = importlib.util.spec_from_file_location(
        "border_bench_utils", BENCH / "python/python_bench_utils.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.get_border_type("CONSTANT_COVERAGE") == "constant_coverage"
