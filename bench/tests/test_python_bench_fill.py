# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for Python benchmark data fills."""

import importlib.util
import sys
import types
from contextlib import nullcontext
from pathlib import Path

import pytest


BENCH_DIR = Path(__file__).resolve().parent.parent


def _load_python_bench_utils(monkeypatch):
    fake_cvcuda = types.ModuleType("cvcuda")
    fake_cvcuda.Tensor = object
    monkeypatch.setitem(sys.modules, "cvcuda", fake_cvcuda)

    fake_cupy = types.ModuleType("cupy")
    fake_cupy.cuda = types.SimpleNamespace(set_allocator=lambda _: None)
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)

    spec = importlib.util.spec_from_file_location(
        "python_bench_utils_fill_test",
        BENCH_DIR / "python" / "python_bench_utils.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_checkerboard_fill_writes_strided_view_without_ravel(monkeypatch):
    bench_utils = _load_python_bench_utils(monkeypatch)
    launches = []

    class StridedArray:
        shape = (3, 11, 17)
        strides = (1056, 96, 4)
        itemsize = 4
        size = 3 * 11 * 17
        dtype = "float32"

        def ravel(self):
            raise AssertionError("pitched image fill must not materialize a dense copy")

    def kernel(grid, block, args):
        launches.append((grid, block, args))

    monkeypatch.setattr(bench_utils, "_checkerboard_kernel_for", lambda _dtype: kernel)

    array = StridedArray()
    bench_utils._checkerboard_fill(array)

    assert len(launches) == 1
    grid, block, args = launches[0]
    assert grid == (3,)
    assert block == (256,)
    assert args[0] is array
    assert tuple(int(value) for value in args[6:10]) == (264, 24, 1, 0)


def test_checkerboard_image_fill_uses_pixel_parity_for_packed_channels(monkeypatch):
    bench_utils = _load_python_bench_utils(monkeypatch)
    fills = []

    class PackedImage:
        shape = (11, 17, 3)

        def __getitem__(self, key):
            return key

    monkeypatch.setattr(bench_utils, "_checkerboard_fill", fills.append)

    bench_utils._checkerboard_fill_image(PackedImage())

    assert fills == [
        (Ellipsis, 0),
        (Ellipsis, 1),
        (Ellipsis, 2),
    ]


def test_create_tensor_preserves_float16_dtype(monkeypatch):
    bench_utils = _load_python_bench_utils(monkeypatch)
    type_names = ("U8", "U16", "U32", "S8", "S16", "S32", "F16", "F32", "F64")
    fake_type = types.SimpleNamespace(**{name: object() for name in type_names})
    bench_utils.cvcuda.Type = fake_type

    fake_cupy = sys.modules["cupy"]
    for name in (
        "uint8",
        "uint16",
        "uint32",
        "int8",
        "int16",
        "int32",
        "float16",
        "float32",
        "float64",
    ):
        setattr(fake_cupy, name, name)
    fake_cupy.cuda.Device = lambda _device: nullcontext()
    allocated = []
    fake_cupy.full = lambda _shape, _value, dtype: allocated.append(dtype) or object()
    bench_utils.cvcuda.as_tensor = lambda array, _layout: array

    bench_utils.create_tensor((1,), fake_type.F16, layout="N", fill_mode=0)

    assert allocated == ["float16"]


@pytest.mark.parametrize(
    ("format_name", "expected_dtype"),
    (("F16", "float16"), ("F64", "float64")),
)
def test_create_varshape_infers_float_dtype(monkeypatch, format_name, expected_dtype):
    bench_utils = _load_python_bench_utils(monkeypatch)
    type_names = ("U8", "U16", "U32", "S8", "S16", "S32", "F16", "F32", "F64")
    fake_type = types.SimpleNamespace(**{name: object() for name in type_names})
    bench_utils.cvcuda.Type = fake_type

    fake_cupy = sys.modules["cupy"]
    for name in (
        "uint8",
        "uint16",
        "uint32",
        "int8",
        "int16",
        "int32",
        "float16",
        "float32",
        "float64",
    ):
        setattr(fake_cupy, name, name)
    fake_cupy.cuda.Device = lambda _device: nullcontext()
    allocated = []
    fake_cupy.empty = lambda _shape, dtype: allocated.append(dtype) or object()

    class FakeView(dict):
        shape = (1, 1)

    fake_cupy.asarray = lambda _buffer: FakeView()
    monkeypatch.setattr(bench_utils, "_lcg_fill", lambda _array, _seed: None)
    bench_utils.cvcuda.Image = lambda _size, _format: types.SimpleNamespace(
        cuda=lambda: object()
    )
    bench_utils.cvcuda.ImageBatchVarShape = lambda _capacity: types.SimpleNamespace(
        pushback=lambda _image: None
    )

    bench_utils.create_image_batch_varshape(
        (1, 1, 1),
        0,
        types.SimpleNamespace(planes=1, name=format_name),
        dtype=None,
        fill_mode="lcg",
    )

    assert allocated == [expected_dtype]


def test_preallocated_varshape_batch_rejects_nonempty_without_mutation(monkeypatch):
    bench_utils = _load_python_bench_utils(monkeypatch)

    class FakeBatch:
        capacity = 2

        def __init__(self):
            self.images = [object()]

        def __len__(self):
            return len(self.images)

        def pushback(self, image):
            if len(self.images) == self.capacity:
                raise RuntimeError("capacity exceeded")
            self.images.append(image)

    type_names = ("U8", "U16", "U32", "S8", "S16", "S32", "F16", "F32", "F64")
    fake_type = types.SimpleNamespace(**{name: object() for name in type_names})
    bench_utils.cvcuda.Type = fake_type
    bench_utils.cvcuda.Image = lambda _size, _format: types.SimpleNamespace(
        cuda=lambda: object()
    )
    fake_cupy = sys.modules["cupy"]
    for name in (
        "uint8",
        "uint16",
        "uint32",
        "int8",
        "int16",
        "int32",
        "float16",
        "float32",
        "float64",
    ):
        setattr(fake_cupy, name, name)
    fake_cupy.cuda.Device = lambda _device: nullcontext()
    fake_cupy.asarray = lambda _buffer: types.SimpleNamespace(fill=lambda _value: None)

    batch = FakeBatch()
    with pytest.raises(ValueError, match="must be empty"):
        bench_utils.create_image_batch_varshape(
            (2, 1, 1),
            0,
            types.SimpleNamespace(planes=1),
            dtype=fake_type.U8,
            fill_mode=0,
            batch=batch,
        )

    assert len(batch) == 1
