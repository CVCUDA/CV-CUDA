# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build-free regression coverage for filter benchmark setup."""

import importlib.util
import json
import sys
import types
from contextlib import nullcontext
from pathlib import Path

import numpy as np


BENCH_DIR = Path(__file__).resolve().parent.parent


def _load_benchmark(monkeypatch, name, fake_cvcuda, fake_utils, fake_cupy=None):
    monkeypatch.setitem(sys.modules, "cvcuda", fake_cvcuda)
    monkeypatch.setitem(sys.modules, "python_bench_utils", fake_utils)
    if fake_cupy is not None:
        monkeypatch.setitem(sys.modules, "cupy", fake_cupy)

    spec = importlib.util.spec_from_file_location(
        f"bench_{name}_test", BENCH_DIR / "python" / "ops" / f"bench_{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_laplacian_f16_configs_pin_supported_kernel_size():
    config = json.loads(
        (BENCH_DIR / "config" / "operators" / "laplacian.json").read_text()
    )["configs"]

    for key in ("laplacian_half3_advanced", "laplacian_scalar_half_advanced"):
        assert config[key]["int64_axes"]["ksize"] == [1]


def test_medianblur_f16_uses_normalized_half_input(monkeypatch):
    f16 = object()
    arrays = []

    fake_cvcuda = types.ModuleType("cvcuda")
    fake_cvcuda.Type = types.SimpleNamespace(
        U8=object(), U16=object(), F16=f16, F32=object()
    )
    fake_cvcuda.as_tensor = lambda data, _layout: arrays.append(data) or object()

    fake_cupy = types.ModuleType("cupy")
    fake_cupy.uint8 = np.uint8
    fake_cupy.uint16 = np.uint16
    fake_cupy.int32 = np.int32
    fake_cupy.float16 = np.float16
    fake_cupy.float32 = np.float32
    fake_cupy.arange = np.arange
    fake_cupy.broadcast_to = np.broadcast_to
    fake_cupy.zeros = np.zeros
    fake_cupy.cuda = types.SimpleNamespace(Device=lambda _device: nullcontext())

    fake_utils = types.ModuleType("python_bench_utils")
    fake_utils.get_input_kind = lambda value: value
    fake_utils.parse_shape = lambda _shape: (1, 2, 4)
    fake_utils.get_dtype = lambda _dtype: f16
    fake_utils.get_num_channels = lambda _dtype: 3
    fake_utils.get_dtype_size = lambda _dtype: 6
    fake_utils.get_format_from_dtype = lambda *_args, **_kwargs: object()
    fake_utils.create_image_batch_varshape = lambda *_args, **_kwargs: object()
    fake_utils.create_stream_cache = lambda: lambda _launch: None
    fake_utils.run_benchmark = lambda *_args, **_kwargs: None

    benchmark = _load_benchmark(
        monkeypatch, "medianblur", fake_cvcuda, fake_utils, fake_cupy
    )

    class State:
        def get_string(self, name):
            return {
                "shape": "1x2x4",
                "InOutDataType": "half3",
                "kernelSize": "5x5",
                "inputKind": "Tensor",
                "layout": "NHWC",
            }[name]

        def get_device(self):
            return 0

        def add_global_memory_reads(self, _bytes):
            pass

        def add_global_memory_writes(self, _bytes):
            pass

    assert callable(benchmark.medianblur(State()))
    assert arrays[0].dtype == np.float16
    assert arrays[0].max() <= 1


def test_jointbilateralfilter_counts_both_input_images(monkeypatch):
    f16 = object()
    fake_cvcuda = types.ModuleType("cvcuda")
    fake_cvcuda.Type = types.SimpleNamespace(F16=f16)

    fake_utils = types.ModuleType("python_bench_utils")
    fake_utils.get_input_kind = lambda value: value
    fake_utils.parse_shape = lambda _shape: (2, 3, 4)
    fake_utils.get_dtype = lambda _dtype: f16
    fake_utils.get_num_channels = lambda _dtype: 3
    fake_utils.get_dtype_size = lambda _dtype: 6
    fake_utils.get_format_from_dtype = lambda *_args, **_kwargs: object()
    fake_utils.get_border_type = lambda _border: object()
    fake_utils.create_tensor = lambda *_args, **_kwargs: object()
    fake_utils.create_image_batch_varshape = lambda *_args, **_kwargs: object()
    fake_utils.create_stream_cache = lambda: lambda _launch: None
    fake_utils.run_benchmark = lambda *_args, **_kwargs: None

    benchmark = _load_benchmark(
        monkeypatch, "jointbilateralfilter", fake_cvcuda, fake_utils
    )

    class State:
        def __init__(self, layout):
            self.layout = layout
            self.reads = []
            self.writes = []

        def get_string(self, name):
            return {
                "shape": "2x3x4",
                "InOutDataType": "half3",
                "border": "REFLECT",
                "inputKind": "Tensor",
                "layout": self.layout,
            }[name]

        def get_int64(self, _name):
            return -1

        def get_float64(self, _name):
            return 1.2

        def get_device(self):
            return 0

        def add_global_memory_reads(self, bytes_):
            self.reads.append(bytes_)

        def add_global_memory_writes(self, bytes_):
            self.writes.append(bytes_)

    image_bytes = 2 * 3 * 4 * 6
    for layout, read_factor, write_factor in (
        ("NHWC", 2, 1),
        ("NCHW_FAKE", 4, 3),
    ):
        state = State(layout)
        assert callable(benchmark.jointbilateralfilter(state))
        assert state.reads == [read_factor * image_bytes]
        assert state.writes == [write_factor * image_bytes]
