# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build-free regression coverage for the ResizeCropConvertReformat benchmark."""

import importlib.util
import sys
import types
from pathlib import Path


BENCH_DIR = Path(__file__).resolve().parent.parent


def _load_resizecropconvertreformat_benchmark(monkeypatch):
    fake_cvcuda = types.ModuleType("cvcuda")
    fake_cvcuda.ChannelManip = types.SimpleNamespace(NO_OP=object())
    fake_cvcuda.Type = types.SimpleNamespace(U8=object())
    monkeypatch.setitem(sys.modules, "cvcuda", fake_cvcuda)

    fake_utils = types.ModuleType("python_bench_utils")
    fake_utils.get_dtype = lambda _dtype: fake_cvcuda.Type.U8
    fake_utils.get_input_kind = lambda _name: "Tensor"
    fake_utils.parse_shape = lambda _shape: (2, 8, 10)
    fake_utils.get_num_channels = lambda _dtype: 3
    fake_utils.get_dtype_size = lambda dtype: {"uchar3": 3, "float16": 2}[dtype]
    fake_utils.get_interpolation_type = lambda _name: object()
    fake_utils.get_format_from_dtype = lambda *_args, **_kwargs: object()
    fake_utils.create_tensor = lambda *_args, **_kwargs: object()
    fake_utils.create_image_batch_varshape = lambda *_args, **_kwargs: object()
    fake_utils.create_stream_cache = lambda: lambda _launch: object()
    fake_utils.run_benchmark = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "python_bench_utils", fake_utils)

    spec = importlib.util.spec_from_file_location(
        "bench_resizecropconvertreformat_test",
        BENCH_DIR / "python" / "ops" / "bench_resizecropconvertreformat.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rgb_f16_output_memory_traffic_counts_channels_once(monkeypatch):
    benchmark = _load_resizecropconvertreformat_benchmark(monkeypatch)

    class State:
        reads = []
        writes = []

        def get_string(self, name):
            return {
                "shape": "2x8x10",
                "InOutDataType": "uchar3",
                "inputKind": "Tensor",
                "interpolation": "LINEAR",
                "layout": "NHWC",
                "outDataType": "float16",
            }[name]

        def get_device(self):
            return 0

        def add_global_memory_reads(self, bytes_):
            self.reads.append(bytes_)

        def add_global_memory_writes(self, bytes_):
            self.writes.append(bytes_)

    state = State()
    assert callable(benchmark.resizecropconvertreformat(state))
    assert state.reads == [2 * 8 * 10 * 3]
    assert state.writes == [2 * 3 * 4 * 3 * 2]
