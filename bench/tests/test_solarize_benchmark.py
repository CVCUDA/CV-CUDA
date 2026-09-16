# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build-free regression coverage for the Solarize benchmark."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest


BENCH_DIR = Path(__file__).resolve().parent.parent


def _load_solarize_benchmark(monkeypatch):
    fake_cvcuda = types.ModuleType("cvcuda")
    fake_cvcuda.Type = types.SimpleNamespace(F16="F16", F32="F32", U16="U16")
    thresholds = []
    fake_cvcuda.solarize_into = (
        lambda _dst, _src, threshold, **_kwargs: thresholds.append(threshold)
    )
    monkeypatch.setitem(sys.modules, "cvcuda", fake_cvcuda)

    dtypes = {
        "float16": (fake_cvcuda.Type.F16, 2, 1),
        "float3": (fake_cvcuda.Type.F32, 12, 3),
        "uint16": (fake_cvcuda.Type.U16, 2, 1),
    }
    base_sizes = {
        fake_cvcuda.Type.F16: 2,
        fake_cvcuda.Type.F32: 4,
        fake_cvcuda.Type.U16: 2,
    }

    fake_utils = types.ModuleType("python_bench_utils")
    fake_utils.get_input_kind = lambda _name: "Tensor"
    fake_utils.parse_shape = lambda _shape: (1, 2, 3)
    fake_utils.get_dtype = lambda name: dtypes[name][0]
    fake_utils.get_dtype_size = lambda value: (
        dtypes[value][1] if value in dtypes else base_sizes[value]
    )
    fake_utils.get_num_channels = lambda name: dtypes[name][2]
    fake_utils.get_format_from_dtype = lambda *_args, **_kwargs: object()
    fake_utils.create_tensor = lambda *_args, **_kwargs: object()
    fake_utils.create_image_batch_varshape = lambda *_args, **_kwargs: object()
    fake_utils.create_stream_cache = lambda: lambda _launch: object()
    fake_utils.run_benchmark = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "python_bench_utils", fake_utils)

    spec = importlib.util.spec_from_file_location(
        "bench_solarize_test", BENCH_DIR / "python" / "ops" / "bench_solarize.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, thresholds


@pytest.mark.parametrize(
    "dtype_str, expected_threshold",
    [("float16", 0.5), ("float3", 0.5), ("uint16", 32767.5)],
)
def test_threshold_matches_cpp_midrange(monkeypatch, dtype_str, expected_threshold):
    bench_solarize, thresholds = _load_solarize_benchmark(monkeypatch)

    class State:
        def get_string(self, name):
            return {
                "shape": "1x2x3",
                "InOutDataType": dtype_str,
                "layout": "NHWC",
                "inputKind": "Tensor",
            }[name]

        def get_device(self):
            return 0

        def add_global_memory_reads(self, _bytes):
            pass

        def add_global_memory_writes(self, _bytes):
            pass

    run = bench_solarize.solarize(State())
    run(object())

    assert thresholds == [expected_threshold]
