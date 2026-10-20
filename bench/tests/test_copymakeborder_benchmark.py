# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build-free regression coverage for the CopyMakeBorder benchmark."""

import importlib.util
import re
import sys
import types
from pathlib import Path
from unittest.mock import Mock

import pytest


BENCH_DIR = Path(__file__).resolve().parent.parent


def _load_python_benchmark(monkeypatch):
    fake_cvcuda = types.ModuleType("cvcuda")
    fake_cvcuda.Type = types.SimpleNamespace(S32="S32")
    fake_cvcuda.copymakeborder_into = Mock()
    fake_cvcuda.copymakeborderstack_into = Mock()
    monkeypatch.setitem(sys.modules, "cvcuda", fake_cvcuda)

    tensor_calls = []
    fake_utils = types.ModuleType("python_bench_utils")
    fake_utils.get_input_kind = lambda value: value
    fake_utils.parse_shape = lambda _shape: (9, 6, 10)
    fake_utils.get_dtype = lambda _dtype: "U8"
    fake_utils.get_dtype_size = lambda _dtype: 1
    fake_utils.get_num_channels = lambda _dtype: 3
    fake_utils.get_format_from_dtype = lambda *_args, **_kwargs: "RGB8"
    fake_utils.get_border_type = lambda value: value

    def create_tensor(*args, **kwargs):
        tensor_calls.append((args, kwargs))
        return object()

    fake_utils.create_tensor = create_tensor
    fake_utils.create_image_batch_varshape = lambda *_args, **_kwargs: object()
    fake_utils.create_stream_cache = lambda: lambda _launch: None
    fake_utils.run_benchmark = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "python_bench_utils", fake_utils)

    spec = importlib.util.spec_from_file_location(
        "bench_copymakeborder_test",
        BENCH_DIR / "python" / "ops" / "bench_copymakeborder.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, tensor_calls, fake_cvcuda


class _State:
    def __init__(self, input_kind="VarShape"):
        self.input_kind = input_kind

    def get_string(self, name):
        return {
            "shape": "9x6x10",
            "InOutDataType": "uchar3",
            "border": "CONSTANT",
            "inputKind": self.input_kind,
            "layout": "NHWC",
        }[name]

    def get_device(self):
        return 0

    def add_global_memory_reads(self, _value):
        pass

    def add_global_memory_writes(self, _value):
        pass


@pytest.mark.parametrize("input_kind", ["VarShape", "VarShapeToTensor"])
def test_python_varshape_parameter_tensors_follow_nhwc_contract(
    monkeypatch, input_kind
):
    benchmark, tensor_calls, cvcuda = _load_python_benchmark(monkeypatch)

    run = benchmark.copymakeborder(_State(input_kind))
    run(None)
    expected = [
        ((1, 1, 9, 1), 3),
        ((1, 1, 9, 1), 5),
    ]
    if input_kind == "VarShapeToTensor":
        expected.append(((9, 9, 15, 3), 0))
    assert cvcuda.copymakeborderstack_into.call_count == (
        input_kind == "VarShapeToTensor"
    )
    assert cvcuda.copymakeborder_into.call_count == (input_kind == "VarShape")
    assert [(args[0], kwargs["fill_mode"]) for args, kwargs in tensor_calls] == expected


def test_cpp_varshape_parameter_tensors_follow_nhwc_contract():
    source = (BENCH_DIR / "cpp" / "ops" / "BenchCopyMakeBorder.cpp").read_text()

    for name in ("topTensor", "leftTensor"):
        assert re.search(
            rf'nvcv::Tensor\s+{name}\s*\(\s*\{{\{{\s*1,\s*1,\s*srcShape\.x,\s*1\s*\}},\s*"NHWC"',
            source,
        )
