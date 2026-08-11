# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build-free regression coverage for the Normalize benchmark."""

import importlib.util
import sys
import types
from pathlib import Path


BENCH_DIR = Path(__file__).resolve().parent.parent


def _load_normalize_benchmark(monkeypatch):
    fake_cvcuda = types.ModuleType("cvcuda")
    fake_cvcuda.NormalizeFlags = types.SimpleNamespace(SCALE_IS_STDDEV=object())
    monkeypatch.setitem(sys.modules, "cvcuda", fake_cvcuda)

    fake_utils = types.ModuleType("python_bench_utils")
    fake_utils.get_input_kind = lambda _name: "Tensor"
    fake_utils.parse_shape = lambda _shape: (2, 3, 4)
    fake_utils.get_dtype = lambda _dtype: object()
    fake_utils.get_dtype_size = lambda _dtype: 1
    fake_utils.get_num_channels = lambda _dtype: 3
    fake_utils.get_format_from_dtype = lambda *_args, **_kwargs: object()

    def unexpected_allocation(*_args, **_kwargs):
        raise AssertionError("NCHW_FAKE TensorScalar must skip before allocation")

    fake_utils.create_tensor = unexpected_allocation
    fake_utils.create_image_batch_varshape = unexpected_allocation
    fake_utils.create_stream_cache = lambda: object()
    fake_utils.run_benchmark = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "python_bench_utils", fake_utils)

    spec = importlib.util.spec_from_file_location(
        "bench_normalize_test",
        BENCH_DIR / "python" / "ops" / "bench_normalize.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tensor_scalar_fake_planar_skips_like_cpp_benchmark(monkeypatch):
    bench_normalize = _load_normalize_benchmark(monkeypatch)

    class State:
        skipped = None

        def get_string(self, name):
            return {
                "shape": "2x3x4",
                "InOutDataType": "U8x3",
                "layout": "NCHW_FAKE",
                "inputKind": "TensorScalar",
            }[name]

        def get_device(self):
            return 0

        def skip(self, reason):
            self.skipped = reason

        def add_global_memory_reads(self, _bytes):
            pass

        def add_global_memory_writes(self, _bytes):
            pass

    state = State()

    assert bench_normalize.normalize(state) is None
    assert state.skipped == (
        "Scalar-parameter normalize benchmark supports only native NHWC and NCHW tensors"
    )
