# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build-free regression coverage for the ChannelReorder benchmark."""

import importlib.util
import sys
import types
from pathlib import Path


BENCH_DIR = Path(__file__).resolve().parent.parent


def _load_channelreorder_benchmark(monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", types.ModuleType("cupy"))
    monkeypatch.setitem(sys.modules, "cvcuda", types.ModuleType("cvcuda"))

    fake_utils = types.ModuleType("python_bench_utils")
    fake_utils.get_input_kind = lambda _name: "VarShape"
    fake_utils.parse_shape = lambda _shape: (2, 3, 4)
    fake_utils.get_dtype = lambda _dtype: object()

    def unexpected_benchmark_setup(*_args, **_kwargs):
        raise AssertionError("NCHW_FAKE VarShape must skip before benchmark setup")

    fake_utils.get_num_channels = unexpected_benchmark_setup
    fake_utils.get_dtype_size = unexpected_benchmark_setup
    fake_utils.get_format_from_dtype = unexpected_benchmark_setup
    fake_utils.create_tensor = unexpected_benchmark_setup
    fake_utils.create_image_batch_varshape = unexpected_benchmark_setup
    fake_utils.create_stream_cache = unexpected_benchmark_setup
    fake_utils.run_benchmark = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "python_bench_utils", fake_utils)

    spec = importlib.util.spec_from_file_location(
        "bench_channelreorder_test",
        BENCH_DIR / "python" / "ops" / "bench_channelreorder.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fake_planar_varshape_skips_before_benchmark_setup(monkeypatch):
    bench_channelreorder = _load_channelreorder_benchmark(monkeypatch)

    class State:
        skipped = None

        def get_string(self, name):
            return {
                "shape": "2x3x4",
                "InOutDataType": "U8x3",
                "layout": "NCHW_FAKE",
                "inputKind": "VarShape",
                "orderPattern": "rotate",
            }[name]

        def get_device(self):
            raise AssertionError("NCHW_FAKE VarShape must skip before device setup")

        def skip(self, reason):
            self.skipped = reason

    state = State()

    assert bench_channelreorder.channelreorder(state) is None
    assert state.skipped == (
        "Fake-planar (NCHW_FAKE) ChannelReorder benchmark is tensor-only"
    )
