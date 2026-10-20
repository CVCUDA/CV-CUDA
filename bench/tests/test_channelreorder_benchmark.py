# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build-free regression coverage for the ChannelReorder benchmark."""

import importlib.util
import json
import sys
import types
from contextlib import nullcontext
from pathlib import Path

import pytest
from _internal.baselines import expected_case_keys_for_entry

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
    fake_utils._checkerboard_fill_image = unexpected_benchmark_setup
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


@pytest.mark.parametrize(
    "input_kind,layout,dtype",
    [
        ("Tensor", "NHWC", "uchar3"),
        ("VarShape", "NHWC", "float3"),
        ("VarShape", "NHWC", "uchar"),
        ("VarShape", "NCHW", "uchar3"),
    ],
)
def test_misaligned_fixture_rejects_unsupported_axes_before_setup(
    monkeypatch, input_kind, layout, dtype
):
    module = _load_channelreorder_benchmark(monkeypatch)
    module.get_input_kind = lambda _name: input_kind

    class State:
        skipped = None

        def get_string(self, name):
            return {
                "shape": "2x3x4",
                "InOutDataType": dtype,
                "layout": layout,
                "inputKind": input_kind,
                "orderPattern": "rotate_misaligned",
            }[name]

        def get_device(self):
            raise AssertionError("Unsupported fixture must skip before device setup")

        def skip(self, reason):
            self.skipped = reason

    state = State()
    assert module.channelreorder(state) is None
    assert state.skipped == "rotate_misaligned requires NHWC VarShape RGB8 or RGBA8"


@pytest.mark.parametrize("channels", [3, 4])
@pytest.mark.parametrize("fill_input", [False, True])
def test_misaligned_fixture_owns_exact_offset_and_pitch(
    monkeypatch, channels, fill_input
):
    module = _load_channelreorder_benchmark(monkeypatch)
    allocations, fills, synchronizations = [], [], []

    class Pointer:
        def __init__(self, ptr):
            self.ptr = ptr

        def __add__(self, offset):
            return Pointer(self.ptr + offset)

    def allocate(size, dtype):
        allocation = types.SimpleNamespace(
            data=Pointer(256 * (1 + len(allocations))), size=size
        )
        allocations.append(allocation)
        return allocation

    def array(shape, dtype, memptr, strides):
        return types.SimpleNamespace(shape=shape, data=memptr, strides=strides)

    def as_image(pixels, _format):
        # Imported DLPack-only buffers need not expose CUDA Array Interface.
        exported = types.SimpleNamespace(pixels=pixels)
        return types.SimpleNamespace(cuda=lambda: exported)

    class Batch(list):
        def __init__(self, capacity):
            self.capacity = capacity

        def pushback(self, image):
            self.append(image)

    module.cp.uint8 = "uint8"
    module.cp.zeros = allocate
    module.cp.ndarray = array
    module.cp.from_dlpack = lambda exported: exported.pixels
    module.cp.cuda = types.SimpleNamespace(
        Device=lambda _device: nullcontext(),
        get_current_stream=lambda: types.SimpleNamespace(
            synchronize=lambda: synchronizations.append(True)
        ),
    )
    module.cvcuda.ImageBatchVarShape = Batch
    module.cvcuda.as_image = as_image
    module._checkerboard_fill_image = fills.append

    batch, owners = module._create_misaligned_batch(
        (2, 5, 33, channels), "format", 0, fill_input
    )
    assert len(batch) == len(owners) == 2
    assert len(fills) == (2 if fill_input else 0)
    assert synchronizations == [True]
    for allocation, pixels, image in owners:
        assert allocation.size == 1 + (33 * channels + 1) * 5
        assert pixels.data.ptr == allocation.data.ptr + 1
        assert pixels.strides == (33 * channels + 1, channels, 1)
        assert image in batch


def test_new_fallback_cases_are_six_explicit_nhwc_varshape_rows():
    configs = json.loads(
        (BENCH_DIR / "config/operators/channelreorder.json").read_text()
    )["configs"]
    names = (
        "channelreorder_aligned_tail127_advanced",
        "channelreorder_aligned_tail129_advanced",
        "channelreorder_misaligned_rows_advanced",
    )
    assert (
        sum(len(expected_case_keys_for_entry(name, configs[name])) for name in names)
        == 6
    )
    for name in names:
        entry = configs[name]
        assert entry["tier"] == "advanced"
        assert entry["dtypes"] == ["uchar3", "uchar4"]
        assert set(entry["string_axes"]) == {
            "shape",
            "layout",
            "orderPattern",
            "inputKind",
        }
        assert entry["string_axes"]["layout"] == ["NHWC"]
        assert entry["string_axes"]["inputKind"] == ["VarShape"]
        assert len(entry["string_axes"]["shape"]) == 1
        assert entry["string_axes"]["orderPattern"] == [
            "rotate_misaligned" if "misaligned" in name else "rotate"
        ]
    for width in (127, 129):
        entry = configs[f"channelreorder_aligned_tail{width}_advanced"]
        assert entry["string_axes"]["shape"][0].split("x")[-1] == str(width)
