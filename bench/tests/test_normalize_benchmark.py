# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build-free regression coverage for the Normalize benchmark."""

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest


BENCH_DIR = Path(__file__).resolve().parent.parent


def _load_normalize_benchmark(monkeypatch):
    fake_cvcuda = types.ModuleType("cvcuda")
    fake_cvcuda.NormalizeFlags = types.SimpleNamespace(SCALE_IS_STDDEV=object())
    fake_cvcuda.Type = types.SimpleNamespace(F16=object(), F32=object())
    monkeypatch.setitem(sys.modules, "cvcuda", fake_cvcuda)

    fake_utils = types.ModuleType("python_bench_utils")
    fake_utils.get_input_kind = lambda name: name
    fake_utils.parse_shape = lambda shape: tuple(map(int, shape.split("x")))
    fake_utils.get_dtype = lambda _dtype: fake_cvcuda.Type.F16
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


@pytest.mark.parametrize(
    "layout,storage,reason",
    [
        (
            "NCHW_FAKE",
            "Native",
            "Scalar-parameter normalize benchmark supports only native NHWC and NCHW tensors",
        ),
        (
            "NCHW",
            "NativePitch",
            "NativePitch storage requires native planar Tensor",
        ),
    ],
)
def test_tensor_scalar_unsupported_storage_skips_like_cpp_benchmark(
    monkeypatch, layout, storage, reason
):
    bench_normalize = _load_normalize_benchmark(monkeypatch)

    class State:
        skipped = None

        def get_string(self, name):
            return {
                "shape": "2x3x4",
                "InOutDataType": "U8x3",
                "layout": layout,
                "storage": storage,
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
    assert state.skipped == reason


@pytest.mark.parametrize("input_kind", ["Tensor", "VarShape"])
def test_strided_fixture_uses_real_offsets_pitches_and_matching_values(
    monkeypatch, input_kind
):
    module = _load_normalize_benchmark(monkeypatch)
    fake_cupy = types.ModuleType("cupy")

    class Pointer:
        def __init__(self, array, offset=0):
            self.array, self.offset = array, offset

        def __add__(self, offset):
            return Pointer(self.array, self.offset + offset)

    class Device:
        def __init__(self, _device):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

    def zeros(count, dtype):
        array = np.zeros(count, dtype=dtype)
        return types.SimpleNamespace(data=Pointer(array))

    fake_cupy.zeros = zeros
    fake_cupy.ndarray = lambda shape, dtype, memptr, strides: np.ndarray(
        shape, dtype, buffer=memptr.array, offset=memptr.offset, strides=strides
    )
    fake_cupy.uint8, fake_cupy.float16 = np.uint8, np.float16
    fake_cupy.cuda = types.SimpleNamespace(
        Device=Device,
        get_current_stream=lambda: types.SimpleNamespace(synchronize=lambda: None),
    )
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)
    sys.modules["python_bench_utils"]._checkerboard_fill = lambda array: np.copyto(
        array, (np.indices(array.shape).sum(axis=0) & 1).astype(np.float16)
    )

    class Batch(list):
        def __init__(self, _capacity):
            super().__init__()

        pushback = list.append

    module.cvcuda.as_tensor = lambda array, _layout: array
    module.cvcuda.as_image = lambda planes, _format: tuple(planes)
    module.cvcuda.ImageBatchVarShape = Batch
    src, dst, owners = module._create_strided_planar(
        (2, 3, 4), 3, input_kind, "half3", 0
    )
    for owner, array in zip(owners[::2], owners[1::2]):
        assert array.ctypes.data - owner.data.array.ctypes.data == 2
        assert array.strides == (90, 30, 10, 2)
        assert owner.data.array[:2].tolist() == [0, 0]
    assert np.count_nonzero(owners[3]) == 0
    if input_kind == "Tensor":
        np.testing.assert_array_equal(src, np.indices(src.shape).sum(axis=0) & 1)
        assert dst is owners[3]
    else:
        assert len(src) == len(dst) == 2
        for image in src:
            for plane in image:
                np.testing.assert_array_equal(
                    plane, np.indices(plane.shape).sum(axis=0) & 1
                )
                assert plane.shape == (3, 4, 1)
                assert plane.strides == (10, 2, 2)


@pytest.mark.parametrize(
    "config_key,base_shape,scale_shape,native_allocation",
    [
        (
            "normalize_planar_half3_spatial_advanced",
            (64, 3, 1080, 1),
            (1, 1, 1, 1920),
            False,
        ),
        (
            "normalize_planar_half3_tensor_tail_advanced",
            (64, 3, 1, 1),
            (64, 3, 1, 1),
            True,
        ),
        (
            "normalize_planar_nchw_1080p_half3_advanced",
            (64, 3, 1, 1),
            (64, 3, 1, 1),
            False,
        ),
    ],
)
def test_tensor_fixtures_submit_expected_parameters_and_storage(
    monkeypatch, config_key, base_shape, scale_shape, native_allocation
):
    module = _load_normalize_benchmark(monkeypatch)
    calls = []

    def create_tensor(shape, dtype, device, **kwargs):
        return shape, dtype, device, kwargs

    module.create_tensor = create_tensor
    module.create_stream_cache = lambda: lambda launch: launch
    module.cvcuda.normalize_into = lambda *args, **kwargs: calls.append((args, kwargs))
    config = json.loads(
        (BENCH_DIR / "config" / "operators" / "normalize.json").read_text()
    )["configs"][config_key]
    values = {name: options[0] for name, options in config["string_axes"].items()}
    values["InOutDataType"] = config["dtypes"][0]
    state = types.SimpleNamespace(
        get_string=values.__getitem__,
        get_device=lambda: 0,
        add_global_memory_reads=lambda _bytes: None,
        add_global_memory_writes=lambda _bytes: None,
        skip=lambda reason: pytest.fail(reason),
    )
    run = module.normalize(state)
    run("stream")
    destination, source, base, scale = calls[0][0][:4]
    assert base[0] == base_shape
    assert scale[0] == scale_shape
    assert base[3]["fill_mode"] == scale[3]["fill_mode"] == "lcg"
    samples, height, width = module.parse_shape(values["shape"])
    assert source[0] == destination[0] == (samples, 3, height, width)
    assert source[3].get("native_allocation", False) == native_allocation
    assert destination[3].get("native_allocation", False) == native_allocation
    assert calls[0][1]["stream"] == "stream"
