# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build-free smoke coverage for the HQResize VLM benchmark row."""

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest


BENCH_DIR = Path(__file__).resolve().parent.parent


def test_vlm_center_crop_uses_duration_calibrated_shared_workload():
    document = json.loads(
        (BENCH_DIR / "config" / "operators" / "hqresize.json").read_text()
    )
    workloads = [
        entry
        for entry in document["configs"].values()
        if "VLM_CENTER_CROP_224" in entry.get("string_axes", {}).get("resizeType", [])
    ]

    assert len(workloads) == 1
    workload = workloads[0]
    assert workload["string_axes"]["shape"] == ["64x1080x1920"]
    assert workload["string_axes"]["inputKind"] == ["VarShape"]
    assert workload["string_axes"]["layout"] == ["NHWC"]
    assert workload["string_axes"]["interpolation"] == ["CUBIC"]
    assert workload["dtypes"] == ["uint8"]
    assert workload["int64_axes"] == {"antialias": [1], "numChannels": [3]}


@pytest.fixture(scope="module")
def benchmark():
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(sys, "path", sys.path[:])
        monkeypatch.syspath_prepend(str(BENCH_DIR / "python"))
        fake_cvcuda = types.ModuleType("cvcuda")
        fake_cvcuda.Tensor = object
        fake_cvcuda.Interp = types.SimpleNamespace(CUBIC="CUBIC", LINEAR="LINEAR")
        fake_cupy = types.ModuleType("cupy")
        fake_cupy.cuda = types.SimpleNamespace(set_allocator=lambda _allocator: None)
        monkeypatch.setitem(sys.modules, "cvcuda", fake_cvcuda)
        monkeypatch.setitem(sys.modules, "cupy", fake_cupy)

        utils_spec = importlib.util.spec_from_file_location(
            "python_bench_utils", BENCH_DIR / "python" / "python_bench_utils.py"
        )
        utils = importlib.util.module_from_spec(utils_spec)
        utils_spec.loader.exec_module(utils)
        monkeypatch.setitem(sys.modules, "python_bench_utils", utils)

        bench_spec = importlib.util.spec_from_file_location(
            "bench_hqresize_test", BENCH_DIR / "python" / "ops" / "bench_hqresize.py"
        )
        module = importlib.util.module_from_spec(bench_spec)
        bench_spec.loader.exec_module(module)
        yield module


class FakeState:
    default_values = {
        "shape": "64x1080x1920",
        "InOutDataType": "uint8",
        "resizeType": "VLM_CENTER_CROP_224",
        "interpolation": "CUBIC",
        "inputKind": "VarShape",
        "layout": "NHWC",
        "antialias": 1,
        "numChannels": 3,
    }

    def __init__(self, **overrides):
        self.values = {**self.default_values, **overrides}
        self.skipped = None
        self.reads = []
        self.writes = []

    def get_string(self, name):
        return self.values[name]

    def get_int64(self, name):
        return self.values[name]

    def get_device(self):
        return 0

    def skip(self, reason):
        self.skipped = reason

    def add_global_memory_reads(self, value):
        self.reads.append(value)

    def add_global_memory_writes(self, value):
        self.writes.append(value)


def test_vlm_center_crop_roi_is_directly_testable(benchmark):
    assert benchmark._vlm_center_crop_roi(1080, 1920, 224, 224) == (
        0.0,
        419.698486328125,
        1080.0,
        1500.301513671875,
    )


@pytest.mark.parametrize(
    "resize_type,dst_h,dst_w,read_w,expected_roi",
    [
        (
            "VLM_CENTER_CROP_224",
            224,
            224,
            1082,
            (0.0, 419.698486328125, 1080.0, 1500.301513671875),
        ),
        (
            "VLM_CENTER_CROP_448",
            448,
            448,
            1082,
            (0.0, 419.698486328125, 1080.0, 1500.301513671875),
        ),
        (
            "VLM_CENTER_CROP_224x352",
            224,
            352,
            1700,
            (0.0, 110.95477294921875, 1080.0, 1809.045166015625),
        ),
        (
            "VLM_CENTER_CROP_224x353",
            224,
            353,
            1704,
            (0.0, 106.13065338134766, 1080.0, 1809.045166015625),
        ),
    ],
)
@pytest.mark.parametrize("batch_size", [16, 32, 64])
def test_vlm_row_forwards_center_crop_roi_and_counts_element_size(
    benchmark, monkeypatch, resize_type, dst_h, dst_w, read_w, expected_roi, batch_size
):
    allocations = []
    launches = []
    monkeypatch.setattr(benchmark, "get_dtype", lambda _dtype: "U8")
    monkeypatch.setattr(benchmark, "get_dtype_size", lambda _dtype: 2)
    monkeypatch.setattr(benchmark, "get_interpolation_type", lambda value: value)
    monkeypatch.setattr(
        benchmark, "get_format_from_dtype", lambda *_args, **_kwargs: "RGB8"
    )
    monkeypatch.setattr(
        benchmark, "create_stream_cache", lambda: lambda launch: f"stream-{launch}"
    )

    def allocate(*args, **kwargs):
        batch = object()
        allocations.append((args, kwargs, batch))
        return batch

    monkeypatch.setattr(benchmark, "create_image_batch_varshape", allocate)
    monkeypatch.setattr(
        benchmark.cvcuda,
        "hq_resize_into",
        lambda *args, **kwargs: launches.append((args, kwargs)),
        raising=False,
    )

    state = FakeState(resizeType=resize_type, shape=f"{batch_size}x1080x1920")
    run = benchmark.hqresize(state)
    assert callable(run)
    assert state.skipped is None
    assert state.reads == [batch_size * 1080 * read_w * 3 * 2]
    assert state.writes == [batch_size * dst_h * dst_w * 3 * 2]
    assert [args[0] for args, _, _ in allocations] == [
        (batch_size, 1080, 1920, 3),
        (batch_size, dst_h, dst_w, 3),
    ]

    run("launch")
    assert launches[0][0] == (allocations[1][2], allocations[0][2])
    assert launches[0][1]["stream"] == "stream-launch"
    assert len(launches) == 1
    assert len(launches[0][1]["roi"]) == batch_size
    assert launches[0][1]["roi"][0] == expected_roi


@pytest.mark.parametrize(
    "overrides,expected_reason",
    [
        ({"inputKind": "Tensor"}, "inputKind=VarShape"),
        ({"layout": "NCHW"}, "layout=NHWC"),
        ({"shape": "64x1920x1080"}, "landscape input"),
        ({"shape": "64x0x1920"}, "positive input height"),
        ({"InOutDataType": "float32"}, "InOutDataType=uint8"),
        ({"numChannels": 1}, "numChannels=3"),
        ({"interpolation": "LINEAR"}, "interpolation=CUBIC"),
        ({"resizeType": "VLM_CENTER_CROP_"}, "positive HxW or square"),
        ({"resizeType": "VLM_CENTER_CROP_0"}, "positive HxW or square"),
        (
            {"resizeType": "VLM_CENTER_CROP_224x352x1"},
            "positive HxW or square",
        ),
        (
            {"resizeType": "VLM_CENTER_CROP_+224"},
            "positive HxW or square",
        ),
        (
            {"resizeType": "VLM_CENTER_CROP_ 224"},
            "positive HxW or square",
        ),
        (
            {"resizeType": "VLM_CENTER_CROP_2147483647"},
            "intermediate dimensions exceed the supported range",
        ),
        (
            {"resizeType": "VLM_CENTER_CROP_2000"},
            "Antialias is no-op for expanding",
        ),
    ],
)
def test_vlm_row_rejects_unsupported_contract(
    benchmark, monkeypatch, overrides, expected_reason
):
    monkeypatch.setattr(benchmark, "get_dtype", lambda _dtype: "U8")
    monkeypatch.setattr(benchmark, "get_dtype_size", lambda _dtype: 1)
    monkeypatch.setattr(benchmark, "get_interpolation_type", lambda value: value)
    monkeypatch.setattr(
        benchmark,
        "create_image_batch_varshape",
        lambda *_args, **_kwargs: pytest.fail("rejected row allocated an image batch"),
    )

    state = FakeState(**overrides)

    assert benchmark.hqresize(state) is None
    assert expected_reason in state.skipped
    assert state.reads == []
    assert state.writes == []
