# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the host-side Python binding benchmark."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock, mock_open

import pytest

BENCHMARK = Path(__file__).resolve().parent.parent / "binding_bench.py"


def _load_benchmark():
    spec = importlib.util.spec_from_file_location("binding_bench", BENCHMARK)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def test_cpu_model_prefers_proc_cpuinfo(monkeypatch):
    benchmark = _load_benchmark()
    monkeypatch.setattr(
        benchmark,
        "open",
        mock_open(read_data="processor : 0\nmodel name : Test CPU\n"),
        raising=False,
    )
    run = Mock(side_effect=AssertionError("lscpu must not run"))
    monkeypatch.setattr(benchmark.subprocess, "run", run)

    assert benchmark._cpu_model() == "Test CPU"
    run.assert_not_called()


def test_cpu_model_identifies_arm_without_proc_model_name(monkeypatch):
    benchmark = _load_benchmark()
    monkeypatch.setattr(
        benchmark,
        "open",
        mock_open(
            read_data="CPU implementer : 0x41\nCPU architecture : 8\nCPU part : 0xd4f\n"
        ),
        raising=False,
    )
    monkeypatch.setattr(benchmark.platform, "processor", lambda: "")
    run = Mock(
        return_value=SimpleNamespace(
            stdout=json.dumps(
                {"lscpu": [{"field": "Model name:", "data": "Neoverse-V2"}]}
            )
        )
    )
    monkeypatch.setattr(benchmark.subprocess, "run", run)

    assert benchmark._cpu_model() == "Neoverse-V2"
    assert run.call_args.args == (["lscpu", "-J"],)
    assert run.call_args.kwargs["env"]["LC_ALL"] == "C"
    assert run.call_args.kwargs["check"] is True
    assert run.call_args.kwargs["capture_output"] is True
    assert run.call_args.kwargs["text"] is True
    assert 0 < run.call_args.kwargs["timeout"] <= 5


@pytest.mark.parametrize("processor", ["Fallback CPU", ""])
@pytest.mark.parametrize(
    "output",
    [
        FileNotFoundError(),
        subprocess.CalledProcessError(1, ["lscpu", "-J"]),
        subprocess.TimeoutExpired(["lscpu", "-J"], 5),
        "not JSON",
        "null",
        "{}",
        '{"lscpu": null}',
        '{"lscpu": [null, {"field": "Model name:", "data": null}]}',
        '{"lscpu": [{"field": "Model name:", "data": "  "}]}',
    ],
)
def test_cpu_model_preserves_fallback_when_lscpu_fails(monkeypatch, processor, output):
    benchmark = _load_benchmark()
    monkeypatch.setattr(
        benchmark, "open", Mock(side_effect=FileNotFoundError()), raising=False
    )
    monkeypatch.setattr(benchmark.platform, "processor", lambda: processor)
    run = (
        Mock(side_effect=output)
        if isinstance(output, Exception)
        else Mock(return_value=SimpleNamespace(stdout=output))
    )
    monkeypatch.setattr(benchmark.subprocess, "run", run)

    assert benchmark._cpu_model() == (processor or "unknown")


def test_summarize_retains_raw_windows_and_robust_statistics():
    benchmark = _load_benchmark()

    result = benchmark.summarize([100, 200, 300], 10)

    assert result["samples"] == 3
    assert result["iterations_per_window"] == 10
    assert result["window_ns"] == [100, 200, 300]
    assert result["ns_per_call"] == [10.0, 20.0, 30.0]
    assert result["p50_ns"] == 20.0
    assert result["mean_ns"] == 20.0
    assert result["stdev_ns"] == 10.0
    assert result["p90_ns"] == 28.0
    assert result["mad_ns"] == 10.0
    assert result["robust_noise_pct"] == pytest.approx(74.13)


def test_build_report_has_stable_schema_and_case_order():
    benchmark = _load_benchmark()
    cases = [
        {"name": "first", "group": "control", "iterations": 2, "window_ns": [4]},
        {"name": "second", "group": "interop", "iterations": 4, "window_ns": [12]},
    ]

    report = benchmark.build_report(cases, {"samples": 1}, {"module": "candidate"})

    assert report["schema_version"] == 1
    assert report["metadata"] == {"module": "candidate"}
    assert report["parameters"] == {"samples": 1}
    assert [result["name"] for result in report["results"]] == ["first", "second"]
    assert [result["p50_ns"] for result in report["results"]] == [2.0, 3.0]


def test_construction_and_wrapping_cases_are_registered():
    benchmark = _load_benchmark()

    assert {
        "construction.tensor.layout_string",
        "construction.tensor.layout_object",
        "construction.reshape.method.layout_string",
        "construction.reshape.method.layout_object",
        "construction.reshape.module.layout_string",
        "construction.reshape.module.layout_object",
        "interop.as_tensor.cai.layout_string",
        "interop.as_tensor.cai.layout_object",
        "interop.as_tensor.dlpack.layout_string",
        "interop.as_tensor.dlpack.layout_object",
        "workflow.flip.as_tensor.cai.explicit.serial",
        "workflow.flip.as_tensor.cai.explicit.queued",
    } <= set(benchmark.CASES)


def test_resource_and_batch_submission_cases_are_registered_and_routed(monkeypatch):
    benchmark = _load_benchmark()
    selected = (
        "submission.threshold.tensor.explicit",
        "submission.cvtcolor.varshape.batch1.explicit",
        "submission.threshold.varshape.explicit",
    )
    assert set(selected) <= set(benchmark.CASES)

    calls = []

    def record(kind):
        def make_case(*args):
            calls.append((kind, args[1:]))
            return kind

        return make_case

    monkeypatch.setattr(
        benchmark, "_make_threshold_tensor_case", record("threshold_tensor")
    )
    monkeypatch.setattr(benchmark, "_make_varshape_case", record("cvtcolor"))
    monkeypatch.setattr(
        benchmark, "_make_threshold_varshape_case", record("threshold_varshape")
    )
    args = SimpleNamespace(
        submission_iterations=11,
        varshape_iterations=7,
        submission_chunk_size=4,
        batch_size=37,
    )

    cases = benchmark._build_cases(object(), selected, args)

    assert cases == ["threshold_tensor", "cvtcolor", "threshold_varshape"]
    assert calls == [
        (
            "threshold_tensor",
            ("submission.threshold.tensor.explicit", 11, 4),
        ),
        (
            "cvtcolor",
            ("submission.cvtcolor.varshape.batch1.explicit", True, 7, 4, 1),
        ),
        (
            "threshold_varshape",
            ("submission.threshold.varshape.explicit", 7, 4, 37),
        ),
    ]


def test_protocol_proxies_select_exactly_one_buffer_interface():
    benchmark = _load_benchmark()
    calls = []

    class Owner:
        __cuda_array_interface__ = {"version": 3}

        def __dlpack__(self, *args, **kwargs):
            calls.append((args, kwargs))
            return "capsule"

        def __dlpack_device__(self):
            return (2, 0)

    owner = Owner()
    cai = benchmark._CudaArrayInterfaceOnly(owner)
    dlpack = benchmark._DLPackOnly(owner)

    assert cai.__cuda_array_interface__ == owner.__cuda_array_interface__
    assert cai.__cuda_array_interface__ is not owner.__cuda_array_interface__
    assert not hasattr(cai, "__dlpack__")
    assert not hasattr(dlpack, "__cuda_array_interface__")
    assert dlpack.__dlpack__(stream=1) == "capsule"
    assert dlpack.__dlpack_device__() == (2, 0)
    assert calls == [((), {"stream": 1})]


@pytest.mark.parametrize("container,dtype", [("tensor", "f4"), ("image", "NV12")])
def test_fresh_export_measures_new_metadata_on_every_call(container, dtype):
    benchmark = _load_benchmark()
    exports, metadata_reads = [], []

    class Buffer:
        @property
        def __cuda_array_interface__(self):
            metadata_reads.append(self)
            return {"typestr": "<f4"}

    def export():
        buffers = [Buffer(), Buffer()] if container == "image" else Buffer()
        exports.append(buffers)
        return buffers

    owner = SimpleNamespace(cuda=export)
    cvcuda = SimpleNamespace(
        Tensor=lambda *args, **kwargs: owner,
        Image=lambda *args, **kwargs: owner,
        Format=SimpleNamespace(NV12="NV12"),
    )
    case = benchmark._make_fresh_export_case(cvcuda, "fresh", container, dtype, 64, 2)
    exports.clear()
    metadata_reads.clear()
    case.call()
    case.call()
    assert len(exports) == 2
    assert exports[0] is not exports[1]
    assert len(metadata_reads) == (4 if container == "image" else 2)
    assert len({id(buffer) for buffer in metadata_reads}) == len(metadata_reads)


def test_fresh_export_rejects_reused_wrapped_buffer():
    benchmark = _load_benchmark()
    buffer = object()
    cvcuda = SimpleNamespace(
        Tensor=lambda *args, **kwargs: SimpleNamespace(cuda=lambda: buffer)
    )
    with pytest.raises(RuntimeError, match="reused an exported buffer"):
        benchmark._make_fresh_export_case(cvcuda, "fresh", "tensor", "u1", 1, 2)


def test_submission_measurement_synchronizes_before_queue_can_fill():
    benchmark = _load_benchmark()
    calls_since_sync = 0
    total_calls = 0
    largest_chunk = 0

    def call():
        nonlocal calls_since_sync, total_calls
        calls_since_sync += 1
        total_calls += 1

    def synchronize():
        nonlocal calls_since_sync, largest_chunk
        largest_chunk = max(largest_chunk, calls_since_sync)
        calls_since_sync = 0

    case = benchmark._BenchmarkCase("chunked", "submission", 10, 4, call, synchronize)
    benchmark._measure_case(case, samples=2, warmups=1)

    # Three chunks per window, each with one untimed steady-state call.
    assert total_calls == 39
    assert largest_chunk == 4


@pytest.mark.parametrize("modern_drain", [False, True])
@pytest.mark.parametrize("chunk_size", [1, 4])
def test_wrap_workflow_primes_only_when_release_drain_is_unavailable(
    monkeypatch, modern_drain, chunk_size
):
    benchmark = _load_benchmark()
    events = []
    result = SimpleNamespace(shape=(1, 32, 32, 4), layout="NHWC")
    owner = SimpleNamespace(cuda=lambda: object())
    stream = SimpleNamespace(sync=lambda: events.append("sync"))
    internal = SimpleNamespace(syncAuxStream=lambda: events.append("legacy-drain"))
    if modern_drain:
        internal.drain_pending_releases = lambda: events.append("drain")

    def flip(*args, **kwargs):
        events.append("call")
        return result

    cvcuda = SimpleNamespace(
        Tensor=lambda *args: owner,
        Type=SimpleNamespace(U8=object()),
        TensorLayout=lambda layout: layout,
        Stream=lambda: stream,
        as_tensor=lambda *args: object(),
        flip=flip,
        internal=internal,
    )
    monkeypatch.setattr(benchmark, "_CudaArrayInterfaceOnly", lambda external: external)
    case = benchmark._make_wrap_and_submit_case(
        cvcuda, "workflow", iterations=4, chunk_size=chunk_size
    )
    events.clear()

    def time_chunk(call, iterations):
        events.append("timed-start")
        for _ in range(iterations):
            call()
        events.append("timed-end")
        return iterations * 3, iterations * 2

    monkeypatch.setattr(benchmark, "_time_chunk", time_chunk)
    measured = benchmark._measure_case(case, samples=2, warmups=1)
    chunks = 4 // chunk_size
    expected_calls = 12 if modern_drain else 12 + 3 * chunks
    assert events.count("call") == expected_calls
    assert measured["window_ns"] == [12, 12]
    assert measured["thread_cpu_window_ns"] == [8, 8]
    if modern_drain:
        for index, event in enumerate(events):
            if event == "timed-start":
                assert events[index - 1] == "drain"
        assert "legacy-drain" not in events


def test_cpu_and_wall_windows_exclude_boundaries_and_warmups(monkeypatch):
    benchmark = _load_benchmark()
    clocks = {"wall": 0, "cpu": 0}

    def call():
        clocks["wall"] += 3
        clocks["cpu"] += 2

    def synchronize():
        clocks["wall"] += 100
        clocks["cpu"] += 50

    def read_cpu_clock():
        clocks["wall"] += 7
        clocks["cpu"] += 5
        return clocks["cpu"]

    monkeypatch.setattr(benchmark.time, "perf_counter_ns", lambda: clocks["wall"])
    monkeypatch.setattr(benchmark.time, "thread_time_ns", read_cpu_clock)
    case = benchmark._BenchmarkCase("chunked", "submission", 10, 4, call, synchronize)
    measured = benchmark._measure_case(case, samples=2, warmups=1)
    assert measured["window_ns"] == [30, 30]
    # CPU-clock overhead is excluded from wall time; CPU time includes one read
    # per chunk, including the final partial chunk.
    assert measured["thread_cpu_window_ns"] == [35, 35]
    report = benchmark.build_report([measured], {}, {})
    assert report["results"][0]["thread_cpu_window_ns"] == [35, 35]


def test_drain_case_uses_legacy_aux_stream_fallback():
    benchmark = _load_benchmark()
    calls = 0

    def sync_aux_stream():
        nonlocal calls
        calls += 1

    cvcuda = SimpleNamespace(internal=SimpleNamespace(syncAuxStream=sync_aux_stream))

    case = benchmark._make_drain_case(cvcuda, "legacy", iterations=2)
    case.call()
    case.synchronize()

    assert calls == 3


def test_submission_chunk_rejects_the_safety_cap_plus_one():
    benchmark = _load_benchmark()

    assert benchmark._submission_chunk_size("64") == 64
    with pytest.raises(argparse.ArgumentTypeError, match="at most 64"):
        benchmark._submission_chunk_size("65")
