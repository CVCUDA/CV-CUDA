# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for Python benchmark state resource lifetime."""

from __future__ import annotations

import gc
import importlib.util
import sys
import types
import weakref
from pathlib import Path

import pytest


BENCH_DIR = Path(__file__).resolve().parent.parent


def _load_python_bench_utils(monkeypatch):
    fake_cvcuda = types.ModuleType("cvcuda")
    fake_cvcuda.Tensor = object
    cache_clears = []
    fake_cvcuda.clear_cache = lambda: cache_clears.append(None)
    fake_cvcuda.Stream = types.SimpleNamespace(
        current=types.SimpleNamespace(sync=lambda: None)
    )
    monkeypatch.setitem(sys.modules, "cvcuda", fake_cvcuda)

    fake_cupy = types.ModuleType("cupy")
    fake_cupy.cuda = types.SimpleNamespace(
        set_allocator=lambda _: None,
        Device=lambda: types.SimpleNamespace(attributes={"MultiProcessorCount": 8}),
        runtime=types.SimpleNamespace(deviceSynchronize=lambda: None),
    )
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)

    spec = importlib.util.spec_from_file_location(
        "python_bench_utils_lifetime_test",
        BENCH_DIR / "python" / "python_bench_utils.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, cache_clears


def _raise_system_exit(code):
    raise SystemExit(code)


def test_run_benchmark_releases_cyclic_resources_before_next_state(monkeypatch, capsys):
    bench_utils, cache_clears = _load_python_bench_utils(monkeypatch)
    fake_cupy = sys.modules["cupy"]
    clock_checks = []
    kernel_compiles = []
    events = []

    class ClockResult:
        def get(self):
            return [1012.0]

    fake_cupy.float64 = object()
    fake_cupy.zeros = lambda _size, dtype: ClockResult()

    def compile_kernel(source, name):
        kernel_compiles.append((source, name))

        def run_kernel(grid, block, _args):
            events.append("clock")
            clock_checks.append((grid, block))

        return run_kernel

    fake_cupy.RawKernel = compile_kernel
    monkeypatch.setattr(bench_utils.os, "_exit", _raise_system_exit)
    monkeypatch.setenv("CVCUDA_BENCH_EXPECT_SM_CLOCK_MHZ", "1005")

    callbacks = []
    fake_bench = types.ModuleType("cuda.bench")

    class Registration:
        def add_string_axis(self, _name, _values):
            return self

    def register(callback):
        callbacks.append(callback)
        return Registration()

    class State:
        def __init__(self, index):
            self.index = index

        def exec(self, callback, sync):
            assert sync is True
            events.append("exec")
            callback(object())

    def run_all_benchmarks(_args):
        for index in range(2):
            callbacks[0](State(index))

    fake_bench.register = register
    fake_bench.run_all_benchmarks = run_all_benchmarks
    fake_cuda = types.ModuleType("cuda")
    fake_cuda.__path__ = []
    fake_cuda.bench = fake_bench
    monkeypatch.setitem(sys.modules, "cuda", fake_cuda)
    monkeypatch.setitem(sys.modules, "cuda.bench", fake_bench)

    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(synchronize=lambda: None)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    config = types.SimpleNamespace(warmup_iterations=1, dtypes=[])
    monkeypatch.setattr(
        bench_utils, "load_operator_config", lambda _operator: (config, [])
    )
    monkeypatch.setattr(
        bench_utils, "register_axes_from_config", lambda _bench, _config: None
    )

    resources = []

    class CyclicResource:
        def __init__(self):
            self.cycle = self

    def benchmark(state):
        if state.index:
            assert resources[-1]() is None
        resource = CyclicResource()
        resources.append(weakref.ref(resource))

        def run(_launch):
            if _launch is None:
                events.append("warmup")
            return resource

        return run

    was_enabled = gc.isenabled()
    gc.collect()
    try:
        bench_utils.run_benchmark("test", benchmark)
    finally:
        gc.collect()
        if was_enabled:
            gc.enable()
        else:
            gc.disable()

    assert gc.isenabled() is was_enabled
    assert len(resources) == 2
    assert all(resource() is None for resource in resources)
    assert len(cache_clears) == 4
    assert len(clock_checks) == 2
    assert events == ["warmup", "clock", "exec"] * 2
    assert kernel_compiles == [(bench_utils._SM_CLOCK_SRC, "measure_sm_clock")]
    assert capsys.readouterr().out.count("SM clock verified: 1012 MHz") == 2
    assert (
        clock_checks
        == [
            (
                (8 * bench_utils._SM_CLOCK_BLOCKS_PER_SM,),
                (bench_utils._SM_CLOCK_THREADS,),
            )
        ]
        * 2
    )


@pytest.mark.parametrize(
    "raw_expected",
    ["", "abc", "0", "-5", "1005 MHz", "0x400", str(2**31)],
)
def test_invalid_expected_sm_clock_is_rejected(monkeypatch, capsys, raw_expected):
    bench_utils, _ = _load_python_bench_utils(monkeypatch)
    monkeypatch.setattr(bench_utils.os, "_exit", _raise_system_exit)
    monkeypatch.setenv("CVCUDA_BENCH_EXPECT_SM_CLOCK_MHZ", raw_expected)

    with pytest.raises(SystemExit) as exc:
        bench_utils._verify_expected_sm_clock()

    assert exc.value.code == 1
    assert (
        f"SM clock verification error: invalid expected clock '{raw_expected}'"
        in capsys.readouterr().err
    )


@pytest.mark.parametrize("raw_expected", ["+1005", " 1005 "])
def test_valid_expected_sm_clock_normalization(monkeypatch, capsys, raw_expected):
    bench_utils, _ = _load_python_bench_utils(monkeypatch)
    fake_cupy = sys.modules["cupy"]
    fake_cupy.float64 = object()
    fake_cupy.zeros = lambda _size, dtype: types.SimpleNamespace(get=lambda: [1012.0])
    fake_cupy.RawKernel = lambda _source, _name: lambda _grid, _block, _args: None
    monkeypatch.setattr(bench_utils.os, "_exit", _raise_system_exit)
    monkeypatch.setenv("CVCUDA_BENCH_EXPECT_SM_CLOCK_MHZ", raw_expected)

    bench_utils._verify_expected_sm_clock()

    assert capsys.readouterr().out == "SM clock verified: 1012 MHz\n"


def test_sm_clock_first_value_outside_tolerance_fails(monkeypatch, capsys):
    bench_utils, _ = _load_python_bench_utils(monkeypatch)
    fake_cupy = sys.modules["cupy"]

    class Result:
        def get(self):
            return [1013.0]

    fake_cupy.float64 = object()
    fake_cupy.zeros = lambda _size, dtype: Result()
    fake_cupy.RawKernel = lambda _source, _name: lambda _grid, _block, _args: None
    monkeypatch.setattr(bench_utils.os, "_exit", _raise_system_exit)
    monkeypatch.setenv("CVCUDA_BENCH_EXPECT_SM_CLOCK_MHZ", "1005")

    with pytest.raises(SystemExit) as exc:
        bench_utils._verify_expected_sm_clock()

    assert exc.value.code == 1
    assert "SM clock mismatch:" in capsys.readouterr().err


def test_lazy_compile_failure_is_a_probe_error(monkeypatch, capsys):
    bench_utils, _ = _load_python_bench_utils(monkeypatch)
    fake_cupy = sys.modules["cupy"]

    class CompileException(Exception):
        pass

    fake_cupy.float64 = object()
    fake_cupy.zeros = lambda _size, dtype: types.SimpleNamespace(get=lambda: [0.0])
    fake_cupy.RawKernel = lambda _source, _name: (
        lambda _grid, _block, _args: (_ for _ in ()).throw(
            CompileException("NVRTC unavailable")
        )
    )
    monkeypatch.setattr(bench_utils.os, "_exit", _raise_system_exit)
    monkeypatch.setenv("CVCUDA_BENCH_EXPECT_SM_CLOCK_MHZ", "1005")

    with pytest.raises(SystemExit):
        bench_utils._verify_expected_sm_clock()

    assert (
        "SM clock verification error: device probe could not run: NVRTC unavailable"
        in capsys.readouterr().err
    )


def test_cuda_runtime_failure_is_a_verification_error(monkeypatch, capsys):
    bench_utils, _ = _load_python_bench_utils(monkeypatch)
    fake_cupy = sys.modules["cupy"]

    class CUDARuntimeError(Exception):
        pass

    fake_cupy.float64 = object()
    fake_cupy.zeros = lambda _size, dtype: types.SimpleNamespace(get=lambda: [0.0])
    fake_cupy.cuda.runtime.CUDARuntimeError = CUDARuntimeError
    fake_cupy.RawKernel = lambda _source, _name: (
        lambda _grid, _block, _args: (_ for _ in ()).throw(
            CUDARuntimeError("launch failed")
        )
    )
    monkeypatch.setattr(bench_utils.os, "_exit", _raise_system_exit)
    monkeypatch.setenv("CVCUDA_BENCH_EXPECT_SM_CLOCK_MHZ", "1005")

    with pytest.raises(SystemExit):
        bench_utils._verify_expected_sm_clock()

    assert "SM clock verification error: device probe could not run: launch failed" in (
        capsys.readouterr().err
    )


def test_invalid_multiprocessor_count_is_a_device_error(monkeypatch, capsys):
    bench_utils, _ = _load_python_bench_utils(monkeypatch)
    fake_cupy = sys.modules["cupy"]
    fake_cupy.float64 = object()
    fake_cupy.zeros = lambda _size, dtype: types.SimpleNamespace(get=lambda: [0.0])
    fake_cupy.RawKernel = lambda _source, _name: lambda _grid, _block, _args: None
    fake_cupy.cuda.Device = lambda: types.SimpleNamespace(
        attributes={"MultiProcessorCount": 0}
    )
    monkeypatch.setattr(bench_utils.os, "_exit", _raise_system_exit)
    monkeypatch.setenv("CVCUDA_BENCH_EXPECT_SM_CLOCK_MHZ", "1005")

    with pytest.raises(SystemExit):
        bench_utils._verify_expected_sm_clock()

    assert "SM clock device error: invalid multiprocessor count: 0" in (
        capsys.readouterr().err
    )


def test_nonpositive_probe_result_is_a_verification_error(monkeypatch, capsys):
    bench_utils, _ = _load_python_bench_utils(monkeypatch)
    fake_cupy = sys.modules["cupy"]
    fake_cupy.float64 = object()
    fake_cupy.zeros = lambda _size, dtype: types.SimpleNamespace(get=lambda: [0.0])
    fake_cupy.RawKernel = lambda _source, _name: lambda _grid, _block, _args: None
    monkeypatch.setattr(bench_utils.os, "_exit", _raise_system_exit)
    monkeypatch.setenv("CVCUDA_BENCH_EXPECT_SM_CLOCK_MHZ", "1005")

    with pytest.raises(SystemExit):
        bench_utils._verify_expected_sm_clock()

    assert "SM clock verification error: device probe returned 0 MHz" in (
        capsys.readouterr().err
    )


def test_ambient_cuda_error_is_not_a_clock_protocol_error(monkeypatch, capsys):
    bench_utils, _ = _load_python_bench_utils(monkeypatch)
    fake_cupy = sys.modules["cupy"]
    fake_cupy.cuda.runtime.deviceSynchronize = lambda: (_ for _ in ()).throw(
        RuntimeError("launch failed")
    )
    monkeypatch.setattr(bench_utils.os, "_exit", _raise_system_exit)
    monkeypatch.setenv("CVCUDA_BENCH_EXPECT_SM_CLOCK_MHZ", "1005")

    with pytest.raises(SystemExit):
        bench_utils._verify_expected_sm_clock()

    assert capsys.readouterr().err == "Pre-existing CUDA error: launch failed\n"


def test_missing_cupy_is_a_probe_setup_error(monkeypatch, capsys):
    bench_utils, _ = _load_python_bench_utils(monkeypatch)
    monkeypatch.setitem(sys.modules, "cupy", None)
    monkeypatch.setattr(bench_utils.os, "_exit", _raise_system_exit)
    monkeypatch.setenv("CVCUDA_BENCH_EXPECT_SM_CLOCK_MHZ", "1005")

    with pytest.raises(SystemExit):
        bench_utils._verify_expected_sm_clock()

    assert "SM clock verification error: probe setup failed:" in (
        capsys.readouterr().err
    )


def test_missing_cuda_synchronize_is_a_probe_setup_error(monkeypatch, capsys):
    bench_utils, _ = _load_python_bench_utils(monkeypatch)
    fake_cupy = sys.modules["cupy"]
    del fake_cupy.cuda.runtime.deviceSynchronize
    monkeypatch.setattr(bench_utils.os, "_exit", _raise_system_exit)
    monkeypatch.setenv("CVCUDA_BENCH_EXPECT_SM_CLOCK_MHZ", "1005")

    with pytest.raises(SystemExit):
        bench_utils._verify_expected_sm_clock()

    assert "probe setup failed: CUDA runtime synchronization unavailable" in (
        capsys.readouterr().err
    )


def test_kernel_construction_failure_is_a_probe_setup_error(monkeypatch, capsys):
    bench_utils, _ = _load_python_bench_utils(monkeypatch)
    fake_cupy = sys.modules["cupy"]
    fake_cupy.RawKernel = lambda *_args: (_ for _ in ()).throw(
        RuntimeError("NVRTC unavailable")
    )
    monkeypatch.setattr(bench_utils.os, "_exit", _raise_system_exit)
    monkeypatch.setenv("CVCUDA_BENCH_EXPECT_SM_CLOCK_MHZ", "1005")

    with pytest.raises(SystemExit):
        bench_utils._verify_expected_sm_clock()

    assert "SM clock verification error: probe setup failed: NVRTC unavailable" in (
        capsys.readouterr().err
    )


def test_unset_expected_sm_clock_skips_probe(monkeypatch):
    bench_utils, _ = _load_python_bench_utils(monkeypatch)
    monkeypatch.setitem(sys.modules, "cupy", None)
    monkeypatch.setattr(bench_utils.os, "_exit", _raise_system_exit)
    monkeypatch.delenv("CVCUDA_BENCH_EXPECT_SM_CLOCK_MHZ", raising=False)

    bench_utils._verify_expected_sm_clock()
