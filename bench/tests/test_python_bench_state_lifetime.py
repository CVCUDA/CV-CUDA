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


BENCH_DIR = Path(__file__).resolve().parent.parent


def _load_python_bench_utils(monkeypatch):
    fake_cvcuda = types.ModuleType("cvcuda")
    fake_cvcuda.Tensor = object
    cache_clears = []
    fake_cvcuda.clear_cache = lambda: cache_clears.append(None)
    monkeypatch.setitem(sys.modules, "cvcuda", fake_cvcuda)

    fake_cupy = types.ModuleType("cupy")
    fake_cupy.cuda = types.SimpleNamespace(set_allocator=lambda _: None)
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)

    spec = importlib.util.spec_from_file_location(
        "python_bench_utils_lifetime_test",
        BENCH_DIR / "python" / "python_bench_utils.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, cache_clears


def test_run_benchmark_releases_cyclic_resources_before_next_state(monkeypatch):
    bench_utils, cache_clears = _load_python_bench_utils(monkeypatch)

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
            callback(None)

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

    config = types.SimpleNamespace(warmup_iterations=0, dtypes=[])
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
