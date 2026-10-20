#!/usr/bin/env python3
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

"""Measure host-side overhead of selected CV-CUDA Python binding paths.

The benchmark preallocates persistent inputs, warms reusable caches, and excludes
blocking completion waits from timed windows. It deliberately reports raw
measurements without pass/fail thresholds so results from two builds can be
retained and compared directly.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from typing import Any

CASES = (
    "python.noop",
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
    "submission.flip.tensor.default",
    "submission.flip.tensor.explicit",
    "submission.threshold.tensor.explicit",
    "submission.cvtcolor.varshape.default",
    "submission.cvtcolor.varshape.explicit",
    "submission.cvtcolor.varshape.batch1.explicit",
    "submission.threshold.varshape.explicit",
    "resourceguard.empty_to_first.explicit",
    "interop.image.cuda.wrapped_writer",
    "interop.dlpack.legacy.stream0",
    "interop.dlpack.versioned.stream0",
    "interop.dlpack.cross_stream",
    "lifecycle.drain_pending_releases.empty",
)

EXPORT_SPECS = {
    f"interop.{container}.fresh_cai.{dtype}.{storage}": (container, dtype, rowalign)
    for container, dtypes in (
        ("tensor", ("u1", "i2", "f2", "f4", "f8", "c16")),
        ("image", ("U8", "F32", "RGB8p", "NV12")),
    )
    for dtype in dtypes
    for storage, rowalign in (("packed", 1), ("padded", 64))
}
SUITES = {"bindings": CASES, "export-metadata": ("python.noop", *EXPORT_SPECS)}


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def summarize(values: Sequence[int], iterations: int) -> dict[str, Any]:
    """Summarize raw timed-window durations as nanoseconds per call."""
    if not values:
        raise ValueError("at least one timed window is required")
    if iterations <= 0:
        raise ValueError("iterations must be positive")

    windows = [int(value) for value in values]
    per_call = [value / iterations for value in windows]
    p50 = float(statistics.median(per_call))
    mad = float(statistics.median(abs(value - p50) for value in per_call))
    robust_noise = 0.0 if p50 == 0 else 100.0 * 1.4826 * mad / p50
    return {
        "samples": len(windows),
        "iterations_per_window": iterations,
        "window_ns": windows,
        "ns_per_call": per_call,
        "p50_ns": p50,
        "mean_ns": float(statistics.mean(per_call)),
        "stdev_ns": float(statistics.stdev(per_call)) if len(per_call) > 1 else 0.0,
        "p90_ns": _percentile(per_call, 0.90),
        "mad_ns": mad,
        "robust_noise_pct": robust_noise,
    }


def build_report(
    cases: Sequence[Mapping[str, Any]],
    parameters: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the stable JSON report from raw case measurements."""
    results = []
    for case in cases:
        result = {
            "name": case["name"],
            "group": case["group"],
        }
        result.update(summarize(case["window_ns"], case["iterations"]))
        if "thread_cpu_window_ns" in case:
            result["thread_cpu_window_ns"] = list(case["thread_cpu_window_ns"])
        results.append(result)
    return {
        "schema_version": 1,
        "metadata": dict(metadata),
        "parameters": dict(parameters),
        "results": results,
    }


@dataclass
class _BenchmarkCase:
    name: str
    group: str
    iterations: int
    chunk_size: int
    call: Callable[[], object]
    synchronize: Callable[[], None]
    prime_before_chunk: bool = True


class _CudaArrayInterfaceOnly:
    """Expose one owner's static CAI without also exposing DLPack."""

    def __init__(self, owner: Any):
        self._owner = owner
        self.__cuda_array_interface__ = dict(owner.__cuda_array_interface__)


class _DLPackOnly:
    """Expose one owner's DLPack protocol without also exposing CAI."""

    def __init__(self, owner: Any):
        self._owner = owner

    def __dlpack__(self, *args: Any, **kwargs: Any):
        return self._owner.__dlpack__(*args, **kwargs)

    def __dlpack_device__(self):
        return self._owner.__dlpack_device__()


def _sync_and_drain(cvcuda: Any, *streams: Any) -> None:
    for stream in streams:
        stream.sync()
    drain = getattr(cvcuda.internal, "drain_pending_releases", None)
    if drain is not None:
        drain()
        return
    sync_aux_stream = getattr(cvcuda.internal, "syncAuxStream", None)
    if sync_aux_stream is not None:
        sync_aux_stream()


def _make_tensor_construction_case(
    cvcuda: Any, name: str, layout_as_string: bool, iterations: int
):
    shape = (1, 32, 32, 4)
    dtype = cvcuda.Type.U8
    expected_layout = cvcuda.TensorLayout("NHWC")
    layout = "NHWC" if layout_as_string else expected_layout

    def call():
        return cvcuda.Tensor(shape, dtype, layout)

    result = call()
    if (
        tuple(result.shape) != shape
        or result.dtype != dtype
        or result.layout != expected_layout
    ):
        raise RuntimeError(f"{name} produced unexpected tensor metadata")
    return _BenchmarkCase(
        name, "construction", iterations, iterations, call, lambda: None
    )


def _make_reshape_case(
    cvcuda: Any,
    name: str,
    method: bool,
    layout_as_string: bool,
    iterations: int,
):
    source = cvcuda.Tensor((1, 32, 32, 4), cvcuda.Type.U8, "NHWC")
    shape = (1, 16, 64, 4)
    expected_layout = cvcuda.TensorLayout("NHWC")
    layout = "NHWC" if layout_as_string else expected_layout

    if method:

        def call():
            return source.reshape(shape, layout)

    else:

        def call():
            return cvcuda.reshape(source, shape, layout)

    result = call()
    if tuple(result.shape) != shape or result.layout != expected_layout:
        raise RuntimeError(f"{name} produced unexpected tensor metadata")
    return _BenchmarkCase(
        name, "construction", iterations, iterations, call, lambda: None
    )


def _make_as_tensor_case(
    cvcuda: Any,
    name: str,
    dlpack: bool,
    layout_as_string: bool,
    iterations: int,
):
    shape = (1, 32, 32, 4)
    dtype = cvcuda.Type.U8
    expected_layout = cvcuda.TensorLayout("NHWC")
    layout = "NHWC" if layout_as_string else expected_layout
    owner = cvcuda.Tensor(shape, dtype, expected_layout)
    external = owner.cuda()
    producer = _DLPackOnly(external) if dlpack else _CudaArrayInterfaceOnly(external)

    def call():
        return cvcuda.as_tensor(producer, layout)

    result = call()
    if (
        tuple(result.shape) != shape
        or result.dtype != dtype
        or result.layout != expected_layout
    ):
        raise RuntimeError(f"{name} produced unexpected tensor metadata")
    return _BenchmarkCase(name, "interop", iterations, iterations, call, lambda: None)


def _make_wrap_and_submit_case(
    cvcuda: Any, name: str, iterations: int, chunk_size: int
):
    shape = (1, 32, 32, 4)
    owner = cvcuda.Tensor(shape, cvcuda.Type.U8, "NHWC")
    producer = _CudaArrayInterfaceOnly(owner.cuda())
    stream = cvcuda.Stream()

    def call():
        return cvcuda.flip(
            cvcuda.as_tensor(producer, "NHWC"), flipCode=0, stream=stream
        )

    result = call()
    _sync_and_drain(cvcuda, stream)
    if tuple(result.shape) != shape or result.layout != cvcuda.TensorLayout("NHWC"):
        raise RuntimeError(f"{name} produced unexpected tensor metadata")
    return _BenchmarkCase(
        name,
        "workflow",
        iterations,
        chunk_size,
        call,
        lambda: _sync_and_drain(cvcuda, stream),
        prime_before_chunk=getattr(cvcuda.internal, "drain_pending_releases", None)
        is None,
    )


def _make_operator_case(
    cvcuda: Any,
    name: str,
    explicit: bool,
    iterations: int,
    chunk_size: int,
):
    src = cvcuda.Tensor((1, 32, 32, 4), cvcuda.Type.U8, "NHWC")
    dst = cvcuda.Tensor(src.shape, src.dtype, src.layout)
    stream = cvcuda.Stream() if explicit else cvcuda.Stream.default

    if explicit:

        def call():
            return cvcuda.flip_into(dst=dst, src=src, flipCode=0, stream=stream)

    else:

        def call():
            return cvcuda.flip_into(dst=dst, src=src, flipCode=0)

    if call() is not dst:
        raise RuntimeError(f"{name} returned a different destination tensor")
    return _BenchmarkCase(
        name,
        "submission",
        iterations,
        chunk_size,
        call,
        lambda: _sync_and_drain(cvcuda, stream),
    )


def _make_threshold_tensor_case(
    cvcuda: Any,
    name: str,
    iterations: int,
    chunk_size: int,
):
    src = cvcuda.Tensor((1, 32, 32, 1), cvcuda.Type.U8, "NHWC")
    dst = cvcuda.Tensor(src.shape, src.dtype, src.layout)
    threshold = cvcuda.Tensor((1,), cvcuda.Type.F64, "N")
    max_value = cvcuda.Tensor((1,), cvcuda.Type.F64, "N")
    stream = cvcuda.Stream()

    def call():
        return cvcuda.threshold_into(
            dst=dst,
            src=src,
            thresh=threshold,
            maxval=max_value,
            type=cvcuda.ThresholdType.BINARY,
            stream=stream,
        )

    if call() is not dst:
        raise RuntimeError(f"{name} returned a different destination tensor")
    return _BenchmarkCase(
        name,
        "submission",
        iterations,
        chunk_size,
        call,
        lambda: _sync_and_drain(cvcuda, stream),
    )


def _make_varshape_case(
    cvcuda: Any,
    name: str,
    explicit: bool,
    iterations: int,
    chunk_size: int,
    batch_size: int,
):
    src = cvcuda.ImageBatchVarShape(batch_size)
    dst = cvcuda.ImageBatchVarShape(batch_size)
    for _ in range(batch_size):
        src.pushback(cvcuda.Image((32, 32), cvcuda.Format.RGB8))
        dst.pushback(cvcuda.Image((32, 32), cvcuda.Format.RGBA8))
    stream = cvcuda.Stream() if explicit else cvcuda.Stream.default

    if explicit:

        def call():
            return cvcuda.cvtcolor_into(
                dst=dst,
                src=src,
                code=cvcuda.ColorConversion.RGB2RGBA,
                stream=stream,
            )

    else:

        def call():
            return cvcuda.cvtcolor_into(
                dst=dst, src=src, code=cvcuda.ColorConversion.RGB2RGBA
            )

    if call() is not dst:
        raise RuntimeError(f"{name} returned a different destination batch")
    return _BenchmarkCase(
        name,
        "submission",
        iterations,
        chunk_size,
        call,
        lambda: _sync_and_drain(cvcuda, stream),
    )


def _make_threshold_varshape_case(
    cvcuda: Any,
    name: str,
    iterations: int,
    chunk_size: int,
    batch_size: int,
):
    src = cvcuda.ImageBatchVarShape(batch_size)
    dst = cvcuda.ImageBatchVarShape(batch_size)
    for _ in range(batch_size):
        src.pushback(cvcuda.Image((32, 32), cvcuda.Format.U8))
        dst.pushback(cvcuda.Image((32, 32), cvcuda.Format.U8))
    threshold = cvcuda.Tensor((batch_size,), cvcuda.Type.F64, "N")
    max_value = cvcuda.Tensor((batch_size,), cvcuda.Type.F64, "N")
    stream = cvcuda.Stream()

    def call():
        return cvcuda.threshold_into(
            dst=dst,
            src=src,
            thresh=threshold,
            maxval=max_value,
            type=cvcuda.ThresholdType.BINARY,
            stream=stream,
        )

    if call() is not dst:
        raise RuntimeError(f"{name} returned a different destination batch")
    return _BenchmarkCase(
        name,
        "submission",
        iterations,
        chunk_size,
        call,
        lambda: _sync_and_drain(cvcuda, stream),
    )


def _make_resourceguard_case(cvcuda: Any, name: str, iterations: int, chunk_size: int):
    stream = cvcuda.Stream()
    resource = cvcuda.Tensor((1, 32, 32, 1), cvcuda.Type.U8, "NHWC")

    def call():
        return cvcuda._test.resourceguard_run_adds_first_resource(stream, resource)

    call()
    return _BenchmarkCase(
        name,
        "resourceguard",
        iterations,
        chunk_size,
        call,
        lambda: _sync_and_drain(cvcuda, stream),
    )


def _make_wrapped_image_case(cvcuda: Any, name: str, iterations: int):
    owner = cvcuda.Image((32, 32), cvcuda.Format.RGB8)
    external = owner.cuda()
    wrapped = cvcuda.as_image(external, cvcuda.Format.RGB8)
    writer = cvcuda.Stream()
    wrapped.submitStreamSync(writer)
    if wrapped.cuda() is not external:
        raise RuntimeError(f"{name} did not preserve wrapped export identity")
    return _BenchmarkCase(
        name,
        "interop",
        iterations,
        iterations,
        wrapped.cuda,
        lambda: _sync_and_drain(cvcuda, writer),
    )


def _make_fresh_export_case(
    cvcuda: Any, name: str, container: str, dtype: str, rowalign: int, iterations: int
):
    import numpy as np

    if container == "tensor":
        owner = cvcuda.Tensor((1, 8, 14, 1), np.dtype(dtype), "NHWC", rowalign=rowalign)

        def call():
            return owner.cuda().__cuda_array_interface__

    else:
        owner = cvcuda.Image((14, 8), getattr(cvcuda.Format, dtype), rowalign=rowalign)

        def call():
            buffers = owner.cuda()
            if not isinstance(buffers, (tuple, list)):
                buffers = (buffers,)
            return tuple(buffer.__cuda_array_interface__ for buffer in buffers)

    # A wrapped Image can return its original buffer and bypass metadata creation.
    # Retain both exports while checking identity to rule out that benchmark trap.
    first, second = owner.cuda(), owner.cuda()
    first = first if isinstance(first, (tuple, list)) else (first,)
    second = second if isinstance(second, (tuple, list)) else (second,)
    if len(first) != len(second) or any(a is b for a, b in zip(first, second)):
        raise RuntimeError(f"{name} reused an exported buffer")
    if call() != call():
        raise RuntimeError(f"{name} produced inconsistent export metadata")
    return _BenchmarkCase(name, "export", iterations, iterations, call, lambda: None)


def _make_dlpack_case(cvcuda: Any, name: str, versioned: bool, iterations: int):
    tensor = cvcuda.Tensor((1, 32, 32, 1), cvcuda.Type.U8, "NHWC")
    external = tensor.cuda()
    if versioned:

        def call():
            return external.__dlpack__(stream=0, max_version=(1, 0))

    else:

        def call():
            return external.__dlpack__(stream=0)

    call()
    return _BenchmarkCase(name, "interop", iterations, iterations, call, lambda: None)


def _make_cross_stream_dlpack_case(
    cvcuda: Any, name: str, iterations: int, chunk_size: int
):
    writer = cvcuda.Stream()
    consumer = cvcuda.Stream()
    src = cvcuda.Tensor((1, 32, 32, 1), cvcuda.Type.U8, "NHWC")
    dst = cvcuda.Tensor(src.shape, src.dtype, src.layout)
    cvcuda.flip_into(dst=dst, src=src, flipCode=0, stream=writer)
    external = dst.cuda()
    writer.sync()

    def call():
        return external.__dlpack__(stream=int(consumer.handle))

    call()
    return _BenchmarkCase(
        name,
        "interop",
        iterations,
        chunk_size,
        call,
        lambda: _sync_and_drain(cvcuda, writer, consumer),
    )


def _make_drain_case(cvcuda: Any, name: str, iterations: int):
    def drain():
        _sync_and_drain(cvcuda)

    drain()
    return _BenchmarkCase(name, "lifecycle", iterations, iterations, drain, drain)


def _build_cases(
    cvcuda: Any, selected: Sequence[str], args: argparse.Namespace
) -> list[_BenchmarkCase]:
    factories = {
        "python.noop": lambda: _BenchmarkCase(
            "python.noop",
            "control",
            args.python_iterations,
            args.python_iterations,
            lambda: None,
            lambda: None,
        ),
        "construction.tensor.layout_string": lambda: _make_tensor_construction_case(
            cvcuda,
            "construction.tensor.layout_string",
            True,
            args.object_iterations,
        ),
        "construction.tensor.layout_object": lambda: _make_tensor_construction_case(
            cvcuda,
            "construction.tensor.layout_object",
            False,
            args.object_iterations,
        ),
        "construction.reshape.method.layout_string": lambda: _make_reshape_case(
            cvcuda,
            "construction.reshape.method.layout_string",
            True,
            True,
            args.object_iterations,
        ),
        "construction.reshape.method.layout_object": lambda: _make_reshape_case(
            cvcuda,
            "construction.reshape.method.layout_object",
            True,
            False,
            args.object_iterations,
        ),
        "construction.reshape.module.layout_string": lambda: _make_reshape_case(
            cvcuda,
            "construction.reshape.module.layout_string",
            False,
            True,
            args.object_iterations,
        ),
        "construction.reshape.module.layout_object": lambda: _make_reshape_case(
            cvcuda,
            "construction.reshape.module.layout_object",
            False,
            False,
            args.object_iterations,
        ),
        "interop.as_tensor.cai.layout_string": lambda: _make_as_tensor_case(
            cvcuda,
            "interop.as_tensor.cai.layout_string",
            False,
            True,
            args.object_iterations,
        ),
        "interop.as_tensor.cai.layout_object": lambda: _make_as_tensor_case(
            cvcuda,
            "interop.as_tensor.cai.layout_object",
            False,
            False,
            args.object_iterations,
        ),
        "interop.as_tensor.dlpack.layout_string": lambda: _make_as_tensor_case(
            cvcuda,
            "interop.as_tensor.dlpack.layout_string",
            True,
            True,
            args.object_iterations,
        ),
        "interop.as_tensor.dlpack.layout_object": lambda: _make_as_tensor_case(
            cvcuda,
            "interop.as_tensor.dlpack.layout_object",
            True,
            False,
            args.object_iterations,
        ),
        "workflow.flip.as_tensor.cai.explicit.serial": lambda: _make_wrap_and_submit_case(
            cvcuda,
            "workflow.flip.as_tensor.cai.explicit.serial",
            args.submission_iterations,
            1,
        ),
        "workflow.flip.as_tensor.cai.explicit.queued": lambda: _make_wrap_and_submit_case(
            cvcuda,
            "workflow.flip.as_tensor.cai.explicit.queued",
            args.submission_iterations,
            args.submission_chunk_size,
        ),
        "submission.flip.tensor.default": lambda: _make_operator_case(
            cvcuda,
            "submission.flip.tensor.default",
            False,
            args.submission_iterations,
            args.submission_chunk_size,
        ),
        "submission.flip.tensor.explicit": lambda: _make_operator_case(
            cvcuda,
            "submission.flip.tensor.explicit",
            True,
            args.submission_iterations,
            args.submission_chunk_size,
        ),
        "submission.threshold.tensor.explicit": lambda: _make_threshold_tensor_case(
            cvcuda,
            "submission.threshold.tensor.explicit",
            args.submission_iterations,
            args.submission_chunk_size,
        ),
        "submission.cvtcolor.varshape.default": lambda: _make_varshape_case(
            cvcuda,
            "submission.cvtcolor.varshape.default",
            False,
            args.varshape_iterations,
            args.submission_chunk_size,
            args.batch_size,
        ),
        "submission.cvtcolor.varshape.explicit": lambda: _make_varshape_case(
            cvcuda,
            "submission.cvtcolor.varshape.explicit",
            True,
            args.varshape_iterations,
            args.submission_chunk_size,
            args.batch_size,
        ),
        "submission.cvtcolor.varshape.batch1.explicit": lambda: _make_varshape_case(
            cvcuda,
            "submission.cvtcolor.varshape.batch1.explicit",
            True,
            args.varshape_iterations,
            args.submission_chunk_size,
            1,
        ),
        "submission.threshold.varshape.explicit": lambda: _make_threshold_varshape_case(
            cvcuda,
            "submission.threshold.varshape.explicit",
            args.varshape_iterations,
            args.submission_chunk_size,
            args.batch_size,
        ),
        "resourceguard.empty_to_first.explicit": lambda: _make_resourceguard_case(
            cvcuda,
            "resourceguard.empty_to_first.explicit",
            args.resource_iterations,
            args.submission_chunk_size,
        ),
        "interop.image.cuda.wrapped_writer": lambda: _make_wrapped_image_case(
            cvcuda, "interop.image.cuda.wrapped_writer", args.interop_iterations
        ),
        "interop.dlpack.legacy.stream0": lambda: _make_dlpack_case(
            cvcuda, "interop.dlpack.legacy.stream0", False, args.interop_iterations
        ),
        "interop.dlpack.versioned.stream0": lambda: _make_dlpack_case(
            cvcuda, "interop.dlpack.versioned.stream0", True, args.interop_iterations
        ),
        "interop.dlpack.cross_stream": lambda: _make_cross_stream_dlpack_case(
            cvcuda,
            "interop.dlpack.cross_stream",
            args.cross_stream_iterations,
            args.submission_chunk_size,
        ),
        "lifecycle.drain_pending_releases.empty": lambda: _make_drain_case(
            cvcuda,
            "lifecycle.drain_pending_releases.empty",
            args.maintenance_iterations,
        ),
    }
    return [
        (
            _make_fresh_export_case(
                cvcuda, name, *EXPORT_SPECS[name], args.object_iterations
            )
            if name in EXPORT_SPECS
            else factories[name]()
        )
        for name in selected
    ]


def _time_chunk(call: Callable[[], object], iterations: int) -> tuple[int, int]:
    gc_was_enabled = gc.isenabled()
    if gc_was_enabled:
        gc.disable()
    try:
        # CPU reads surround wall timing so their overhead cannot enter its metric.
        cpu_start = time.thread_time_ns()
        start = time.perf_counter_ns()
        for _ in range(iterations):
            call()
        elapsed = time.perf_counter_ns() - start
        cpu_elapsed = time.thread_time_ns() - cpu_start
        return elapsed, cpu_elapsed
    finally:
        if gc_was_enabled:
            gc.enable()


def _measure_case(case: _BenchmarkCase, samples: int, warmups: int) -> dict[str, Any]:
    def run_window() -> tuple[int, int]:
        remaining = case.iterations
        elapsed = 0
        cpu_elapsed = 0
        case.synchronize()
        try:
            while remaining > 0:
                chunk = min(remaining, case.chunk_size)
                # Start each chunk from steady retirement state. On releases
                # that use the legacy auxiliary callback stream, synchronization
                # makes the previous closures ready but the next submission is
                # what clears them. Keep that boundary cleanup out of timing.
                if case.prime_before_chunk:
                    case.call()
                    case.synchronize()
                chunk_elapsed, chunk_cpu_elapsed = _time_chunk(case.call, chunk)
                elapsed += chunk_elapsed
                cpu_elapsed += chunk_cpu_elapsed
                remaining -= chunk
                case.synchronize()
            return elapsed, cpu_elapsed
        finally:
            case.synchronize()

    for _ in range(warmups):
        run_window()
    windows = [run_window() for _ in range(samples)]
    return {
        "name": case.name,
        "group": case.group,
        "iterations": case.iterations,
        "window_ns": [wall for wall, _ in windows],
        "thread_cpu_window_ns": [cpu for _, cpu in windows],
    }


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as cpuinfo:
            for line in cpuinfo:
                if line.lower().startswith("model name"):
                    return line.partition(":")[2].strip()
    except OSError:
        pass
    # ARM /proc/cpuinfo may expose only numeric implementer/part identifiers.
    try:
        result = subprocess.run(
            ["lscpu", "-J"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
            env={**os.environ, "LC_ALL": "C"},
        )
        inventory = json.loads(result.stdout)
        if isinstance(inventory, dict) and isinstance(inventory.get("lscpu"), list):
            for entry in inventory["lscpu"]:
                if isinstance(entry, dict) and entry.get("field") == "Model name:":
                    model = entry.get("data")
                    if isinstance(model, str) and model.strip():
                        return model.strip()
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    return platform.processor() or "unknown"


def _gpu_inventory() -> list[str]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,pci.bus_id,driver_version,compute_cap",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _cpu_frequency_khz() -> int | None:
    if not hasattr(os, "sched_getaffinity"):
        return None
    affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) != 1:
        return None
    path = Path(f"/sys/devices/system/cpu/cpu{affinity[0]}/cpufreq/scaling_cur_freq")
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metadata(cvcuda: Any, cpu_frequency_before_khz: int | None) -> dict[str, Any]:
    affinity = (
        sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    )
    extension = Path(cvcuda._cvcuda.__file__).resolve()
    benchmark = Path(__file__).resolve()
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_script": str(benchmark),
        "benchmark_script_sha256": _sha256(benchmark),
        "cvcuda_package": str(Path(cvcuda.__file__).resolve()),
        "cvcuda_extension": str(extension),
        "cvcuda_extension_sha256": _sha256(extension),
        "cvcuda_version": getattr(cvcuda, "__version__", "unknown"),
        "python_version": platform.python_version(),
        "python_build": sys.version,
        "gil_enabled": getattr(sys, "_is_gil_enabled", lambda: True)(),
        "runtime_image": os.environ.get("CVCUDA_BINDING_BENCH_IMAGE"),
        "cuda_major": os.environ.get("CVCUDA_BINDING_BENCH_CUDA_MAJOR"),
        "python_executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "cpu_model": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "cpu_affinity": affinity,
        "cpu_frequency_before_khz": cpu_frequency_before_khz,
        "cpu_frequency_after_khz": _cpu_frequency_khz(),
        "gpu_inventory": _gpu_inventory(),
        "cuda_version_environment": os.environ.get("CUDA_VERSION"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_visible_devices": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
        "worker_hostname": platform.node(),
        "k8s_node_name": os.environ.get("K8S_NODE_NAME"),
        "timer": "time.perf_counter_ns",
        "timer_resolution_ns": time.get_clock_info("perf_counter").resolution * 1e9,
    }


def _print_table(report: Mapping[str, Any]) -> None:
    print("CV-CUDA Python binding host overhead (blocking completion waits excluded)")
    print(f"module: {report['metadata']['cvcuda_extension']}")
    print(f"samples: {report['parameters']['samples']}")
    print()
    print(
        f"{'case':48} {'p50 ns':>12} {'mean ns':>12} {'p90 ns':>12} {'stdev':>12} {'noise %':>10}"
    )
    for result in report["results"]:
        print(
            f"{result['name']:48} {result['p50_ns']:12.1f} {result['mean_ns']:12.1f} "
            f"{result['p90_ns']:12.1f} {result['stdev_ns']:12.1f} {result['robust_noise_pct']:10.2f}"
        )
    noisy = [
        result["name"]
        for result in report["results"]
        if result["robust_noise_pct"] > 10.0
    ]
    if noisy:
        print(f"\nwarning: robust noise exceeds 10% for {', '.join(noisy)}")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _submission_chunk_size(value: str) -> int:
    parsed = _positive_int(value)
    if parsed > 64:
        raise argparse.ArgumentTypeError(
            "must be at most 64 to avoid asynchronous queue backpressure"
        )
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  Measure every case and save the raw-window artifact:
    binding_bench.py --output candidate.json

  Measure one case on one logical CPU:
    binding_bench.py --case submission.flip.tensor.explicit --cpu 3 --output flip.json

Exit status:
  0 if all selected cases completed and the JSON artifact was written;
  nonzero for invalid arguments, unavailable benchmark features, or runtime errors.
""",
    )
    parser.add_argument("--suite", choices=SUITES, default="bindings")
    parser.add_argument(
        "--case",
        action="append",
        choices=tuple(
            dict.fromkeys(name for cases in SUITES.values() for name in cases)
        ),
        help="case to run; repeat to select several",
    )
    parser.add_argument(
        "--samples",
        type=_positive_int,
        default=20,
        help="timed windows per case (default: 20)",
    )
    parser.add_argument(
        "--warmups",
        type=int,
        default=5,
        help="discarded warm-up windows per case (default: 5)",
    )
    parser.add_argument("--python-iterations", type=_positive_int, default=1_000_000)
    parser.add_argument(
        "--object-iterations",
        type=_positive_int,
        default=20_000,
        help="calls per construction, wrapping, or reshape window (default: 20000)",
    )
    parser.add_argument("--submission-iterations", type=_positive_int, default=12_800)
    parser.add_argument("--varshape-iterations", type=_positive_int, default=3_200)
    parser.add_argument("--resource-iterations", type=_positive_int, default=64_000)
    parser.add_argument("--interop-iterations", type=_positive_int, default=200_000)
    parser.add_argument("--cross-stream-iterations", type=_positive_int, default=10_000)
    parser.add_argument("--maintenance-iterations", type=_positive_int, default=500_000)
    parser.add_argument(
        "--submission-chunk-size",
        type=_submission_chunk_size,
        default=64,
        help="calls between excluded synchronizations, at most 64 (default: 64)",
    )
    parser.add_argument(
        "--batch-size",
        type=_positive_int,
        default=256,
        help=(
            "batch size for configurable VarShape cases; the fixed batch-1 "
            "probe is unchanged (default: 256)"
        ),
    )
    affinity = parser.add_mutually_exclusive_group()
    affinity.add_argument(
        "--cpu",
        type=int,
        help="logical CPU to use (default: first CPU in the allowed affinity mask)",
    )
    affinity.add_argument(
        "--no-pin", action="store_true", help="do not pin the benchmark process"
    )
    parser.add_argument(
        "--precondition-ms",
        type=_nonnegative_float,
        default=250.0,
        help="CPU preconditioning duration before measurement (default: 250)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("python-binding-benchmark.json")
    )
    return parser


def _pin_benchmark_cpu(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> None:
    if args.no_pin:
        return
    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        if args.cpu is not None:
            parser.error("--cpu is not supported on this platform")
        return
    allowed = sorted(os.sched_getaffinity(0))
    cpu = args.cpu if args.cpu is not None else allowed[0]
    if cpu not in allowed:
        parser.error(f"CPU {cpu} is outside the allowed affinity mask {allowed}")
    try:
        os.sched_setaffinity(0, {cpu})
    except OSError as error:
        parser.error(f"cannot pin to CPU {cpu}: {error}")


def _precondition_cpu(milliseconds: float) -> None:
    deadline = time.perf_counter_ns() + int(milliseconds * 1_000_000)
    value = 1
    while time.perf_counter_ns() < deadline:
        value = (value * 6364136223846793005 + 1) & 0xFFFFFFFFFFFFFFFF
    if value < 0:  # pragma: no cover - keeps the loop result observably live.
        raise AssertionError


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.warmups < 0:
        parser.error("--warmups must be non-negative")
    args.output.unlink(missing_ok=True)

    import cvcuda

    selected = tuple(dict.fromkeys(args.case or SUITES[args.suite]))
    if not set(selected) <= set(SUITES[args.suite]):
        parser.error("--case must belong to the selected --suite")
    cases = _build_cases(cvcuda, selected, args)
    # Pin only after CUDA initialization so runtime worker threads do not inherit
    # the benchmark thread's single-CPU affinity.
    _pin_benchmark_cpu(args, parser)
    _precondition_cpu(args.precondition_ms)
    cpu_frequency_before_khz = _cpu_frequency_khz()
    raw_results = [_measure_case(case, args.samples, args.warmups) for case in cases]
    parameters = {
        "selected_cases": list(selected),
        "samples": args.samples,
        "warmups": args.warmups,
        "python_iterations": args.python_iterations,
        "object_iterations": args.object_iterations,
        "submission_iterations": args.submission_iterations,
        "varshape_iterations": args.varshape_iterations,
        "resource_iterations": args.resource_iterations,
        "interop_iterations": args.interop_iterations,
        "cross_stream_iterations": args.cross_stream_iterations,
        "maintenance_iterations": args.maintenance_iterations,
        "submission_chunk_size": args.submission_chunk_size,
        "batch_size": args.batch_size,
        "precondition_ms": args.precondition_ms,
        "timing_policy": {
            "blocking_completion_waits": "excluded",
            "thread_cpu_time": "diagnostic; surrounding chunk CPU time including clock overhead",
            "window_boundary_stream_synchronization": "excluded",
            "intrinsic_dependency_commands": "included",
            "steady_state_boundary_call": "excluded",
            "workflow_boundary_priming": "only when explicit release drain is unavailable",
            "garbage_collection_during_timed_chunks": "disabled",
            "return_value_python_decref": "included",
            "construction_allocation_cache": "warmed",
            "reshape_and_wrapping_previous_wrapper_cleanup": "included",
            "submission_native_resource_reclamation": "excluded at chunk boundaries",
            "workflow_serial_chunk_size": 1,
            "workflow_queued_chunk_size": args.submission_chunk_size,
        },
    }
    report = build_report(
        raw_results, parameters, _metadata(cvcuda, cpu_frequency_before_khz)
    )
    _print_table(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
