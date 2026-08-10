#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Utility functions for CV-CUDA Python benchmarks using nvbench.
Python equivalent of C++ CppBenchUtils.hpp
"""

import sys
from pathlib import Path
from typing import Tuple, Union

# Workaround for cuda-pathfinder >= 1.4 rejecting 'nvperf_target' and
# 'nvperf_host' as unknown library names.  cuda.bench.__init__ tries to
# load these unconditionally via load_nvidia_dynamic_lib(); we patch the
# loader so unrecognised names are silently skipped instead of raising.
try:
    import cuda.pathfinder as _pf

    _orig_load = _pf.load_nvidia_dynamic_lib

    def _tolerant_load(libname, *args, **kwargs):
        try:
            return _orig_load(libname, *args, **kwargs)
        except Exception:
            pass

    _pf.load_nvidia_dynamic_lib = _tolerant_load
except ImportError:
    pass

import cvcuda

# Add config directory to path for imports (handles both source and build layouts)
_this_dir = Path(__file__).resolve().parent
_config_paths = [
    str(_this_dir / "config"),  # Installed/build: bin/config/
    str(_this_dir.parent / "config"),  # Source: bench/config/
]
for _p in _config_paths:
    sys.path.insert(0, _p)

_support_paths = [str(_this_dir), str(_this_dir.parent)]
for _p in _support_paths:
    sys.path.insert(0, _p)

from _internal.warmup import resolve_warmup_iterations  # noqa: E402

# Re-export config loading functions so benchmarks can import from python_bench_utils
# (maintains backwards compatibility)
try:
    from load_config import (  # noqa: E402, F401
        BenchmarkConfig,
        ConfigLoader,
        load_bench_config,
        load_operator_config,
        register_axes_from_config,
        generate_axis_args,
        get_operator_from_benchmark_name,
        get_configs_for_benchmark,
    )
except ImportError as e:
    import warnings

    warnings.warn(
        f"Failed to import load_config. Searched paths: {_config_paths}. "
        f"python_bench_utils.py location: {Path(__file__).resolve()}. Error: {e}"
    )
    raise


def _disable_cupy_memory_pool() -> None:
    """Route all cuPy allocations through ``cudaMalloc`` instead of the cached pool.

    cuPy's default allocator caches freed blocks and reuses them at the same
    physical pages.  C++ benchmarks allocate via NVCV, which goes straight to
    ``cudaMalloc``.  Two different allocator policies on the two sides produce
    systematically different physical-DRAM layouts for the kernel's working
    set.  On memory-bandwidth-bound operators (e.g. remap with >100 MB/kernel
    streamed reads), DRAM bank-row conflicts depend on physical placement and
    drive multi-percent wall-time differences that ncu cannot see — it counts
    bytes and SM cycles, not bank-conflict latency.

    Switching cuPy to direct ``cudaMalloc`` removes the systematic placement
    bias between the two paths.  This is a partial fix: separate processes
    still get different physical pages from the kernel-mode driver, so the
    parity gate still has a process-level noise floor.  The complete fix is a
    single-process harness that runs both paths in one CUDA context.
    """
    import cupy as cp

    cp.cuda.set_allocator(None)


_disable_cupy_memory_pool()


# nvcv is now part of cvcuda in 0.16.0


def parse_shape(shape_str: str, delimiter: str = "x") -> Tuple[int, ...]:
    """
    Parse shape string into tuple.
    Equivalent to C++ GetShape<N> function.

    Args:
        shape_str: Shape string like "1x1080x1920"
        delimiter: Delimiter character (default: 'x')

    Returns:
        Tuple of integers representing the shape

    Examples:
        >>> parse_shape("1x1080x1920")
        (1, 1080, 1920)
    """
    try:
        return tuple(int(x) for x in shape_str.split(delimiter))
    except ValueError as e:
        raise ValueError(f"Invalid shape string '{shape_str}': {e}")


def get_resize_output_shape(shape: Tuple[int, int, int], resize_type: str):
    """Resolve legacy relative resize modes or an exact ``TARGET_HxW`` size."""
    N, H, W = shape
    if resize_type == "EXPAND":
        return N, H * 2, W * 2
    if resize_type == "CONTRACT":
        return N, H // 2, W // 2
    if resize_type.startswith("TARGET_"):
        target = parse_shape(resize_type.removeprefix("TARGET_"))
        if len(target) != 2 or any(dim <= 0 for dim in target):
            raise ValueError(
                f"Resize target must contain two positive dimensions: {resize_type}"
            )
        return N, *target
    raise ValueError(f"Invalid resizeType: {resize_type}")


def get_dtype(dtype_str: str):
    """
    Map dtype string to cvcuda.Type.
    Equivalent to C++ GetDataType<T> function.

    Supports both Python-style names (uint8, float32) and C++-style vector types
    (uchar, uchar3, short4, etc.) which are mapped to their base types.

    Args:
        dtype_str: String like "uint8", "float32", "float", "int32", "uchar3", "short4"

    Returns:
        cvcuda.Type value (e.g., cvcuda.Type.U8, cvcuda.Type.F32)
    """
    # C++ vector type to base type mapping
    # uchar, uchar3, uchar4 -> uint8
    # char, char3, char4 -> int8
    # ushort, ushort3, ushort4 -> uint16
    # short, short3, short4 -> int16
    # uint, uint3, uint4 -> uint32
    # int, int3, int4 -> int32
    # float, float3, float4 -> float32
    # double, double3, double4 -> float64

    # Strip vector suffix (1-4) if present
    import re

    base_type = re.sub(r"[1-4]$", "", dtype_str)

    dtype_map = {
        # Standard names
        "uint8": cvcuda.Type.U8,
        "uint16": cvcuda.Type.U16,
        "uint32": cvcuda.Type.U32,
        "int8": cvcuda.Type.S8,
        "int16": cvcuda.Type.S16,
        "int32": cvcuda.Type.S32,
        "int": cvcuda.Type.S32,
        "float32": cvcuda.Type.F32,
        "float": cvcuda.Type.F32,
        "float64": cvcuda.Type.F64,
        "double": cvcuda.Type.F64,
        # C++ type names (base)
        "uchar": cvcuda.Type.U8,
        "char": cvcuda.Type.S8,
        "ushort": cvcuda.Type.U16,
        "short": cvcuda.Type.S16,
        "uint": cvcuda.Type.U32,
    }

    # Try exact match first, then base type
    if dtype_str in dtype_map:
        return dtype_map[dtype_str]
    elif base_type in dtype_map:
        return dtype_map[base_type]
    else:
        raise ValueError(
            f"Unsupported dtype: {dtype_str}. " f"Supported: {list(dtype_map.keys())}"
        )


def get_input_kind(value: str) -> str:
    """Validate and return the ``inputKind`` config axis value.

    Selects which nvcv container a benchmark builds for its batch: ``Tensor``
    (one dense tensor) or ``VarShape`` (ImageBatchVarShape). Mirrors
    ``get_interpolation_type``: maps the axis string to the accepted set and
    raises on anything unexpected (no silent fallback).
    """
    if value not in ("Tensor", "VarShape"):
        raise ValueError(f"Unexpected inputKind = {value}")
    return value


def get_num_channels(dtype_str: str) -> int:
    """
    Get number of channels for a dtype string.

    Args:
        dtype_str: String like "uint8", "uchar", "uchar3", "uchar4", "float3"

    Returns:
        Number of channels (1, 3, or 4)

    Examples:
        >>> get_num_channels("uint8")  # 1
        >>> get_num_channels("uchar3")  # 3
        >>> get_num_channels("uchar4")  # 4
    """
    import re

    # Extract channel count from vector types (uchar3, float4, etc.)
    # Look for patterns like "uchar3", "float4", etc. (not "uint8", "uint16")
    # Match only if preceded by a letter (to avoid matching digits in uint8, uint16, etc.)
    match = re.search(r"[a-z](\d)$", dtype_str, re.IGNORECASE)
    if match:
        channel_count = int(match.group(1))
        # Only return if it's a valid channel count (2, 3 or 4)
        if channel_count in [2, 3, 4]:
            return channel_count
    return 1


def get_dtype_size(dtype) -> int:
    """
    Get element size in bytes for a dtype (including vector types).

    Args:
        dtype: cvcuda.Type or dtype string (e.g., cvcuda.Type.U8, "uint8", "uchar4")

    Returns:
        Total size in bytes (base_size * num_channels for string types)
    """
    if isinstance(dtype, str):
        base_dtype = get_dtype(dtype)  # Gets base type
        num_channels = get_num_channels(dtype)
        base_size = {
            cvcuda.Type.U8: 1,
            cvcuda.Type.S8: 1,
            cvcuda.Type.U16: 2,
            cvcuda.Type.S16: 2,
            cvcuda.Type.U32: 4,
            cvcuda.Type.S32: 4,
            cvcuda.Type.F32: 4,
            cvcuda.Type.F64: 8,
        }.get(base_dtype, 4)
        return base_size * num_channels
    else:
        # Existing logic for cvcuda.Type
        size_map = {
            cvcuda.Type.U8: 1,
            cvcuda.Type.S8: 1,
            cvcuda.Type.U16: 2,
            cvcuda.Type.S16: 2,
            cvcuda.Type.U32: 4,
            cvcuda.Type.S32: 4,
            cvcuda.Type.F32: 4,
            cvcuda.Type.F64: 8,
        }
        return size_map.get(dtype, 4)


def get_format_from_dtype(
    dtype_str: str, num_channels: int = None, planar: bool = False
):
    """
    Get cvcuda.Format from dtype string and channel count.

    Args:
        dtype_str: Base dtype like "uint8", "float32"
        num_channels: Number of channels (if None, inferred from dtype_str)
        planar: Return a planar multi-plane format when available.

    Returns:
        cvcuda.Format

    Examples:
        >>> get_format_from_dtype("uint8", 1)  # Format.U8
        >>> get_format_from_dtype("uchar4")    # Format.RGBA8
        >>> get_format_from_dtype("float32", 3) # Format.RGBf32
        >>> get_format_from_dtype("uchar3", planar=True) # Format.RGB8p
    """
    if num_channels is None:
        num_channels = get_num_channels(dtype_str)

    base_dtype = get_dtype(dtype_str)

    # Map (base_type, channels) to Format
    format_map = {
        (cvcuda.Type.U8, 1): cvcuda.Format.U8,
        (cvcuda.Type.U8, 3): cvcuda.Format.RGB8,
        (cvcuda.Type.U8, 4): cvcuda.Format.RGBA8,
        (cvcuda.Type.F32, 1): cvcuda.Format.F32,
        (cvcuda.Type.F32, 2): cvcuda.Format._2F32,
        (cvcuda.Type.F32, 3): cvcuda.Format.RGBf32,
        (cvcuda.Type.F32, 4): cvcuda.Format.RGBAf32,
        (cvcuda.Type.U16, 1): cvcuda.Format.U16,
        (cvcuda.Type.S16, 1): cvcuda.Format.S16,
        (cvcuda.Type.S16, 2): cvcuda.Format._2S16,
        (cvcuda.Type.S32, 1): cvcuda.Format.S32,
        (cvcuda.Type.F64, 1): cvcuda.Format.F64,
    }

    key = (base_dtype, num_channels)
    if planar:
        planar_format_map = {
            (cvcuda.Type.U8, 3): cvcuda.Format.RGB8p,
            (cvcuda.Type.U8, 4): cvcuda.Format.RGBA8p,
            (cvcuda.Type.F32, 3): cvcuda.Format.RGBf32p,
            (cvcuda.Type.F32, 4): cvcuda.Format.RGBAf32p,
        }
        if key in planar_format_map:
            return planar_format_map[key]

    if key in format_map:
        return format_map[key]
    raise ValueError(
        f"Unsupported {'planar ' if planar else ''}format for "
        f"{dtype_str} with {num_channels} channels"
    )


def get_stream(launch):
    """
    Get/create cached CUDA stream from launch object.

    Caches stream per unique stream address to avoid recreating on each call,
    but recreates if the stream address changes (e.g., between benchmark configs).

    Args:
        launch: cuda.bench launch object, or None for warmup iterations

    Returns:
        cvcuda stream object

    Usage in benchmarks:
        from python_bench_utils import get_stream
        # In run function:
        operator(..., stream=get_stream(launch))
    """
    # During warmup, launch is None - use current CUDA stream
    if launch is None:
        if not hasattr(get_stream, "warmup_stream"):
            get_stream.warmup_stream = cvcuda.Stream.current
        return get_stream.warmup_stream

    stream_addr = launch.get_stream().addressof()
    if not hasattr(get_stream, "stream") or get_stream.stream_addr != stream_addr:
        get_stream.stream = cvcuda.as_stream(stream_addr)
        get_stream.stream_addr = stream_addr
    return get_stream.stream


def create_stream_cache():
    """
    Create a local stream cache for use in benchmark run functions.

    This is faster than get_stream() because it uses a closure-based cache
    that avoids the overhead of hasattr() and global attribute lookups on
    every iteration. Caches the warmup stream and the measurement stream
    separately, since they ARE different streams: warmup runs on
    cvcuda.Stream.current (typically the null stream), while nvbench's
    state.exec hands launch objects whose get_stream() is the dedicated
    measurement stream nvbench records its CUDA events on.

    Returning the warmup-time Stream.current for measurement runs is a
    correctness bug: kernels submit to the null stream, but nvbench
    measures launch.get_stream(), producing a bogus very-short measured
    interval (event start/stop fire back-to-back on an idle stream).
    Data-dependent ops (histogram, remap, sift, …) surface this as
    Python-faster-than-C++ parity failures.

    Usage in benchmarks:
        get_cached_stream = create_stream_cache()

        def run(launch):
            operator(..., stream=get_cached_stream(launch))

        # Warmup (launch will be None)
        run_warmup(run, warmup_iterations)
        state.exec(run, sync=True)
    """
    warmup_stream = [None]
    measure_stream = [None, None]  # [stream, addr]

    def get_cached_stream(launch):
        if launch is None:
            # Warmup phase: nvbench hasn't created its measurement stream yet.
            # Use the current CUDA stream so warmup ordering is well-defined.
            if warmup_stream[0] is None:
                warmup_stream[0] = cvcuda.Stream.current
            return warmup_stream[0]
        # Measurement phase: ALWAYS submit to launch.get_stream() so the
        # kernel ends up on the same stream nvbench is timing.
        addr = launch.get_stream().addressof()
        if measure_stream[1] != addr:
            measure_stream[0] = cvcuda.as_stream(addr)
            measure_stream[1] = addr
        return measure_stream[0]

    return get_cached_stream


def run_warmup(run_fn, iterations: int):
    """
    Run warmup iterations before benchmark measurement.

    This mirrors the C++ warmup functionality in CppBenchUtils.hpp.
    Warmup helps stabilize GPU clocks and ensures caches are warm
    before actual measurements begin.

    Args:
        run_fn: The benchmark function to run (takes launch as argument).
                When called with None as launch, the run function should
                use get_stream(None) or create_stream_cache()(None) which
                will return the current CUDA stream.
        iterations: Number of warmup iterations to execute

    Usage in benchmarks:
        get_stream = create_stream_cache()

        def run(launch):
            operator(..., stream=get_stream(launch))

        # Run warmup before measurement (pass None as launch during warmup)
        run_warmup(run, config.warmup_iterations)
        state.exec(run, sync=True)
    """
    iterations = resolve_warmup_iterations(iterations)
    if iterations <= 0:
        return

    import torch

    for _ in range(iterations):
        run_fn(None)  # Pass None as launch during warmup (no timing needed)

    # Synchronize to ensure all warmup work is complete
    torch.cuda.synchronize()


def as_torch_cuda_stream(cuda_stream, device_id: int = 0):
    """
    Convert a CUDA stream to a PyTorch CUDA stream.

    Args:
        cuda_stream: A CUDA stream object (e.g., from cuda.bench launch.get_stream())
        device_id: The GPU device ID

    Returns:
        torch.cuda.Stream: A PyTorch CUDA stream wrapping the given stream
    """
    import torch

    return torch.cuda.ExternalStream(
        cuda_stream.addressof(),
        device=torch.device(f"cuda:{device_id}"),
    )


def run_benchmark(operator_name: str, benchmark_func):
    """
    Standard entry point for benchmark scripts.

    Handles config loading, axis registration, warmup, and benchmark execution.
    The benchmark function should return its run function instead of calling exec.

    Args:
        operator_name: The operator name for config lookup (e.g., "resize", "gaussian")
        benchmark_func: The benchmark function that returns a run(launch) callable

    Usage:
        from python_bench_utils import run_benchmark

        def my_benchmark(state):
            # Setup tensors, etc.
            ...
            def run(launch):
                cvcuda.operator_into(dst, src, stream=get_stream(launch))
            return run  # Return run function for framework to handle

        if __name__ == "__main__":
            run_benchmark("my_operator", my_benchmark)
    """
    import gc
    import cuda.bench as bench

    gc.disable()

    config, bench_args = load_operator_config(operator_name)
    warmup_iterations = config.warmup_iterations

    def wrapped_benchmark(state):
        """Wrapper that handles warmup and execution."""
        # Keep setup and teardown state-local, matching the C++ harness. Automatic
        # GC stays disabled while timing, so collect cycles explicitly afterward.
        cvcuda.clear_cache()
        run_fn = None
        try:
            run_fn = benchmark_func(state)
            if run_fn is not None:
                run_warmup(run_fn, warmup_iterations)
                state.exec(run_fn, sync=True)
        finally:
            run_fn = None
            cvcuda.clear_cache()
            gc.collect()

    # Copy function name BEFORE registration (nvbench captures name at register time)
    wrapped_benchmark.__name__ = benchmark_func.__name__
    b = bench.register(wrapped_benchmark)
    b.add_string_axis("InOutDataType", config.dtypes)
    register_axes_from_config(b, config)
    bench.run_all_benchmarks(bench_args)


def get_interpolation_type(interp_str: str):
    """
    Map interpolation string to CV-CUDA interpolation type.
    Equivalent to C++ GetInterpolationType function.

    Args:
        interp_str: String like "NEAREST", "LINEAR", "CUBIC", "AREA"

    Returns:
        CV-CUDA interpolation type (NVCV_INTERP_*)
    """
    interp_map = {
        "NEAREST": cvcuda.Interp.NEAREST,
        "LINEAR": cvcuda.Interp.LINEAR,
        "CUBIC": cvcuda.Interp.CUBIC,
        "AREA": cvcuda.Interp.AREA,
        "LANCZOS": cvcuda.Interp.LANCZOS,
        "GAUSSIAN": cvcuda.Interp.GAUSSIAN,
        "HAMMING": cvcuda.Interp.HAMMING,
        "BOX": cvcuda.Interp.BOX,
    }
    if interp_str not in interp_map:
        raise ValueError(
            f"Unsupported interpolation: {interp_str}. "
            f"Supported: {list(interp_map.keys())}"
        )
    return interp_map[interp_str]


def get_border_type(border_str: str):
    """
    Map border string to CV-CUDA border type.
    Equivalent to C++ GetBorderType function.

    Args:
        border_str: String like "CONSTANT", "REPLICATE", "REFLECT", "WRAP", "REFLECT101"

    Returns:
        CV-CUDA border type (NVCV_BORDER_*)
    """
    border_map = {
        "CONSTANT": cvcuda.Border.CONSTANT,
        "REPLICATE": cvcuda.Border.REPLICATE,
        "REFLECT": cvcuda.Border.REFLECT,
        "WRAP": cvcuda.Border.WRAP,
        "REFLECT101": cvcuda.Border.REFLECT101,
    }
    if border_str not in border_map:
        raise ValueError(
            f"Unsupported border: {border_str}. "
            f"Supported: {list(border_map.keys())}"
        )
    return border_map[border_str]


# --- Shared deterministic LCG fill ----------------------------------------
#
# Mirrors bench/cpp/BenchFillKernels.cu's randomFillTypedKernel<T>: same hash32
# (low-32(seed XOR idx) XOR high-32(seed), 3 rounds of Numerical Recipes LCG)
# and same sample_typed mapping (truncate-to-T for ints, -1 + s * 2/4294967295
# for floats). Both sides use the same seed → byte-identical buffers.
#
# Verified bit-for-bit equivalent to the C++ output across all supported types
# in /tmp/claude/dump_lcg_{cpp,py}.* (one-shot dev tool).

_LCG_SEED = 0x9E3779B97F4A7C15  # matches CppBenchUtils.hpp FillTensor seed

# (tag, cupy dtype, C type, is_float, kInv literal at C precision)
_LCG_TYPE_SPECS = (
    ("u8", "uint8", "unsigned char", False, None),
    ("u16", "uint16", "unsigned short", False, None),
    ("u32", "uint32", "unsigned int", False, None),
    ("i8", "int8", "signed char", False, None),
    ("i16", "int16", "short", False, None),
    ("i32", "int32", "int", False, None),
    ("f32", "float32", "float", True, "2.0f / 4294967295.0f"),
    ("f64", "float64", "double", True, "2.0 / 4294967295.0"),
)

_LCG_SRC_INT = r"""
extern "C" __global__
void lcg_fill_{tag}({T} *data, unsigned long long n_elements, unsigned long long seed)
{{
    unsigned long long idx = (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x
                             + (unsigned long long)threadIdx.x;
    if (idx >= n_elements) return;
    unsigned int s = (unsigned int)(seed ^ idx) ^ (unsigned int)(seed >> 32);
    s = s * 1664525u + 1013904223u;
    s = s * 1664525u + 1013904223u;
    s = s * 1664525u + 1013904223u;
    data[idx] = ({T})s;
}}
"""

_LCG_SRC_FLT = r"""
extern "C" __global__
void lcg_fill_{tag}({T} *data, unsigned long long n_elements, unsigned long long seed)
{{
    unsigned long long idx = (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x
                             + (unsigned long long)threadIdx.x;
    if (idx >= n_elements) return;
    unsigned int s = (unsigned int)(seed ^ idx) ^ (unsigned int)(seed >> 32);
    s = s * 1664525u + 1013904223u;
    s = s * 1664525u + 1013904223u;
    s = s * 1664525u + 1013904223u;
    const {T} kInv = ({T})({kInv_lit});
    data[idx] = ({T})(-1) + ({T})s * kInv;
}}
"""

_lcg_kernel_cache = {}


def _lcg_kernel_for(dtype):
    import cupy as cp

    np_dtype = cp.dtype(dtype)
    spec = next(s for s in _LCG_TYPE_SPECS if cp.dtype(s[1]) == np_dtype)
    tag, _, T, is_float, kInv_lit = spec
    if tag not in _lcg_kernel_cache:
        tmpl = _LCG_SRC_FLT if is_float else _LCG_SRC_INT
        src = tmpl.format(tag=tag, T=T, kInv_lit=kInv_lit or "")
        _lcg_kernel_cache[tag] = cp.RawKernel(src, f"lcg_fill_{tag}")
    return _lcg_kernel_cache[tag]


def _lcg_fill(arr, seed: int = _LCG_SEED) -> None:
    """Fill `arr` (a cupy ndarray of any shape) in place with the bit-deterministic
    LCG bytes. Output is byte-identical to BenchFillKernels' launchRandomFillTyped<T>
    with the same seed."""
    import numpy as np

    flat = arr.ravel()
    kernel = _lcg_kernel_for(arr.dtype)
    n = flat.size
    threads = 256
    blocks = (n + threads - 1) // threads
    kernel((blocks,), (threads,), (flat, np.uint64(n), np.uint64(seed)))


# --- Shared deterministic checkerboard fill --------------------------------

_CHECKERBOARD_TYPE_SPECS = (
    ("u8", "uint8", "unsigned char", "255"),
    ("u16", "uint16", "unsigned short", "65535"),
    ("u32", "uint32", "unsigned int", "4294967295u"),
    ("i8", "int8", "signed char", "127"),
    ("i16", "int16", "short", "32767"),
    ("i32", "int32", "int", "2147483647"),
    ("f32", "float32", "float", "1.0f"),
    ("f64", "float64", "double", "1.0"),
)

_CHECKERBOARD_SRC = r"""
extern "C" __global__
void checkerboard_fill_{tag}({T} *data,
                             unsigned long long n_elements,
                             unsigned long long d0,
                             unsigned long long d1,
                             unsigned long long d2,
                             unsigned long long d3,
                             unsigned long long s0,
                             unsigned long long s1,
                             unsigned long long s2,
                             unsigned long long s3,
                             int rank)
{{
    unsigned long long idx = (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x
                             + (unsigned long long)threadIdx.x;
    if (idx >= n_elements) return;

    unsigned long long tmp = idx;
    unsigned long long i0 = 0, i1 = 0, i2 = 0, i3 = 0;
    unsigned int parity = 0u;
    if (rank == 4)
    {{
        i3 = tmp % d3; tmp /= d3;
        i2 = tmp % d2; tmp /= d2;
        i1 = tmp % d1; tmp /= d1;
        i0 = tmp % d0;
    }}
    else if (rank == 3)
    {{
        i2 = tmp % d2; tmp /= d2;
        i1 = tmp % d1; tmp /= d1;
        i0 = tmp % d0;
    }}
    else
    {{
        i1 = tmp % d1; tmp /= d1;
        i0 = tmp % d0;
    }}

    parity = (unsigned int)((i0 + i1 + i2 + i3) & 1ull);
    data[i0 * s0 + i1 * s1 + i2 * s2 + i3 * s3]
        = parity ? ({T})({hi}) : ({T})0;
}}
"""

_checkerboard_kernel_cache = {}


def _checkerboard_kernel_for(dtype):
    import cupy as cp

    np_dtype = cp.dtype(dtype)
    spec = next(s for s in _CHECKERBOARD_TYPE_SPECS if cp.dtype(s[1]) == np_dtype)
    tag, _, T, hi = spec
    if tag not in _checkerboard_kernel_cache:
        src = _CHECKERBOARD_SRC.format(tag=tag, T=T, hi=hi)
        _checkerboard_kernel_cache[tag] = cp.RawKernel(src, f"checkerboard_fill_{tag}")
    return _checkerboard_kernel_cache[tag]


def _checkerboard_fill(arr) -> None:
    """Fill `arr` in place with the per-coordinate checkerboard pattern used by
    benchutils::CheckerboardValues<T>, without materializing temporary grids."""
    import numpy as np

    shape = arr.shape
    if len(shape) not in (2, 3, 4):
        raise ValueError(f"checkerboard not supported for shape {shape}")

    dims = list(shape) + [1] * (4 - len(shape))
    strides = [stride // arr.itemsize for stride in arr.strides]
    strides += [0] * (4 - len(strides))
    kernel = _checkerboard_kernel_for(arr.dtype)
    n = arr.size
    threads = 256
    blocks = (n + threads - 1) // threads
    kernel(
        (blocks,),
        (threads,),
        (
            arr,
            np.uint64(n),
            np.uint64(dims[0]),
            np.uint64(dims[1]),
            np.uint64(dims[2]),
            np.uint64(dims[3]),
            np.uint64(strides[0]),
            np.uint64(strides[1]),
            np.uint64(strides[2]),
            np.uint64(strides[3]),
            np.int32(len(shape)),
        ),
    )


def _checkerboard_fill_image(arr) -> None:
    """Fill an image with checkerboard parity based on pixel coordinates."""
    if len(arr.shape) == 3 and arr.shape[-1] > 1:
        for channel in range(arr.shape[-1]):
            _checkerboard_fill(arr[..., channel])
    else:
        _checkerboard_fill(arr)


def create_tensor(
    shape: Tuple[int, ...],
    dtype,
    device: int = 0,
    layout: str = "NHWC",
    fill_mode: Union[str, float, int, Tuple, list] = "random",
) -> cvcuda.Tensor:
    """
    Create a tensor for benchmarking with specified fill pattern.
    Equivalent to C++ FillTensor function.

    Uses CuPy for array creation.

    Args:
        shape: Tensor shape tuple (e.g., (1, 1080, 1920, 1))
        dtype: cvcuda.Type or dtype string (e.g., cvcuda.Type.U8, "uint8", "float32")
        device: Device ID
        layout: Tensor layout ("NHWC", "NCHW", "NC", "N", etc.)
        fill_mode: How to fill the tensor:
            - "random": Random values (0-255 for uint8, 0-1 for float)
            - numeric value: Fill with specific constant
            - tuple/list: Broadcast to shape (e.g., [3, 5] for shape (N, 2) creates N rows of [3, 5])

    Returns:
        CV-CUDA tensor with data filled according to fill_mode
    """
    import cupy as cp

    # Convert string to cvcuda.Type if needed
    if isinstance(dtype, str):
        cvcuda_dtype = get_dtype(dtype)
    else:
        cvcuda_dtype = dtype

    # Map cvcuda.Type to cupy dtype
    dtype_map = {
        cvcuda.Type.U8: cp.uint8,
        cvcuda.Type.U16: cp.uint16,
        cvcuda.Type.U32: cp.uint32,
        cvcuda.Type.S8: cp.int8,
        cvcuda.Type.S16: cp.int16,
        cvcuda.Type.S32: cp.int32,
        cvcuda.Type.F32: cp.float32,
        cvcuda.Type.F64: cp.float64,
    }
    cupy_dtype = dtype_map.get(cvcuda_dtype, cp.float32)

    # Determine if this is an integer type
    is_integer = cvcuda_dtype in [
        cvcuda.Type.U8,
        cvcuda.Type.U16,
        cvcuda.Type.U32,
        cvcuda.Type.S8,
        cvcuda.Type.S16,
        cvcuda.Type.S32,
    ]

    # Set CuPy device
    with cp.cuda.Device(device):
        # Fill tensor according to fill_mode
        if fill_mode == "lcg":
            # Deterministic varied fill that's byte-identical to the C++ side's
            # launchRandomFillTyped (BenchFillKernels.cu). Use this in place of
            # "random" whenever the C++ bench uses RandomValues<T>(): it gives
            # the same statistical properties (uniform over full type range for
            # ints, [-1, +1] for floats) but produces matching bytes.
            array = cp.empty(shape, dtype=cupy_dtype)
            _lcg_fill(array, _LCG_SEED)
        elif fill_mode == "random":
            # Generate random data
            if is_integer:
                # Integer types: random in valid range
                max_val = 256 if cvcuda_dtype == cvcuda.Type.U8 else 65536
                array = cp.random.randint(0, max_val, shape, dtype=cupy_dtype)
            else:
                # Float types: random in [0, 1]
                array = cp.random.random(shape).astype(cupy_dtype)
        elif fill_mode == "checkerboard":
            # Per-element checkerboard: (sum-of-coords & 1) ? hi : lo. Matches the
            # C++ helper benchutils::CheckerboardValues<T> in CppBenchUtils.hpp,
            # which uses (x + y + z + w) & 1 across all dims (incl. channel).
            #
            # Channel must participate in the alternation: data-dependent ops
            # like RGB2HSV branch on min(R,G,B)==max(R,G,B). A per-pixel pattern
            # (broadcasting across C) makes every pixel either pure-black or
            # pure-white, hitting the kernel's uniform-RGB fast path and giving
            # Python a ~10% spurious speedup vs the per-element C++ fill.
            array = cp.empty(shape, dtype=cupy_dtype)
            _checkerboard_fill(array)
        elif fill_mode == "gradient_h":
            # Match C++ benchmark image patterns for layout-parity runs.
            if len(shape) != len(layout):
                raise ValueError(
                    f"gradient_h shape/layout mismatch: shape={shape}, layout={layout}"
                )
            try:
                width_axis = layout.index("W")
            except ValueError as exc:
                raise ValueError(
                    f"gradient_h not supported for layout {layout}"
                ) from exc

            W = shape[width_axis]
            gradient = (255 - (cp.arange(W, dtype=cp.int32) % 256)).astype(cupy_dtype)
            if not is_integer:
                gradient = gradient / 255.0
            gradient_shape = [1] * len(shape)
            gradient_shape[width_axis] = W
            array = cp.broadcast_to(
                gradient.reshape(tuple(gradient_shape)), shape
            ).copy()
        elif isinstance(fill_mode, dict) and "random_int" in fill_mode:
            # Random integers in specified range (matching C++ RandomValues<T>(low, high))
            low, high = fill_mode["random_int"]
            array = cp.random.randint(low, high + 1, shape, dtype=cupy_dtype)
        elif isinstance(fill_mode, (list, tuple)):
            # Broadcast tuple/list to shape (e.g., [3, 5] repeated for each row)
            # Convert to array and broadcast/tile to match shape
            fill_array = cp.array(fill_mode, dtype=cupy_dtype)
            # If shape is (N, len(fill_mode)), tile the fill_array N times
            if len(shape) == 2 and shape[1] == len(fill_mode):
                array = cp.tile(fill_array, (shape[0], 1))
            else:
                # General broadcast
                array = cp.broadcast_to(fill_array, shape).copy()
        else:
            # Fill with constant value
            array = cp.full(shape, fill_mode, dtype=cupy_dtype)

    return cvcuda.as_tensor(array, layout)


def create_image_batch_varshape(
    shape: Tuple[int, ...],
    size_variation: int,
    img_format,
    dtype=None,
    device: int = 0,
    fill_mode: Union[str, float, int] = "random",
    batch=None,
):
    """
    Create an ImageBatchVarShape with specified fill pattern.
    Equivalent to C++ FillImageBatch function.

    Args:
        shape: Image batch shape (N, H, W) or (N, H, W, C)
            - N: batch size
            - H: base height for all images (or fixed height if size_variation=0)
            - W: base width for all images (or fixed width if size_variation=0)
            - C: number of channels (optional, defaults to 1)
        size_variation: Size variation parameter:
            - 0: All images same size (base_height x base_width)
            - >0: Variable sizes around base, ±size_variation per image
        img_format: cvcuda.Format for images (e.g., Format.U8, Format.RGB8)
        dtype: cvcuda.Type or dtype string (e.g., cvcuda.Type.U8, "uint8"). If None, inferred from img_format.
        device: Device ID
        fill_mode: How to fill the images:
            - "random": Random values (0-255 for uint8, 0-1 for float)
            - numeric value: Fill with specific constant
        batch: Optional preallocated cvcuda.ImageBatchVarShape. Use this when
            multiple batches must allocate metadata before any image storage.

    Returns:
        cvcuda.ImageBatchVarShape with images filled according to fill_mode

    Example:
        >>> # Create batch with 32 images, 1080x1920 base, random RGB data
        >>> batch = create_image_batch_varshape(
        ...     (32, 1080, 1920, 3), 0, cvcuda.Format.RGB8,
        ...     dtype=cvcuda.Type.U8, device=0, fill_mode="random"
        ... )
    """
    import cupy as cp

    # Parse shape
    if len(shape) == 3:
        batch_size, base_height, base_width = shape
        num_channels = 1
    elif len(shape) == 4:
        batch_size, base_height, base_width, num_channels = shape
    else:
        raise ValueError(f"shape must be (N, H, W) or (N, H, W, C), got {shape}")

    # Convert string to cvcuda.Type if needed (matches create_tensor pattern)
    if dtype is None:
        # Check if format suggests uint8 or float
        format_name = str(img_format)
        if "F32" in format_name or "F64" in format_name:
            cvcuda_dtype = cvcuda.Type.F32
        else:
            cvcuda_dtype = cvcuda.Type.U8
    elif isinstance(dtype, str):
        cvcuda_dtype = get_dtype(dtype)
    else:
        cvcuda_dtype = dtype

    # Map cvcuda.Type to cupy dtype
    dtype_map = {
        cvcuda.Type.U8: cp.uint8,
        cvcuda.Type.U16: cp.uint16,
        cvcuda.Type.U32: cp.uint32,
        cvcuda.Type.S8: cp.int8,
        cvcuda.Type.S16: cp.int16,
        cvcuda.Type.S32: cp.int32,
        cvcuda.Type.F32: cp.float32,
        cvcuda.Type.F64: cp.float64,
    }
    cupy_dtype = dtype_map.get(cvcuda_dtype, cp.uint8)

    # Determine if this is an integer type
    is_integer = cvcuda_dtype in [
        cvcuda.Type.U8,
        cvcuda.Type.U16,
        cvcuda.Type.U32,
        cvcuda.Type.S8,
        cvcuda.Type.S16,
        cvcuda.Type.S32,
    ]

    # Helper to calculate image size for each index (matches C++ FillImageBatch)
    def get_image_size(i):
        if size_variation == 0:
            return base_height, base_width
        else:
            img_h = base_height + (
                i * size_variation
                if i < batch_size // 2
                else -(i - batch_size // 2) * size_variation
            )
            img_w = base_width + (
                i * size_variation
                if i < batch_size // 2
                else -(i - batch_size // 2) * size_variation
            )
            return max(1, img_h), max(1, img_w)

    if batch is None:
        batch = cvcuda.ImageBatchVarShape(batch_size)
    elif batch.capacity != batch_size:
        raise ValueError(
            f"batch capacity ({batch.capacity}) does not match shape ({batch_size})"
        )
    elif len(batch) != 0:
        raise ValueError(f"preallocated batch must be empty, got {len(batch)} images")

    # Set CuPy device (matches create_tensor pattern)
    with cp.cuda.Device(device):
        for i in range(batch_size):
            img_h, img_w = get_image_size(i)

            def make_cupy_data(img_shape):
                # Generate fill data according to fill_mode
                if fill_mode == "lcg":
                    # Per-image deterministic LCG fill, byte-identical to the C++
                    # FillImageBatch GPU fast path (BenchFillKernels.cu): seed is
                    # _LCG_SEED + i to match the per-image offset on the C++ side.
                    cupy_data = cp.empty(img_shape, dtype=cupy_dtype)
                    _lcg_fill(cupy_data, _LCG_SEED + i)
                elif fill_mode == "random":
                    if is_integer:
                        max_val = 256 if cvcuda_dtype == cvcuda.Type.U8 else 65536
                        cupy_data = cp.random.randint(
                            0, max_val, img_shape, dtype=cupy_dtype
                        )
                    else:
                        cupy_data = cp.random.random(img_shape).astype(cupy_dtype)
                elif fill_mode == "gradient_h":
                    # Match C++ benchmark image patterns for per-image parity runs.
                    gradient = (255 - (cp.arange(img_w, dtype=cp.int32) % 256)).astype(
                        cupy_dtype
                    )
                    if not is_integer:
                        gradient = gradient / 255.0  # normalize to [0, 1] for float
                    if len(img_shape) == 2:
                        cupy_data = cp.broadcast_to(
                            gradient.reshape(1, img_w), img_shape
                        ).copy()
                    else:
                        cupy_data = cp.broadcast_to(
                            gradient.reshape(1, img_w, 1), img_shape
                        ).copy()
                elif isinstance(fill_mode, dict) and "random_int" in fill_mode:
                    # Random integers in specified range (matching C++ RandomValues<T>(low, high))
                    low, high = fill_mode["random_int"]
                    cupy_data = cp.random.randint(
                        low, high + 1, img_shape, dtype=cupy_dtype
                    )
                else:
                    # Fill with constant value
                    cupy_data = cp.full(img_shape, fill_mode, dtype=cupy_dtype)

                return cupy_data

            def fill_image_data(target):
                if fill_mode == "checkerboard":
                    _checkerboard_fill_image(target)
                elif isinstance(fill_mode, (int, float)):
                    target.fill(fill_mode)
                else:
                    target[...] = make_cupy_data(target.shape)

            num_planes = img_format.planes
            if len(shape) == 4 and num_planes > 1 and num_channels != num_planes:
                raise ValueError(
                    f"shape channel count ({num_channels}) does not match "
                    f"{img_format} plane count ({num_planes})"
                )
            if num_planes == 1:
                # Allocate via NVCV so the row pitch is padded to NVCV's aligned
                # stride (matching the C++ FillImageBatch images). Wrapping a dense
                # cupy array instead leaves unaligned rows when W*channels is not
                # pitch-aligned, which slows memory-bound kernels on the Python side
                # only and produces a spurious parity gap.
                img = cvcuda.Image((img_w, img_h), img_format)
                view = cp.asarray(img.cuda())
                fill_image_data(view)
            else:
                # Keep the same per-plane deterministic data as the previous
                # as_image([plane...]) path, but use NVCV allocation so planar
                # images get the same row pitch and allocator behavior as C++.
                img = cvcuda.Image((img_w, img_h), img_format)
                view = cp.asarray(img.cuda())
                for plane in range(num_planes):
                    fill_image_data(view[plane, ...])

            batch.pushback(img)

    return batch
