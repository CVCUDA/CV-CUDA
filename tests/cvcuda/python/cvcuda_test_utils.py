# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
import os
import sysconfig

import pytest


class _CudaStreamProtocol:
    def __init__(self, handle):
        self.handle = int(handle)

    def __cuda_stream__(self):
        return 0, self.handle


def as_cupy_stream(handle):
    """Borrow a CUDA stream handle without taking ownership of the stream."""
    import cupy

    if hasattr(cupy.cuda.Stream, "from_external"):
        return cupy.cuda.Stream.from_external(_CudaStreamProtocol(handle))
    return cupy.cuda.ExternalStream(int(handle))


def subprocess_timeout_scale() -> float:
    configured = os.environ.get("CVCUDA_TEST_TIMEOUT_SCALE")
    if configured is not None:
        try:
            scale = float(configured)
        except ValueError:
            pytest.fail(
                f"CVCUDA_TEST_TIMEOUT_SCALE must be a positive number, got {configured!r}"
            )
        if not math.isfinite(scale) or scale <= 0:
            pytest.fail(
                f"CVCUDA_TEST_TIMEOUT_SCALE must be positive, got {configured!r}"
            )
        return scale

    config_args = sysconfig.get_config_var("CONFIG_ARGS") or ""
    if "--with-thread-sanitizer" in config_args or "TSAN_OPTIONS" in os.environ:
        return 10.0
    return 1.0
