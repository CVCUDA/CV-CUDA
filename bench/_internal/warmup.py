# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared runtime policy for capping benchmark warmup iterations."""

from __future__ import annotations

import os
from collections.abc import Mapping


WARMUP_CAP_ENV = "CVCUDA_BENCH_WARMUP_CAP"
_MAX_CXX_INT = 2_147_483_647


def parse_warmup_cap(raw_value: str) -> int:
    """Parse the nonnegative decimal cap accepted by both harnesses."""
    if not raw_value or any(ch < "0" or ch > "9" for ch in raw_value):
        raise ValueError(f"{WARMUP_CAP_ENV} must be a nonnegative decimal integer")

    cap = int(raw_value)
    if cap > _MAX_CXX_INT:
        raise ValueError(f"{WARMUP_CAP_ENV} must not exceed {_MAX_CXX_INT}")
    return cap


def resolve_warmup_iterations(
    configured_iterations: int, environ: Mapping[str, str] | None = None
) -> int:
    """Apply the optional process-level cap to a configured warmup count."""
    if configured_iterations <= 0:
        return configured_iterations

    env = os.environ if environ is None else environ
    raw_cap = env.get(WARMUP_CAP_ENV)
    if raw_cap is None:
        return configured_iterations
    return min(configured_iterations, parse_warmup_cap(raw_cap))
