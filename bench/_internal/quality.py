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

"""Shared benchmark quality thresholds.

The default values match the CI benchmark gate.  Runtime validation, committed
baseline verification, and baseline imports must all read from this module so
the quality policy cannot drift between collection and persistence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


_FLOAT_TOLERANCE = 1e-9


def exceeds_limit(value: float, limit: float) -> bool:
    return value > limit + _FLOAT_TOLERANCE


@dataclass(frozen=True)
class BenchmarkQualityCriteria:
    max_noise_pct: float
    max_perf_diff_pct: float
    max_perf_diff_us: float

    def noise_pct(self, gpu_time_us: float, gpu_noise_us: float) -> float:
        return gpu_noise_us / gpu_time_us * 100.0

    def parity_deltas(
        self, cpp_time_us: float, python_time_us: float
    ) -> Tuple[float, float]:
        return (
            (python_time_us / cpp_time_us - 1.0) * 100.0,
            python_time_us - cpp_time_us,
        )

    def noise_exceeds_limit(self, noise_pct: float) -> bool:
        return exceeds_limit(noise_pct, self.max_noise_pct)

    def relative_parity_exceeds_limit(self, diff_pct: float) -> bool:
        return exceeds_limit(abs(diff_pct), self.max_perf_diff_pct)

    def absolute_parity_exceeds_limit(self, diff_us: float) -> bool:
        return exceeds_limit(abs(diff_us), self.max_perf_diff_us)

    def parity_exceeds_limit(self, diff_pct: float, diff_us: float) -> bool:
        return self.relative_parity_exceeds_limit(
            diff_pct
        ) or self.absolute_parity_exceeds_limit(diff_us)


DEFAULT_BENCHMARK_QUALITY = BenchmarkQualityCriteria(
    max_noise_pct=10.0,
    max_perf_diff_pct=10.0,
    max_perf_diff_us=100.0,
)
