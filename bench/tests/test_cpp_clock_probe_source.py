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

"""Cross-language constants for the SM clock probes."""

import re
from pathlib import Path


BENCH_DIR = Path(__file__).resolve().parent.parent
CPP_DIR = BENCH_DIR / "cpp"


def _integer(source, name):
    match = re.search(rf"\b{name}\s*=\s*([\d'_]+)", source)
    assert match is not None, f"missing integer constant {name}"
    return int(match.group(1).replace("'", "").replace("_", ""))


def _string(source, name):
    match = re.search(rf'\b{name}\s*=\s*"([^"]+)"', source)
    assert match is not None, f"missing string constant {name}"
    return match.group(1)


def test_clock_probe_constants_match_python():
    source = (CPP_DIR / "BenchClockProbe.cu").read_text()
    python_source = (BENCH_DIR / "python" / "python_bench_utils.py").read_text()
    runner_source = (BENCH_DIR / "run_bench.py").read_text()

    for producer in (source, python_source):
        assert "CVCUDA_BENCH_EXPECT_SM_CLOCK_MHZ" in producer
    assert 'env["CVCUDA_BENCH_EXPECT_SM_CLOCK_MHZ"]' in runner_source

    python_tolerance = _integer(python_source, "_SM_CLOCK_MEASUREMENT_TOLERANCE_MHZ")
    assert _integer(source, "kToleranceMHz") == python_tolerance

    python_settle = re.search(
        r"start_ns\s*-\s*settle_start_ns\s*<\s*(\d+)ULL", python_source
    )
    python_sample = re.search(r"end_ns\s*-\s*start_ns\s*<\s*(\d+)ULL", python_source)
    assert python_settle is not None
    assert python_sample is not None
    assert _integer(source, "kSettleNs") == int(python_settle.group(1))
    assert _integer(source, "kSampleNs") == int(python_sample.group(1))
    assert _integer(source, "kThreads") == _integer(python_source, "_SM_CLOCK_THREADS")
    assert _integer(source, "kBlocksPerSm") == _integer(
        python_source, "_SM_CLOCK_BLOCKS_PER_SM"
    )

    for cpp_name, python_name in (
        ("kVerifiedPrefix", "_SM_CLOCK_VERIFIED_PREFIX"),
        ("kMismatchPrefix", "_SM_CLOCK_MISMATCH_PREFIX"),
        ("kVerificationErrorPrefix", "_SM_CLOCK_VERIFICATION_ERROR_PREFIX"),
        ("kDeviceErrorPrefix", "_SM_CLOCK_DEVICE_ERROR_PREFIX"),
    ):
        expected = _string(runner_source, python_name)
        assert _string(python_source, python_name) == expected
        assert _string(source, cpp_name) == expected
