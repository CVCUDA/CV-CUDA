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

import subprocess

import pytest

from _internal import check_bench_sku as cbs
from _internal.check_bench_sku import (
    EXIT_ELIGIBLE,
    EXIT_ERROR,
    EXIT_INELIGIBLE,
    decide,
    predict_locked_clock,
)

ENTRIES = [
    {"gpu_name": "NVIDIA H100 PCIe", "power_cap_w": 350, "locked_sm_clock_mhz": 1095},
    {
        "gpu_name": "NVIDIA A100-PCIE-40GB",
        "power_cap_w": 250,
        "locked_sm_clock_mhz": 1095,
    },
]


@pytest.mark.parametrize(
    "name,power,clock,expected",
    [
        ("NVIDIA H100 PCIe", 350, 1095, EXIT_ELIGIBLE),  # canonical H100
        ("NVIDIA H100 PCIe", 350, 1005, EXIT_INELIGIBLE),  # clamp silicon
        ("NVIDIA H100 PCIe", 310, 1095, EXIT_INELIGIBLE),  # low-TDP variant
        ("NVIDIA A100-PCIE-40GB", 250, 1095, EXIT_ELIGIBLE),
        ("NVIDIA L40", 300, 1095, EXIT_ELIGIBLE),  # not in map -> fail-open
        (None, None, None, EXIT_ELIGIBLE),  # nvidia-smi unavailable -> fail-open
    ],
)
def test_decide(name, power, clock, expected):
    code, msg = decide(name, power, clock, ENTRIES)
    assert code == expected
    assert msg  # always explains the decision


def test_decide_ineligible_lists_allowed_skus():
    code, msg = decide("NVIDIA H100 PCIe", 310, 1095, ENTRIES)
    assert code == EXIT_INELIGIBLE
    assert "(350, 1095)" in msg  # surfaces the allowed combo for triage


@pytest.mark.parametrize(
    "supported,expected",
    [
        ({750, 900, 1005, 1095}, 1095),  # 1095 available -> highest preferred
        ({750, 900, 1005}, 1005),  # 1095 absent -> clamps to 1005
        ({750, 900}, 900),
        (None, 1095),  # cannot query -> request preferred[0] (run_bench behavior)
        (set(), None),  # nothing supported in preferred list
    ],
)
def test_predict_locked_clock(supported, expected):
    assert predict_locked_clock(supported, preferred=[1095, 1005, 900, 750]) == expected


def test_smi_returns_none_when_binary_absent(monkeypatch):
    def _raise(*a, **k):
        raise FileNotFoundError()

    monkeypatch.setattr(cbs.subprocess, "run", _raise)
    assert cbs._smi("--query-gpu=gpu_name") is None


def test_smi_raises_on_nonzero_exit(monkeypatch):
    class _R:
        returncode = 1
        stdout = ""
        stderr = "boom"

    monkeypatch.setattr(cbs.subprocess, "run", lambda *a, **k: _R())
    with pytest.raises(cbs.NvidiaSmiError):
        cbs._smi("--query-gpu=gpu_name")


def test_smi_raises_on_timeout(monkeypatch):
    def _timeout(*a, **k):
        raise subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=10)

    monkeypatch.setattr(cbs.subprocess, "run", _timeout)
    with pytest.raises(cbs.NvidiaSmiError):
        cbs._smi("--query-gpu=gpu_name")


def test_main_fails_fast_when_nvidia_smi_errors(monkeypatch):
    # nvidia-smi present but failing -> EXIT_ERROR (fail fast), not eligible.
    def _raise():
        raise cbs.NvidiaSmiError("boom")

    monkeypatch.setattr(cbs, "_gpu_name", _raise)
    assert cbs.main() == EXIT_ERROR


def test_main_fail_open_when_binary_absent(monkeypatch):
    # nvidia-smi absent -> _gpu_name() is None -> fail-open (eligible).
    monkeypatch.setattr(cbs, "_gpu_name", lambda: None)
    assert cbs.main() == EXIT_ELIGIBLE
