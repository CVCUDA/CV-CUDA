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

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from _internal import check_bench_sku as cbs
from _internal.baselines import validate_sku_map_cuda_coverage
from _internal.check_bench_sku import (
    EXIT_ELIGIBLE,
    EXIT_ERROR,
    EXIT_INELIGIBLE,
    _PROBE_LOCK_FAILED,
    _PROBE_NO_TARGET,
    _PROBE_OK,
)


def test_script_mode_imports_and_fails_open_without_nvidia_smi(tmp_path):
    bench_dir = Path(__file__).resolve().parent.parent
    env = os.environ.copy()
    env["PATH"] = str(tmp_path)

    result = subprocess.run(
        [sys.executable, "bench/_internal/check_bench_sku.py", "--cuda-major", "13"],
        cwd=bench_dir.parent,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == EXIT_ELIGIBLE, result.stderr
    assert "nvidia-smi unavailable" in result.stdout


def test_committed_sku_map_has_cuda_12_and_13_for_each_gpu_power_profile():
    sku_map_path = (
        Path(cbs.__file__).resolve().parent.parent / "config" / "sku_map.json"
    )
    sku_map = cbs.load_sku_map(sku_map_path, strict=True)
    validate_sku_map_cuda_coverage(sku_map)


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
    monkeypatch.setattr("sys.argv", ["check_bench_sku.py", "--cuda-major", "12"])
    assert cbs.main() == EXIT_ERROR


def test_main_fail_open_when_binary_absent(monkeypatch):
    # nvidia-smi absent -> _gpu_name() is None -> fail-open (eligible).
    monkeypatch.setattr(cbs, "_gpu_name", lambda: None)
    monkeypatch.setattr("sys.argv", ["check_bench_sku.py", "--cuda-major", "12"])
    assert cbs.main() == EXIT_ELIGIBLE


# Fixture profile used by the monkeypatched SKU map below.
_H100_NAME = "NVIDIA H100 PCIe"
_H100_POWER = 350


def _patch_h100(monkeypatch):
    """Patch a stable H100 profile for preflight behavior tests."""
    monkeypatch.setattr(cbs, "_gpu_name", lambda: _H100_NAME)
    monkeypatch.setattr(cbs, "_power_cap_w", lambda: _H100_POWER)
    monkeypatch.setattr(
        cbs,
        "load_sku_map",
        lambda *_args, **_kwargs: {
            (_H100_NAME, _H100_POWER, 1095, 12): "H100_CUDA12",
            (_H100_NAME, _H100_POWER, 1095, 13): "H100_CUDA13",
        },
    )
    monkeypatch.setattr("sys.argv", ["check_bench_sku.py", "--cuda-major", "12"])


def test_main_probe_matches_exits_eligible(monkeypatch):
    """The preflight probes only the clock configured for this SKU."""
    _patch_h100(monkeypatch)
    targets = []

    def _probe(target):
        targets.append(target)
        return _PROBE_OK

    monkeypatch.setattr(cbs, "_probe_sm_clock", _probe)
    assert cbs.main() == EXIT_ELIGIBLE
    assert targets == [1095]


@pytest.mark.parametrize("cuda_major", [12, 13])
@pytest.mark.parametrize("power_cap", [310, 350])
def test_configured_h100_accepts_any_power_cap(monkeypatch, cuda_major, power_cap):
    monkeypatch.setattr(cbs, "_gpu_name", lambda: _H100_NAME)
    monkeypatch.setattr(cbs, "_power_cap_w", lambda: power_cap)
    targets = []

    def probe(target):
        targets.append(target)
        return _PROBE_OK

    monkeypatch.setattr(cbs, "_probe_sm_clock", probe)
    monkeypatch.setattr(
        "sys.argv", ["check_bench_sku.py", "--cuda-major", str(cuda_major)]
    )
    assert cbs.main() == EXIT_ELIGIBLE
    assert targets == [1005]


def test_required_clock_filters_the_full_profile():
    sku_map = {
        (name, power, clock, cuda): f"sku_{index}"
        for index, (name, power, clock, cuda) in enumerate(
            [
                (_H100_NAME, 350, 1005, 13),
                ("NVIDIA A100-PCIE-40GB", 350, 1095, 13),
                (_H100_NAME, 310, 900, 13),
                (_H100_NAME, 350, 1095, 12),
            ]
        )
    }

    assert (
        cbs.required_clock_for_profile(
            sku_map,
            gpu_name=_H100_NAME,
            power_cap_w=350,
            cuda_major=13,
        )
        == 1005
    )


def test_main_missing_hardware_profile_reports_allowed_profiles(monkeypatch, capsys):
    _patch_h100(monkeypatch)
    monkeypatch.setattr(cbs, "_power_cap_w", lambda: 310)
    monkeypatch.setattr(
        cbs,
        "_probe_sm_clock",
        lambda _target: pytest.fail("clock probe must not run"),
    )

    assert cbs.main() == EXIT_INELIGIBLE
    assert "allowed for this model: [(350, 1095, 12), (350, 1095, 13)]" in (
        capsys.readouterr().err
    )


def test_main_cuda_major_mismatch_fails_fast_as_configuration_error(
    monkeypatch, capsys
):
    monkeypatch.setattr(cbs, "_gpu_name", lambda: _H100_NAME)
    monkeypatch.setattr(cbs, "_power_cap_w", lambda: _H100_POWER)
    monkeypatch.setattr(
        cbs,
        "load_sku_map",
        lambda *_args, **_kwargs: {(_H100_NAME, _H100_POWER, 1095, 13): "H100_CUDA13"},
    )
    monkeypatch.setattr(
        cbs,
        "_probe_sm_clock",
        lambda _target: pytest.fail("clock probe must not run"),
    )
    monkeypatch.setattr("sys.argv", ["check_bench_sku.py", "--cuda-major", "12"])

    assert cbs.main() == EXIT_ERROR
    assert "configured CUDA majors: [13]" in capsys.readouterr().err


def test_main_missing_power_cap_exits_ineligible(monkeypatch):
    _patch_h100(monkeypatch)
    monkeypatch.setattr(cbs, "_power_cap_w", lambda: None)
    monkeypatch.setattr(
        cbs,
        "required_clock_for_profile",
        lambda *_args, **_kwargs: pytest.fail("profile lookup must not run"),
    )

    assert cbs.main() == EXIT_INELIGIBLE


def test_main_ambiguous_clock_targets_exit_error(monkeypatch, tmp_path, capsys):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "sku_map.json").write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "gpu_name": _H100_NAME,
                        "power_cap_w": _H100_POWER,
                        "locked_sm_clock_mhz": clock,
                        "cuda_major": 13,
                        "stem": f"H100_{clock}",
                    }
                    for clock in (1005, 1095)
                ]
            }
        )
    )
    monkeypatch.setattr(cbs, "__file__", str(tmp_path / "_internal" / "check.py"))
    monkeypatch.setattr("sys.argv", ["check_bench_sku.py", "--cuda-major", "13"])
    monkeypatch.setattr(
        cbs,
        "_probe_sm_clock",
        lambda _target: pytest.fail("clock probe must not run"),
    )

    assert cbs.main() == EXIT_ERROR
    assert "multiple SM clock targets" in capsys.readouterr().err


def test_main_unsupported_target_exits_ineligible(monkeypatch):
    _patch_h100(monkeypatch)
    monkeypatch.setattr(cbs, "_probe_sm_clock", lambda _target: _PROBE_NO_TARGET)
    assert cbs.main() == EXIT_INELIGIBLE


def test_main_probe_lock_fails_exits_error(monkeypatch):
    _patch_h100(monkeypatch)
    monkeypatch.setattr(cbs, "_probe_sm_clock", lambda _target: _PROBE_LOCK_FAILED)
    assert cbs.main() == EXIT_ERROR


def test_main_unknown_gpu_skips_probe(monkeypatch):
    probe_called = []

    def _track_probe(_target):
        probe_called.append(True)
        return _PROBE_OK

    monkeypatch.setattr(cbs, "_gpu_name", lambda: "NVIDIA L40")
    monkeypatch.setattr(cbs, "_power_cap_w", lambda: 300)
    monkeypatch.setattr(cbs, "_probe_sm_clock", _track_probe)
    monkeypatch.setattr("sys.argv", ["check_bench_sku.py", "--cuda-major", "12"])

    assert cbs.main() == EXIT_ELIGIBLE
    assert not probe_called, "probe must not run for a GPU not in sku_map.json"


def _make_run_privileged_tracker(lock_succeeds=True, lock_detail=""):
    """Returns a fake _run_privileged that records calls and controls -lgc outcome."""
    calls = []

    def _fake(args, timeout=10):
        calls.append(list(args))
        if "-lgc" in args:
            return cbs.ClockResult(lock_succeeds, "" if lock_succeeds else lock_detail)
        return cbs.ClockResult(True)  # -acp, -pm, -rgc all succeed

    _fake.calls = calls
    return _fake


def test_probe_reports_the_driver_explanation_for_a_failed_lock(monkeypatch, capsys):
    """A board-side clock failure leaves nvidia-smi's message as its only clue."""
    tracker = _make_run_privileged_tracker(
        lock_succeeds=False, lock_detail="Insufficient Permissions"
    )
    monkeypatch.setattr(cbs, "_run_privileged", tracker)
    monkeypatch.setattr(cbs, "_supported_sm_clocks", lambda: {1095, 1005})

    assert cbs._probe_sm_clock(1095) == cbs._PROBE_LOCK_FAILED
    assert "Insufficient Permissions" in capsys.readouterr().err


def test_probe_reset_called_when_lock_succeeds(monkeypatch):
    tracker = _make_run_privileged_tracker(lock_succeeds=True)
    monkeypatch.setattr(cbs, "_run_privileged", tracker)
    monkeypatch.setattr(cbs, "_supported_sm_clocks", lambda: {1095, 1005, 900, 750})

    status = cbs._probe_sm_clock(1095)

    assert status == _PROBE_OK
    assert ["-rgc", "-i", "0"] in tracker.calls


def test_main_reset_failure_after_lock_exits_ineligible(monkeypatch):
    _patch_h100(monkeypatch)
    tracker = _make_run_privileged_tracker(lock_succeeds=True)

    def _fail_reset(args, timeout=10):
        if "-rgc" in args:
            tracker.calls.append(list(args))
            return cbs.ClockResult(False)
        return tracker(args, timeout)

    monkeypatch.setattr(cbs, "_run_privileged", _fail_reset)
    monkeypatch.setattr(cbs, "_supported_sm_clocks", lambda: {1095})

    assert cbs.main() == EXIT_INELIGIBLE
    assert tracker.calls.count(["-rgc", "-i", "0"]) == cbs._CLOCK_RESET_ATTEMPTS


def test_probe_retries_transient_reset_failure(monkeypatch):
    calls = []

    def _transient_reset(args, timeout=10):
        calls.append(list(args))
        return cbs.ClockResult(
            "-rgc" not in args or calls.count(["-rgc", "-i", "0"]) == 2
        )

    monkeypatch.setattr(cbs, "_run_privileged", _transient_reset)
    monkeypatch.setattr(cbs, "_supported_sm_clocks", lambda: {1095})

    assert cbs._probe_sm_clock(1095) == _PROBE_OK
    assert calls.count(["-rgc", "-i", "0"]) == 2


def test_probe_does_not_reset_when_lock_fails(monkeypatch):
    tracker = _make_run_privileged_tracker(lock_succeeds=False)
    monkeypatch.setattr(cbs, "_run_privileged", tracker)
    monkeypatch.setattr(cbs, "_supported_sm_clocks", lambda: {1095, 1005, 900, 750})

    status = cbs._probe_sm_clock(1095)

    assert status == _PROBE_LOCK_FAILED
    assert ["-rgc", "-i", "0"] not in tracker.calls


def test_main_lock_failure_exits_environmental_without_reset(monkeypatch):
    _patch_h100(monkeypatch)
    calls = []

    def _fail_lock_and_reset(args, timeout=10):
        calls.append(list(args))
        return cbs.ClockResult("-lgc" not in args and "-rgc" not in args)

    monkeypatch.setattr(cbs, "_run_privileged", _fail_lock_and_reset)
    monkeypatch.setattr(cbs, "_supported_sm_clocks", lambda: {1095})

    assert cbs.main() == EXIT_ERROR
    assert ["-rgc", "-i", "0"] not in calls


def test_probe_does_not_reset_when_target_is_unsupported(monkeypatch):
    tracker = _make_run_privileged_tracker(lock_succeeds=True)
    monkeypatch.setattr(cbs, "_run_privileged", tracker)
    monkeypatch.setattr(cbs, "_supported_sm_clocks", lambda: {300, 200})

    status = cbs._probe_sm_clock(1095)

    assert status == _PROBE_NO_TARGET
    assert ["-rgc", "-i", "0"] not in tracker.calls


def test_probe_does_not_reset_on_smi_error(monkeypatch):
    tracker = _make_run_privileged_tracker(lock_succeeds=True)
    monkeypatch.setattr(cbs, "_run_privileged", tracker)
    monkeypatch.setattr(
        cbs,
        "_supported_sm_clocks",
        lambda: (_ for _ in ()).throw(cbs.NvidiaSmiError("query timeout")),
    )

    with pytest.raises(cbs.NvidiaSmiError):
        cbs._probe_sm_clock(1095)

    assert ["-rgc", "-i", "0"] not in tracker.calls


def test_probe_rejects_unparseable_supported_clocks(monkeypatch):
    tracker = _make_run_privileged_tracker()
    monkeypatch.setattr(cbs, "_run_privileged", tracker)
    monkeypatch.setattr(cbs, "_supported_sm_clocks", lambda: None)

    with pytest.raises(cbs.NvidiaSmiError, match="could not parse"):
        cbs._probe_sm_clock(1095)

    assert ["-rgc", "-i", "0"] not in tracker.calls


def test_load_sku_map_ignores_explicit_routing_entries(tmp_path):
    # Explicit-routed SKUs (Jetson) carry no device metadata to match on, so
    # leaving them in the routing map raises a KeyError during the preflight.
    sku_map_path = tmp_path / "sku_map.json"
    sku_map_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "gpu_name": _H100_NAME,
                        "power_cap_w": _H100_POWER,
                        "locked_sm_clock_mhz": 1005,
                        "cuda_major": 12,
                        "stem": "H100_CUDA12",
                    },
                    {
                        "routing": "explicit",
                        "cuda_major": 12,
                        "stem": "Jetson_Orin_R36_2_MAXN_CUDA12",
                    },
                ]
            }
        )
    )

    assert cbs.load_sku_map(sku_map_path, strict=True) == {
        (_H100_NAME, _H100_POWER, 1005, 12): "H100_CUDA12"
    }
