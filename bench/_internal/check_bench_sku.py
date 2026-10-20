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

"""Pre-flight benchmark-SKU eligibility check for CI.

The benchmark regression check (``compare_to_baseline.py``) routes a run to a
per-SKU JSON baseline by device name, power cap, locked SM clock, and CUDA major
via ``bench/config/sku_map.json``. A null power cap in that map accepts any
measured power cap while retaining the model, clock, and CUDA checks.
This preflight checks that the hardware
profile has one exact baseline clock target, that the target is supported, and
that the driver accepts the lock request. ``clocks.current.sm`` is not a
validity oracle: it can report an idle P-state or the requested clock while
device code runs at another bin. ``run_bench.py`` therefore verifies the active
clock from device code before every benchmark state.

Exit codes:
  0  eligible (profile is in sku_map.json), or fail-open: the GPU model is not
     covered by the map, or nvidia-smi is *absent* (FileNotFoundError). These
     are genuinely "cannot/should not gate" cases — do not block scheduling.
  3  ineligible: the mapped GPU's power cap is unreadable, its power profile or
     supported clocks have no matching baseline, or its clock reset fails after
     a successful lock. These are node-local failures; callers should exclude
     this node and retry elsewhere.
  1  error: the SKU map is invalid or ambiguous, nvidia-smi is present but a
     query failed or timed out, the mapped GPU/power profile omits the requested
     CUDA major, supported-clock output cannot be parsed, or the clock request
     (``-lgc``) failed on all privilege levels. These are config, tooling, or
     environment-wide failures; callers should fail fast.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _internal.baselines import (  # noqa: E402
    BaselineError,
    MissingClockConfigurationError,
    MissingClockProfileError,
    SUPPORTED_CUDA_MAJORS,
    load_sku_map,
    profiles_for_gpu,
    required_clock_for_profile,
    sku_map_gpu_names,
)

EXIT_ELIGIBLE = 0
EXIT_ERROR = 1
EXIT_INELIGIBLE = 3

_PROBE_OK = "ok"
_PROBE_NO_TARGET = "no_target"  # node does not support its baseline clock
_PROBE_LOCK_FAILED = "lock_failed"  # -lgc failed on all privilege levels
_CLOCK_RESET_ATTEMPTS = 3


class NvidiaSmiError(RuntimeError):
    """nvidia-smi is present but a query failed — distinct from the binary being
    absent, so the caller can fail fast instead of silently passing eligibility."""


class ClockResetError(NvidiaSmiError):
    """A node-local failure to restore clocks after a successful probe lock."""


def _smi(*args):
    """Run nvidia-smi. Returns stdout on success, or None only when the binary
    is absent (FileNotFoundError) — the sole "can't check, skip" case. A
    present-but-failing nvidia-smi (non-zero exit or timeout) raises
    NvidiaSmiError so the caller fails fast instead of silently passing."""
    try:
        r = subprocess.run(
            ["nvidia-smi", *args], capture_output=True, text=True, timeout=10
        )
    except FileNotFoundError:
        return None
    except subprocess.TimeoutExpired as exc:
        raise NvidiaSmiError(f"nvidia-smi timed out for args {args}") from exc
    if r.returncode != 0:
        raise NvidiaSmiError(
            f"nvidia-smi failed for args {args}: rc={r.returncode}, "
            f"stderr={r.stderr.strip()}"
        )
    return r.stdout


def _gpu_name():
    out = _smi("--query-gpu=gpu_name", "--format=csv,noheader", "-i", "0")
    return out.strip().splitlines()[0].strip() if out else None


def _power_cap_w():
    out = _smi(
        "--query-gpu=power.max_limit", "--format=csv,noheader,nounits", "-i", "0"
    )
    if not out:
        return None
    try:
        return int(round(float(out.strip().splitlines()[0])))
    except (IndexError, ValueError):
        return None


def _supported_sm_clocks():
    """Graphics (SM) clocks the driver will accept for -lgc, via
    ``nvidia-smi -q -d SUPPORTED_CLOCKS``. Returns a set, or None on failure."""
    out = _smi("-q", "-d", "SUPPORTED_CLOCKS", "-i", "0")
    if not out:
        return None
    clocks = set()
    in_graphics = False
    for line in out.splitlines():
        s = line.strip()
        if s.startswith("Graphics"):
            in_graphics = True
            head = s.split(":", 1)
            if len(head) == 2 and head[1].strip():
                try:
                    clocks.add(int(head[1].split()[0]))
                except (IndexError, ValueError):
                    pass
        elif s.startswith("Memory") or s.startswith("SM"):
            in_graphics = False
        elif in_graphics and s.endswith("MHz"):
            try:
                clocks.add(int(s.split()[0]))
            except (IndexError, ValueError):
                pass
    return clocks or None


class ClockResult(NamedTuple):
    """Outcome of a clock command, carrying why the last attempt failed.

    A bare boolean loses the driver's own explanation, which is the only clue a
    board-side clock failure leaves behind in a CI log.
    """

    ok: bool
    detail: str = ""

    def __bool__(self):
        return self.ok


def _run_privileged(args, timeout=10):
    """Try ``sudo -n`` then unprivileged for the given nvidia-smi args.
    Returns a ``ClockResult``; truthy if either attempt succeeded (rc=0).
    Environmental failures (binary absent, timeout) are treated as failure."""
    detail = "nvidia-smi not available"
    for prefix in (["sudo", "-n"], []):
        try:
            r = subprocess.run(
                prefix + ["nvidia-smi"] + list(args),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except FileNotFoundError:
            continue
        except subprocess.TimeoutExpired:
            detail = f"nvidia-smi timed out after {timeout}s"
            continue
        if r.returncode == 0:
            return ClockResult(True)
        detail = (
            r.stderr or r.stdout or ""
        ).strip() or f"nvidia-smi exited {r.returncode}"
    return ClockResult(False, detail)


def _probe_sm_clock(target):
    """Apply the real SM clock request, then reset it.

    Mirrors run_bench.py's lock sequence. The benchmark verifies the active
    frequency from device code. Returns ``_PROBE_OK``, ``_PROBE_NO_TARGET``
    when the node does not support ``target``, or ``_PROBE_LOCK_FAILED`` when
    ``-lgc`` fails at every privilege level.
    """
    _run_privileged(["-acp", "UNRESTRICTED", "-i", "0"])  # best-effort prep
    _run_privileged(["-pm", "ENABLED", "-i", "0"])

    supported = _supported_sm_clocks()
    if supported is None:
        raise NvidiaSmiError("could not parse supported SM clocks")
    if target not in supported:
        return _PROBE_NO_TARGET

    lock = _run_privileged(["-lgc", f"{target},{target}", "-i", "0"])
    if not lock.ok:
        # nvidia-smi's own message is the only clue a board-side clock
        # failure leaves in a CI log, and the status alone discards it.
        print(f"[bench-sku] -lgc {target} MHz: {lock.detail}", file=sys.stderr)
        return _PROBE_LOCK_FAILED
    for _ in range(_CLOCK_RESET_ATTEMPTS):
        if _run_privileged(["-rgc", "-i", "0"]):
            break
    else:
        raise ClockResetError(
            f"failed to reset SM clock after {_CLOCK_RESET_ATTEMPTS} attempts; "
            "the clock may remain locked"
        )
    return _PROBE_OK


def _report(code, msg):
    """Print the decision and return its exit code. Anything but eligible goes to stderr."""
    print(
        f"[bench-sku] {msg}", file=sys.stdout if code == EXIT_ELIGIBLE else sys.stderr
    )
    return code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda-major", type=int, choices=SUPPORTED_CUDA_MAJORS, required=True
    )
    args = parser.parse_args()
    sku_map_path = Path(__file__).resolve().parent.parent / "config" / "sku_map.json"

    try:
        sku_map = load_sku_map(sku_map_path, strict=True)
        name = _gpu_name()
        power = _power_cap_w() if name is not None else None

        if name is None:
            return _report(
                EXIT_ELIGIBLE,
                "nvidia-smi unavailable; skipping SKU eligibility check.",
            )
        if name not in sku_map_gpu_names(sku_map):
            return _report(
                EXIT_ELIGIBLE,
                f"'{name}' is not in sku_map.json; skipping (fail-open).",
            )
        if power is None:
            return _report(
                EXIT_INELIGIBLE,
                f"INELIGIBLE: could not determine power cap for mapped GPU "
                f"'{name}'. Node should be excluded.",
            )

        try:
            target = required_clock_for_profile(
                sku_map,
                gpu_name=name,
                power_cap_w=power,
                cuda_major=args.cuda_major,
            )
        except MissingClockProfileError:
            allowed = profiles_for_gpu(sku_map, name)
            return _report(
                EXIT_INELIGIBLE,
                f"INELIGIBLE: {name} @ {power}W / CUDA {args.cuda_major} has no "
                f"baseline (allowed for this model: {allowed}).",
            )
        status = _probe_sm_clock(target)
    except ClockResetError as exc:
        return _report(EXIT_INELIGIBLE, f"INELIGIBLE: {exc}. Node should be excluded.")
    except MissingClockConfigurationError as exc:
        return _report(EXIT_ERROR, f"ERROR: {exc}")
    except (BaselineError, NvidiaSmiError) as exc:
        return _report(EXIT_ERROR, f"ERROR: {exc}")

    if status == _PROBE_NO_TARGET:
        return _report(
            EXIT_INELIGIBLE,
            f"INELIGIBLE: {name} @ {power}W does not support the expected SM clock "
            f"({target} MHz). Node should be excluded.",
        )

    if status == _PROBE_LOCK_FAILED:
        return _report(EXIT_ERROR, "ERROR: -lgc failed at every privilege level.")

    return _report(
        EXIT_ELIGIBLE,
        f"OK: {name} @ {power}W / {target}MHz / CUDA {args.cuda_major} is a "
        "baseline-backed SKU.",
    )


if __name__ == "__main__":
    sys.exit(main())
