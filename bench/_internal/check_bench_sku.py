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
per-SKU JSON baseline by the ``(Device Name, Power Cap (W), Locked SM Clock (MHz))``
triple via ``bench/config/sku_map.json``. Only SKUs present in that map have
a baseline, so a benchmark leg that lands on a GPU outside the map (e.g. an H100
PCIe at a non-canonical 310 W TDP, or silicon that clamps ``-lgc 1095`` to
1005 MHz) cannot be compared and should be retried on a different node.

This script predicts the routing key *without* locking the clock — which is why
it is safe to run in the lightweight pre-bench environment stage:

* Power cap comes straight from ``nvidia-smi --query-gpu=power.max_limit``
  (the same value ``run_bench.py`` stamps as ``Power Cap (W)``).
* The locked SM clock ``run_bench.py`` will commit to is the highest entry of
  its preferred list (``BENCH_LOCK_SM_CLOCK_MHZ``, default 1095,1005,900,750)
  that the device actually exposes in ``clocks.gr.supported`` — so a card whose
  supported list lacks 1095 is exactly the one that would clamp to 1005.

Exit codes:
  0  eligible (triple is in sku_map.json), or fail-open: the GPU model is not
     covered by the map, or nvidia-smi is *absent* (FileNotFoundError). These
     are genuinely "cannot/should not gate" cases — do not block scheduling.
  3  ineligible: the GPU model IS in the map but its predicted (power, clock)
     has no baseline. Callers should exclude this node and retry elsewhere.
  1  error: nvidia-smi is present but a query failed or timed out. Callers
     should fail fast rather than mislabel the node as ineligible — a tooling
     failure is not a bad node, and retrying elsewhere would not fix it.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

EXIT_ELIGIBLE = 0
EXIT_ERROR = 1
EXIT_INELIGIBLE = 3


class NvidiaSmiError(RuntimeError):
    """nvidia-smi is present but a query failed — distinct from the binary being
    absent, so the caller can fail fast instead of silently passing eligibility."""


# Mirrors run_bench.py's _CLOCK_LOCK_PREFERRED_MHZ so the predicted lock target
# matches what the bench run will actually commit to.
_PREFERRED_MHZ = [
    int(x)
    for x in os.environ.get("BENCH_LOCK_SM_CLOCK_MHZ", "1095,1005,900,750").split(",")
    if x.strip()
]


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


def predict_locked_clock(supported, preferred=None):
    """Highest preferred clock present in `supported` (what -lgc lands on).
    `supported` is a set of MHz ints, or None if it couldn't be queried — in
    which case fall back to preferred[0] (what run_bench.py requests blindly)."""
    pref = preferred if preferred is not None else _PREFERRED_MHZ
    if supported is None:
        return pref[0] if pref else None
    for mhz in pref:
        if mhz in supported:
            return mhz
    return None


def decide(name, power, clock, entries):
    """Pure eligibility decision. Returns (exit_code, message)."""
    if name is None:
        return EXIT_ELIGIBLE, "nvidia-smi unavailable; skipping SKU eligibility check."

    known_names = {e["gpu_name"] for e in entries}
    if name not in known_names:
        # Not a baseline-backed GPU model at all — not this gate's concern.
        return EXIT_ELIGIBLE, f"'{name}' is not in sku_map.json; skipping (fail-open)."

    allowed = [
        (e["power_cap_w"], e["locked_sm_clock_mhz"])
        for e in entries
        if e["gpu_name"] == name
    ]
    if (power, clock) in allowed:
        return (
            EXIT_ELIGIBLE,
            f"OK: {name} @ {power}W / {clock}MHz is a baseline-backed SKU.",
        )

    return EXIT_INELIGIBLE, (
        f"INELIGIBLE: {name} @ {power}W / predicted {clock}MHz has no baseline "
        f"(allowed for this model: {allowed}). Node should be excluded and the "
        f"leg retried elsewhere."
    )


def main():
    sku_map_path = Path(__file__).resolve().parent.parent / "config" / "sku_map.json"
    entries = json.loads(sku_map_path.read_text()).get("entries", [])

    try:
        name = _gpu_name()
        power = _power_cap_w() if name is not None else None
        clock = (
            predict_locked_clock(_supported_sm_clocks()) if name is not None else None
        )
    except NvidiaSmiError as exc:
        print(f"[bench-sku] ERROR: {exc}", file=sys.stderr)
        return EXIT_ERROR

    code, msg = decide(name, power, clock, entries)
    print(
        f"[bench-sku] {msg}", file=sys.stderr if code != EXIT_ELIGIBLE else sys.stdout
    )
    return code


if __name__ == "__main__":
    sys.exit(main())
