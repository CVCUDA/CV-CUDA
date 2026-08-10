#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Run selected CV-CUDA C++ and Python benchmarks and aggregate their results.

Results are written as a combined CSV by default. JSON output can be consumed
by the baseline comparison and update commands.
"""

import os
import sys
import time
import argparse
import atexit
import json
import difflib
import subprocess
import threading
import pandas as pd
from pathlib import Path
from typing import List, Optional, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from _internal.warmup import WARMUP_CAP_ENV, parse_warmup_cap

from _internal.axes import format_axis_name, format_axis_value
from _internal.baselines import (
    DEFAULT_CONFIG_DIR,
    DEFAULT_SKU_MAP_PATH,
    BaselineError,
    baseline_payload_from_dataframe,
    load_config_index,
)
from _internal.quality import (
    BenchmarkQualityCriteria,
    DEFAULT_BENCHMARK_QUALITY,
    exceeds_limit,
)

# =============================================================================
# GPU CLOCK CONTROL — lock SM clocks during the run + log live clocks
# =============================================================================
#
# Why: across the K8s GPU pool we observed individual nodes whose SM clocks did
# not boost (entire bench wave 40–75% slower than the rest of the wave on the
# *same* SHA + same SKU label).  Locking the SM clock to a fixed value before
# the bench starts eliminates frequency drift as a noise source and makes
# build-to-build comparisons deterministic.  Locking requires CAP_SYS_ADMIN
# (or root) — we attempt it and fall back gracefully if denied.
#
# Independently, we sample the live SM clock + power + temperature throughout
# the run and write JSONL next to the bench output.  Even when locking is
# denied, the log makes it obvious in retrospect why a build's kernel times
# drifted.
#
# Target clock: BENCH_LOCK_SM_CLOCK_MHZ accepts a comma-separated preference
# list (highest-first).  We intersect it with the device's `clocks.gr.supported`
# list and pick the highest value present in both.  Default 1095 is the
# documented base SM clock for both A100 PCIe 40GB and H100 PCIe; we fall back
# through {1005, 900, 750} for H100 PCIe silicon variants whose supported list
# does not include 1095 (the driver silently clamps `-lgc 1095,1095` to the
# closest supported value below, so we have to check ourselves).

_CLOCK_LOCK_PREFERRED_MHZ = [
    int(x)
    for x in os.environ.get("BENCH_LOCK_SM_CLOCK_MHZ", "1095,1005,900,750").split(",")
    if x.strip()
]
_CLOCK_SAMPLE_INTERVAL_S = float(os.environ.get("BENCH_CLOCK_SAMPLE_INTERVAL_S", "0.5"))
_CLOCK_LOG_PATH = os.environ.get("BENCH_CLOCK_LOG", "")  # set after parse_args

_clock_lock_acquired = False
_clock_lock_applied_mhz: Optional[int] = None
_clock_sampler_stop = threading.Event()
_clock_sampler_thread: Optional[threading.Thread] = None


def _nvidia_smi(*args: str, timeout: float = 5.0) -> Optional[str]:
    """Run nvidia-smi with given args, return stdout or None on failure."""
    try:
        r = subprocess.run(
            ["nvidia-smi", *args], capture_output=True, text=True, timeout=timeout
        )
        return r.stdout if r.returncode == 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _query_supported_sm_clocks_mhz() -> Optional[List[int]]:
    """Parse `nvidia-smi -q -d SUPPORTED_CLOCKS` to extract the SM (graphics)
    clock list the driver will actually accept for `-lgc`.

    Different H100 PCIe silicon variants ship with different supported lists;
    `-lgc <mhz>,<mhz>` silently clamps to the nearest value in the list and
    reports success either way, so picking a target without consulting this
    list can land at an unintended clock.  Returns None on parse failure.
    """
    out = _nvidia_smi("-q", "-d", "SUPPORTED_CLOCKS", "-i", "0", timeout=5.0)
    if not out:
        return None
    clocks: List[int] = []
    in_graphics = False
    for line in out.splitlines():
        s = line.strip()
        if s.startswith("Graphics"):
            in_graphics = True
            try:
                val = s.split(":", 1)[1].strip().split()[0]
                clocks.append(int(val))
            except (IndexError, ValueError):
                pass
        elif s.startswith("Memory") or s.startswith("SM"):
            in_graphics = False
        elif in_graphics and s.endswith("MHz"):
            try:
                clocks.append(int(s.split()[0]))
            except (IndexError, ValueError):
                pass
    return sorted(set(clocks)) if clocks else None


def _query_current_sm_clock_mhz() -> Optional[int]:
    """Sample `clocks.current.sm` once.  Returns None on failure."""
    out = _nvidia_smi(
        "--query-gpu=clocks.current.sm",
        "--format=csv,noheader,nounits",
        "-i",
        "0",
        timeout=5.0,
    )
    if not out:
        return None
    try:
        return int(out.strip().splitlines()[0].split()[0])
    except (IndexError, ValueError):
        return None


def _pick_lock_target(preferred: List[int]) -> Optional[int]:
    """Pick the highest value from `preferred` that's in the device's
    supported-clocks list, so `-lgc` lands deterministically.  Falls back
    to `preferred[0]` if the supported list cannot be queried (e.g.
    older driver), accepting the risk of silent clamping.
    """
    supported = _query_supported_sm_clocks_mhz()
    if supported is None:
        return preferred[0] if preferred else None
    supported_set = set(supported)
    for target in preferred:
        if target in supported_set:
            return target
    return None


def _try_lock_sm_clock(preferred: List[int]) -> Tuple[bool, Optional[int]]:
    """Attempt to lock the SM clock to a value in `preferred` that the device
    actually supports.

    Follows the CI GPU-clock-control recipe:
      1. -acp UNRESTRICTED   (allow per-process clock changes)
      2. -pm ENABLED         (persistence mode — keeps the driver loaded so
                              the lock survives idle gaps without snapping
                              back to default boost/throttle policy)
      3. -lgc <mhz>,<mhz>    (the actual lock)

    Each step tries `sudo -n` first (CI runners with a sudoers rule) then
    unprivileged (CI pods with SYS_ADMIN injected via the mutating
    admission webhook).  Prep-step failures are tolerated.

    Returns (locked, applied_mhz). When `locked` is True, `applied_mhz` is
    the verified `clocks.current.sm` post-lock — which may differ from the
    requested target if the driver clamped, so callers should trust this
    over the requested value.  When `locked` is False, `applied_mhz` is
    the freerunning sampled clock (best-effort) for diagnostic logging.
    """

    def _run(args):
        for prefix in (["sudo", "-n"], []):
            try:
                r = subprocess.run(
                    prefix + ["nvidia-smi"] + list(args),
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if r.returncode == 0:
                    return True, prefix
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue
        return False, None

    # Prep — best effort, ignore failures
    _run(["-acp", "UNRESTRICTED", "-i", "0"])
    _run(["-pm", "ENABLED", "-i", "0"])

    target = _pick_lock_target(preferred)
    if target is None:
        print(
            f"[clock-lock] WARN: no preferred clock {preferred} is in the "
            f"device's supported list; skipping lock",
            file=sys.stderr,
        )
        return False, _query_current_sm_clock_mhz()

    ok, prefix = _run(["-lgc", f"{target},{target}", "-i", "0"])
    if ok:
        applied = _query_current_sm_clock_mhz()
        mode = "sudo" if prefix == ["sudo", "-n"] else "unprivileged"
        if applied == target:
            print(
                f"[clock-lock] Locked SM clock to {target} MHz ({mode})",
                file=sys.stderr,
            )
        else:
            print(
                f"[clock-lock] Locked SM clock to {target} MHz ({mode}); "
                f"driver clamped to {applied} MHz",
                file=sys.stderr,
            )
        return True, applied

    print(
        f"[clock-lock] WARN: could not lock SM clock to {target} MHz "
        f"(insufficient permissions); benchmarks may show frequency drift",
        file=sys.stderr,
    )
    return False, _query_current_sm_clock_mhz()


def _try_reset_sm_clock() -> None:
    """Restore default clock policy.  Safe to call even if lock failed."""
    for prefix in (["sudo", "-n"], []):
        try:
            r = subprocess.run(
                prefix + ["nvidia-smi", "-rgc"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if r.returncode == 0:
                return
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue


def _clock_sampler_loop(out_path: str, interval_s: float) -> None:
    """Background thread: periodically sample SM clock / power / temp.

    Writes one JSON object per sample. Robust to nvidia-smi transient
    failures (skips the sample on parse error).
    """
    fields = [
        "wall_time_s",
        "sm_clock_mhz",
        "mem_clock_mhz",
        "gpu_util_pct",
        "mem_util_pct",
        "power_w",
        "power_limit_w",
        "gpu_temp_c",
        "throttle_reasons",
    ]
    query_cols = (
        "clocks.current.sm,clocks.current.memory,utilization.gpu,utilization.memory,"
        "power.draw,power.limit,temperature.gpu,clocks_throttle_reasons.active"
    )

    t0 = time.time()
    try:
        with open(out_path, "w") as f:
            while not _clock_sampler_stop.is_set():
                out = _nvidia_smi(
                    f"--query-gpu={query_cols}",
                    "--format=csv,noheader,nounits",
                    "-i",
                    "0",
                    timeout=2.0,
                )
                if out:
                    line = out.strip().splitlines()[0] if out.strip() else ""
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 8:
                        row = dict(zip(fields, [round(time.time() - t0, 2), *parts]))
                        f.write(json.dumps(row) + "\n")
                        f.flush()
                _clock_sampler_stop.wait(interval_s)
    except Exception as e:
        print(f"[clock-sampler] thread error: {e}", file=sys.stderr)


def init_clock_control(out_dir: str) -> None:
    """Lock SM clock + dump GPU fingerprint + start the live-clock sampler.
    Idempotent on cleanup.
    """
    global _clock_lock_acquired, _clock_lock_applied_mhz, _clock_sampler_thread
    if _clock_lock_acquired or _clock_sampler_thread is not None:
        return  # already initialized
    _clock_lock_acquired, _clock_lock_applied_mhz = _try_lock_sm_clock(
        _CLOCK_LOCK_PREFERRED_MHZ
    )
    _write_gpu_fingerprint(out_dir)
    log_path = _CLOCK_LOG_PATH or os.path.join(out_dir, "clock_log.jsonl")
    _clock_sampler_thread = threading.Thread(
        target=_clock_sampler_loop,
        args=(log_path, _CLOCK_SAMPLE_INTERVAL_S),
        daemon=True,
    )
    _clock_sampler_thread.start()
    print(
        f"[clock-sampler] sampling SM clock + power + temp every "
        f"{_CLOCK_SAMPLE_INTERVAL_S:.2f}s -> {log_path}",
        file=sys.stderr,
    )
    atexit.register(shutdown_clock_control)


def shutdown_clock_control() -> None:
    """Stop sampler thread and reset clock lock.  Safe to call multiple times."""
    global _clock_sampler_thread, _clock_lock_acquired
    if _clock_sampler_thread is not None:
        _clock_sampler_stop.set()
        _clock_sampler_thread.join(timeout=3.0)
        _clock_sampler_thread = None
    if _clock_lock_acquired:
        _try_reset_sm_clock()
        _clock_lock_acquired = False


try:
    from tabulate import tabulate
except ImportError:
    tabulate = None


# =============================================================================
# SCHEMA DEFINITIONS
# =============================================================================

CONFIG_KEY_COLUMN = "config_key"


@dataclass
class ColumnDef:
    """Definition of a single column in benchmark results."""

    name: str
    category: str  # 'config', 'metric', 'derived', 'metadata'


# Single source of truth for all column definitions
SCHEMA = [
    # Configuration columns (used for grouping/matching)
    ColumnDef("Benchmark", "config"),
    ColumnDef("config_key", "config"),
    ColumnDef("Language", "config"),
    ColumnDef("InOutDataType", "config"),
    ColumnDef("shape", "config"),
    ColumnDef("outDataType", "config"),
    ColumnDef("distanceType", "config"),
    ColumnDef("inputKind", "config"),
    ColumnDef("batch", "config"),
    ColumnDef("antialias", "config"),
    ColumnDef("numBoxes", "config"),
    ColumnDef("kernelSize", "config"),
    ColumnDef("diameter", "config"),
    ColumnDef("cropMode", "config"),
    ColumnDef("flagsMode", "config"),
    ColumnDef("randomMode", "config"),
    ColumnDef("maskMode", "config"),
    ColumnDef("inpaintRadius", "config"),
    ColumnDef("ksize", "config"),
    ColumnDef("iteration", "config"),
    ColumnDef("numErase", "config"),
    ColumnDef("maxLocations", "config"),
    ColumnDef("numOctaveLayers", "config"),
    ColumnDef("matchesPerPoint", "config"),
    ColumnDef("numElem", "config"),
    ColumnDef("maxCapacity", "config"),
    ColumnDef("blockSize", "config"),
    # Metric columns (raw measurements)
    ColumnDef("GPU Time (µs)", "metric"),
    ColumnDef("GPU Noise (%)", "metric"),
    ColumnDef("GPU Noise (µs)", "metric"),
    ColumnDef("CPU Time (µs)", "metric"),
    ColumnDef("CPU Noise (%)", "metric"),
    ColumnDef("CPU Noise (µs)", "metric"),
    ColumnDef("GlobalMem BW (bytes/sec)", "metric"),
    ColumnDef("BWUtil", "metric"),
    # Derived columns (computed from metrics)
    ColumnDef("Py overhead (%)", "derived"),
    ColumnDef("Py overhead (µs)", "derived"),
    ColumnDef("Status", "derived"),
    # Metadata columns (informational only)
    ColumnDef("Device Name", "metadata"),
    ColumnDef("Power Cap (W)", "metadata"),
    ColumnDef("Locked SM Clock (MHz)", "metadata"),
    ColumnDef("vBIOS Version", "metadata"),
    ColumnDef("Samples", "metadata"),
    ColumnDef("tier", "metadata"),
]

# Columns that are NOT config columns (used for grouping/matching)
NON_CONFIG_COLS = frozenset(c.name for c in SCHEMA if c.category != "config") | {
    "Device",
    "Device Name",
    "Skipped",
    "Samples",
    "Samples.1",
    "Batch GPU (sec)",
}

# Integer config columns that need normalization to Int64
# (inputKind is a string axis -- Tensor/VarShape -- so it is intentionally absent here)
INT_COLUMNS = [
    "batch",
    "antialias",
    "numBoxes",
    "kernelSize",
    "diameter",
    "ksize",
    "iteration",
    "numErase",
    "maxLocations",
    "numOctaveLayers",
    "matchesPerPoint",
    "numElem",
    "maxCapacity",
    "blockSize",
]

# C++ nvbench type names → Python numpy names
DTYPE_MAP = {
    "U8": "uint8",
    "U16": "uint16",
    "U32": "uint32",
    "U64": "uint64",
    "I8": "int8",
    "I16": "int16",
    "I32": "int32",
    "I64": "int64",
    "F32": "float32",
    "F64": "float64",
}

# Column display order (priority columns first)
PRIORITY_COLUMNS = [
    "Benchmark",
    "tier",
    "config_key",
    "Language",
    "GPU Time (µs)",
    "GPU Noise (%)",
    "GPU Noise (µs)",
    "CPU Time (µs)",
    "CPU Noise (%)",
    "CPU Noise (µs)",
    "Py overhead (%)",
    "Py overhead (µs)",
    "Status",
    "GlobalMem BW (bytes/sec)",
    "BWUtil",
]

# Internal nvbench scratch columns we require.
REQUIRED_SCRATCH_COLS = {"Benchmark", "BWUtil", "Skipped", "GPU Time (sec)"}


def get_config_cols(df: pd.DataFrame) -> List[str]:
    """Get configuration columns present in dataframe (excludes Language)."""
    return [c for c in df.columns if c not in NON_CONFIG_COLS and c != "Language"]


_POWER_CAP_W_CACHE: Optional[int] = None
_POWER_CAP_W_QUERIED = False


def query_locked_sm_clock_mhz() -> Optional[int]:
    """Return the SM clock the bench is effectively running at.

    When init_clock_control() has applied a lock, this is the verified
    `clocks.current.sm` post-lock (the value the driver actually committed,
    which may differ from the requested target on H100 silicon variants
    whose supported-clocks list does not include the requested value).

    When no lock is in effect, this samples the live clock once. The JSON
    artifact field derived from this is informational on locked runs and noisy
    on unlocked runs; baselines should be reseeded only from locked runs.
    """
    if _clock_lock_applied_mhz is not None:
        return _clock_lock_applied_mhz
    return _query_current_sm_clock_mhz()


_VBIOS_CACHE: Optional[str] = None
_VBIOS_QUERIED = False


def query_vbios_version() -> Optional[str]:
    """Query the GPU's vBIOS version via nvidia-smi, cached per process.

    Used as an additional baseline-routing dimension because the silicon-side
    behaviour (atomic throughput, DRAM placement quirks, supported-clocks
    list) varies across H100 PCIe revisions even when Device Name + Power Cap
    + Locked SM Clock all match.  We observed 12% spread on histogrameq
    between two H100 PCIe 350W pools that locked identically to 1095 MHz.
    The vBIOS string (e.g. "96.00.51.00.00") is durable per silicon revision.
    """
    global _VBIOS_CACHE, _VBIOS_QUERIED
    if _VBIOS_QUERIED:
        return _VBIOS_CACHE
    _VBIOS_QUERIED = True
    out = _nvidia_smi(
        "--query-gpu=vbios_version",
        "--format=csv,noheader",
        "-i",
        "0",
        timeout=5.0,
    )
    if not out:
        return None
    val = out.strip().splitlines()[0].strip() if out.strip() else None
    _VBIOS_CACHE = val or None
    return _VBIOS_CACHE


def _write_gpu_fingerprint(out_dir: str) -> None:
    """Dump a richer hardware fingerprint alongside clock_log.jsonl.

    Captures vBIOS, UUID, product name/brand, the full `clocks.gr.supported`
    list, and the K8s `spec.nodeName` (per-build ephemeral name; the
    stable host name lives in the node's `kubernetes.io/hostname`
    label, which the pod's default ServiceAccount can't read — physical
    host identification is done out of band).  Every
    signal we have for distinguishing silicon variants in a multi-host
    K8s pool.  Purely diagnostic; routing keys do not consume these
    fields.
    """
    path = os.path.join(out_dir, "gpu_fingerprint.txt")
    queries = [
        ("vbios_version", "--query-gpu=vbios_version"),
        ("uuid", "--query-gpu=uuid"),
        ("name", "--query-gpu=name"),
        ("driver_version", "--query-gpu=driver_version"),
        ("memory.total", "--query-gpu=memory.total"),
        ("compute_cap", "--query-gpu=compute_cap"),
    ]
    try:
        with open(path, "w") as f:
            for label, q in queries:
                out = _nvidia_smi(q, "--format=csv,noheader", "-i", "0", timeout=5.0)
                f.write(f"{label}: {(out or '').strip()}\n")
            supported = _query_supported_sm_clocks_mhz() or []
            f.write(f"clocks.gr.supported: {supported}\n")
            # Read K8S_NODE_NAME (not NODE_NAME) — the CI agent reserves
            # NODE_NAME and overwrites the pod-level Downward API var of
            # that name with the agent's own name (= the pod name).
            # K8S_NODE_NAME is the Downward API spec.nodeName value the
            # K8s scheduler placed the pod on.  That is the ephemeral
            # agent name, not the underlying physical host; the stable
            # host name is in the node's `kubernetes.io/hostname` label,
            # readable only via the K8s API (RBAC denied here, so it is
            # looked up out of band).
            f.write(f"k8s_node_name: {os.environ.get('K8S_NODE_NAME', '')}\n")
    except OSError as e:
        print(f"[gpu-fingerprint] failed to write {path}: {e}", file=sys.stderr)


def query_power_cap_w() -> Optional[int]:
    """Query the GPU's max power cap (W) via nvidia-smi, cached per process.

    Used to key per-(SKU, power-cap) baselines: the same Device Name can
    appear at different TDP configurations (e.g. H100 PCIe at 350W vs 310W),
    and they perform meaningfully differently.
    """
    global _POWER_CAP_W_CACHE, _POWER_CAP_W_QUERIED
    if _POWER_CAP_W_QUERIED:
        return _POWER_CAP_W_CACHE
    _POWER_CAP_W_QUERIED = True
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=power.max_limit",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        first_line = out.strip().splitlines()[0]
        _POWER_CAP_W_CACHE = int(round(float(first_line)))
    except (subprocess.SubprocessError, ValueError, IndexError, FileNotFoundError):
        _POWER_CAP_W_CACHE = None
    return _POWER_CAP_W_CACHE


# =============================================================================
# DISPLAY UTILITIES
# =============================================================================


class Colors:
    """ANSI color codes for terminal output."""

    RESET = "\033[0m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    GRAY = "\033[90m"
    BOLD = "\033[1m"

    @staticmethod
    def disable():
        """Disable all colors."""
        Colors.RESET = ""
        Colors.GREEN = ""
        Colors.RED = ""
        Colors.YELLOW = ""
        Colors.CYAN = ""
        Colors.GRAY = ""
        Colors.BOLD = ""


# Symbols
SUCCESS = "✓"
ERROR = "✗"
WARNING = "⚠"
INFO = "→"

# Global output control
VERBOSE = False
QUIET = False


def log_info(msg: str):
    if not QUIET:
        print(f"{Colors.GRAY}{INFO}{Colors.RESET} {msg}")


def log_success(msg: str):
    if not QUIET:
        print(f"{Colors.GREEN}{SUCCESS}{Colors.RESET} {msg}")


def log_warning(msg: str):
    print(f"{Colors.YELLOW}{WARNING}{Colors.RESET} {msg}")


def log_error(msg: str):
    print(f"{Colors.RED}{ERROR}{Colors.RESET} {msg}")


def print_table(rows: List[List[str]], headers: List[str]):
    """Print table using tabulate or fallback to simple formatting."""
    if tabulate:
        print(tabulate(rows, headers=headers, tablefmt="pipe"))
    else:
        print("| " + " | ".join(headers) + " |")
        print("|" + "|".join(["-" * (len(h) + 2) for h in headers]) + "|")
        for row in rows:
            print("| " + " | ".join(str(v) for v in row) + " |")


# =============================================================================
# VALIDATION
# =============================================================================


@dataclass
class ValidationError:
    """A single validation error."""

    error_type: str  # 'high_noise', 'missing_cpp', 'missing_python', 'perf_diff'
    config: str
    details: dict = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result of all validation checks."""

    noise_errors: List[ValidationError] = field(default_factory=list)
    config_errors: List[ValidationError] = field(default_factory=list)
    perf_errors: List[ValidationError] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return bool(self.noise_errors or self.config_errors or self.perf_errors)

    @property
    def total_errors(self) -> int:
        return len(self.noise_errors) + len(self.config_errors) + len(self.perf_errors)


# =============================================================================
# RESULTS PROCESSOR
# =============================================================================


class ResultsProcessor:
    """Processes benchmark results through a clean transform pipeline."""

    def __init__(
        self, max_noise_pct: float, max_perf_diff_pct: float, max_perf_diff_us: float
    ):
        self.quality = BenchmarkQualityCriteria(
            max_noise_pct=max_noise_pct,
            max_perf_diff_pct=max_perf_diff_pct,
            max_perf_diff_us=max_perf_diff_us,
        )
        self.max_noise_pct = self.quality.max_noise_pct
        self.max_perf_diff_pct = self.quality.max_perf_diff_pct
        # Absolute Py-overhead floor. The relative gate alone misses anomalies
        # on long kernels; those are typically real path divergences (different
        # kernel selection, different launch params, or bench harness mismatch).
        self.max_perf_diff_us = self.quality.max_perf_diff_us

    def process(
        self, dfs: List[pd.DataFrame], compute_parity: bool = True
    ) -> pd.DataFrame:
        """Run full transform pipeline on list of result dataframes."""
        if not dfs:
            return pd.DataFrame()

        df = pd.concat(dfs, ignore_index=True)
        df = self._normalize_dtypes(df)
        df = self._normalize_int_columns(df)
        df = self._compute_noise_absolute(df)
        df = self._compute_status_and_parity(df, compute_parity)
        df = self._sort_results(df)
        df = self._reorder_columns(df)
        return df

    def _normalize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize C++ dtype names to Python names."""
        if "InOutDataType" in df.columns:
            df["InOutDataType"] = df["InOutDataType"].replace(DTYPE_MAP)
        return df

    def _normalize_int_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert integer columns to nullable Int64 to avoid float/int mismatches."""
        for col in INT_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        return df

    def _compute_noise_absolute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute absolute noise values in microseconds."""
        if "GPU Noise (%)" in df.columns and "GPU Time (µs)" in df.columns:
            df["GPU Noise (µs)"] = (
                df["GPU Time (µs)"] * df["GPU Noise (%)"] / 100
            ).round(2)
        if "CPU Noise (%)" in df.columns and "CPU Time (µs)" in df.columns:
            df["CPU Noise (µs)"] = (
                df["CPU Time (µs)"] * df["CPU Noise (%)"] / 100
            ).round(2)
        return df

    def _compute_status_and_parity(
        self, df: pd.DataFrame, compute_parity: bool
    ) -> pd.DataFrame:
        """Compute status for all rows, and parity metrics for paired configs.

        For unpaired rows (or when compute_parity=False): status based on noise only.
        For paired rows: status includes noise and parity checks.
        """
        # Initialize status and parity columns
        df["Status"] = "PASS"
        df["Py overhead (%)"] = pd.NA
        df["Py overhead (µs)"] = pd.NA

        # Track which rows have been processed (for paired configs)
        processed_indices = set()

        if compute_parity:
            config_cols = get_config_cols(df)
            if config_cols:
                for _, group in df.groupby(config_cols, dropna=False):
                    cpp_rows = group[group["Language"] == "cpp"]
                    py_rows = group[group["Language"] == "python"]

                    if len(cpp_rows) > 0 and len(py_rows) > 0:
                        cpp_idx, py_idx = cpp_rows.index[0], py_rows.index[0]
                        cpp_time = df.at[cpp_idx, "GPU Time (µs)"]
                        py_time = df.at[py_idx, "GPU Time (µs)"]
                        cpp_noise = df.at[cpp_idx, "GPU Noise (%)"]
                        py_noise = df.at[py_idx, "GPU Noise (%)"]

                        # Check for missing/invalid timing data
                        cpp_time_valid = pd.notna(cpp_time) and cpp_time > 0
                        py_time_valid = pd.notna(py_time) and py_time > 0

                        if not cpp_time_valid or not py_time_valid:
                            # Mark as FAIL if timing data is missing
                            fail_msg = "FAIL (missing timing data:"
                            if not cpp_time_valid:
                                fail_msg += " C++"
                            if not py_time_valid:
                                fail_msg += " Python"
                            fail_msg += ")"
                            df.at[cpp_idx, "Status"] = fail_msg
                            df.at[py_idx, "Status"] = fail_msg
                            processed_indices.update([cpp_idx, py_idx])
                        elif cpp_time_valid and py_time_valid:
                            overhead_pct = (py_time / cpp_time - 1.0) * 100
                            overhead_us = py_time - cpp_time

                            # Set overhead on both rows
                            for idx in [cpp_idx, py_idx]:
                                df.at[idx, "Py overhead (%)"] = overhead_pct
                                df.at[idx, "Py overhead (µs)"] = overhead_us

                            # Compute combined status for paired rows
                            fail_reasons = []
                            if self.quality.noise_exceeds_limit(cpp_noise):
                                fail_reasons.append(f"C++ noise {cpp_noise:.1f}%")
                            if self.quality.noise_exceeds_limit(py_noise):
                                fail_reasons.append(f"Py noise {py_noise:.1f}%")
                            if self.quality.relative_parity_exceeds_limit(overhead_pct):
                                fail_reasons.append(f"parity {overhead_pct:+.1f}%")
                            # Absolute parity floor — catches large path
                            # divergences on long kernels that the relative
                            # gate would miss.
                            if self.quality.absolute_parity_exceeds_limit(overhead_us):
                                fail_reasons.append(f"parity {overhead_us:+.0f}us")

                            status = (
                                f"FAIL ({', '.join(fail_reasons)})"
                                if fail_reasons
                                else "PASS"
                            )
                            df.at[cpp_idx, "Status"] = status
                            df.at[py_idx, "Status"] = status
                            processed_indices.update([cpp_idx, py_idx])

        # Set status for unpaired rows based on noise and missing data
        for idx in df.index:
            if idx not in processed_indices:
                gpu_time = df.at[idx, "GPU Time (µs)"]
                noise = df.at[idx, "GPU Noise (%)"]
                lang = df.at[idx, "Language"]

                # Check for missing timing data in THIS row
                if pd.isna(gpu_time) or gpu_time <= 0:
                    df.at[idx, "Status"] = f"FAIL (missing {lang} timing)"
                # In parity mode, unpaired rows mean the counterpart is missing
                elif compute_parity:
                    other = "C++" if lang == "python" else "Python"
                    df.at[idx, "Status"] = f"FAIL (missing {other})"
                elif pd.notna(noise) and self.quality.noise_exceeds_limit(noise):
                    df.at[idx, "Status"] = f"FAIL (noise {noise:.1f}%)"

        return df

    def _sort_results(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sort to interleave C++ and Python entries for easy comparison."""
        config_cols = get_config_cols(df)
        sort_cols = [c for c in config_cols + ["Language"] if c in df.columns]
        return df.sort_values(by=sort_cols, ignore_index=True) if sort_cols else df

    def _reorder_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reorder columns: priority first, then remaining."""
        all_cols = df.columns.tolist()
        ordered = [c for c in PRIORITY_COLUMNS if c in all_cols]
        ordered += [c for c in all_cols if c not in ordered]
        return df[ordered]

    def finalize_for_output(self, df: pd.DataFrame) -> pd.DataFrame:
        """Final cleanup for CSV output and display summaries."""
        # Drop Device index column if present
        if "Device" in df.columns:
            df = df.drop(columns=["Device"])

        # Round numeric columns
        round_2 = [
            "GPU Time (µs)",
            "CPU Time (µs)",
            "GPU Noise (%)",
            "CPU Noise (%)",
            "GPU Noise (µs)",
            "CPU Noise (µs)",
            "Py overhead (%)",
            "Py overhead (µs)",
        ]
        for col in round_2:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").round(2)
        if "BWUtil" in df.columns:
            df["BWUtil"] = pd.to_numeric(df["BWUtil"], errors="coerce").round(4)

        return df

    # -------------------------------------------------------------------------
    # Validation methods
    # -------------------------------------------------------------------------

    def validate(self, df: pd.DataFrame, check_parity: bool = True) -> ValidationResult:
        """Run all validations and return structured result."""
        result = ValidationResult()
        result.noise_errors = self._check_noise(df)
        if check_parity:
            result.config_errors = self._check_config_parity(df)
            result.perf_errors = self._check_perf_parity(df)
        return result

    def _build_config_str(self, row: pd.Series) -> str:
        """Build human-readable config string from row."""
        parts = [row.get("Benchmark", "unknown")]
        for col in ["InOutDataType", "shape", "inputKind"]:
            if col in row and pd.notna(row[col]):
                parts.append(
                    f"{format_axis_name(col)}={format_axis_value(col, row[col])}"
                )
        return ", ".join(parts)

    def _check_noise(self, df: pd.DataFrame) -> List[ValidationError]:
        """Check for excessive noise/stddev in benchmark results."""
        errors = []
        for _, row in df.iterrows():
            noise = row.get("GPU Noise (%)", 0.0)
            if self.quality.noise_exceeds_limit(noise):
                lang = row.get("Language", "unknown")
                config = f"{self._build_config_str(row)} ({lang})"
                errors.append(
                    ValidationError(
                        "high_noise",
                        config,
                        {"stddev": noise, "threshold": self.max_noise_pct},
                    )
                )
        return errors

    def _check_config_parity(self, df: pd.DataFrame) -> List[ValidationError]:
        """Check that each C++ config has a matching Python config."""
        errors = []
        config_cols = get_config_cols(df)
        if not config_cols:
            return errors

        for _, group in df.groupby(config_cols, dropna=False):
            cpp_rows = group[group["Language"] == "cpp"]
            py_rows = group[group["Language"] == "python"]
            row = cpp_rows.iloc[0] if len(cpp_rows) > 0 else py_rows.iloc[0]
            config = self._build_config_str(row)

            if len(cpp_rows) == 0:
                errors.append(ValidationError("missing_cpp", config))
            elif len(py_rows) == 0:
                errors.append(ValidationError("missing_python", config))
        return errors

    def _check_perf_parity(self, df: pd.DataFrame) -> List[ValidationError]:
        """Check for performance differences between C++ and Python."""
        errors = []
        config_cols = get_config_cols(df)
        if not config_cols:
            return errors

        for _, group in df.groupby(config_cols, dropna=False):
            cpp_rows = group[group["Language"] == "cpp"]
            py_rows = group[group["Language"] == "python"]
            if len(cpp_rows) == 0 or len(py_rows) == 0:
                continue

            cpp_time = cpp_rows.iloc[0].get("GPU Time (µs)", 0)
            py_time = py_rows.iloc[0].get("GPU Time (µs)", 0)
            if cpp_time == 0:
                continue

            ratio = py_time / cpp_time
            diff_pct = abs(ratio - 1.0) * 100
            diff_us = py_time - cpp_time
            if self.quality.parity_exceeds_limit(diff_pct, diff_us):
                config = self._build_config_str(cpp_rows.iloc[0])
                errors.append(
                    ValidationError(
                        "perf_diff",
                        config,
                        {
                            "cpp_time": cpp_time,
                            "python_time": py_time,
                            "ratio": ratio,
                            "diff_pct": diff_pct,
                            "diff_us": diff_us,
                            "threshold_pct": self.max_perf_diff_pct,
                            "threshold_us": self.max_perf_diff_us,
                        },
                    )
                )
        return errors

    def print_errors(self, result: ValidationResult):
        """Print validation errors in a clear format."""
        print(f"{Colors.RED}{Colors.BOLD}{'=' * 60}{Colors.RESET}")
        print(f"{Colors.RED}{Colors.BOLD}  VALIDATION ERRORS{Colors.RESET}")
        print(f"{Colors.RED}{Colors.BOLD}{'=' * 60}{Colors.RESET}\n")

        if result.config_errors:
            print(f"{Colors.RED}{Colors.BOLD}Configuration Mismatches:{Colors.RESET}")
            for err in result.config_errors:
                if err.error_type == "missing_cpp":
                    print(f"  - {err.config}: Missing C++ benchmark")
                elif err.error_type == "missing_python":
                    print(f"  - {err.config}: Missing Python benchmark")
            print()

        if result.noise_errors:
            threshold = result.noise_errors[0].details["threshold"]
            print(
                f"{Colors.RED}{Colors.BOLD}High Noise/StdDev (>{threshold}%):{Colors.RESET}"
            )
            for err in result.noise_errors:
                print(f"  - {err.config}: {err.details['stddev']:.2f}%")
            print()

        if result.perf_errors:
            d0 = result.perf_errors[0].details
            tpct = d0["threshold_pct"]
            tus = d0["threshold_us"]
            print(
                f"{Colors.RED}{Colors.BOLD}Performance Differences "
                f"(>{tpct:.1f}% relative or >{tus:.0f}us absolute):{Colors.RESET}"
            )
            for err in result.perf_errors:
                d = err.details
                direction = "faster" if d["ratio"] < 1.0 else "slower"
                print(
                    f"  - {err.config}: Python {d['ratio']:.3f}x {direction} "
                    f"({d['diff_pct']:.1f}% / {d['diff_us']:+.0f}us)"
                )
            print()

        print(f"{Colors.RED}Total errors: {result.total_errors}{Colors.RESET}")


# =============================================================================
# DISPLAY FORMATTERS
# =============================================================================


def print_parity_summary_table(df: pd.DataFrame):
    """Print a summary table comparing C++ and Python performance.

    Uses pre-computed columns from ResultsProcessor (Status, Py overhead, etc.).
    Shows N/A for missing language data when running single-language mode.
    """
    config_cols = get_config_cols(df)
    if not config_cols:
        return

    def fmt(val):
        return "" if pd.isna(val) else str(val)

    def fmt_time(t):
        return "N/A" if t is None or pd.isna(t) else f"{t:,.1f}"

    def fmt_noise(noise, time):
        if noise is None or time is None or pd.isna(noise) or pd.isna(time):
            return "N/A"
        return f"±{noise:.2f}% (±{time * noise / 100:.0f}µs)"

    def fmt_overhead(pct, us):
        if pct is None or pd.isna(pct):
            return "N/A"
        return f"{pct:+.1f}% ({us:+.0f}µs)"

    def fmt_bwutil(bw):
        if bw is None or pd.isna(bw):
            return "N/A"
        return f"{bw:.1%}"

    table_rows = []
    for _, group in df.groupby(config_cols, dropna=False):
        cpp_rows = group[group["Language"] == "cpp"]
        py_rows = group[group["Language"] == "python"]
        if len(cpp_rows) == 0 and len(py_rows) == 0:
            continue

        ref_row = cpp_rows.iloc[0] if len(cpp_rows) > 0 else py_rows.iloc[0]
        cpp_time = cpp_rows.iloc[0].get("GPU Time (µs)") if len(cpp_rows) > 0 else None
        py_time = py_rows.iloc[0].get("GPU Time (µs)") if len(py_rows) > 0 else None
        cpp_noise = cpp_rows.iloc[0].get("GPU Noise (%)") if len(cpp_rows) > 0 else None
        py_noise = py_rows.iloc[0].get("GPU Noise (%)") if len(py_rows) > 0 else None
        cpp_bwutil = cpp_rows.iloc[0].get("BWUtil") if len(cpp_rows) > 0 else None
        py_bwutil = py_rows.iloc[0].get("BWUtil") if len(py_rows) > 0 else None

        # Use pre-computed values from ResultsProcessor
        overhead_pct = ref_row.get("Py overhead (%)")
        overhead_us = ref_row.get("Py overhead (µs)")
        status = ref_row.get("Status", "PASS")
        is_pass = status == "PASS"

        table_rows.append(
            [format_axis_value(c, ref_row.get(c)) for c in config_cols]
            + [
                fmt_time(cpp_time),
                fmt_time(py_time),
                fmt_overhead(overhead_pct, 0 if pd.isna(overhead_us) else overhead_us),
                fmt_noise(cpp_noise, cpp_time),
                fmt_noise(py_noise, py_time),
                fmt_bwutil(cpp_bwutil),
                fmt_bwutil(py_bwutil),
                (
                    f"{Colors.GREEN}PASS{Colors.RESET}"
                    if is_pass
                    else f"{Colors.RED}FAIL{Colors.RESET}"
                ),
            ]
        )

    if table_rows:
        headers = [format_axis_name(c) for c in config_cols] + [
            "C++ (µs)",
            "Py (µs)",
            "Py overhead",
            "C++ Noise",
            "Py Noise",
            "C++ BWUtil",
            "Py BWUtil",
            "Status",
        ]
        print_table(table_rows, headers)
        print()


# =============================================================================
# BENCHMARK RUNNERS
# =============================================================================


def get_output_filename(benchmark_name: str, language: str) -> str:
    """Generate a unique internal nvbench scratch filename."""
    clean_name = benchmark_name.replace(".py", "").replace("bench_", "")
    return f"out_{clean_name}_{language}.nvbench"


@dataclass
class BenchmarkSummary:
    """Track benchmark execution statistics."""

    language: str
    total: int = 0
    successful: int = 0
    failed: int = 0


class BenchmarkRunner(ABC):
    """Base class for benchmark runners."""

    def __init__(
        self,
        bench_prefix: str,
        bench_folder: str,
        operators: Optional[List[str]],
        language: str,
        keep_outputs: bool = False,
        max_noise_pct: float = DEFAULT_BENCHMARK_QUALITY.max_noise_pct,
        config_file: Optional[str] = None,
        tiers: Optional[set] = None,
        config_keys: Optional[List[str]] = None,
        warmup_cap: Optional[int] = None,
    ):
        self.bench_prefix = bench_prefix
        self.bench_folder = bench_folder
        self.operators = operators or []
        self.language = language
        self.keep_outputs = keep_outputs
        self.max_noise_pct = max_noise_pct
        self.config_file = config_file
        self.tiers = tiers  # None = no tier filter
        self.config_keys = config_keys
        self.warmup_cap = warmup_cap
        self.operator_manifest = {}
        self.results: List[pd.DataFrame] = []
        self.output_files: List[str] = []
        self.summary = BenchmarkSummary(language)

    def _load_config(self, need_axis_args: bool = False):
        """Load benchmark configuration from JSON file.

        Args:
            need_axis_args: If True, also load generate_axis_args (C++ only).
        """
        try:
            sys.path.insert(0, str(Path(__file__).parent / "config"))
            from load_config import (
                load_bench_config,
                load_bench_manifest,
                get_operator_from_benchmark_name,
                get_configs_for_benchmark,
            )

            self.get_operator_from_benchmark_name = get_operator_from_benchmark_name
            self.get_configs_for_benchmark = get_configs_for_benchmark
            self.operator_manifest = load_bench_manifest(self.config_file)

            if need_axis_args:
                from load_config import generate_axis_args

                self.generate_axis_args = generate_axis_args
                cfg_path = self.config_file or "bench/config/bench_params.json"
                log_info(f"Loaded benchmark configuration from {cfg_path}")

            return load_bench_config(self.config_file)
        except Exception as e:
            log_error(f"Failed to load benchmark config: {e}")
            sys.exit(2)

    def discover_benchmarks(self) -> List[tuple]:
        """Map configured operators to config keys and benchmark files."""
        if self.config_keys is not None:
            return self._discover_selected_config_keys()

        selected_operators = self.operators or list(self.operator_manifest)
        pairs = []
        missing_tier = []
        for operator in selected_operators:
            keys = self.get_configs_for_benchmark(
                operator, self.config, tiers=self.tiers
            )
            if keys:
                path = self._benchmark_path_for_operator(operator)
                pairs.extend((k, path) for k in keys)
            elif self.operators:
                missing_tier.append(operator)

        if missing_tier:
            tiers = ",".join(sorted(self.tiers)) if self.tiers else "all"
            log_error(
                f"No config entries for operator(s) {missing_tier} in tier(s) {tiers}"
            )
            sys.exit(1)

        return pairs

    def _discover_selected_config_keys(self) -> List[tuple]:
        """Map exact config keys to benchmark files for this language."""
        pairs = []
        errors = []
        for config_key in self.config_keys:
            entry = self.config.get(config_key)
            if not isinstance(entry, dict):
                errors.append(f"Unknown config key: {config_key}")
                continue
            benchmark = entry.get("benchmark")
            if not benchmark:
                errors.append(
                    f"Config key {config_key!r} is missing required 'benchmark' field"
                )
                continue

            try:
                path = self._benchmark_path_for_operator(benchmark)
            except KeyError:
                errors.append(
                    f"Config key {config_key!r} targets benchmark {benchmark!r}, "
                    "which is not listed in bench_params.json"
                )
                continue
            pairs.append((config_key, path))

        if errors:
            for error in errors:
                log_error(error)
            sys.exit(2)

        return pairs

    def _benchmark_path_for_operator(self, operator: str) -> str:
        """Return the configured benchmark file path for this runner language."""
        spec = self.operator_manifest.get(operator)
        if spec is None:
            raise KeyError(operator)

        benchmark_file = spec.get(self.language)
        if not benchmark_file:
            log_error(
                f"Operator {operator!r} has no {self.language} benchmark in "
                "bench_params.json"
            )
            sys.exit(1)

        benchmark_path = os.path.join(self.bench_folder, benchmark_file)
        if not os.path.isfile(benchmark_path):
            log_error(
                f"Configured {self.language} benchmark for operator {operator!r} "
                f"does not exist: {benchmark_path}"
            )
            sys.exit(1)

        return benchmark_path

    @abstractmethod
    def build_command(
        self,
        benchmark_path: str,
        extra_args: List[str],
        output_file: str,
        config_key: Optional[str] = None,
    ) -> List[str]:
        """Build command to execute benchmark."""
        pass

    def benchmark_env(self) -> Optional[dict]:
        """Environment for benchmark subprocesses."""
        env = os.environ.copy()
        if self.warmup_cap is None:
            env.pop(WARMUP_CAP_ENV, None)
        else:
            env[WARMUP_CAP_ENV] = str(self.warmup_cap)
        return env

    def run_benchmark(
        self,
        benchmark_path: str,
        extra_args: List[str],
        config_key: Optional[str] = None,
    ) -> tuple:
        """Run a single benchmark and return (output_filename, all_pass)."""
        benchmark_name = config_key if config_key else os.path.basename(benchmark_path)
        output_file = get_output_filename(benchmark_name, self.language)
        output_path = os.path.join(self.bench_folder, output_file)
        cmd_list = self.build_command(
            benchmark_path, extra_args, output_path, config_key=config_key
        )
        if VERBOSE:
            log_info(f'Running: "{" ".join(cmd_list)}"')

        beg = time.time()

        try:
            result = subprocess.run(
                cmd_list,
                shell=False,
                check=False,
                capture_output=True,
                text=True,
                timeout=300,
                env=self.benchmark_env(),
            )
            elapsed = time.time() - beg

            if result.returncode != 0:
                if not QUIET:
                    print(
                        f"{Colors.RED}{ERROR}{Colors.RESET} "
                        f"(exit code {result.returncode}, {elapsed:.2f}s)"
                    )
                    if result.stderr:
                        print(result.stderr)
                if os.path.exists(output_path):
                    os.remove(output_path)
                return None, False

            # Check noise from the internal nvbench scratch output.
            all_pass = self._check_noise_from_scratch(output_path)

            return output_file, all_pass

        except subprocess.TimeoutExpired:
            if not QUIET:
                print(f"{Colors.RED}TIMEOUT{Colors.RESET} ({time.time() - beg:.2f}s)")
            return None, False
        except Exception as e:
            if not QUIET:
                print(
                    f"{Colors.RED}ERROR{Colors.RESET}: {e} ({time.time() - beg:.2f}s)"
                )
            return None, False

    def _check_noise_from_scratch(self, scratch_path: str) -> bool:
        """Check if all noise values in internal nvbench output are within threshold."""
        try:
            if not os.path.exists(scratch_path):
                return False  # Missing scratch output is a failure
            df = pd.read_csv(scratch_path)
            if len(df) == 0:
                return False  # Empty scratch output is a failure
            # Filter out skipped rows
            df = df[df.get("Skipped", "No") == "No"]
            if len(df) == 0:
                return False  # All configs skipped is a failure
            if "Noise.1" in df.columns:  # GPU noise in nvbench scratch output.
                return not any(
                    exceeds_limit(float(noise) * 100.0, self.max_noise_pct)
                    for noise in df["Noise.1"].dropna()
                )
            return True
        except Exception:
            return False

    def collect_results(self, output_file: str, config_key: str) -> bool:
        """Collect results from an internal nvbench scratch output file."""
        filepath = os.path.join(self.bench_folder, output_file)

        if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
            VERBOSE and log_warning(
                f"Skipping '{output_file}': does not exist or is empty"
            )
            return False

        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            VERBOSE and log_warning(f"Error reading '{output_file}': {e}")
            return False

        entry = self.config.get(config_key, {}) if self.config else {}
        benchmark_name = entry.get(
            "benchmark", self.get_operator_from_benchmark_name(config_key)
        )
        if "Benchmark" not in df.columns:
            df.insert(0, "Benchmark", benchmark_name)
        else:
            df["Benchmark"] = benchmark_name

        if not REQUIRED_SCRATCH_COLS.issubset(df.columns):
            missing = REQUIRED_SCRATCH_COLS - set(df.columns)
            VERBOSE and log_warning(f"Skipping {config_key}: Missing columns {missing}")
            return False

        df_filtered = df[df["Skipped"] == "No"].copy()
        if len(df_filtered) == 0:
            VERBOSE and log_warning(f"No valid results found in {config_key}")
            return False

        df_filtered.insert(0, "Language", self.language)
        df_filtered.insert(0, CONFIG_KEY_COLUMN, config_key)
        # Stamp the config entry's tier so combined-tier outputs (and any
        # downstream filtering / per-tier baseline compare) can disambiguate
        # rows by tier.
        df_filtered["tier"] = entry.get("tier")
        df_filtered = self._transform_scratch_columns(df_filtered)

        has_noise_failures = False
        if "GPU Noise (%)" in df_filtered.columns:
            has_noise_failures = any(
                exceeds_limit(float(noise), self.max_noise_pct)
                for noise in df_filtered["GPU Noise (%)"].dropna()
            )

        self.results.append(df_filtered)
        self.output_files.append(filepath)
        return not has_noise_failures

    def _transform_scratch_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform nvbench scratch columns to standardized format."""
        # Time: seconds → microseconds
        for col in ["GPU Time (sec)", "CPU Time (sec)"]:
            if col in df.columns:
                new_col = col.replace("(sec)", "(µs)")
                df[new_col] = df[col] * 1_000_000
                df.drop(columns=[col], inplace=True)

        # Noise: fractions → percentages
        rename_map = {"Noise": "CPU Noise (%)", "Noise.1": "GPU Noise (%)"}
        for old_col, new_col in rename_map.items():
            if old_col in df.columns:
                df[new_col] = df[old_col] * 100
                df.drop(columns=[old_col], inplace=True)

        # Stamp the GPU power cap (W) so baselines can split per configuration.
        # nvbench's Device Name alone is ambiguous: e.g. H100 PCIe boxes ship at
        # both 350W and 310W TDP and perform ~26% apart on identical kernels.
        df["Power Cap (W)"] = query_power_cap_w()

        # Stamp the actually-applied SM clock.  H100 PCIe pools contain silicon
        # variants whose supported-clocks lists differ; even with the same
        # `-lgc` request, half the pool can land at a different clock and
        # produce a bimodal baseline.  Recording the committed value lets
        # baselines route per (SKU, Power Cap, Locked SM Clock).
        df["Locked SM Clock (MHz)"] = query_locked_sm_clock_mhz()

        # Stamp the vBIOS version.  Even within a single (SKU, Power Cap,
        # Locked SM Clock) bucket, H100 PCIe silicon revisions perform
        # measurably differently on bandwidth/atomics-bound ops (observed
        # 12% spread on histogrameq between two pools both locked to
        # 1095 MHz).  Routing by vBIOS isolates each revision into its own
        # baseline.
        df["vBIOS Version"] = query_vbios_version()

        # Note: dtype normalization is done in ResultsProcessor._normalize_dtypes()
        return df


class CppBenchmarkRunner(BenchmarkRunner):
    """Runner for C++ benchmarks."""

    def __init__(
        self,
        bench_folder: str,
        operators: Optional[List[str]],
        keep_outputs: bool = False,
        max_noise_pct: float = DEFAULT_BENCHMARK_QUALITY.max_noise_pct,
        config_file: Optional[str] = None,
        tiers: Optional[set] = None,
        config_keys: Optional[List[str]] = None,
        warmup_cap: Optional[int] = None,
    ):
        super().__init__(
            "bench_",
            bench_folder,
            operators,
            "cpp",
            keep_outputs,
            max_noise_pct,
            config_file=config_file,
            tiers=tiers,
            config_keys=config_keys,
            warmup_cap=warmup_cap,
        )
        self.config = self._load_config(need_axis_args=True)

    def build_command(
        self,
        benchmark_path: str,
        extra_args: List[str],
        output_file: str,
        config_key: Optional[str] = None,
    ) -> List[str]:
        """Build command for C++ benchmark with --axis arguments from config."""
        if config_key is None:
            config_key = self.get_operator_from_benchmark_name(
                os.path.basename(benchmark_path)
            )
        axis_args = self.generate_axis_args(config_key, self.config)
        return [benchmark_path] + axis_args + extra_args + ["--csv", output_file]


class PythonBenchmarkRunner(BenchmarkRunner):
    """Runner for Python benchmarks."""

    def __init__(
        self,
        bench_folder: str,
        operators: Optional[List[str]],
        keep_outputs: bool = False,
        max_noise_pct: float = DEFAULT_BENCHMARK_QUALITY.max_noise_pct,
        config_file: Optional[str] = None,
        tiers: Optional[set] = None,
        config_keys: Optional[List[str]] = None,
        warmup_cap: Optional[int] = None,
    ):
        super().__init__(
            "bench_",
            bench_folder,
            operators,
            "python",
            keep_outputs,
            max_noise_pct,
            config_file=config_file,
            tiers=tiers,
            config_keys=config_keys,
            warmup_cap=warmup_cap,
        )
        self.config = self._load_config(need_axis_args=False)
        self._verify_cuda_available()

    def _python_package_path(self) -> Optional[Path]:
        """Return the build-tree Python package path when available.

        Source-tree runs against build-rel/bin should exercise the freshly
        built cvcuda extension instead of an older user-site install. Installed
        benchmark layouts do not have this sibling directory, so they keep the
        ambient Python environment.
        """
        candidate = Path(self.bench_folder).resolve().parent / "python3"
        if (candidate / "cvcuda" / "__init__.py").exists():
            return candidate
        return None

    def benchmark_env(self) -> Optional[dict]:
        env = super().benchmark_env()
        package_path = self._python_package_path()
        if package_path is not None:
            existing = env.get("PYTHONPATH")
            env["PYTHONPATH"] = (
                str(package_path)
                if not existing
                else os.pathsep.join([str(package_path), existing])
            )
        return env

    def _verify_cuda_available(self):
        """Verify CUDA is accessible from Python before running benchmarks."""
        try:
            # Run a quick import check in a subprocess.  We add the bench directory
            # to sys.path so that python_bench_utils can be imported — it applies
            # the cuda.pathfinder patch (for cuda-pathfinder >= 1.4) before importing
            # cvcuda, ensuring the patch is active when cuda.bench is loaded.
            # Note: we only check cvcuda and cuda.bench (pynvbench), not cuda.cuda
            # which requires the cuda-python package.
            bench_dir = str(Path(__file__).parent)
            check_script = f"""\
import sys
sys.path.insert(0, {bench_dir!r})
print(f'Python: {{sys.executable}}')
try:
    import python_bench_utils  # applies cuda.pathfinder patch; imports cvcuda
    import cvcuda
    print(f'cvcuda: {{cvcuda.__file__}}')
except ImportError as e:
    print(f'cvcuda import error: {{e}}')
    sys.exit(1)
try:
    import cuda.bench
    print(f'cuda.bench (pynvbench): OK')
except ImportError as e:
    print(f'cuda.bench import error: {{e}}')
    sys.exit(1)
"""
            result = subprocess.run(
                [sys.executable, "-c", check_script],
                capture_output=True,
                text=True,
                timeout=30,
                env=self.benchmark_env(),
            )
            if result.returncode != 0:
                log_warning("Python environment check failed:")
                for line in result.stdout.strip().split("\n"):
                    print(f"  {line}")
                if result.stderr:
                    print(f"  stderr: {result.stderr.strip()}")
            else:
                # Always show this info to help debug CI issues
                log_info("Python benchmark environment:")
                for line in result.stdout.strip().split("\n"):
                    print(f"  {line}")
        except Exception as e:
            log_warning(f"Python environment check failed: {e}")

    def build_command(
        self,
        benchmark_path: str,
        extra_args: List[str],
        output_file: str,
        config_key: Optional[str] = None,
    ) -> List[str]:
        """Build command for Python benchmark with --config-key for aliasing."""
        cmd = [sys.executable, benchmark_path, "--csv", output_file]
        if config_key:
            cmd.extend(["--config-key", config_key])
        config_file = self.config_file or str(
            Path(__file__).resolve().parent / "config" / "bench_params.json"
        )
        cmd.extend(["--config-file", config_file])
        cmd.extend(extra_args)
        return cmd


# =============================================================================
# PAIRED BENCHMARK DISCOVERY
# =============================================================================


def discover_operator_pairs(
    cpp_runner: Optional[BenchmarkRunner],
    python_runner: Optional[BenchmarkRunner],
    config_keys: Optional[List[str]] = None,
) -> List[tuple]:
    """Discover all operators and their C++/Python benchmark paths.

    Returns list of (config_key, cpp_path_or_None, python_path_or_None).
    """
    cpp_map = dict(cpp_runner.discover_benchmarks()) if cpp_runner else {}
    python_map = dict(python_runner.discover_benchmarks()) if python_runner else {}
    if config_keys is not None:
        all_keys = [key for key in config_keys if key in cpp_map or key in python_map]
    elif (cpp_runner and cpp_runner.operators) or (
        python_runner and python_runner.operators
    ):
        all_keys = []
        seen = set()
        for key in list(cpp_map) + list(python_map):
            if key not in seen:
                all_keys.append(key)
                seen.add(key)
    else:
        all_keys = sorted(set(cpp_map.keys()) | set(python_map.keys()))
    return [(key, cpp_map.get(key), python_map.get(key)) for key in all_keys]


# =============================================================================
# CLI ENTRY POINT
# =============================================================================


def _format_unknown_operator_error(unknown: List[str], available: List[str]) -> str:
    details = []
    for operator_name in unknown:
        suggestions = difflib.get_close_matches(operator_name, available, n=3)
        suffix = f" (did you mean {', '.join(suggestions)}?)" if suggestions else ""
        details.append(f"{operator_name}{suffix}")
    return (
        f"--operator: unknown operator(s): {', '.join(details)}. "
        "Use --list-operators to see valid names."
    )


def parse_args():
    """Parse command line arguments for unified benchmark runner."""
    parser = argparse.ArgumentParser(
        description="Run selected CV-CUDA C++ and Python benchmarks and write one result.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  run_bench.py --lang python --operator resize
  run_bench.py --output bench_output.json
  run_bench.py . -- --axis shape=1x1080x1920

Output:
  CSV by default; use a .json path for the baseline tools. GPU diagnostic
  files are written beside the result.

Exit status:
  0  A result was written and final validation passed or was skipped.
  1  The run could not complete, output failed, or validation failed.
  2  Arguments or benchmark configuration could not be parsed.
""",
    )
    parser.add_argument(
        "--lang",
        choices=["cpp", "python", "both"],
        default="both",
        help="Language to run: cpp, python, or both (default: both)",
    )
    parser.add_argument(
        "--operator",
        "--operators",
        dest="operator",
        type=str,
        default=None,
        help=(
            "Run exact operator name(s), comma-separated or quoted "
            "space-separated. Names must be listed in "
            "bench/config/bench_params.json. Example: --operator resize,gaussian"
        ),
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        default=None,
        help=(
            "Deprecated alias for --operator. Matching is exact against "
            "operator names; substring matching is no longer supported."
        ),
    )
    parser.add_argument(
        "--list-operators",
        action="store_true",
        default=False,
        help="List operator names available in the benchmark manifest and exit.",
    )
    parser.add_argument(
        "--keep-outputs",
        action="store_true",
        default=False,
        help="Keep individual benchmark output CSV files (for debugging)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Show verbose output including full command lines",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="Suppress per-benchmark output, show only summary",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        default=False,
        help="Disable colored output (useful for CI/logging)",
    )
    parser.add_argument(
        "--max-noise-pct",
        type=float,
        default=DEFAULT_BENCHMARK_QUALITY.max_noise_pct,
        help="Maximum allowed noise/stddev percentage (default: %(default)s)",
    )
    parser.add_argument(
        "--max-perf-diff-pct",
        type=float,
        default=DEFAULT_BENCHMARK_QUALITY.max_perf_diff_pct,
        help=(
            "Maximum allowed performance difference between C++ and Python "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--max-perf-diff-us",
        type=float,
        default=DEFAULT_BENCHMARK_QUALITY.max_perf_diff_us,
        help=(
            "Maximum absolute C++/Python timing difference in microseconds "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        default=False,
        help=(
            "Do not fail the final run on noise or C++/Python parity checks. "
            "Result rows still include validation statuses."
        ),
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=0,
        help="Max retries per operator if validation fails (default: 0)",
    )
    parser.add_argument(
        "--bench-min-time",
        type=float,
        default=0.25,
        help="nvbench --min-time per-config measurement floor in seconds "
        "(default: 0.25; nvbench's own default is 0.5). Lower = faster runs; "
        "stays well below the --max-noise-pct gate. Pass 0 to use nvbench's default.",
    )
    parser.add_argument(
        "--bench-max-noise",
        type=float,
        default=1.0,
        help="nvbench --max-noise stopping criterion in percent rel-stddev "
        "(default: 1.0; nvbench's own default is 0.5). Higher = faster runs; "
        "must stay below the configured --max-noise-pct gate. "
        "Pass 0 to use nvbench's default.",
    )
    parser.add_argument(
        "--warmup-cap",
        type=parse_warmup_cap,
        default=None,
        help=(
            "Cap configured warmup iterations for every C++ and Python "
            "benchmark. Pass 0 to disable warmup; omit to preserve each "
            "operator's configured count."
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help=(
            "Output file path. .csv writes the legacy combined CSV, .json writes "
            "the baseline JSON payload. Default: bench_output.csv in bench_folder, "
            "or bench_output_<stem>.csv when --config-file is set."
        ),
    )
    parser.add_argument(
        "--tier",
        type=str,
        default="basic",
        help="Tiers: basic, advanced, or basic,advanced (default: basic).",
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default=None,
        help=(
            "Path to a bench_params.json config file. Defaults to "
            "bench/config/bench_params.json. Power-user escape hatch — most "
            "users want --tier or --config-key instead."
        ),
    )
    parser.add_argument(
        "--config-key",
        type=str,
        default=None,
        help=(
            "Run exact config key(s), comma-separated. This selection bypasses "
            "--tier and --operator."
        ),
    )
    parser.add_argument(
        "bench_folder",
        nargs="?",
        default=None,
        metavar="BENCH_FOLDER",
        help=(
            "Directory containing benchmark executables and scripts (default: "
            "this script's directory). Supply it before forwarded arguments."
        ),
    )
    parser.add_argument(
        "bench_args",
        nargs=argparse.REMAINDER,
        metavar="BENCH_ARG",
        help=(
            "Arguments forwarded to each benchmark. Put BENCH_FOLDER and -- "
            "before them."
        ),
    )

    args = parser.parse_args()

    if args.bench_folder is None:
        args.bench_folder = str(Path(__file__).parent)

    if args.operator and args.benchmarks:
        log_error("Use only one of --operator or --benchmarks")
        sys.exit(2)

    operator_arg = args.operator or args.benchmarks

    # Parse selectors and config metadata; fail fast on typos.
    sys.path.insert(0, str(Path(__file__).parent / "config"))
    from load_config import (
        load_bench_config,
        load_bench_manifest,
        parse_config_key_arg,
        parse_operator_arg,
        parse_tier_arg,
    )

    try:
        manifest = load_bench_manifest(args.config_file)
    except Exception as e:
        log_error(f"Failed to load benchmark manifest: {e}")
        sys.exit(2)

    if args.list_operators:
        for operator_name in manifest:
            print(operator_name)
        sys.exit(0)

    args.operators = None
    if operator_arg:
        try:
            args.operators = parse_operator_arg(operator_arg)
        except ValueError as e:
            option_name = "--benchmarks" if args.benchmarks else "--operator"
            log_error(str(e).replace("--operator", option_name))
            sys.exit(2)

        available = list(manifest)
        unknown = [
            operator_name
            for operator_name in args.operators
            if operator_name not in manifest
        ]
        if unknown:
            option_name = "--benchmarks" if args.benchmarks else "--operator"
            message = _format_unknown_operator_error(unknown, available).replace(
                "--operator", option_name
            )
            log_error(message)
            sys.exit(2)

    try:
        args.tiers = parse_tier_arg(args.tier)
    except ValueError as e:
        log_error(str(e))
        sys.exit(2)

    args.config_keys = None
    if args.config_key:
        try:
            args.config_keys = parse_config_key_arg(args.config_key)
            config = load_bench_config(args.config_file)
        except ValueError as e:
            log_error(str(e))
            sys.exit(2)
        except Exception as e:
            log_error(f"Failed to load benchmark config: {e}")
            sys.exit(2)

        unknown = [key for key in args.config_keys if key not in config]
        if unknown:
            log_error(
                f"--config-key: unknown key(s) {unknown}; "
                "run with a key present in bench/config/bench_params.json"
            )
            sys.exit(2)

        missing_benchmark = [
            key
            for key in args.config_keys
            if not isinstance(config.get(key), dict) or "benchmark" not in config[key]
        ]
        if missing_benchmark:
            log_error(
                "--config-key: key(s) missing required benchmark field "
                f"{missing_benchmark}"
            )
            sys.exit(2)

    # Set default output path if not specified.
    #   --tier basic       -> bench_output.csv              (legacy default)
    # An explicit --config-file overrides this with stem-based naming, for
    # power users running ad-hoc custom configs.
    if args.output is None:
        if args.config_file:
            stem = Path(args.config_file).stem  # e.g. "bench_params_smoke"
            suffix = stem.replace("bench_params", "").lstrip("_")
            if suffix:
                args.output = os.path.join(
                    args.bench_folder, f"bench_output_{suffix}.csv"
                )
            else:
                args.output = os.path.join(args.bench_folder, "bench_output.csv")
        else:
            args.output = os.path.join(args.bench_folder, "bench_output.csv")

    # Apply global settings
    global VERBOSE, QUIET
    VERBOSE = args.verbose
    QUIET = args.quiet
    if args.no_color:
        Colors.disable()

    # Inject nvbench stopping-criterion overrides into extra args, unless the
    # user already passed the same flag explicitly. This trims per-config
    # measurement time on the all-operator run while keeping noise well below
    # the --max-noise-pct gate.
    user_extra = list(args.bench_args or [])
    nvbench_prefix: List[str] = []
    if (
        args.bench_min_time
        and args.bench_min_time > 0
        and "--min-time" not in user_extra
    ):
        nvbench_prefix += ["--min-time", str(args.bench_min_time)]
    if (
        args.bench_max_noise
        and args.bench_max_noise > 0
        and "--max-noise" not in user_extra
    ):
        nvbench_prefix += ["--max-noise", str(args.bench_max_noise)]
    args.bench_args = nvbench_prefix + user_extra

    return args


def _print_banner(title: str):
    """Print a colored section banner."""
    print(f"\n{Colors.CYAN}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.CYAN}{title:<60}{Colors.RESET}")
    print(f"{Colors.CYAN}{'=' * 60}{Colors.RESET}\n")


def _get_gpu_info() -> str:
    """Get GPU name using nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip().split("\n")[0]  # First GPU
    except Exception:
        pass
    return "Unknown GPU"


def _baseline_config_paths(config_file: Optional[str]) -> Tuple[List[Path], Path]:
    sys.path.insert(0, str(Path(__file__).parent / "config"))
    from load_config import load_bench_manifest

    if config_file is None:
        base_dir = DEFAULT_CONFIG_DIR
    else:
        config_path = Path(config_file).resolve()
        base_dir = config_path if config_path.is_dir() else config_path.parent

    manifest = load_bench_manifest(config_file)
    paths: List[Path] = []
    seen = set()
    for spec in manifest.values():
        raw_path = Path(spec["config"])
        if raw_path.is_absolute():
            path = raw_path
        elif raw_path.exists():
            path = raw_path.resolve()
        else:
            path = (base_dir / raw_path).resolve()
        if path not in seen:
            seen.add(path)
            paths.append(path)

    sku_map_path = base_dir / "sku_map.json"
    if not sku_map_path.is_file():
        sku_map_path = DEFAULT_SKU_MAP_PATH
    return paths, sku_map_path


def _write_output_json(args, df_combined: pd.DataFrame) -> None:
    config_paths, sku_map_path = _baseline_config_paths(args.config_file)
    index = load_config_index(paths=config_paths)
    payload = baseline_payload_from_dataframe(
        df_combined,
        index=index,
        sku_map_path=sku_map_path,
        source=Path(args.output),
    )
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=4) + "\n")
    log_success(f"Results written to {out_path}")


def _write_output_csv(args, df_combined: pd.DataFrame) -> None:
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ResultsProcessor(
        args.max_noise_pct, args.max_perf_diff_pct, args.max_perf_diff_us
    ).finalize_for_output(df_combined.copy()).to_csv(out_path, index=False)
    log_success(f"Results written to {out_path}")


def _write_output(args, df_combined: pd.DataFrame) -> None:
    suffix = Path(args.output).suffix.lower()
    if suffix == ".json":
        _write_output_json(args, df_combined)
    else:
        _write_output_csv(args, df_combined)


def _run_paired_benchmarks(args) -> tuple:
    """Run C++/Python benchmark pairs per operator with immediate parity check."""
    # Create runners
    cpp_runner = (
        CppBenchmarkRunner(
            bench_folder=args.bench_folder,
            operators=args.operators,
            keep_outputs=args.keep_outputs,
            max_noise_pct=args.max_noise_pct,
            config_file=args.config_file,
            tiers=args.tiers,
            config_keys=args.config_keys,
            warmup_cap=args.warmup_cap,
        )
        if args.lang in ["cpp", "both"]
        else None
    )

    python_runner = (
        PythonBenchmarkRunner(
            bench_folder=args.bench_folder,
            operators=args.operators,
            keep_outputs=args.keep_outputs,
            max_noise_pct=args.max_noise_pct,
            config_file=args.config_file,
            tiers=args.tiers,
            config_keys=args.config_keys,
            warmup_cap=args.warmup_cap,
        )
        if args.lang in ["python", "both"]
        else None
    )

    pairs = discover_operator_pairs(cpp_runner, python_runner, args.config_keys)
    if not pairs:
        log_error("No benchmarks found to run")
        sys.exit(1)

    _print_banner("=== Running Benchmarks ===")
    log_info(f"GPU: {_get_gpu_info()}")
    log_info(f"Found {len(pairs)} benchmarks")
    if args.warmup_cap is not None:
        log_info(f"Warmup iterations capped at {args.warmup_cap}")

    processor = ResultsProcessor(
        args.max_noise_pct, args.max_perf_diff_pct, args.max_perf_diff_us
    )

    def run_one(runner, path, config_key):
        """Run a single benchmark and update runner stats. Returns df or None."""
        output_file, noise_pass = runner.run_benchmark(
            path, args.bench_args, config_key
        )
        runner.summary.total += 1
        if output_file:
            results_before = len(runner.results)
            collect_pass = runner.collect_results(output_file, config_key)
            # Always return df if results were collected, regardless of noise
            if len(runner.results) > results_before:
                runner.summary.successful += 1 if (noise_pass and collect_pass) else 0
                runner.summary.failed += 0 if (noise_pass and collect_pass) else 1
                return runner.results[-1]
        runner.summary.failed += 1
        return None

    # Collect all operator results in memory, then write one JSON artifact.
    total_retries = 0
    max_attempts = args.max_retries + 1  # +1 for initial attempt

    def check_validation_passed(df_op: pd.DataFrame) -> bool:
        """Check if all rows in processed dataframe passed validation."""
        if df_op is None or len(df_op) == 0:
            return False
        if "Status" not in df_op.columns:
            return True
        return not df_op["Status"].str.contains("FAIL").any()

    def get_failure_reasons(df_op: pd.DataFrame) -> List[str]:
        """Extract failure reasons from Status column for failed rows."""
        if df_op is None or "Status" not in df_op.columns:
            return []
        failed = df_op[df_op["Status"].str.contains("FAIL", na=False)]
        reasons = []
        for _, row in failed.drop_duplicates(subset=["Status"]).iterrows():
            status = row["Status"]
            # Extract the reason from "FAIL (reason)" format
            if "(" in status and ")" in status:
                _, _, after = status.partition("(")
                reason, _, _ = after.rpartition(")")
            else:
                reason = "unknown"
            # Build a descriptive message
            config_parts = []
            if "InOutDataType" in row and pd.notna(row["InOutDataType"]):
                config_parts.append(str(row["InOutDataType"]))
            if "shape" in row and pd.notna(row["shape"]):
                config_parts.append(str(row["shape"]))
            config_str = ", ".join(config_parts) if config_parts else ""
            reasons.append(f"{config_str}: {reason}" if config_str else reason)
        return reasons

    # Run each operator pair with retry logic
    for idx, (config_key, cpp_path, py_path) in enumerate(pairs, 1):
        if not QUIET:
            print(f"\n{Colors.BOLD}[{idx}/{len(pairs)}] {config_key}{Colors.RESET}")

        best_df_op = None
        validation_passed = False

        for attempt in range(1, max_attempts + 1):
            if attempt > 1:
                total_retries += 1
                if not QUIET:
                    print(
                        f"{Colors.YELLOW}  Retry {attempt - 1}/{args.max_retries} "
                        f"for {config_key}{Colors.RESET}"
                    )

            op_start = time.time()

            # Save state before running (for potential rollback on retry)
            cpp_results_before = len(cpp_runner.results) if cpp_runner else 0
            py_results_before = len(python_runner.results) if python_runner else 0

            cpp_df = (
                run_one(cpp_runner, cpp_path, config_key)
                if cpp_runner and cpp_path
                else None
            )
            py_df = (
                run_one(python_runner, py_path, config_key)
                if python_runner and py_path
                else None
            )
            op_elapsed = time.time() - op_start

            # Process results for this operator
            dfs = [df for df in [cpp_df, py_df] if df is not None]
            if dfs:
                has_both = cpp_df is not None and py_df is not None
                df_op = processor.process(dfs, compute_parity=has_both)

                validation_passed = check_validation_passed(df_op)

                if validation_passed or attempt == max_attempts:
                    # Keep these results (passed or last attempt)
                    best_df_op = df_op
                    break
                else:
                    # Print failure reasons before retrying
                    if not QUIET:
                        reasons = get_failure_reasons(df_op)
                        for reason in reasons:
                            print(f"{Colors.RED}    ✗ {reason}{Colors.RESET}")
                    # Rollback: remove results added in this attempt for retry
                    if cpp_runner and len(cpp_runner.results) > cpp_results_before:
                        cpp_runner.results.pop()
                    if python_runner and len(python_runner.results) > py_results_before:
                        python_runner.results.pop()
            else:
                # No results collected, can't retry
                break

        if best_df_op is not None:
            if not QUIET:
                print_parity_summary_table(best_df_op)

        if not QUIET:
            retry_info = f" (after {attempt - 1} retries)" if attempt > 1 else ""
            print(
                f"{Colors.GRAY}  Completed in {op_elapsed:.2f}s{retry_info}{Colors.RESET}"
            )

    if total_retries > 0:
        log_info(f"Total retries across all operators: {total_retries}")

    return cpp_runner, python_runner


def _process_and_validate(args, cpp_runner, python_runner) -> tuple:
    """Process results and run validation checks."""
    all_results = []
    if cpp_runner and cpp_runner.results:
        all_results.extend(cpp_runner.results)
    if python_runner and python_runner.results:
        all_results.extend(python_runner.results)

    if not all_results:
        log_warning("No benchmark results were successfully processed.")
        sys.exit(1)

    compute_parity = args.lang == "both" and cpp_runner and python_runner
    processor = ResultsProcessor(
        args.max_noise_pct, args.max_perf_diff_pct, args.max_perf_diff_us
    )
    df_combined = processor.process(all_results, compute_parity=compute_parity)

    validation_failed = False
    if not args.skip_validation:
        result = processor.validate(df_combined, check_parity=compute_parity)
        if result.has_errors:
            validation_failed = True
            processor.print_errors(result)
        else:
            log_success(
                f"All validations passed (noise <{args.max_noise_pct}%, "
                f"perf diff <{args.max_perf_diff_pct}% AND <{args.max_perf_diff_us:.0f}us)"
            )

    return df_combined, validation_failed


def _print_final_summary(args, df_combined):
    """Print final summary statistics."""
    print(f"\n{Colors.BOLD}--- Summary ---{Colors.RESET}")
    num_operators = (
        df_combined["Benchmark"].nunique() if "Benchmark" in df_combined.columns else 0
    )
    config_cols = get_config_cols(df_combined)
    num_configs = (
        len(df_combined.groupby(config_cols, dropna=False))
        if config_cols
        else len(df_combined)
    )

    failed_configs = 0
    if "Status" in df_combined.columns and config_cols:
        for _, group in df_combined.groupby(config_cols, dropna=False):
            if group["Status"].str.contains("FAIL").any():
                failed_configs += 1

    passed = num_configs - failed_configs
    print(
        f"  Operators: {num_operators}, Configurations: {num_configs} "
        f"({passed} passed, {failed_configs} failed)"
    )
    print(f"  Output: {args.output}")

    # Compute and display performance statistics
    if "Language" in df_combined.columns and "GPU Noise (%)" in df_combined.columns:
        cpp_rows = df_combined[df_combined["Language"] == "cpp"]
        py_rows = df_combined[df_combined["Language"] == "python"]

        stats = []

        # C++ noise statistics
        if len(cpp_rows) > 0:
            cpp_noise_pct = cpp_rows["GPU Noise (%)"].mean()
            cpp_noise_us = (
                cpp_rows["GPU Noise (µs)"].mean()
                if "GPU Noise (µs)" in cpp_rows.columns
                else None
            )
            if cpp_noise_us is not None:
                stats.append(
                    f"C++ noise: ±{cpp_noise_pct:.2f}% (±{cpp_noise_us:.1f}µs)"
                )
            else:
                stats.append(f"C++ noise: ±{cpp_noise_pct:.2f}%")

        # Python noise statistics
        if len(py_rows) > 0:
            py_noise_pct = py_rows["GPU Noise (%)"].mean()
            py_noise_us = (
                py_rows["GPU Noise (µs)"].mean()
                if "GPU Noise (µs)" in py_rows.columns
                else None
            )
            if py_noise_us is not None:
                stats.append(f"Py noise: ±{py_noise_pct:.2f}% (±{py_noise_us:.1f}µs)")
            else:
                stats.append(f"Py noise: ±{py_noise_pct:.2f}%")

        # Python overhead statistics (only when both languages present)
        if "Py overhead (%)" in df_combined.columns:
            overhead_pct = df_combined["Py overhead (%)"].dropna()
            overhead_us = (
                df_combined["Py overhead (µs)"].dropna()
                if "Py overhead (µs)" in df_combined.columns
                else None
            )
            if len(overhead_pct) > 0:
                mean_overhead_pct = overhead_pct.mean()
                if overhead_us is not None and len(overhead_us) > 0:
                    mean_overhead_us = overhead_us.mean()
                    stats.append(
                        f"Py overhead: +{mean_overhead_pct:.1f}% (+{mean_overhead_us:.0f}µs)"
                    )
                else:
                    stats.append(f"Py overhead: +{mean_overhead_pct:.1f}%")

        if stats:
            print(f"  {', '.join(stats)}")

    print(f"{Colors.GRAY}  View full results: cat {args.output}{Colors.RESET}")
    print(f"\n{Colors.BOLD}--- Benchmarking Complete ---{Colors.RESET}")


def main():
    args = parse_args()

    # Lock SM clock + start clock sampler before any kernels run.  Cleanup
    # registered via atexit so we always reset the lock and flush the log.
    init_clock_control(out_dir=str(Path(args.output).parent or "."))

    cpp_runner, python_runner = _run_paired_benchmarks(args)
    df_combined, validation_failed = _process_and_validate(
        args, cpp_runner, python_runner
    )

    output_failed = False
    try:
        _write_output(args, df_combined)
    except BaselineError as exc:
        output_failed = True
        log_error(f"Failed to write benchmark output: {exc}")

    # Cleanup internal nvbench scratch files unless requested for debugging.
    all_output_files = (cpp_runner.output_files if cpp_runner else []) + (
        python_runner.output_files if python_runner else []
    )
    if not args.keep_outputs:
        for f in all_output_files:
            try:
                os.path.exists(f) and os.remove(f)
            except Exception:
                pass
        all_output_files and log_info(
            f"Cleaned up {len(all_output_files)} individual output file(s)"
        )

    _print_final_summary(args, df_combined)
    if validation_failed:
        log_error("Benchmarking completed with validation errors")
    sys.exit(1 if validation_failed or output_failed else 0)


if __name__ == "__main__":
    main()
