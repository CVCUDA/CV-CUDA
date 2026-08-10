#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark two CV-CUDA Python wheels with the same current harness.

The command preserves each raw benchmark run and writes a compatibility and
performance comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime
from email.parser import Parser
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd

from _internal.axes import format_axes
from _internal.baselines import (
    CONFIG_KEY_COLUMN,
    DEFAULT_MAX_BASELINE_REGRESSION_PCT,
    DEVICE_NAME_COLUMN,
    GPU_TIME_COLUMN,
    LOCKED_CLOCK_COLUMN,
    POWER_CAP_COLUMN,
    VBIOS_COLUMN,
    BaselineError,
    ConfigIndex,
    case_key_from_row,
    expected_case_keys_for_entry,
    load_config_index,
    parse_case_key,
)
from config.load_config import (
    load_bench_manifest,
    parse_config_key_arg,
    parse_operator_arg,
    parse_tier_arg,
)
from _internal.quality import DEFAULT_BENCHMARK_QUALITY, exceeds_limit
from _internal.warmup import parse_warmup_cap

BENCH_DIR = Path(__file__).resolve().parent
RUNNER_FILES = ("run_bench.py",)
INTERNAL_FILES = (
    "__init__.py",
    "axes.py",
    "baselines.py",
    "quality.py",
    "warmup.py",
)
CONFIG_FILES = ("bench_params.json", "sku_map.json", "load_config.py", "axis_order.py")
GPU_NOISE_COLUMN = "GPU Noise (%)"
STATUS_COLUMN = "Status"
STABLE_FINGERPRINT_COLUMNS = (DEVICE_NAME_COLUMN, POWER_CAP_COLUMN, VBIOS_COLUMN)
REQUIRED_RUN_COLUMNS = frozenset(
    {
        "Benchmark",
        CONFIG_KEY_COLUMN,
        "Language",
        GPU_TIME_COLUMN,
        GPU_NOISE_COLUMN,
        STATUS_COLUMN,
        DEVICE_NAME_COLUMN,
        POWER_CAP_COLUMN,
        LOCKED_CLOCK_COLUMN,
    }
)
NORMALIZED_COLUMNS = (
    "benchmark",
    "config_key",
    "case_key",
    "axes",
    "gpu_us",
    "noise_pct",
    "status",
    "device_name",
    "power_cap",
    "vbios",
    "clock_mhz",
)
COMPARISON_COLUMNS = (
    "Benchmark",
    "config_key",
    "case_key",
    "axes",
    "baseline_gpu_us",
    "candidate_gpu_us",
    "candidate_delta_pct",
    "candidate_speedup",
    "baseline_noise_pct",
    "candidate_noise_pct",
    "baseline_status",
    "candidate_status",
    "comparison_status",
)
OPERATOR_RESULT_LIMIT = 5


PROBE_SCRIPT = r"""
import importlib.metadata
import json
import sys
from pathlib import Path

expected_name, expected_version, output_path = sys.argv[1:]

# python_bench_utils applies the cuda-pathfinder patch used by the benchmarks.
import pandas
import python_bench_utils  # noqa: F401
import cuda.bench
import cvcuda
import cupy

dist = importlib.metadata.distribution(expected_name)
environment = Path(sys.prefix).resolve()
package = Path(dist.locate_file("cvcuda")).resolve()
module = Path(cvcuda.__file__).resolve()
extension = Path(cvcuda._cvcuda.__file__).resolve()
errors = []
if dist.version != expected_version:
    errors.append(f"installed version {dist.version!r}, expected {expected_version!r}")
for label, path in (("package", package), ("module", module), ("extension", extension)):
    try:
        path.relative_to(environment)
    except ValueError:
        errors.append(f"cvcuda {label} is outside the wheel environment: {path}")
try:
    module.relative_to(package)
    extension.relative_to(package)
except ValueError:
    errors.append("cvcuda module/extension do not belong to the installed distribution")

# Exercise device-backed creation before either long run starts.
stream = cvcuda.Stream()
tensor = cvcuda.Tensor((1, 1, 1, 1), cvcuda.Type.U8, "NHWC")
stream.sync()

payload = {
    "passed": not errors,
    "errors": errors,
    "python": sys.executable,
    "distribution": dist.metadata["Name"],
    "distribution_version": dist.version,
    "cvcuda_version": getattr(cvcuda, "__version__", None),
    "cvcuda_module": str(module),
    "cvcuda_extension": str(extension),
    "dependencies": {
        "pandas": pandas.__version__,
        "cupy": cupy.__version__,
        "cuda_bench": cuda.bench.__file__,
    },
}
Path(output_path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
if errors:
    raise SystemExit("; ".join(errors))
"""


class WheelComparisonError(RuntimeError):
    """Raised when wheel setup or benchmark artifacts are invalid."""


@dataclass(frozen=True)
class WheelMetadata:
    path: Path
    name: str
    canonical_name: str
    version: str
    sha256: str

    @property
    def label(self) -> str:
        return f"{self.name} {self.version} [{self.sha256[:12]}] ({self.path})"


@dataclass
class Comparison:
    table: pd.DataFrame
    expected_case_operators: dict[str, str]
    missing_baseline_cases: tuple[str, ...]
    missing_candidate_cases: tuple[str, ...]
    baseline_fingerprint: Optional[tuple[Optional[str], ...]]
    candidate_fingerprint: Optional[tuple[Optional[str], ...]]
    baseline_clocks: tuple[str, ...]
    candidate_clocks: tuple[str, ...]
    baseline_row_count: int
    candidate_row_count: int
    threshold: float

    def rows(self, status: str) -> pd.DataFrame:
        return self.table[self.table["comparison_status"] == status]

    @property
    def regressions(self) -> pd.DataFrame:
        return self.rows("regression").sort_values(
            ["candidate_delta_pct", "case_key"], ascending=[False, True]
        )

    @property
    def improvements(self) -> pd.DataFrame:
        return self.rows("improvement").sort_values(["candidate_delta_pct", "case_key"])

    @property
    def comparable(self) -> pd.DataFrame:
        return self.table[
            self.table["comparison_status"].isin(
                ("regression", "improvement", "neutral")
            )
        ]

    @property
    def operator_geomean_speedups(self) -> pd.Series:
        if self.comparable.empty:
            return pd.Series(dtype=float)
        return self.comparable.groupby("Benchmark", sort=True)["candidate_speedup"].agg(
            lambda values: math.exp(
                sum(math.log(value) for value in values) / len(values)
            )
        )

    @property
    def mean_operator_speedup(self) -> Optional[float]:
        operator_speedups = self.operator_geomean_speedups
        if operator_speedups.empty:
            return None
        return float(operator_speedups.mean())

    @property
    def shared_expected_cases(self) -> set[str]:
        both_present = self.table[
            self.table["comparison_status"].isin(
                ("regression", "improvement", "neutral", "excluded")
            )
        ]
        return set(both_present["case_key"]) & set(self.expected_case_operators)

    @property
    def operator_compatibility(self) -> dict[str, str]:
        shared_cases = self.shared_expected_cases
        cases_by_operator: dict[str, set[str]] = {}
        for case_key, operator in self.expected_case_operators.items():
            cases_by_operator.setdefault(operator, set()).add(case_key)

        compatibility = {}
        for operator, case_keys in sorted(cases_by_operator.items()):
            shared_count = len(case_keys & shared_cases)
            if shared_count == len(case_keys):
                status = "fully_compatible"
            elif shared_count:
                status = "partially_compatible"
            else:
                status = "incompatible_or_not_present"
            compatibility[operator] = status
        return compatibility

    @property
    def failed(self) -> bool:
        failures = {
            "regression",
            "excluded",
            "missing_in_candidate",
            "only_in_candidate",
            "missing_from_both",
        }
        return bool(
            not self.baseline_row_count
            or not self.candidate_row_count
            or self.table["comparison_status"].isin(failures).any()
            or self.missing_baseline_cases
            or self.missing_candidate_cases
        )


def _canonical_distribution_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as wheel_file:
        for block in iter(lambda: wheel_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_wheel_metadata(path: Path) -> WheelMetadata:
    path = path.expanduser().resolve()
    if not path.is_file() or path.suffix != ".whl":
        raise WheelComparisonError(
            f"wheel does not exist or is not a .whl file: {path}"
        )
    try:
        with zipfile.ZipFile(path) as wheel:
            metadata_paths = [
                name
                for name in wheel.namelist()
                if name.endswith(".dist-info/METADATA") and name.count("/") == 1
            ]
            if len(metadata_paths) != 1:
                raise WheelComparisonError(
                    f"{path}: expected one top-level .dist-info/METADATA, "
                    f"found {len(metadata_paths)}"
                )
            metadata = Parser().parsestr(wheel.read(metadata_paths[0]).decode())
    except (OSError, UnicodeDecodeError, zipfile.BadZipFile) as exc:
        raise WheelComparisonError(
            f"could not read wheel metadata from {path}: {exc}"
        ) from exc

    name, version = metadata.get("Name"), metadata.get("Version")
    if not name or not version:
        raise WheelComparisonError(
            f"{path}: wheel metadata must contain Name and Version"
        )
    canonical_name = _canonical_distribution_name(name)
    if canonical_name != "cvcuda" and not canonical_name.startswith("cvcuda-cu"):
        raise WheelComparisonError(
            f"{path}: {name!r} is not a CV-CUDA wheel distribution"
        )
    return WheelMetadata(path, name, canonical_name, version, _sha256(path))


def _resolve_asset(source_dir: Path, *relative_paths: str) -> Path:
    for relative_path in relative_paths:
        candidate = source_dir / relative_path
        if candidate.is_file():
            return candidate
    choices = ", ".join(str(source_dir / path) for path in relative_paths)
    raise WheelComparisonError(f"benchmark asset not found; checked {choices}")


def stage_benchmark_harness(source_dir: Path, destination: Path) -> None:
    """Stage the flattened Python benchmark layout installed by CMake."""
    source_dir = source_dir.resolve()
    destination.mkdir(parents=True)
    for filename in RUNNER_FILES:
        shutil.copy2(_resolve_asset(source_dir, filename), destination / filename)
    staged_internal = destination / "_internal"
    staged_internal.mkdir()
    for filename in INTERNAL_FILES:
        shutil.copy2(
            _resolve_asset(source_dir / "_internal", filename),
            staged_internal / filename,
        )

    source_config, staged_config = source_dir / "config", destination / "config"
    staged_config.mkdir()
    for filename in CONFIG_FILES:
        shutil.copy2(_resolve_asset(source_config, filename), staged_config / filename)
    shutil.copytree(source_config / "operators", staged_config / "operators")
    shutil.copy2(
        _resolve_asset(
            source_dir, "python_bench_utils.py", "python/python_bench_utils.py"
        ),
        destination / "python_bench_utils.py",
    )

    manifest_path = source_config / "bench_params.json"
    try:
        manifest = load_bench_manifest(str(manifest_path))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise WheelComparisonError(
            f"invalid benchmark manifest {manifest_path}: {exc}"
        ) from exc
    for operator, spec in manifest.items():
        filename = spec.get("python")
        if not filename:
            raise WheelComparisonError(
                f"benchmark manifest operator {operator!r} has no Python script"
            )
        shutil.copy2(
            _resolve_asset(source_dir, filename, f"python/ops/{filename}"),
            destination / filename,
        )


def _clean_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _run_logged(
    command: Sequence[str],
    log_path: Path,
    *,
    cwd: Optional[Path] = None,
    append: bool = False,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a" if append else "w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n\n")
        log.flush()
        try:
            return subprocess.run(
                list(command),
                cwd=cwd,
                env=_clean_subprocess_env(),
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            ).returncode
        except OSError as exc:
            log.write(f"\n{exc}\n")
            return 127


def create_wheel_environment(
    base_python: Path, environment_dir: Path, wheel: WheelMetadata, log_path: Path
) -> Path:
    create = [
        str(base_python),
        "-m",
        "venv",
        "--system-site-packages",
        str(environment_dir),
    ]
    if _run_logged(create, log_path) != 0:
        raise WheelComparisonError(
            f"could not create wheel environment; see {log_path}"
        )
    environment_python = environment_dir / "bin" / "python"
    install = [
        str(environment_python),
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-input",
        "--force-reinstall",
        "--no-deps",
        str(wheel.path),
    ]
    if _run_logged(install, log_path, append=True) != 0:
        raise WheelComparisonError(f"could not install {wheel.path}; see {log_path}")
    return environment_python


def probe_wheel_environment(
    environment_python: Path,
    harness_dir: Path,
    wheel: WheelMetadata,
    output_path: Path,
    log_path: Path,
) -> None:
    command = [
        str(environment_python),
        "-c",
        PROBE_SCRIPT,
        wheel.name,
        wheel.version,
        str(output_path),
    ]
    returncode = _run_logged(command, log_path, cwd=harness_dir)
    if returncode != 0 or not output_path.is_file():
        raise WheelComparisonError(
            f"wheel environment did not pass import/dependency verification; see {log_path}"
        )
    payload = json.loads(output_path.read_text())
    if not payload.get("passed"):
        raise WheelComparisonError(
            f"wheel environment verification failed: {payload.get('errors', [])}"
        )


def build_runner_command(
    args: argparse.Namespace,
    environment_python: Path,
    harness_dir: Path,
    output_csv: Path,
) -> list[str]:
    command = [
        str(environment_python),
        str(harness_dir / "run_bench.py"),
        "--lang",
        "python",
        "--no-color",
        "--output",
        str(output_csv),
        "--tier",
        args.tier,
        "--max-noise-pct",
        str(args.max_noise_pct),
    ]
    for flag, value in (
        ("--operator", args.operator),
        ("--config-key", args.config_key),
        ("--warmup-cap", args.warmup_cap),
        ("--bench-min-time", args.bench_min_time),
        ("--bench-max-noise", args.bench_max_noise),
    ):
        if value is not None:
            command.extend([flag, str(value)])
    return command + [str(harness_dir)]


def run_benchmark(
    args: argparse.Namespace,
    environment_python: Path,
    harness_dir: Path,
    output_csv: Path,
    log_path: Path,
) -> int:
    return _run_logged(
        build_runner_command(args, environment_python, harness_dir, output_csv),
        log_path,
        cwd=harness_dir,
    )


def _optional_text(value: Any) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip() or None


def load_measurements(path: Path, index: ConfigIndex) -> pd.DataFrame:
    try:
        dataframe = pd.read_csv(path)
    except Exception as exc:
        raise WheelComparisonError(
            f"could not read benchmark CSV {path}: {exc}"
        ) from exc
    missing_columns = sorted(REQUIRED_RUN_COLUMNS - set(dataframe.columns))
    if missing_columns:
        raise WheelComparisonError(
            f"{path}: missing required columns {missing_columns}"
        )

    records, errors = [], []
    for position, (_, row) in enumerate(dataframe.iterrows(), start=2):
        try:
            config_key = str(row[CONFIG_KEY_COLUMN]).strip()
            ref = index.require(config_key)
            benchmark = str(row["Benchmark"]).strip()
            if benchmark != ref.benchmark:
                raise BaselineError(
                    f"benchmark {benchmark!r} does not match config operator {ref.benchmark!r}"
                )
            if str(row["Language"]).strip() != "python":
                raise BaselineError(
                    f"expected Language='python', got {row['Language']!r}"
                )
            case_key = case_key_from_row(row, ref)
            _, axes = parse_case_key(case_key)
            gpu_us, noise_pct = float(row[GPU_TIME_COLUMN]), float(
                row[GPU_NOISE_COLUMN]
            )
            if not math.isfinite(gpu_us) or gpu_us <= 0:
                raise BaselineError(
                    f"GPU time must be finite and positive, got {gpu_us}"
                )
            if not math.isfinite(noise_pct) or noise_pct < 0:
                raise BaselineError(
                    f"GPU noise must be finite and nonnegative, got {noise_pct}"
                )
            status = _optional_text(row[STATUS_COLUMN])
            if not status:
                raise BaselineError("Status must not be empty")
            fingerprint = tuple(
                _optional_text(row[column]) if column in dataframe.columns else None
                for column in STABLE_FINGERPRINT_COLUMNS
            )
            records.append(
                {
                    "benchmark": benchmark,
                    "config_key": config_key,
                    "case_key": case_key,
                    "axes": format_axes(axes),
                    "gpu_us": gpu_us,
                    "noise_pct": noise_pct,
                    "status": status,
                    "device_name": fingerprint[0],
                    "power_cap": fingerprint[1],
                    "vbios": fingerprint[2],
                    "clock_mhz": _optional_text(row[LOCKED_CLOCK_COLUMN]),
                }
            )
        except Exception as exc:
            errors.append(f"{path}:{position}: {exc}")
    if errors:
        raise WheelComparisonError("invalid benchmark rows:\n  " + "\n  ".join(errors))

    normalized = pd.DataFrame.from_records(records, columns=NORMALIZED_COLUMNS)
    duplicates = normalized[normalized["case_key"].duplicated()]["case_key"].tolist()
    if duplicates:
        raise WheelComparisonError(f"{path}: duplicate benchmark row(s): {duplicates}")
    fingerprints = normalized[["device_name", "power_cap", "vbios"]].drop_duplicates()
    if len(fingerprints) > 1:
        raise WheelComparisonError(
            f"{path}: benchmark rows contain inconsistent stable GPU fingerprints"
        )
    return normalized


def selected_config_keys(index: ConfigIndex, args: argparse.Namespace) -> set[str]:
    if args.config_key:
        selected = set(parse_config_key_arg(args.config_key))
        unknown = selected - set(index.refs_by_key)
        if unknown:
            raise WheelComparisonError(f"unknown config key(s): {sorted(unknown)}")
        return selected

    tiers = parse_tier_arg(args.tier)
    operators = (
        set(parse_operator_arg(args.operator))
        if args.operator
        else set(index.refs_by_operator)
    )
    unknown = operators - set(index.refs_by_operator)
    if unknown:
        raise WheelComparisonError(f"unknown operator(s): {sorted(unknown)}")
    selected = {
        ref.key
        for operator in operators
        for ref in index.refs_by_operator[operator]
        if ref.tier in tiers
    }
    if not selected:
        raise WheelComparisonError(
            f"no configs selected for operators {sorted(operators)} and tiers {sorted(tiers)}"
        )
    return selected


def expected_case_keys(index: ConfigIndex, config_keys: set[str]) -> set[str]:
    return set(expected_case_operators(index, config_keys))


def expected_case_operators(
    index: ConfigIndex, config_keys: set[str]
) -> dict[str, str]:
    cases = {}
    for config_key in config_keys:
        ref = index.require(config_key)
        for case_key in expected_case_keys_for_entry(config_key, ref.entry):
            cases[case_key] = ref.benchmark
    return cases


def _metadata(
    frame: pd.DataFrame,
) -> tuple[Optional[tuple[Optional[str], ...]], tuple[str, ...]]:
    if frame.empty:
        return None, ()
    first = frame.iloc[0]
    fingerprint = (first["device_name"], first["power_cap"], first["vbios"])
    clocks = tuple(sorted(value for value in frame["clock_mhz"].dropna().unique()))
    return fingerprint, clocks


def compare_measurements(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    expected_case_operators: dict[str, str],
    threshold: float,
) -> Comparison:
    baseline_fingerprint, baseline_clocks = _metadata(baseline)
    candidate_fingerprint, candidate_clocks = _metadata(candidate)
    if (
        baseline_fingerprint is not None
        and candidate_fingerprint is not None
        and baseline_fingerprint != candidate_fingerprint
    ):
        raise WheelComparisonError(
            "benchmark GPU fingerprints differ: "
            f"baseline={baseline_fingerprint!r}, candidate={candidate_fingerprint!r}"
        )

    baseline_rows = {row.case_key: row for row in baseline.itertuples(index=False)}
    candidate_rows = {row.case_key: row for row in candidate.itertuples(index=False)}
    expected_cases = set(expected_case_operators)
    records = []
    for case_key in sorted(set(baseline_rows) | set(candidate_rows) | expected_cases):
        base, current = baseline_rows.get(case_key), candidate_rows.get(case_key)
        source = base or current
        if source is None:
            config_key, axes = parse_case_key(case_key)
            records.append(
                {
                    "Benchmark": expected_case_operators[case_key],
                    "config_key": config_key,
                    "case_key": case_key,
                    "axes": format_axes(axes),
                    "comparison_status": "missing_from_both",
                }
            )
            continue
        record = {
            "Benchmark": source.benchmark,
            "config_key": source.config_key,
            "case_key": case_key,
            "axes": source.axes,
            "baseline_gpu_us": base.gpu_us if base else None,
            "candidate_gpu_us": current.gpu_us if current else None,
            "baseline_noise_pct": base.noise_pct if base else None,
            "candidate_noise_pct": current.noise_pct if current else None,
            "baseline_status": base.status if base else None,
            "candidate_status": current.status if current else None,
        }
        if base is None:
            record["comparison_status"] = "only_in_candidate"
        elif current is None:
            record["comparison_status"] = "missing_in_candidate"
        else:
            fraction = current.gpu_us / base.gpu_us - 1.0
            record["candidate_delta_pct"] = fraction * 100.0
            record["candidate_speedup"] = base.gpu_us / current.gpu_us
            if base.status != "PASS" or current.status != "PASS":
                record["comparison_status"] = "excluded"
            elif exceeds_limit(fraction, threshold):
                record["comparison_status"] = "regression"
            elif exceeds_limit(-fraction, threshold):
                record["comparison_status"] = "improvement"
            else:
                record["comparison_status"] = "neutral"
        records.append(record)

    baseline_keys, candidate_keys = set(baseline_rows), set(candidate_rows)
    table = pd.DataFrame.from_records(records, columns=COMPARISON_COLUMNS)
    return Comparison(
        table=table,
        expected_case_operators=dict(expected_case_operators),
        missing_baseline_cases=tuple(sorted(expected_cases - baseline_keys)),
        missing_candidate_cases=tuple(sorted(expected_cases - candidate_keys)),
        baseline_fingerprint=baseline_fingerprint,
        candidate_fingerprint=candidate_fingerprint,
        baseline_clocks=baseline_clocks,
        candidate_clocks=candidate_clocks,
        baseline_row_count=len(baseline),
        candidate_row_count=len(candidate),
        threshold=threshold,
    )


def _result_failed(comparison: Comparison, baseline_rc: int, candidate_rc: int) -> bool:
    return bool(comparison.failed or baseline_rc or candidate_rc)


def _summary_items(
    comparison: Comparison, baseline_rc: int, candidate_rc: int
) -> list[tuple[str, str]]:
    count = lambda status: str(len(comparison.rows(status)))  # noqa: E731
    speedup = comparison.mean_operator_speedup
    operator_total = len(set(comparison.expected_case_operators.values()))
    configuration_total = len(comparison.expected_case_operators)
    run_status = lambda returncode: (  # noqa: E731
        f"**{'PASS' if returncode == 0 else 'FAIL'}** (exit code {returncode})"
    )
    return [
        (
            "overall candidate speedup vs baseline",
            (
                f"**{speedup:.4f}x** (mean of per-operator geomeans across "
                "matched benchmarks)"
                if speedup is not None
                else "N/A"
            ),
        ),
        (
            "rows",
            f"baseline={comparison.baseline_row_count}, candidate={comparison.candidate_row_count}",
        ),
        (
            "valid compatible operators",
            f"{len(comparison.operator_geomean_speedups)}/{operator_total}",
        ),
        (
            "valid compatible configurations",
            f"{len(comparison.comparable)}/{configuration_total}",
        ),
        ("configuration regressions over threshold", count("regression")),
        ("configuration improvements over threshold", count("improvement")),
        ("reference `run_bench.py`", run_status(baseline_rc)),
        ("candidate `run_bench.py`", run_status(candidate_rc)),
    ]


def _compatibility_items(comparison: Comparison) -> list[tuple[str, str]]:
    operator_compatibility = comparison.operator_compatibility
    operator_total = len(operator_compatibility)
    operators = {
        status: tuple(
            operator
            for operator, actual_status in operator_compatibility.items()
            if actual_status == status
        )
        for status in (
            "incompatible_or_not_present",
            "partially_compatible",
            "fully_compatible",
        )
    }
    configuration_total = len(comparison.expected_case_operators)
    shared_configurations = len(comparison.shared_expected_cases)
    expected_rows = comparison.table[
        comparison.table["case_key"].isin(comparison.expected_case_operators)
    ]
    count = lambda status: int(  # noqa: E731
        (expected_rows["comparison_status"] == status).sum()
    )
    names = lambda status: (  # noqa: E731
        ", ".join(f"`{operator}`" for operator in operators[status]) or "_none_"
    )
    operators_with_any_incompatibility = len(
        operators["incompatible_or_not_present"]
    ) + len(operators["partially_compatible"])
    return [
        (
            "compatibility denominator",
            f"selected current-harness surface ({operator_total} operators, "
            f"{configuration_total} expanded configurations)",
        ),
        (
            "operators present in both",
            f"{operator_total - len(operators['incompatible_or_not_present'])}/{operator_total}",
        ),
        (
            "operators with any incompatible or missing configurations",
            f"{operators_with_any_incompatibility}/{operator_total}",
        ),
        (
            "operators with no shared configurations",
            f"{len(operators['incompatible_or_not_present'])}/{operator_total}",
        ),
        (
            "operators partially compatible",
            f"{len(operators['partially_compatible'])}/{operator_total}",
        ),
        (
            "operators fully compatible",
            f"{len(operators['fully_compatible'])}/{operator_total}",
        ),
        (
            "operator names with no shared configurations",
            names("incompatible_or_not_present"),
        ),
        ("partially compatible operator names", names("partially_compatible")),
        (
            "expanded benchmark configurations incompatible or not present",
            f"{configuration_total - shared_configurations}/{configuration_total}",
        ),
        (
            "expanded benchmark configurations present in both",
            f"{shared_configurations}/{configuration_total}",
        ),
        ("only in baseline", str(count("missing_in_candidate"))),
        ("only in candidate", str(count("only_in_candidate"))),
        ("missing from both", str(count("missing_from_both"))),
        ("present in both but non-PASS", str(count("excluded"))),
    ]


def _operator_result_lines(comparison: Comparison, *, improvements: bool) -> list[str]:
    speedups = comparison.operator_geomean_speedups
    speedups = speedups[speedups > 1.0] if improvements else speedups[speedups < 1.0]
    speedups = speedups.sort_values(ascending=not improvements).head(
        OPERATOR_RESULT_LIMIT
    )
    if speedups.empty:
        return ["_none_"]

    case_counts = comparison.comparable.groupby("Benchmark").size()
    lines = [
        "| Operator | Speedup | Candidate time delta | Matched configurations |",
        "|---|---:|---:|---:|",
    ]
    for operator, speedup in speedups.items():
        delta = (1.0 / speedup - 1.0) * 100.0
        lines.append(
            f"| {operator} | {speedup:.4f}x | {delta:+.2f}% | "
            f"{case_counts[operator]} |"
        )
    return lines


def _comparison_status(
    comparison: Comparison, baseline_rc: int, candidate_rc: int
) -> str:
    if not _result_failed(comparison, baseline_rc, candidate_rc):
        return "**PASS**"

    reasons = []
    if baseline_rc:
        reasons.append(f"reference `run_bench.py` exited {baseline_rc}")
    if candidate_rc:
        reasons.append(f"candidate `run_bench.py` exited {candidate_rc}")
    if not comparison.baseline_row_count:
        reasons.append("reference produced no benchmark rows")
    if not comparison.candidate_row_count:
        reasons.append("candidate produced no benchmark rows")

    for status, description in (
        ("regression", "regression over threshold"),
        ("missing_in_candidate", "only in reference"),
        ("only_in_candidate", "only in candidate"),
        ("missing_from_both", "missing from both"),
        ("excluded", "non-PASS"),
    ):
        count = len(comparison.rows(status))
        if count:
            noun = "configuration" if count == 1 else "configurations"
            reasons.append(f"{count} {noun} {description}")

    if not reasons:
        reasons.append("comparison completeness checks failed")
    return "**FAIL** — " + "; ".join(reasons)


def format_report(
    baseline_wheel: WheelMetadata,
    candidate_wheel: WheelMetadata,
    comparison: Comparison,
    *,
    baseline_rc: int,
    candidate_rc: int,
    output_dir: Path,
) -> str:
    lines = [
        "# CV-CUDA Python wheel benchmark comparison",
        "",
        f"- baseline: `{baseline_wheel.label}`",
        f"- candidate: `{candidate_wheel.label}`",
        f"- GPU fingerprint: `{comparison.baseline_fingerprint or comparison.candidate_fingerprint}`",
        f"- sampled SM clocks (MHz): baseline={list(comparison.baseline_clocks)}, "
        f"candidate={list(comparison.candidate_clocks)}",
        f"- regression threshold: `{comparison.threshold * 100:.2f}%`",
        f"- comparison status: {_comparison_status(comparison, baseline_rc, candidate_rc)}",
        f"- artifacts: `{output_dir}`",
        "  - reference benchmark CSV: `baseline/bench_output.csv`",
        "  - candidate benchmark CSV: `candidate/bench_output.csv`",
        "  - comparison CSV: `comparison.csv`",
        "",
        "## Summary",
        "",
    ]
    lines.extend(
        f"- {label}: {value}"
        for label, value in _summary_items(comparison, baseline_rc, candidate_rc)
    )
    lines.extend(["", "## Compatibility", ""])
    lines.extend(
        f"- {label}: {value}" for label, value in _compatibility_items(comparison)
    )
    lines.extend(["", "## Top 5 operator improvements", ""])
    lines.extend(_operator_result_lines(comparison, improvements=True))
    lines.extend(["", "## Top 5 operator regressions", ""])
    lines.extend(_operator_result_lines(comparison, improvements=False))
    return "\n".join(lines) + "\n"


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return parsed


def _warmup_cap(value: str) -> int:
    try:
        return parse_warmup_cap(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two CV-CUDA Python wheels with the current benchmark suite.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  compare_wheels.py reference.whl candidate.whl
  compare_wheels.py reference.whl candidate.whl \\
    --operator resize,gaussian --output-dir wheel-comparison

Outputs under DIR:
  baseline/bench_output.csv, candidate/bench_output.csv, comparison.csv,
  summary.md, and per-run logs and wheel metadata.

Exit status:
  0  Both runs and the comparison passed.
  1  A benchmark run or comparison check failed; the report was still written.
  2  Inputs, wheel setup, or artifacts are invalid.
""",
    )
    parser.add_argument(
        "baseline_wheel",
        metavar="REFERENCE_WHEEL",
        type=Path,
        help="Reference CV-CUDA wheel (.whl).",
    )
    parser.add_argument(
        "candidate_wheel",
        metavar="CANDIDATE_WHEEL",
        type=Path,
        help="Candidate CV-CUDA wheel (.whl) to compare with the reference.",
    )
    selectors = parser.add_mutually_exclusive_group()
    selectors.add_argument(
        "--operator", help="Run exact operator name(s), comma-separated."
    )
    selectors.add_argument(
        "--config-key",
        help="Run exact config key(s), comma-separated; bypasses --tier.",
    )
    parser.add_argument(
        "--tier",
        default="basic",
        help="Tiers: basic, advanced, or basic,advanced (default: basic).",
    )
    parser.add_argument(
        "--output-dir",
        metavar="DIR",
        type=Path,
        help=(
            "Write all artifacts to DIR. It must be new or empty "
            "(default: wheel-benchmark-<timestamp>)."
        ),
    )
    parser.add_argument(
        "--python",
        metavar="PYTHON",
        type=Path,
        default=Path(sys.executable),
        help=(
            "Interpreter used for both wheel runs; it must provide the benchmark "
            "dependencies (default: current interpreter)."
        ),
    )
    parser.add_argument(
        "--regression-threshold-pct",
        metavar="PCT",
        type=_nonnegative_float,
        default=DEFAULT_MAX_BASELINE_REGRESSION_PCT,
        help=(
            "Fail when any matched candidate configuration is slower by more "
            "than PCT percent (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--max-noise-pct",
        metavar="PCT",
        type=_nonnegative_float,
        default=DEFAULT_BENCHMARK_QUALITY.max_noise_pct,
        help="Maximum allowed measurement noise in percent (default: %(default)s).",
    )
    parser.add_argument(
        "--warmup-cap",
        metavar="N",
        type=_warmup_cap,
        help="Cap configured warmup iterations; 0 disables warmup.",
    )
    parser.add_argument(
        "--bench-min-time",
        metavar="SECONDS",
        type=_nonnegative_float,
        help="Override the minimum measurement time for each configuration.",
    )
    parser.add_argument(
        "--bench-max-noise",
        metavar="PCT",
        type=_nonnegative_float,
        help="Override the benchmark stopping-noise target in percent.",
    )
    return parser.parse_args(argv)


def _prepare_output_dir(requested: Optional[Path]) -> Path:
    output_dir = requested or Path(
        "wheel-benchmark-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        if not output_dir.is_dir():
            raise WheelComparisonError(f"output path is not a directory: {output_dir}")
        if any(output_dir.iterdir()):
            raise WheelComparisonError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def run(args: argparse.Namespace) -> int:
    baseline_wheel = read_wheel_metadata(args.baseline_wheel)
    candidate_wheel = read_wheel_metadata(args.candidate_wheel)
    if baseline_wheel.canonical_name != candidate_wheel.canonical_name:
        raise WheelComparisonError(
            f"wheel distributions differ: {baseline_wheel.name!r} vs {candidate_wheel.name!r}"
        )
    base_python = args.python.expanduser().resolve()
    if not base_python.is_file():
        raise WheelComparisonError(f"Python executable does not exist: {base_python}")

    output_dir = _prepare_output_dir(args.output_dir)
    artifact_dirs = {side: output_dir / side for side in ("baseline", "candidate")}
    for artifact_dir in artifact_dirs.values():
        artifact_dir.mkdir()
    workdir = Path(tempfile.mkdtemp(prefix="work-", dir=output_dir))
    try:
        harness_dir = workdir / "harness-bin"
        stage_benchmark_harness(BENCH_DIR, harness_dir)
        index = load_config_index(harness_dir / "config" / "operators")
        declared_cases = expected_case_operators(
            index, selected_config_keys(index, args)
        )

        wheels = {"baseline": baseline_wheel, "candidate": candidate_wheel}
        environments = {}
        for side in ("baseline", "candidate"):
            print(f"Preparing {side} environment for {wheels[side].label}")
            environments[side] = create_wheel_environment(
                base_python,
                workdir / f"{side}-env",
                wheels[side],
                artifact_dirs[side] / "install.log",
            )
            probe_wheel_environment(
                environments[side],
                harness_dir,
                wheels[side],
                artifact_dirs[side] / "wheel.json",
                artifact_dirs[side] / "probe.log",
            )

        results = {}
        for side in ("baseline", "candidate"):
            print(f"Running {side} benchmarks...")
            output_csv = artifact_dirs[side] / "bench_output.csv"
            returncode = run_benchmark(
                args,
                environments[side],
                harness_dir,
                output_csv,
                artifact_dirs[side] / "run.log",
            )
            measurements = (
                load_measurements(output_csv, index)
                if output_csv.is_file()
                else pd.DataFrame(columns=NORMALIZED_COLUMNS)
            )
            results[side] = (returncode, measurements)

        baseline_rc, baseline = results["baseline"]
        candidate_rc, candidate = results["candidate"]
        comparison = compare_measurements(
            baseline,
            candidate,
            expected_case_operators=declared_cases,
            threshold=args.regression_threshold_pct / 100.0,
        )
        comparison.table.to_csv(output_dir / "comparison.csv", index=False)
        report = format_report(
            baseline_wheel,
            candidate_wheel,
            comparison,
            baseline_rc=baseline_rc,
            candidate_rc=candidate_rc,
            output_dir=output_dir,
        )
        (output_dir / "summary.md").write_text(report)
        print("\n" + report)
        if comparison.baseline_clocks != comparison.candidate_clocks:
            print(
                "WARNING: sampled SM clocks differ; clocks are diagnostic, "
                "not part of the stable GPU fingerprint"
            )
        return 1 if _result_failed(comparison, baseline_rc, candidate_rc) else 0
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        return run(parse_args(argv))
    except (
        WheelComparisonError,
        BaselineError,
        ValueError,
        json.JSONDecodeError,
        OSError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
