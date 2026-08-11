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

"""Shared helpers for JSON-backed benchmark baselines.

The benchmark config files are human-authored; the nested ``baselines`` blocks
are machine-owned. This module centralizes the rules that turn benchmark result
rows into stable case keys and validate the JSON baseline payloads so compare,
update, and validation all agree on row identity.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

if TYPE_CHECKING:
    import pandas as pd

from .quality import BenchmarkQualityCriteria, DEFAULT_BENCHMARK_QUALITY

BENCH_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_DIR = BENCH_DIR / "config"
DEFAULT_OPERATORS_DIR = DEFAULT_CONFIG_DIR / "operators"
DEFAULT_SKU_MAP_PATH = DEFAULT_CONFIG_DIR / "sku_map.json"
DEFAULT_MAX_BASELINE_REGRESSION_PCT = 10.0

CONFIG_KEY_COLUMN = "config_key"
DEVICE_NAME_COLUMN = "Device Name"
POWER_CAP_COLUMN = "Power Cap (W)"
LOCKED_CLOCK_COLUMN = "Locked SM Clock (MHz)"
VBIOS_COLUMN = "vBIOS Version"

LANGUAGES = frozenset({"cpp", "python"})

GPU_TIME_COLUMN = "GPU Time (µs)"
GPU_NOISE_US_COLUMN = "GPU Noise (µs)"
BWUTIL_COLUMN = "BWUtil"

RUN_REQUIRED_COLUMNS = frozenset(
    {
        CONFIG_KEY_COLUMN,
        "Benchmark",
        "Language",
        GPU_TIME_COLUMN,
        GPU_NOISE_US_COLUMN,
        BWUTIL_COLUMN,
        DEVICE_NAME_COLUMN,
        POWER_CAP_COLUMN,
        LOCKED_CLOCK_COLUMN,
    }
)

RUN_METADATA_COLUMNS = frozenset(
    {
        CONFIG_KEY_COLUMN,
        "Benchmark",
        "Language",
        "tier",
        GPU_TIME_COLUMN,
        "GPU Noise (%)",
        GPU_NOISE_US_COLUMN,
        "CPU Time (µs)",
        "CPU Noise (%)",
        "CPU Noise (µs)",
        "GlobalMem BW (bytes/sec)",
        BWUTIL_COLUMN,
        "Py overhead (%)",
        "Py overhead (µs)",
        "Status",
        DEVICE_NAME_COLUMN,
        POWER_CAP_COLUMN,
        LOCKED_CLOCK_COLUMN,
        VBIOS_COLUMN,
        "Device",
        "Skipped",
        "Samples",
        "Samples.1",
        "Batch GPU (sec)",
    }
)

METRIC_FIELDS_BY_LANGUAGE = {
    "cpp": ("gpu_time_us_cpp", "gpu_noise_us_cpp", "gpu_bwutil_cpp"),
    "python": ("gpu_time_us_python", "gpu_noise_us_python", "gpu_bwutil_python"),
}

BASELINE_METRIC_FIELDS = (
    "n_runs",
    "gpu_time_us_cpp",
    "gpu_time_us_python",
    "gpu_noise_us_cpp",
    "gpu_noise_us_python",
    "gpu_bwutil_cpp",
    "gpu_bwutil_python",
)
GPU_GAP_STDDEV_FIELD = "gpu_gap_stddev_us"

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

_CASE_KEY_RE = re.compile(r"^([^\[\]]+)((?:\[[^\[\]=]+=[^\[\]]+\])*)$")
_CASE_AXIS_RE = re.compile(r"\[([^\[\]=]+)=([^\[\]]+)\]")


class BaselineError(RuntimeError):
    """Raised when baseline/config/run data violates the hard schema."""


@dataclass(frozen=True)
class AxisSpec:
    name: str
    kind: str
    values: Tuple[Any, ...]


@dataclass
class OperatorConfigFile:
    path: Path
    benchmark: str
    configs: Dict[str, Dict[str, Any]]
    raw: Dict[str, Any]
    new_shape: bool


@dataclass(frozen=True)
class ConfigRef:
    key: str
    benchmark: str
    path: Path
    entry: Dict[str, Any]

    @property
    def tier(self) -> Optional[str]:
        tier = self.entry.get("tier")
        return str(tier) if tier is not None else None

    @property
    def operator(self) -> str:
        return self.benchmark

    @property
    def case_axis_specs(self) -> Tuple[AxisSpec, ...]:
        return case_axis_specs(self.entry)


@dataclass
class ConfigIndex:
    refs_by_key: Dict[str, ConfigRef]
    refs_by_operator: Dict[str, List[ConfigRef]]
    docs_by_path: Dict[Path, OperatorConfigFile]

    def require(self, config_key: str) -> ConfigRef:
        ref = self.refs_by_key.get(config_key)
        if ref is None:
            raise BaselineError(
                f"config_key {config_key!r} does not exist in exactly one operator JSON"
            )
        return ref


@dataclass(frozen=True)
class RunMeasurement:
    config_key: str
    benchmark: str
    language: str
    case_key: str
    sku: str
    gpu_time_us: float
    gpu_noise_us: float
    gpu_bwutil: float
    source: Path
    row_number: int
    tier: Optional[str] = None


@dataclass(frozen=True)
class LanguageMetric:
    gpu_time_us: float
    gpu_noise_us: float
    gpu_bwutil: float


@dataclass(frozen=True)
class BaselineUpdate:
    config_key: str
    benchmark: str
    case_key: str
    sku: str
    n_runs: int
    language_metrics: Dict[str, LanguageMetric]
    # Unbiased sample standard deviation across paired artifact-level
    # (Python GPU time - C++ GPU time) gaps. It is absent when fewer than two
    # paired artifacts are available or the input aggregation lost pairing.
    gpu_gap_stddev_us: Optional[float] = None


def _gpu_gap_stats(
    update: BaselineUpdate,
) -> Optional[Tuple[int, float, float]]:
    """Return (count, mean, M2) when artifact-level gap dispersion is known."""
    cpp = update.language_metrics.get("cpp")
    python = update.language_metrics.get("python")
    if cpp is None or python is None:
        return None

    gap_mean = python.gpu_time_us - cpp.gpu_time_us
    if update.n_runs == 1:
        # A raw benchmark artifact is one valid paired gap observation, but one
        # observation cannot independently expose a sample standard deviation.
        return 1, gap_mean, 0.0
    if update.gpu_gap_stddev_us is None:
        return None
    return (
        update.n_runs,
        gap_mean,
        update.gpu_gap_stddev_us**2 * (update.n_runs - 1),
    )


def _pooled_gpu_gap_stddev(updates: Sequence[BaselineUpdate]) -> Optional[float]:
    """Pool paired-gap sample variances without assuming language independence."""
    count = 0
    mean = 0.0
    m2 = 0.0
    for update in updates:
        stats = _gpu_gap_stats(update)
        if stats is None:
            return None
        item_count, item_mean, item_m2 = stats
        combined_count = count + item_count
        if count:
            delta = item_mean - mean
            m2 += item_m2 + delta * delta * count * item_count / combined_count
            mean += delta * item_count / combined_count
        else:
            mean = item_mean
            m2 = item_m2
        count = combined_count

    if count < 2:
        return None
    return math.sqrt(max(m2, 0.0) / (count - 1))


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        # Keep case-key expansion and JSON validation stdlib-only for lightweight
        # lifecycle checks. Dataframe import/update paths load pandas on demand.
        import pandas as pd

        missing = pd.isna(value)
    except (ImportError, TypeError, ValueError):
        missing = False
    if isinstance(missing, bool):
        if missing:
            return True
    if isinstance(value, str):
        return value.strip() == ""
    return False


def _finite_float(value: Any, *, label: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise BaselineError(f"{label} must be numeric, got {value!r}") from exc
    if not math.isfinite(out):
        raise BaselineError(f"{label} must be finite, got {value!r}")
    return out


def _canonical_int(value: Any) -> str:
    raw = str(value).strip()
    try:
        number = float(raw)
    except ValueError as exc:
        raise BaselineError(f"expected integer axis value, got {value!r}") from exc
    if not math.isfinite(number) or not number.is_integer():
        raise BaselineError(f"expected integer axis value, got {value!r}")
    return str(int(number))


def _canonical_float(value: Any) -> str:
    raw = str(value).strip()
    try:
        number = float(raw)
    except ValueError as exc:
        raise BaselineError(
            f"expected floating-point axis value, got {value!r}"
        ) from exc
    if not math.isfinite(number):
        raise BaselineError(f"expected finite floating-point axis value, got {value!r}")
    return str(number)


def _declared_value_strings(spec: AxisSpec) -> List[str]:
    out: List[str] = []
    for value in spec.values:
        if spec.kind == "int64":
            out.append(_canonical_int(value))
        elif spec.kind == "float64":
            out.append(_canonical_float(value))
        elif spec.kind == "dtype":
            out.append(DTYPE_MAP.get(str(value).strip(), str(value).strip()))
        else:
            out.append(str(value).strip())
    return out


def canonical_axis_value(spec: AxisSpec, value: Any) -> str:
    if _is_missing(value):
        raise BaselineError(f"missing value for axis {spec.name!r}")

    if spec.kind == "int64":
        candidate = _canonical_int(value)
    elif spec.kind == "float64":
        candidate = _canonical_float(value)
    elif spec.kind == "dtype":
        candidate = DTYPE_MAP.get(str(value).strip(), str(value).strip())
    else:
        candidate = str(value).strip()

    declared = _declared_value_strings(spec)
    if candidate not in declared:
        raise BaselineError(
            f"axis {spec.name!r} value {candidate!r} is not declared "
            f"(allowed: {declared})"
        )
    return candidate


def case_axis_specs(entry: Dict[str, Any]) -> Tuple[AxisSpec, ...]:
    specs: List[AxisSpec] = [
        AxisSpec("InOutDataType", "dtype", tuple(entry.get("dtypes", [])))
    ]
    for group_name, kind in (
        ("string_axes", "string"),
        ("int64_axes", "int64"),
        ("float64_axes", "float64"),
    ):
        axes = entry.get(group_name, {})
        if not isinstance(axes, dict):
            raise BaselineError(f"{group_name} must be a dictionary")
        for axis_name, values in axes.items():
            if not isinstance(values, list):
                raise BaselineError(f"axis {axis_name!r} values must be a list")
            specs.append(AxisSpec(axis_name, kind, tuple(values)))
    return tuple(specs)


def case_key_from_axis_values(
    config_key: str, axis_values: Sequence[Tuple[str, str]]
) -> str:
    return str(config_key) + "".join(
        f"[{axis_name}={axis_value}]" for axis_name, axis_value in axis_values
    )


def expected_case_keys_for_entry(config_key: str, entry: Dict[str, Any]) -> List[str]:
    specs = case_axis_specs(entry)
    return [
        case_key_from_axis_values(
            config_key,
            [(spec.name, value) for spec, value in zip(specs, values)],
        )
        for values in product(*(_declared_value_strings(spec) for spec in specs))
    ]


def fake_planar_pairing_issues(
    configs: Mapping[str, Mapping[str, Any]],
) -> Tuple[str, ...]:
    """Return deterministic violations of the FakePlanar comparison contract.

    Every FakePlanar case is an advanced-tier, Tensor-only reference row.  Its
    signature must have exactly one same-tier native-planar case after removing
    only the layout axis.  Native-only signatures are allowed because a benchmark
    may select a representative subset for layout-conversion comparisons.
    """

    native_layouts = {"NCHW", "CHW"}
    fake_layouts = {"NCHW_FAKE", "CHW_FAKE"}
    groups: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], Dict[str, List[str]]] = {}

    for config_key in sorted(configs):
        entry = configs[config_key]
        if not isinstance(entry, Mapping):
            raise BaselineError(f"config {config_key!r} must be an object")
        tier = str(entry.get("tier", ""))
        for case_key in expected_case_keys_for_entry(config_key, dict(entry)):
            _, axes = parse_case_key(case_key)
            axis_map = dict(axes)
            layout = axis_map.get("layout")
            if layout not in native_layouts | fake_layouts:
                continue
            signature = (
                tier,
                tuple(
                    sorted((name, value) for name, value in axes if name != "layout")
                ),
            )
            kind = "fake" if layout in fake_layouts else "native"
            groups.setdefault(signature, {"fake": [], "native": []})[kind].append(
                case_key
            )

    issues = []
    for (tier, identity), cases in sorted(groups.items(), key=lambda item: item[0]):
        fake_cases = sorted(cases["fake"])
        if not fake_cases:
            continue
        native_cases = sorted(cases["native"])
        reasons = []
        if tier != "advanced":
            reasons.append(f"tier is {tier or '<missing>'}, expected advanced")
        if dict(identity).get("inputKind") != "Tensor":
            reasons.append("inputKind is not Tensor")
        if len(fake_cases) != 1:
            reasons.append(
                f"signature expands to {len(fake_cases)} FakePlanar cases, expected exactly 1"
            )
        if len(native_cases) != 1:
            reasons.append(
                f"found {len(native_cases)} same-tier native NCHW/CHW matches, expected exactly 1"
            )
        if reasons:
            issues.append(f"{', '.join(fake_cases)}: {'; '.join(reasons)}")
    return tuple(issues)


def parse_case_key(case_key: str) -> Tuple[str, Tuple[Tuple[str, str], ...]]:
    match = _CASE_KEY_RE.fullmatch(case_key)
    if not match:
        raise BaselineError(
            f"invalid case key syntax {case_key!r}; expected "
            "'<config_key>[axis=value]...'"
        )
    config_key, axis_blob = match.groups()
    axes = tuple(_CASE_AXIS_RE.findall(axis_blob))
    if not axes or "".join(f"[{name}={value}]" for name, value in axes) != axis_blob:
        raise BaselineError(f"invalid axis syntax in case key {case_key!r}")
    return config_key, axes


def validate_case_key_for_config(
    case_key: str, ref: ConfigRef
) -> Tuple[Tuple[str, str], ...]:
    parsed_config_key, axes = parse_case_key(case_key)
    if parsed_config_key != ref.key:
        raise BaselineError(
            f"case key config {parsed_config_key!r} does not match nested "
            f"config {ref.key!r}"
        )

    specs = ref.case_axis_specs
    expected_names = tuple(spec.name for spec in specs)
    actual_names = tuple(name for name, _ in axes)
    if actual_names != expected_names:
        raise BaselineError(
            f"case key {case_key!r} has axes {list(actual_names)}, expected "
            f"{list(expected_names)}"
        )

    canonical_axes: List[Tuple[str, str]] = []
    for spec, (_, raw_value) in zip(specs, axes):
        canonical = canonical_axis_value(spec, raw_value)
        if canonical != raw_value:
            raise BaselineError(
                f"case key {case_key!r} uses non-canonical value {raw_value!r} "
                f"for axis {spec.name!r}; expected {canonical!r}"
            )
        canonical_axes.append((spec.name, canonical))
    return tuple(canonical_axes)


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise BaselineError(f"{path}: JSON root must be an object")
    return raw


def _entry_without_benchmark(entry: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in entry.items() if k != "benchmark"}


def _entry_without_baselines(entry: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in entry.items() if k != "baselines"}


def split_operator_payload(path: Path, raw: Dict[str, Any]) -> OperatorConfigFile:
    if "configs" in raw or "benchmark" in raw:
        benchmark = raw.get("benchmark")
        configs = raw.get("configs")
        if not isinstance(benchmark, str) or not benchmark:
            raise BaselineError(f"{path}: top-level 'benchmark' must be a string")
        if not isinstance(configs, dict):
            raise BaselineError(f"{path}: top-level 'configs' must be an object")
        for key, entry in configs.items():
            if not isinstance(key, str) or not key:
                raise BaselineError(f"{path}: config keys must be non-empty strings")
            if not isinstance(entry, dict):
                raise BaselineError(f"{path}: config {key!r} must be an object")
        return OperatorConfigFile(path, benchmark, configs, raw, True)

    benchmark_values = {
        entry.get("benchmark")
        for entry in raw.values()
        if isinstance(entry, dict) and isinstance(entry.get("benchmark"), str)
    }
    if len(benchmark_values) != 1:
        raise BaselineError(
            f"{path}: old-shape operator config must contain exactly one "
            f"benchmark value, found {sorted(benchmark_values)!r}"
        )
    configs = {}
    for key, entry in raw.items():
        if not isinstance(key, str) or not key:
            raise BaselineError(f"{path}: config keys must be non-empty strings")
        if not isinstance(entry, dict):
            raise BaselineError(f"{path}: config {key!r} must be an object")
        configs[key] = _entry_without_benchmark(entry)
    return OperatorConfigFile(path, next(iter(benchmark_values)), configs, raw, False)


def split_operator_document(path: Path) -> OperatorConfigFile:
    return split_operator_payload(path, _read_json(path))


def operator_config_paths(operators_dir: Path = DEFAULT_OPERATORS_DIR) -> List[Path]:
    return sorted(Path(operators_dir).glob("*.json"))


def load_config_index(
    operators_dir: Path = DEFAULT_OPERATORS_DIR,
    paths: Optional[Sequence[Path]] = None,
) -> ConfigIndex:
    docs: Dict[Path, OperatorConfigFile] = {}
    refs_by_key: Dict[str, ConfigRef] = {}
    refs_by_operator: Dict[str, List[ConfigRef]] = {}
    duplicates: Dict[str, List[Path]] = {}

    selected_paths = [
        Path(p)
        for p in (paths if paths is not None else operator_config_paths(operators_dir))
    ]
    for path in selected_paths:
        doc = split_operator_document(path)
        docs[path] = doc
        for config_key, entry in doc.configs.items():
            if config_key in refs_by_key:
                duplicates.setdefault(
                    config_key, [refs_by_key[config_key].path]
                ).append(path)
                continue
            ref = ConfigRef(config_key, doc.benchmark, path, entry)
            refs_by_key[config_key] = ref
            refs_by_operator.setdefault(doc.benchmark, []).append(ref)

    if duplicates:
        details = ", ".join(
            f"{key}: {[str(p) for p in paths]}" for key, paths in duplicates.items()
        )
        raise BaselineError(f"duplicate config key(s): {details}")

    return ConfigIndex(refs_by_key, refs_by_operator, docs)


def operator_document_to_new_shape(doc: OperatorConfigFile) -> Dict[str, Any]:
    if doc.new_shape:
        return doc.raw

    configs: Dict[str, Dict[str, Any]] = {}
    for key, entry in doc.configs.items():
        configs[key] = _entry_without_benchmark(entry)
    return {"benchmark": doc.benchmark, "configs": configs}


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=4) + "\n")


def row_axis_values(row: pd.Series, ref: ConfigRef) -> Tuple[Tuple[str, str], ...]:
    expected_specs = ref.case_axis_specs
    expected_names = {spec.name for spec in expected_specs}

    extra_axes: List[str] = []
    for column in row.index:
        if column in RUN_METADATA_COLUMNS:
            continue
        if column.startswith("Unnamed:"):
            continue
        if column in expected_names:
            continue
        if not _is_missing(row[column]):
            extra_axes.append(column)
    if extra_axes:
        raise BaselineError(
            f"unknown non-metadata axis column(s) for config {ref.key!r}: "
            f"{extra_axes}"
        )

    axis_values: List[Tuple[str, str]] = []
    for spec in expected_specs:
        if spec.name not in row.index or _is_missing(row[spec.name]):
            if len(spec.values) != 1:
                raise BaselineError(
                    f"row for config {ref.key!r} is missing axis column "
                    f"{spec.name!r}"
                )
            raw_value = spec.values[0]
        else:
            raw_value = row[spec.name]
        axis_values.append((spec.name, canonical_axis_value(spec, raw_value)))
    return tuple(axis_values)


def case_key_from_row(row: pd.Series, ref: ConfigRef) -> str:
    return case_key_from_axis_values(ref.key, row_axis_values(row, ref))


def load_sku_map(
    sku_map_path: Optional[Path] = DEFAULT_SKU_MAP_PATH, *, strict: bool = False
) -> Dict[Tuple[str, int, int], str]:
    if not sku_map_path or not Path(sku_map_path).is_file():
        if strict:
            raise BaselineError(f"sku_map.json not found at {sku_map_path}")
        return {}
    try:
        raw = json.loads(Path(sku_map_path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        if strict:
            raise BaselineError(
                f"could not parse sku_map.json at {sku_map_path}"
            ) from exc
        return {}

    out: Dict[Tuple[str, int, int], str] = {}
    for idx, entry in enumerate(raw.get("entries", [])):
        try:
            name = str(entry["gpu_name"])
            cap = int(round(float(entry["power_cap_w"])))
            clock = int(round(float(entry["locked_sm_clock_mhz"])))
            stem = str(entry["stem"])
        except (KeyError, TypeError, ValueError) as exc:
            if strict:
                raise BaselineError(
                    f"invalid sku_map.json entry #{idx}: {entry!r}"
                ) from exc
            continue
        out[(name, cap, clock)] = stem
    return out


def sku_stems(
    sku_map_path: Optional[Path] = DEFAULT_SKU_MAP_PATH, *, strict: bool = False
) -> Set[str]:
    return set(load_sku_map(sku_map_path, strict=strict).values())


def resolve_sku_for_row(
    row: pd.Series, sku_map: Dict[Tuple[str, int, int], str]
) -> str:
    missing = [
        column
        for column in (DEVICE_NAME_COLUMN, POWER_CAP_COLUMN, LOCKED_CLOCK_COLUMN)
        if column not in row.index or _is_missing(row[column])
    ]
    if missing:
        raise BaselineError(f"missing SKU routing column(s): {missing}")
    name = str(row[DEVICE_NAME_COLUMN]).strip()
    cap = int(round(_finite_float(row[POWER_CAP_COLUMN], label=POWER_CAP_COLUMN)))
    clock = int(
        round(_finite_float(row[LOCKED_CLOCK_COLUMN], label=LOCKED_CLOCK_COLUMN))
    )
    sku = sku_map.get((name, cap, clock))
    if not sku:
        raise BaselineError(f"({name!r}, {cap!r}, {clock!r}) is not in sku_map.json")
    return sku


def _validate_required_run_columns(df: pd.DataFrame, source: Path) -> None:
    missing = sorted(RUN_REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise BaselineError(f"{source}: missing required column(s): {missing}")


def _row_label(path: Path, row_number: int) -> str:
    return f"{path}:{row_number}"


def measurement_from_row(
    row: pd.Series,
    *,
    source: Path,
    row_number: int,
    index: ConfigIndex,
    sku_map: Dict[Tuple[str, int, int], str],
) -> RunMeasurement:
    config_key = str(row[CONFIG_KEY_COLUMN]).strip()
    ref = index.require(config_key)

    benchmark = str(row["Benchmark"]).strip()
    if benchmark != ref.benchmark:
        raise BaselineError(
            f"Benchmark {benchmark!r} does not match config {config_key!r}'s "
            f"operator {ref.benchmark!r}"
        )

    language = str(row["Language"]).strip()
    if language not in LANGUAGES:
        raise BaselineError(
            f"Language must be one of {sorted(LANGUAGES)}, got {language!r}"
        )

    sku = resolve_sku_for_row(row, sku_map)
    gpu_time_us = _finite_float(row[GPU_TIME_COLUMN], label=GPU_TIME_COLUMN)
    gpu_noise_us = _finite_float(row[GPU_NOISE_US_COLUMN], label=GPU_NOISE_US_COLUMN)
    gpu_bwutil = _finite_float(row[BWUTIL_COLUMN], label=BWUTIL_COLUMN)
    if gpu_time_us <= 0:
        raise BaselineError(f"{GPU_TIME_COLUMN} must be > 0, got {gpu_time_us}")
    if gpu_noise_us < 0:
        raise BaselineError(f"{GPU_NOISE_US_COLUMN} must be >= 0, got {gpu_noise_us}")
    if gpu_bwutil < 0:
        raise BaselineError(f"{BWUTIL_COLUMN} must be >= 0, got {gpu_bwutil}")

    case_key = case_key_from_row(row, ref)
    return RunMeasurement(
        config_key=config_key,
        benchmark=benchmark,
        language=language,
        case_key=case_key,
        sku=sku,
        gpu_time_us=gpu_time_us,
        gpu_noise_us=gpu_noise_us,
        gpu_bwutil=gpu_bwutil,
        source=source,
        row_number=row_number,
        tier=ref.tier,
    )


def aggregate_measurements(
    measurements: Sequence[RunMeasurement],
) -> Dict[Tuple[str, str, str], BaselineUpdate]:
    grouped: Dict[Tuple[str, str, str, str], List[RunMeasurement]] = {}
    for measurement in measurements:
        grouped.setdefault(
            (
                measurement.config_key,
                measurement.case_key,
                measurement.sku,
                measurement.language,
            ),
            [],
        ).append(measurement)

    by_case: Dict[
        Tuple[str, str, str], Dict[str, Tuple[int, float, float, float, str]]
    ] = {}
    benchmark_by_case: Dict[Tuple[str, str, str], str] = {}
    errors: List[str] = []

    for (config_key, case_key, sku, language), rows in grouped.items():
        n = len(rows)
        avg_time = sum(row.gpu_time_us for row in rows) / n
        avg_noise = sum(row.gpu_noise_us for row in rows) / n
        avg_bwutil = sum(row.gpu_bwutil for row in rows) / n
        case = (config_key, case_key, sku)
        benchmark_by_case[case] = rows[0].benchmark
        by_case.setdefault(case, {})[language] = (
            n,
            avg_time,
            avg_noise,
            avg_bwutil,
            language,
        )

    updates: Dict[Tuple[str, str, str], BaselineUpdate] = {}
    for case, per_language in by_case.items():
        counts = {payload[0] for payload in per_language.values()}
        if len(counts) != 1:
            counts_by_language = {lang: data[0] for lang, data in per_language.items()}
            errors.append(
                f"{case}: repeated-run counts differ by language: "
                f"{counts_by_language}"
            )
            continue
        n_runs = counts.pop()
        updates[case] = BaselineUpdate(
            config_key=case[0],
            benchmark=benchmark_by_case[case],
            case_key=case[1],
            sku=case[2],
            n_runs=n_runs,
            language_metrics={
                lang: LanguageMetric(
                    gpu_time_us=payload[1],
                    gpu_noise_us=payload[2],
                    gpu_bwutil=payload[3],
                )
                for lang, payload in per_language.items()
            },
        )

    if errors:
        raise BaselineError("invalid repeated-run groups:\n  " + "\n  ".join(errors))
    return updates


def merge_baseline_updates(
    updates: Iterable[BaselineUpdate],
) -> Dict[Tuple[str, str, str], BaselineUpdate]:
    grouped: Dict[Tuple[str, str, str], List[BaselineUpdate]] = {}
    for update in updates:
        grouped.setdefault((update.config_key, update.case_key, update.sku), []).append(
            update
        )

    merged: Dict[Tuple[str, str, str], BaselineUpdate] = {}
    errors: List[str] = []
    for key, items in grouped.items():
        benchmarks = {item.benchmark for item in items}
        if len(benchmarks) != 1:
            errors.append(f"{key}: conflicting benchmark names {sorted(benchmarks)}")
            continue

        per_language: Dict[str, Tuple[int, float, float, float]] = {}
        for item in items:
            for language, metric in item.language_metrics.items():
                n_prev, time_prev, noise_prev, bwutil_prev = per_language.get(
                    language, (0, 0.0, 0.0, 0.0)
                )
                per_language[language] = (
                    n_prev + item.n_runs,
                    time_prev + metric.gpu_time_us * item.n_runs,
                    noise_prev + metric.gpu_noise_us * item.n_runs,
                    bwutil_prev + metric.gpu_bwutil * item.n_runs,
                )

        counts = {payload[0] for payload in per_language.values()}
        if len(counts) != 1:
            counts_by_language = {lang: data[0] for lang, data in per_language.items()}
            errors.append(
                f"{key}: repeated-run counts differ by language: "
                f"{counts_by_language}"
            )
            continue
        n_runs = counts.pop()
        merged[key] = BaselineUpdate(
            config_key=key[0],
            benchmark=next(iter(benchmarks)),
            case_key=key[1],
            sku=key[2],
            n_runs=n_runs,
            language_metrics={
                language: LanguageMetric(
                    gpu_time_us=time_sum / n_runs,
                    gpu_noise_us=noise_sum / n_runs,
                    gpu_bwutil=bwutil_sum / n_runs,
                )
                for language, (
                    _count,
                    time_sum,
                    noise_sum,
                    bwutil_sum,
                ) in per_language.items()
            },
            gpu_gap_stddev_us=_pooled_gpu_gap_stddev(items),
        )

    if errors:
        raise BaselineError("invalid baseline update groups:\n  " + "\n  ".join(errors))
    return merged


def measurements_from_dataframe(
    df: pd.DataFrame,
    *,
    index: ConfigIndex,
    sku_map_path: Path = DEFAULT_SKU_MAP_PATH,
    source: Path = Path("<dataframe>"),
) -> List[RunMeasurement]:
    sku_map = load_sku_map(sku_map_path, strict=True)
    errors: List[str] = []
    try:
        _validate_required_run_columns(df, source)
    except Exception as exc:
        errors.append(f"{source}: {exc}")

    measurements: List[RunMeasurement] = []
    if not errors:
        for row_idx, row in df.iterrows():
            row_number = int(row_idx) + 2
            try:
                measurements.append(
                    measurement_from_row(
                        row,
                        source=source,
                        row_number=row_number,
                        index=index,
                        sku_map=sku_map,
                    )
                )
            except Exception as exc:
                errors.append(f"{_row_label(source, row_number)}: {exc}")

    if errors:
        raise BaselineError("invalid benchmark row(s):\n  " + "\n  ".join(errors))
    return measurements


def baseline_payload_from_updates(
    updates: Iterable[BaselineUpdate],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    payload: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for update in sorted(
        updates,
        key=lambda item: (item.config_key, item.case_key, item.sku),
    ):
        case_payload = payload.setdefault(update.case_key, {})
        metrics: Dict[str, Any] = {"n_runs": update.n_runs}
        for language in sorted(update.language_metrics):
            time_field, noise_field, bwutil_field = METRIC_FIELDS_BY_LANGUAGE[language]
            metric = update.language_metrics[language]
            metrics[time_field] = metric.gpu_time_us
            metrics[noise_field] = metric.gpu_noise_us
            metrics[bwutil_field] = metric.gpu_bwutil
        if update.gpu_gap_stddev_us is not None:
            metrics[GPU_GAP_STDDEV_FIELD] = update.gpu_gap_stddev_us
        case_payload[update.sku] = _ordered_metric_object(metrics)
    return payload


def baseline_payload_from_dataframe(
    df: pd.DataFrame,
    *,
    index: ConfigIndex,
    sku_map_path: Path = DEFAULT_SKU_MAP_PATH,
    source: Path = Path("<dataframe>"),
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    measurements = measurements_from_dataframe(
        df, index=index, sku_map_path=sku_map_path, source=source
    )
    return baseline_payload_from_updates(aggregate_measurements(measurements).values())


def _positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise BaselineError(f"{label} must be a positive integer, got {value!r}")
    return value


def baseline_updates_from_payload(
    payload: Dict[str, Any],
    *,
    index: ConfigIndex,
    sku_stem_set: Set[str],
    source: Path = Path("<baseline-json>"),
) -> List[BaselineUpdate]:
    if not isinstance(payload, dict):
        raise BaselineError(f"{source}: baseline JSON root must be an object")

    updates: List[BaselineUpdate] = []
    errors: List[str] = []
    for case_key, case_payload in payload.items():
        try:
            config_key, _ = parse_case_key(case_key)
            ref = index.require(config_key)
            validate_case_key_for_config(case_key, ref)
            if not isinstance(case_payload, dict):
                raise BaselineError("SKU map must be an object")

            for sku, metric_payload in case_payload.items():
                if sku not in sku_stem_set:
                    raise BaselineError(f"unknown SKU {sku!r}")
                if not isinstance(metric_payload, dict):
                    raise BaselineError(f"{sku}: metrics must be an object")

                n_runs = _positive_int(
                    metric_payload.get("n_runs"), label=f"{sku}: n_runs"
                )
                language_metrics: Dict[str, LanguageMetric] = {}
                for (
                    language,
                    (time_field, noise_field, bwutil_field),
                ) in METRIC_FIELDS_BY_LANGUAGE.items():
                    has_time = time_field in metric_payload
                    has_noise = noise_field in metric_payload
                    has_bwutil = bwutil_field in metric_payload
                    if len({has_time, has_noise, has_bwutil}) != 1:
                        raise BaselineError(
                            f"{sku}: fields {time_field!r}, {noise_field!r}, "
                            f"and {bwutil_field!r} "
                            "must be provided together"
                        )
                    if not has_time:
                        continue
                    time_us = _finite_float(
                        metric_payload[time_field], label=f"{sku}: {time_field}"
                    )
                    noise_us = _finite_float(
                        metric_payload[noise_field], label=f"{sku}: {noise_field}"
                    )
                    bwutil = _finite_float(
                        metric_payload[bwutil_field], label=f"{sku}: {bwutil_field}"
                    )
                    if time_us <= 0:
                        raise BaselineError(
                            f"{sku}: {time_field} must be > 0, got {time_us}"
                        )
                    if noise_us < 0:
                        raise BaselineError(
                            f"{sku}: {noise_field} must be >= 0, got {noise_us}"
                        )
                    if bwutil < 0:
                        raise BaselineError(
                            f"{sku}: {bwutil_field} must be >= 0, got {bwutil}"
                        )
                    language_metrics[language] = LanguageMetric(
                        gpu_time_us=time_us,
                        gpu_noise_us=noise_us,
                        gpu_bwutil=bwutil,
                    )

                if not language_metrics:
                    raise BaselineError(
                        f"{sku}: expected at least one complete language metric pair"
                    )

                gpu_gap_stddev_us = None
                if GPU_GAP_STDDEV_FIELD in metric_payload:
                    gpu_gap_stddev_us = _finite_float(
                        metric_payload[GPU_GAP_STDDEV_FIELD],
                        label=f"{sku}: {GPU_GAP_STDDEV_FIELD}",
                    )
                    if gpu_gap_stddev_us < 0:
                        raise BaselineError(
                            f"{sku}: {GPU_GAP_STDDEV_FIELD} must be >= 0, "
                            f"got {gpu_gap_stddev_us}"
                        )
                    if n_runs < 2:
                        raise BaselineError(
                            f"{sku}: {GPU_GAP_STDDEV_FIELD} requires n_runs >= 2"
                        )
                    missing_gap_languages = sorted(LANGUAGES - set(language_metrics))
                    if missing_gap_languages:
                        raise BaselineError(
                            f"{sku}: {GPU_GAP_STDDEV_FIELD} requires paired "
                            f"C++/Python metrics; missing {missing_gap_languages}"
                        )

                updates.append(
                    BaselineUpdate(
                        config_key=config_key,
                        benchmark=ref.benchmark,
                        case_key=case_key,
                        sku=sku,
                        n_runs=n_runs,
                        language_metrics=language_metrics,
                        gpu_gap_stddev_us=gpu_gap_stddev_us,
                    )
                )
        except Exception as exc:
            errors.append(f"{source}: {case_key}: {exc}")

    if errors:
        raise BaselineError("invalid baseline JSON:\n  " + "\n  ".join(errors))
    return updates


def baseline_updates_from_jsons(
    paths: Sequence[Path],
    *,
    index: ConfigIndex,
    sku_map_path: Path = DEFAULT_SKU_MAP_PATH,
) -> Dict[Tuple[str, str, str], BaselineUpdate]:
    stems = sku_stems(sku_map_path, strict=True)
    updates: List[BaselineUpdate] = []
    errors: List[str] = []
    for path in paths:
        try:
            raw = json.loads(Path(path).read_text())
            updates.extend(
                baseline_updates_from_payload(
                    raw,
                    index=index,
                    sku_stem_set=stems,
                    source=Path(path),
                )
            )
        except Exception as exc:
            errors.append(f"{path}: {exc}")

    if errors:
        raise BaselineError("invalid baseline JSON input(s):\n  " + "\n  ".join(errors))
    return merge_baseline_updates(updates)


def _ordered_metric_object(metrics: Dict[str, Any]) -> Dict[str, Any]:
    ordered = {
        field: metrics[field] for field in BASELINE_METRIC_FIELDS if field in metrics
    }
    if GPU_GAP_STDDEV_FIELD in metrics:
        ordered[GPU_GAP_STDDEV_FIELD] = metrics[GPU_GAP_STDDEV_FIELD]
    for key, value in metrics.items():
        if key not in ordered:
            ordered[key] = value
    return ordered


def apply_updates_to_document(
    doc: OperatorConfigFile,
    updates: Iterable[BaselineUpdate],
) -> Dict[str, Any]:
    raw = operator_document_to_new_shape(doc)
    configs = raw["configs"]

    for update in updates:
        entry = configs[update.config_key]
        baselines = entry.setdefault("baselines", {})
        if not isinstance(baselines, dict):
            raise BaselineError(
                f"{doc.path}: config {update.config_key!r} has non-object baselines"
            )
        case_payload = baselines.setdefault(update.case_key, {})
        if not isinstance(case_payload, dict):
            raise BaselineError(
                f"{doc.path}: case {update.case_key!r} has non-object SKU map"
            )
        metric_payload = dict(case_payload.get(update.sku, {}))
        metric_payload["n_runs"] = update.n_runs
        for language, metric in update.language_metrics.items():
            time_field, noise_field, bwutil_field = METRIC_FIELDS_BY_LANGUAGE[language]
            metric_payload[time_field] = metric.gpu_time_us
            metric_payload[noise_field] = metric.gpu_noise_us
            metric_payload[bwutil_field] = metric.gpu_bwutil
        if update.gpu_gap_stddev_us is None:
            # Never retain dispersion from an older baseline when the new
            # source did not preserve artifact-level C++/Python pairing.
            metric_payload.pop(GPU_GAP_STDDEV_FIELD, None)
        else:
            metric_payload[GPU_GAP_STDDEV_FIELD] = update.gpu_gap_stddev_us
        case_payload[update.sku] = _ordered_metric_object(metric_payload)

    return raw


def validate_baseline_update_quality(
    update: BaselineUpdate,
    *,
    criteria: BenchmarkQualityCriteria = DEFAULT_BENCHMARK_QUALITY,
) -> List[str]:
    errors: List[str] = []
    missing_languages = sorted(LANGUAGES - set(update.language_metrics))
    if missing_languages:
        errors.append(
            f"{update.case_key}: {update.sku}: missing language metric(s) "
            f"{missing_languages}; baseline quality requires C++/Python parity"
        )
        return errors

    for language in sorted(update.language_metrics):
        metric = update.language_metrics[language]
        if metric.gpu_time_us <= 0:
            errors.append(
                f"{update.case_key}: {update.sku}: {language} GPU time must be > 0, "
                f"got {metric.gpu_time_us}"
            )
            continue
        noise_pct = criteria.noise_pct(metric.gpu_time_us, metric.gpu_noise_us)
        if criteria.noise_exceeds_limit(noise_pct):
            errors.append(
                f"{update.case_key}: {update.sku}: {language} noise "
                f"{noise_pct:.2f}% exceeds {criteria.max_noise_pct:.2f}% "
                f"({metric.gpu_noise_us:.6g}us / {metric.gpu_time_us:.6g}us)"
            )

    cpp = update.language_metrics.get("cpp")
    python = update.language_metrics.get("python")
    if cpp is None or python is None or cpp.gpu_time_us <= 0:
        return errors

    diff_pct, diff_us = criteria.parity_deltas(cpp.gpu_time_us, python.gpu_time_us)
    if criteria.relative_parity_exceeds_limit(diff_pct):
        errors.append(
            f"{update.case_key}: {update.sku}: C++/Python parity {diff_pct:+.2f}% "
            f"exceeds {criteria.max_perf_diff_pct:.2f}% "
            f"(cpp={cpp.gpu_time_us:.6g}us, python={python.gpu_time_us:.6g}us)"
        )
    if criteria.absolute_parity_exceeds_limit(diff_us):
        errors.append(
            f"{update.case_key}: {update.sku}: C++/Python parity {diff_us:+.2f}us "
            f"exceeds {criteria.max_perf_diff_us:.2f}us "
            f"(cpp={cpp.gpu_time_us:.6g}us, python={python.gpu_time_us:.6g}us)"
        )
    return errors


def validate_baseline_updates_quality(
    updates: Iterable[BaselineUpdate],
    *,
    criteria: BenchmarkQualityCriteria = DEFAULT_BENCHMARK_QUALITY,
) -> List[str]:
    errors: List[str] = []
    for update in updates:
        errors.extend(validate_baseline_update_quality(update, criteria=criteria))
    return errors


def validate_baselines_in_document(
    doc: OperatorConfigFile,
    index: ConfigIndex,
    sku_stem_set: Set[str],
    *,
    config_key_filter: Optional[Set[str]] = None,
    criteria: BenchmarkQualityCriteria = DEFAULT_BENCHMARK_QUALITY,
) -> List[str]:
    errors: List[str] = []
    for config_key, entry in doc.configs.items():
        if config_key_filter is not None and config_key not in config_key_filter:
            continue
        ref = index.refs_by_key.get(config_key)
        if ref is None:
            errors.append(f"{doc.path}: config {config_key!r} is not indexed")
            continue
        baselines = entry.get("baselines", {})
        if baselines is None:
            baselines = {}
        if not isinstance(baselines, dict):
            errors.append(f"{doc.path}: {config_key}: baselines must be an object")
            continue
        try:
            expected_case_keys = expected_case_keys_for_entry(config_key, entry)
        except BaselineError as exc:
            errors.append(f"{doc.path}: {config_key}: {exc}")
            continue
        for case_key in expected_case_keys:
            case_payload = baselines.get(case_key)
            if case_payload is None:
                errors.append(
                    f"{doc.path}: {config_key}: missing baseline case {case_key}"
                )
                continue
            if isinstance(case_payload, dict):
                missing_skus = sorted(sku_stem_set - set(case_payload))
                if missing_skus:
                    errors.append(
                        f"{doc.path}: {config_key}: {case_key}: "
                        f"missing SKU baseline(s) {missing_skus}"
                    )
        for case_key, case_payload in baselines.items():
            try:
                validate_case_key_for_config(case_key, ref)
            except BaselineError as exc:
                errors.append(f"{doc.path}: {config_key}: {case_key}: {exc}")
                continue
            if not isinstance(case_payload, dict):
                errors.append(
                    f"{doc.path}: {config_key}: {case_key}: SKU map must be an object"
                )
                continue
            for sku, metric_payload in case_payload.items():
                if sku not in sku_stem_set:
                    errors.append(
                        f"{doc.path}: {config_key}: {case_key}: unknown SKU {sku!r}"
                    )
                if not isinstance(metric_payload, dict):
                    errors.append(
                        f"{doc.path}: {config_key}: {case_key}: {sku}: "
                        "metrics must be an object"
                    )
                    continue
                missing = [
                    field
                    for field in BASELINE_METRIC_FIELDS
                    if field not in metric_payload
                ]
                if missing:
                    errors.append(
                        f"{doc.path}: {config_key}: {case_key}: {sku}: "
                        f"missing metric field(s) {missing}"
                    )
                    continue
                metric_errors: List[str] = []
                n_runs = metric_payload["n_runs"]
                if (
                    isinstance(n_runs, bool)
                    or not isinstance(n_runs, int)
                    or n_runs <= 0
                ):
                    metric_errors.append(
                        f"{doc.path}: {config_key}: {case_key}: {sku}: "
                        f"n_runs must be a positive integer, got {n_runs!r}"
                    )
                language_metrics: Dict[str, LanguageMetric] = {}
                for field_name in BASELINE_METRIC_FIELDS[1:]:
                    value = metric_payload[field_name]
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        metric_errors.append(
                            f"{doc.path}: {config_key}: {case_key}: {sku}: "
                            f"{field_name} must be numeric, got {value!r}"
                        )
                        continue
                    if not math.isfinite(float(value)) or float(value) < 0:
                        metric_errors.append(
                            f"{doc.path}: {config_key}: {case_key}: {sku}: "
                            f"{field_name} must be finite and >= 0, got {value!r}"
                        )
                gpu_gap_stddev_us = None
                if GPU_GAP_STDDEV_FIELD in metric_payload:
                    value = metric_payload[GPU_GAP_STDDEV_FIELD]
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        metric_errors.append(
                            f"{doc.path}: {config_key}: {case_key}: {sku}: "
                            f"{GPU_GAP_STDDEV_FIELD} must be numeric, got {value!r}"
                        )
                    elif not math.isfinite(float(value)) or float(value) < 0:
                        metric_errors.append(
                            f"{doc.path}: {config_key}: {case_key}: {sku}: "
                            f"{GPU_GAP_STDDEV_FIELD} must be finite and >= 0, "
                            f"got {value!r}"
                        )
                    elif isinstance(n_runs, int) and not isinstance(n_runs, bool):
                        if n_runs < 2:
                            metric_errors.append(
                                f"{doc.path}: {config_key}: {case_key}: {sku}: "
                                f"{GPU_GAP_STDDEV_FIELD} requires n_runs >= 2"
                            )
                        else:
                            gpu_gap_stddev_us = float(value)
                if metric_errors:
                    errors.extend(metric_errors)
                    continue
                for language, fields in METRIC_FIELDS_BY_LANGUAGE.items():
                    time_field, noise_field, bwutil_field = fields
                    language_metrics[language] = LanguageMetric(
                        gpu_time_us=float(metric_payload[time_field]),
                        gpu_noise_us=float(metric_payload[noise_field]),
                        gpu_bwutil=float(metric_payload[bwutil_field]),
                    )
                update = BaselineUpdate(
                    config_key=config_key,
                    benchmark=ref.benchmark,
                    case_key=case_key,
                    sku=sku,
                    n_runs=n_runs,
                    language_metrics=language_metrics,
                    gpu_gap_stddev_us=gpu_gap_stddev_us,
                )
                for quality_error in validate_baseline_update_quality(
                    update, criteria=criteria
                ):
                    errors.append(f"{doc.path}: {config_key}: {quality_error}")
    return errors
