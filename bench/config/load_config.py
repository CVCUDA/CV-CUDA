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

"""
Unified benchmark configuration loader for CV-CUDA benchmarks.

Provides configuration loading for both:
- Python benchmarks (structured config objects)
- C++ benchmark runners (CLI argument generation)
"""

import json
import warnings
from pathlib import Path
from typing import Dict, List, Any, Optional, Set

try:
    from .axis_order import order_axis_names
except ImportError:  # Standalone scripts add bench/config directly to sys.path.
    from axis_order import order_axis_names


# Allow-list for the per-entry `tier` field. The default `basic` suite is the
# broad CI-gated set. The `advanced` tier holds deeper operator profiles and is
# only run when explicitly requested.
VALID_TIERS = frozenset({"basic", "advanced"})


def _default_config_path() -> Path:
    script_dir = Path(__file__).parent
    config_path = script_dir / "bench_params.json"
    if config_path.exists():
        return config_path

    alt_path = script_dir.parent / "bench_params.json"
    if alt_path.exists():
        return alt_path

    raise FileNotFoundError(
        f"Cannot find bench_params.json. Searched:\n"
        f"  1. {config_path}\n"
        f"  2. {alt_path}\n"
        f"  load_config.py location: {Path(__file__).resolve()}"
    )


def _merge_config_entries(
    merged: Dict[str, Any], incoming: Dict[str, Any], source_path: Path
) -> None:
    duplicates = [key for key in incoming if key in merged]
    if duplicates:
        raise ValueError(f"Duplicate config key(s) in {source_path}: {duplicates}")
    merged.update(incoming)


def _expand_include_patterns(manifest_path: Path, patterns: List[str]) -> List[Path]:
    included_paths = []
    for pattern in patterns:
        matches = sorted(manifest_path.parent.glob(pattern))
        if not matches:
            raise FileNotFoundError(
                f"Config include {pattern!r} from {manifest_path} matched no files"
            )
        included_paths.extend(path for path in matches if path.is_file())
    return included_paths


def _default_operator_spec(operator_name: str) -> Dict[str, str]:
    return {
        "config": f"operators/{operator_name}.json",
        "cpp": f"bench_{operator_name}",
        "python": f"bench_{operator_name}.py",
    }


def _normalize_operator_manifest(
    manifest_path: Path, operators: Dict[str, Any]
) -> Dict[str, Dict[str, str]]:
    if not isinstance(operators, dict):
        raise ValueError(f"{manifest_path}: 'operators' must be a dictionary")

    normalized: Dict[str, Dict[str, str]] = {}
    for operator_name, spec in operators.items():
        if not isinstance(operator_name, str) or not operator_name:
            raise ValueError(
                f"{manifest_path}: operator names must be non-empty strings"
            )
        if not isinstance(spec, dict):
            raise ValueError(
                f"{manifest_path}: operator {operator_name!r} must map to a dictionary"
            )

        defaults = _default_operator_spec(operator_name)
        normalized_spec: Dict[str, str] = {}
        for field in ("config", "cpp", "python"):
            value = spec.get(field, defaults[field] if field != "config" else None)
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"{manifest_path}: operator {operator_name!r} missing "
                    f"non-empty string field {field!r}"
                )
            normalized_spec[field] = value

        normalized[operator_name] = normalized_spec

    return normalized


def _config_path_from_manifest_spec(manifest_path: Path, spec: Dict[str, str]) -> Path:
    config_path = Path(spec["config"])
    if not config_path.is_absolute():
        config_path = manifest_path.parent / config_path
    return config_path


def _infer_manifest_from_config(
    config: Dict[str, Any], config_path: Optional[Path] = None
) -> Dict[str, Dict[str, str]]:
    if _is_operator_config_document(config):
        benchmark = config["benchmark"]
        spec = _default_operator_spec(benchmark)
        if config_path is not None:
            spec["config"] = str(config_path)
        return {benchmark: spec}

    operators: Dict[str, Dict[str, str]] = {}
    for entry in config.values():
        if not isinstance(entry, dict):
            continue
        benchmark = entry.get("benchmark")
        if not isinstance(benchmark, str) or not benchmark:
            continue
        if benchmark in operators:
            continue
        spec = _default_operator_spec(benchmark)
        if config_path is not None:
            spec["config"] = str(config_path)
        operators[benchmark] = spec
    return operators


def _is_operator_config_document(config: Dict[str, Any]) -> bool:
    return "benchmark" in config and "configs" in config


def _strip_machine_owned_fields(entry: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in entry.items() if key != "baselines"}


def _flatten_operator_config_document(
    config: Dict[str, Any], config_path: Path
) -> Dict[str, Any]:
    benchmark = config.get("benchmark")
    configs = config.get("configs")
    if not isinstance(benchmark, str) or not benchmark:
        raise ValueError(f"{config_path}: top-level 'benchmark' must be a string")
    if not isinstance(configs, dict):
        raise ValueError(f"{config_path}: top-level 'configs' must be a dictionary")

    flattened: Dict[str, Any] = {}
    for config_key, entry in configs.items():
        if not isinstance(config_key, str) or not config_key:
            raise ValueError(f"{config_path}: config keys must be non-empty strings")
        if not isinstance(entry, dict):
            raise ValueError(
                f"{config_path}: config entry {config_key!r} must be a dictionary"
            )
        cleaned = {"benchmark": benchmark}
        cleaned.update(_strip_machine_owned_fields(entry))
        flattened[config_key] = cleaned
    return flattened


def _flatten_legacy_config_document(
    config: Dict[str, Any], config_path: Path
) -> Dict[str, Any]:
    flattened: Dict[str, Any] = {}
    for config_key, entry in config.items():
        if not isinstance(entry, dict):
            flattened[config_key] = entry
            continue
        flattened[config_key] = _strip_machine_owned_fields(entry)
    return flattened


def _load_manifest_path(config_path: Path) -> Dict[str, Dict[str, str]]:
    config_path = Path(config_path)

    if config_path.is_dir():
        manifest = config_path / "bench_params.json"
        if manifest.exists():
            return _load_manifest_path(manifest)

        inferred: Dict[str, Dict[str, str]] = {}
        for path in sorted(config_path.glob("*.json")):
            inferred.update(_load_manifest_path(path))
        return inferred

    with open(config_path, "r") as f:
        cfg = json.load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"{config_path}: benchmark config must be a dictionary")

    has_operators = "operators" in cfg
    has_include = "include" in cfg
    if has_operators and has_include:
        raise ValueError(
            f"{config_path}: use either 'operators' or 'include', not both"
        )

    if has_operators:
        return _normalize_operator_manifest(config_path, cfg["operators"])

    if has_include:
        include_patterns = cfg["include"]
        if not isinstance(include_patterns, list) or not all(
            isinstance(pattern, str) for pattern in include_patterns
        ):
            raise ValueError(f"{config_path}: 'include' must be a list of strings")

        inferred: Dict[str, Dict[str, str]] = {}
        for path in _expand_include_patterns(config_path, include_patterns):
            inferred.update(_load_manifest_path(path))
        return inferred

    return _infer_manifest_from_config(cfg, config_path)


def _load_config_path(config_path: Path) -> Dict[str, Any]:
    config_path = Path(config_path)

    if config_path.is_dir():
        manifest = config_path / "bench_params.json"
        if manifest.exists():
            return _load_config_path(manifest)

        merged: Dict[str, Any] = {}
        for path in sorted(config_path.glob("*.json")):
            _merge_config_entries(merged, _load_config_path(path), path)
        return merged

    with open(config_path, "r") as f:
        cfg = json.load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"{config_path}: benchmark config must be a dictionary")

    has_operators = "operators" in cfg
    has_include = "include" in cfg
    if has_operators and has_include:
        raise ValueError(
            f"{config_path}: use either 'operators' or 'include', not both"
        )

    if has_operators:
        merged: Dict[str, Any] = {}
        for spec in _normalize_operator_manifest(
            config_path, cfg["operators"]
        ).values():
            path = _config_path_from_manifest_spec(config_path, spec)
            _merge_config_entries(merged, _load_config_path(path), path)
        return merged

    include_patterns = cfg.get("include")
    if include_patterns is None:
        if _is_operator_config_document(cfg):
            return _flatten_operator_config_document(cfg, config_path)
        return _flatten_legacy_config_document(cfg, config_path)
    if not isinstance(include_patterns, list) or not all(
        isinstance(pattern, str) for pattern in include_patterns
    ):
        raise ValueError(f"{config_path}: 'include' must be a list of strings")

    merged: Dict[str, Any] = {}
    for path in _expand_include_patterns(config_path, include_patterns):
        _merge_config_entries(merged, _load_config_path(path), path)
    return merged


def _validate_tiers(config: Dict[str, Any]) -> None:
    """Hard-fail if any entry is missing `tier` or has an unknown value.

    Run at config-load time so a typo (or a forgotten `tier` on a freshly
    added entry) surfaces before any benchmark runs, with the offending key
    in the error.
    """
    bad = []
    for key, entry in config.items():
        if not isinstance(entry, dict):
            continue
        tier = entry.get("tier")
        if tier is None:
            bad.append(f"{key!r}: missing required 'tier' field")
        elif tier not in VALID_TIERS:
            bad.append(f"{key!r}: tier={tier!r} not in {sorted(VALID_TIERS)}")
    if bad:
        raise ValueError("Invalid bench config values:\n  " + "\n  ".join(bad))


def parse_tier_arg(value: str) -> Set[str]:
    """Convert a --tier CLI value into a set of tier names.

    Accepts:
        "basic"            → {"basic"}
        "advanced"         → {"advanced"}
        "basic,advanced"   → {"basic", "advanced"}

    Raises ValueError on unknown tier names.
    """
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        raise ValueError(f"--tier: empty value {value!r}")
    unknown = [p for p in parts if p not in VALID_TIERS]
    if unknown:
        raise ValueError(
            f"--tier: unknown tier(s) {unknown}; "
            f"valid options: {sorted(VALID_TIERS)}"
        )
    return set(parts)


def parse_config_key_arg(value: str) -> List[str]:
    """Convert a --config-key CLI value into an ordered list of config keys.

    Accepts:
        "resize_advanced"                  -> ["resize_advanced"]
        "resize_advanced,gaussian_advanced" -> ["resize_advanced", "gaussian_advanced"]

    Raises ValueError on empty values or duplicate keys.
    """
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        raise ValueError(f"--config-key: empty value {value!r}")

    seen = set()
    duplicates = []
    for key in parts:
        if key in seen and key not in duplicates:
            duplicates.append(key)
        seen.add(key)
    if duplicates:
        raise ValueError(f"--config-key: duplicate key(s) {duplicates}")

    return parts


def parse_operator_arg(value: str) -> List[str]:
    """Convert a --operator CLI value into an ordered list of operator names.

    Accepts comma-separated and/or space-separated values. Matching is exact
    against the operator names advertised by bench_params.json.
    """
    parts = [p.strip() for p in value.replace(",", " ").split() if p.strip()]
    if not parts:
        raise ValueError(f"--operator: empty value {value!r}")

    seen = set()
    duplicates = []
    for operator_name in parts:
        if operator_name in seen and operator_name not in duplicates:
            duplicates.append(operator_name)
        seen.add(operator_name)
    if duplicates:
        raise ValueError(f"--operator: duplicate operator(s) {duplicates}")

    return parts


class BenchmarkConfig:
    """Generic benchmark configuration container."""

    def __init__(self, config_dict: Dict[str, Any]):
        self.dtypes = config_dict.get("dtypes", [])
        self.string_axes = config_dict.get("string_axes", {})
        self.int64_axes = config_dict.get("int64_axes", {})
        self.float64_axes = config_dict.get("float64_axes", {})
        self.metadata = config_dict.get("metadata", {})
        self.warmup_iterations = config_dict.get("warmup_iterations", 100)

    def get_string_axis(self, name: str) -> List[str]:
        """Get values for a string axis."""
        return self.string_axes.get(name, [])

    def get_int64_axis(self, name: str) -> List[int]:
        """Get values for an int64 axis."""
        return self.int64_axes.get(name, [])

    def get_float64_axis(self, name: str) -> List[float]:
        """Get values for a float64 axis."""
        return self.float64_axes.get(name, [])

    def has_axis(self, name: str) -> bool:
        """Check if an axis exists in any category."""
        return (
            name in self.string_axes
            or name in self.int64_axes
            or name in self.float64_axes
        )

    def get_ordered_axes(self):
        """
        Get all axes in standardized order matching C++ benchmarks.

        Returns a list of tuples: (axis_name, axis_type, values)
        where axis_type is 'string', 'int64', or 'float64'.
        """
        ordered = []

        for axis_name in order_axis_names(
            (self.string_axes, self.int64_axes, self.float64_axes)
        ):
            if axis_name in self.string_axes:
                ordered.append((axis_name, "string", self.string_axes[axis_name]))
            elif axis_name in self.int64_axes:
                ordered.append((axis_name, "int64", self.int64_axes[axis_name]))
            else:
                ordered.append((axis_name, "float64", self.float64_axes[axis_name]))

        return ordered


class ConfigLoader:
    """Loads benchmark configurations from JSON file or manifest."""

    def __init__(self, config_path: Optional[str] = None):
        import os

        if config_path is None:
            config_path = _default_config_path()

        # Debug output for path issues
        if os.environ.get("BENCH_DEBUG"):
            print(f"[DEBUG] ConfigLoader: loading from {config_path}")

        self.config = _load_config_path(Path(config_path))

        _validate_tiers(self.config)

        if os.environ.get("BENCH_DEBUG"):
            print(f"[DEBUG] ConfigLoader: loaded {len(self.config)} config entries")

    def get_operator_config(self, operator_name: str) -> BenchmarkConfig:
        """Get configuration for a specific operator."""
        # Direct hit (a fully-qualified key like "resize_expand_basic").
        if operator_name in self.config:
            return BenchmarkConfig(self.config[operator_name])

        # Standalone-run convenience: a developer running
        # `python3 bench_resize.py` with no --config-key gets
        # operator_name="resize" by default. Since every entry now carries
        # a tier suffix, fall back to "<name>_basic" so the basic suite
        # stays directly invokable; if that's still not unique enough
        # (operator has no entry that's exactly "<name>_basic"), point
        # them at --config-key.
        fallback = f"{operator_name}_basic"
        if fallback in self.config:
            return BenchmarkConfig(self.config[fallback])

        raise KeyError(
            f"Operator not found in config: {operator_name!r}. "
            f"Tried {operator_name!r} and {fallback!r}. "
            f"Pass --config-key to select a specific entry."
        )


def load_bench_config(config_file: str = None) -> Dict[str, Any]:
    """
    Load benchmark configuration JSON.

    Args:
        config_file: Path to a flat config file, split-config manifest, or
            directory. If None, looks for bench_params.json next to this script.

    Returns:
        Dictionary of operator configurations
    """
    if config_file is None:
        config_file = _default_config_path()

    cfg = _load_config_path(Path(config_file))
    _validate_tiers(cfg)
    return cfg


def load_bench_manifest(config_file: Optional[str] = None) -> Dict[str, Dict[str, str]]:
    """
    Load the benchmark operator manifest.

    Args:
        config_file: Path to bench_params.json, a flat config file, split-config
            manifest, or directory. If None, uses the default bench_params.json.

    Returns:
        Dictionary keyed by exact operator name. Each entry provides the
        operator config file and the C++/Python benchmark file names.
    """
    if config_file is None:
        config_file = _default_config_path()

    return _load_manifest_path(Path(config_file))


def load_operator_config(operator_name: str, args: list = None) -> tuple:
    """
    Load configuration for a specific operator with --config-key aliasing support.

    This is the single entry point for Python benchmarks to load their config.
    It handles:
    1. Parsing --config-key from command line args (uses sys.argv[1:] by default)
    2. Loading the config by key (falling back to operator_name if no --config-key)
    3. Returning cleaned args for nvbench

    Args:
        operator_name: Default operator name (e.g., "resize")
        args: Command line args (default: sys.argv[1:], excluding script path)

    Returns:
        Tuple of (BenchmarkConfig, remaining_args) where remaining_args
        has --config-key removed and can be passed to nvbench.

    Example usage in benchmark:
        ```python
        from load_config import load_operator_config, register_axes_from_config

        config, bench_args = load_operator_config("resize")

        b = bench.register(my_benchmark)
        b.add_string_axis("InOutDataType", config.dtypes)
        register_axes_from_config(b, config)
        bench.run_all_benchmarks(bench_args)
        ```
    """
    import sys

    if args is None:
        # Skip argv[0] (script path) - nvbench expects arguments only
        args = sys.argv[1:]

    # Parse --config-key and --config-file from args
    remaining_args = []
    config_key = operator_name
    config_file = None
    i = 0
    while i < len(args):
        if args[i] == "--config-key" and i + 1 < len(args):
            config_key = args[i + 1]
            i += 2  # Skip both --config-key and its value
        elif args[i] == "--config-file" and i + 1 < len(args):
            config_file = args[i + 1]
            i += 2  # Skip both --config-file and its value
        else:
            remaining_args.append(args[i])
            i += 1

    # Load config by key (from --config-file if provided, else default)
    loader = ConfigLoader(config_file) if config_file else ConfigLoader()
    config = loader.get_operator_config(config_key)

    # Debug output (only if BENCH_DEBUG env var is set)
    import os

    if os.environ.get("BENCH_DEBUG"):
        print(f"[DEBUG] load_operator_config: key={config_key}")
        print(f"[DEBUG]   dtypes={config.dtypes}")
        print(f"[DEBUG]   string_axes={config.string_axes}")
        print(f"[DEBUG]   int64_axes={config.int64_axes}")

    return config, remaining_args


def register_axes_from_config(benchmark, config: BenchmarkConfig):
    """
    Register benchmark axes from config in standardized order matching C++.

    Args:
        benchmark: nvbench benchmark object
        config: BenchmarkConfig object

    Usage:
        config = load_operator_config("resize")
        b = bench.register(resize_benchmark)
        b.add_string_axis("dtype", config.dtypes)  # Type axis first
        register_axes_from_config(b, config)       # Then ordered axes
    """
    ordered = config.get_ordered_axes()

    # Register axes in standardized order (interleaving types as needed)
    for axis_name, axis_type, values in ordered:
        if axis_type == "string":
            benchmark.add_string_axis(axis_name, values)
        elif axis_type == "int64":
            benchmark.add_int64_axis(axis_name, values)
        elif axis_type == "float64":
            benchmark.add_float64_axis(axis_name, values)


# Map Python dtype names to nvbench type axis names
# Used by generate_axis_args() to filter C++ benchmarks by dtype at runtime
DTYPE_TO_NVBENCH = {
    "uint8": "U8",
    "uint16": "U16",
    "uint32": "U32",
    "uint64": "U64",
    "int8": "I8",
    "int16": "I16",
    "int32": "I32",
    "int64": "I64",
    "float32": "F32",
    "float64": "F64",
}


def generate_axis_args(operator_name: str, config: Dict[str, Any]) -> List[str]:
    """
    Generate --axis CLI arguments from operator config.

    Used by C++ benchmark runner to pass config values as command-line arguments.

    Args:
        operator_name: Name of operator (e.g., "resize")
        config: Full config dictionary

    Returns:
        List of CLI arguments like ["--axis", "shape=[1x1080x1920]", ...]
    """
    if operator_name not in config:
        return []

    op_config = config[operator_name]
    args = []

    # Add dtype filter if specified in config
    # This ensures C++ runs only the dtypes specified in this config entry
    if "dtypes" in op_config and op_config["dtypes"]:
        dtypes = op_config["dtypes"]
        # Map Python dtype names to nvbench type names
        nvbench_types = [DTYPE_TO_NVBENCH.get(dt, dt) for dt in dtypes]
        if len(nvbench_types) == 1:
            args.extend(["--axis", f"InOutDataType={nvbench_types[0]}"])
        else:
            types_str = ",".join(nvbench_types)
            args.extend(["--axis", f"InOutDataType=[{types_str}]"])

    # Add string axes
    if "string_axes" in op_config:
        for axis_name, values in op_config["string_axes"].items():
            if len(values) == 1:
                args.extend(["--axis", f"{axis_name}={values[0]}"])
            else:
                values_str = ",".join(values)
                args.extend(["--axis", f"{axis_name}=[{values_str}]"])

    # Add int64 axes
    if "int64_axes" in op_config:
        for axis_name, values in op_config["int64_axes"].items():
            if len(values) == 1:
                args.extend(["--axis", f"{axis_name}={values[0]}"])
            else:
                values_str = ",".join(str(v) for v in values)
                args.extend(["--axis", f"{axis_name}=[{values_str}]"])

    # Add float64 axes
    if "float64_axes" in op_config:
        for axis_name, values in op_config["float64_axes"].items():
            if len(values) == 1:
                args.extend(["--axis", f"{axis_name}={values[0]}"])
            else:
                values_str = ",".join(str(v) for v in values)
                args.extend(["--axis", f"{axis_name}=[{values_str}]"])

    return args


def get_operator_from_benchmark_name(bench_name: str) -> str:
    """
    Extract operator name from benchmark executable name.

    Examples:
        bench_resize -> resize
        bench_resize.py -> resize
        bench_gaussian -> gaussian
    """
    name = bench_name
    # Remove .py extension if present
    if name.endswith(".py"):
        name = name[:-3]
    # Remove bench_ prefix
    if name.startswith("bench_"):
        name = name[len("bench_") :]  # noqa: E203
    return name


def get_configs_for_benchmark(
    benchmark_name: str,
    config: Dict[str, Any],
    tiers: Optional[Set[str]] = None,
) -> List[str]:
    """
    Get all config keys that target a specific benchmark.

    Returns a list of config keys where the "benchmark" field matches
    the given benchmark_name. Optionally filtered to a subset of tiers.

    Args:
        benchmark_name: The benchmark name (e.g., "resize")
        config: Full config dictionary
        tiers: Optional set of tier names (e.g. {"basic"}). None → no tier
            filter.

    Returns:
        List of config keys that target this benchmark
    """
    matching_configs = []

    for config_key, op_config in config.items():
        # Skip non-dict entries (shouldn't happen but be safe)
        if not isinstance(op_config, dict):
            continue

        # Skip entries without benchmark field (invalid config)
        if "benchmark" not in op_config:
            warnings.warn(
                f"Config entry '{config_key}' missing required 'benchmark' field, skipping"
            )
            continue

        # Match on the benchmark field
        if op_config["benchmark"] != benchmark_name:
            continue

        # Tier filter: tier was already validated at load time, so we can
        # trust it's present and one of VALID_TIERS.
        if tiers is not None and op_config.get("tier") not in tiers:
            continue

        matching_configs.append(config_key)

    return matching_configs
