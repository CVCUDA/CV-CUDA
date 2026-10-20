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

"""Validate committed benchmark baseline schema, quality, and parity."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

# Support direct source-tree invocation from any working directory.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _internal.axes import format_axes
from _internal.baselines import (
    DEFAULT_CONFIG_DIR,
    DEFAULT_MAX_BASELINE_REGRESSION_PCT,
    BaselineError,
    OperatorConfigFile,
    load_config_index,
    load_sku_map,
    load_sku_policies,
    normalize_operator_names,
    operator_config_paths,
    parse_case_key,
    split_operator_payload,
    validate_sku_map_cuda_coverage,
    validate_baselines_in_document,
)

BaselineMetricKey = Tuple[str, str, str, str]
BaselineMetricValue = Tuple[float, Tuple[Tuple[str, str], ...]]
GIT_COMMAND_TIMEOUT_SECONDS = 30


def _selected_paths(args: argparse.Namespace) -> List[Path]:
    operators_dir = args.config_dir / "operators"
    if args.paths:
        return [Path(path) for path in args.paths]
    operators = normalize_operator_names(args.operator)
    if operators is None:
        return operator_config_paths(operators_dir)
    return [operators_dir / f"{operator}.json" for operator in sorted(operators)]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _git_command(*args: str, repo: Optional[Path] = None) -> List[str]:
    repo_root = repo or _repo_root()
    # The CI runner may check out files under a UID different from the benchmark container's UID.
    return ["git", "-c", f"safe.directory={repo_root}", *args]


def _path_at_git_ref(path: Path, repo_root: Path) -> str:
    try:
        relpath = path.resolve().relative_to(repo_root.resolve())
    except ValueError as exc:
        raise BaselineError(
            f"{path}: path is outside git repository {repo_root}"
        ) from exc
    return relpath.as_posix()


def resolve_git_ref(ref: str, *, repo: Optional[Path] = None) -> str:
    """Expand a ref to its full commit SHA, failing if it does not exist."""

    # The ^{commit} suffix forces annotated tags and other refs to resolve to a commit object.
    try:
        proc = subprocess.run(
            _git_command(
                "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}", repo=repo
            ),
            cwd=repo or _repo_root(),
            capture_output=True,
            text=True,
            timeout=GIT_COMMAND_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise BaselineError(
            f"timed out verifying baseline regression ref: {ref}"
        ) from exc
    if proc.returncode != 0:
        raise BaselineError(f"baseline regression ref does not exist: {ref}")
    return proc.stdout.strip()


def verify_git_ref(ref: str, *, repo: Optional[Path] = None) -> None:
    resolve_git_ref(ref, repo=repo)


def operator_names_at_ref(ref: str, *, repo: Optional[Path] = None) -> Set[str]:
    """Operator config stems present at a ref, independent of the checkout."""

    directory = DEFAULT_CONFIG_DIR.name + "/operators"
    try:
        proc = subprocess.run(
            _git_command(
                "ls-tree", "--name-only", ref, f"bench/{directory}/", repo=repo
            ),
            cwd=repo or _repo_root(),
            capture_output=True,
            text=True,
            timeout=GIT_COMMAND_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise BaselineError(f"timed out listing operator configs at {ref}") from exc
    if proc.returncode != 0:
        return set()
    return {
        Path(line.strip()).stem
        for line in proc.stdout.splitlines()
        if line.strip().endswith(".json")
    }


def operator_doc_at_ref(
    ref: str, path: Path, *, repo: Optional[Path] = None
) -> Optional[OperatorConfigFile]:
    git_path = _path_at_git_ref(path, repo or _repo_root())
    try:
        proc = subprocess.run(
            _git_command("show", f"{ref}:{git_path}", repo=repo),
            cwd=repo or _repo_root(),
            capture_output=True,
            text=True,
            timeout=GIT_COMMAND_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise BaselineError(f"timed out reading {ref}:{git_path}") from exc
    if proc.returncode != 0:
        return None
    try:
        raw = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise BaselineError(f"{ref}:{git_path}: invalid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise BaselineError(f"{ref}:{git_path}: JSON root must be an object")
    return split_operator_payload(path, raw)


def baseline_metrics(
    doc: OperatorConfigFile,
    config_key_filter: Optional[Set[str]] = None,
) -> Dict[BaselineMetricKey, BaselineMetricValue]:
    rows: Dict[BaselineMetricKey, BaselineMetricValue] = {}
    for config_key, entry in doc.configs.items():
        if config_key_filter is not None and config_key not in config_key_filter:
            continue
        baselines = entry.get("baselines", {})
        if not isinstance(baselines, dict):
            continue
        for case_key, case_payload in baselines.items():
            if not isinstance(case_payload, dict):
                continue
            try:
                _, axes = parse_case_key(case_key)
            except BaselineError:
                axes = ()
            for sku, metric_payload in case_payload.items():
                if not isinstance(metric_payload, dict):
                    continue
                for language, field_name in (
                    ("cpp", "gpu_time_us_cpp"),
                    ("python", "gpu_time_us_python"),
                ):
                    value = metric_payload.get(field_name)
                    if isinstance(value, (int, float)):
                        rows[(config_key, case_key, str(sku), language)] = (
                            float(value),
                            axes,
                        )
    return rows


def _baseline_regression_errors_against_ref(
    docs_by_path: Dict[Path, OperatorConfigFile],
    paths: Sequence[Path],
    ref: str,
    max_regression_pct: float,
    config_key_filter: Optional[Set[str]],
) -> List[str]:
    verify_git_ref(ref)
    errors: List[str] = []
    for path in paths:
        current_doc = docs_by_path[path]
        base_doc = operator_doc_at_ref(ref, path)
        if base_doc is None:
            continue
        base_rows = baseline_metrics(base_doc, config_key_filter)
        for key, (current_us, axes) in baseline_metrics(
            current_doc, config_key_filter
        ).items():
            base_row = base_rows.get(key)
            if base_row is None:
                continue
            base_us, _ = base_row
            if base_us <= 0:
                continue
            pct = (current_us / base_us - 1.0) * 100.0
            if current_us <= base_us * (1.0 + max_regression_pct / 100.0):
                continue
            config_key, case_key, sku, language = key
            errors.append(
                f"{path}: {sku} {config_key} [{language}] "
                f"({format_axes(axes)}): same-key baseline regressed {pct:+.2f}% "
                f"vs {ref} ({base_us:.2f} -> {current_us:.2f} us, "
                f"case={case_key})"
            )
    return errors


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate committed baselines and optionally compare them with a git ref.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  bench/_internal/validate_baselines.py
  bench/_internal/validate_baselines.py --operator resize,gaussian
  bench/_internal/validate_baselines.py --reject-regressions-from origin/main

Exit status:
  0  All selected baselines are valid.
  1  Baseline validation or regression checks failed.
  2  Inputs, configuration, or git reference are invalid.
""",
    )
    p.add_argument(
        "paths",
        nargs="*",
        metavar="OPERATOR_JSON",
        type=Path,
        help=(
            "Operator JSON file(s) to validate (default: every file under "
            "<config-dir>/operators). Takes precedence over --operator."
        ),
    )
    p.add_argument(
        "--config-dir",
        metavar="DIR",
        type=Path,
        default=DEFAULT_CONFIG_DIR,
        help=(
            "Benchmark config directory containing operators/ and sku_map.json "
            "(default: bench/config)."
        ),
    )
    p.add_argument(
        "--sku-map",
        metavar="JSON",
        type=Path,
        default=None,
        help="SKU map to use (default: <config-dir>/sku_map.json).",
    )
    p.add_argument(
        "--operator",
        action="append",
        default=None,
        metavar="NAME",
        help="Validate named operator(s); repeatable or comma-separated.",
    )
    p.add_argument(
        "--config-key",
        action="append",
        default=None,
        metavar="KEY",
        help="Validate named config key(s); repeatable or comma-separated.",
    )
    p.add_argument(
        "--reject-regressions-from",
        metavar="REF",
        default=None,
        help=(
            "Compare same-key timings with git REF and fail on slowdowns above "
            "the threshold."
        ),
    )
    p.add_argument(
        "--max-regression-pct",
        type=float,
        default=DEFAULT_MAX_BASELINE_REGRESSION_PCT,
        help=(
            "Maximum allowed same-key committed-baseline slowdown when "
            f"--reject-regressions-from is set. Default: "
            f"{DEFAULT_MAX_BASELINE_REGRESSION_PCT}."
        ),
    )
    p.add_argument(
        "--allow-regressions",
        action="store_true",
        help=(
            "Permit reviewed timing slowdowns; requires "
            "--reject-regressions-from. Schema, quality, and parity checks "
            "still apply."
        ),
    )
    return p.parse_args(argv)


def _first_seed_path(scope: str, paths: Sequence[Path]) -> Path:
    """Fail closed: preparation is local and limited to one uncommitted new operator."""
    if any(
        name in {"CI", "CONTINUOUS_INTEGRATION"} or name.startswith("CI_")
        for name in os.environ
    ):
        raise BaselineError("local first-seed preparation is forbidden in CI")
    if not re.fullmatch(r"bench/config/operators/[a-z][a-z0-9_]*\.json", scope):
        raise BaselineError("first-seed scope must be one canonical operator JSON path")
    repo = _repo_root()
    path = repo / scope
    if path.resolve() != path.absolute() or path.resolve() not in {
        selected.resolve() for selected in paths
    }:
        raise BaselineError(
            "first-seed path must be selected and must not be a symlink"
        )
    for ref in ("origin/main", "HEAD"):
        sha = resolve_git_ref(ref)
        proc = subprocess.run(
            _git_command("ls-tree", "--name-only", sha, "--", scope),
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=GIT_COMMAND_TIMEOUT_SECONDS,
        )
        if proc.returncode != 0:
            raise BaselineError(
                f"cannot verify first-seed absence at {ref}: {proc.stderr}"
            )
        if proc.stdout.strip():
            raise BaselineError(f"first-seed operator already exists at {ref}: {scope}")
    return path.resolve()


def main(
    argv: Optional[Sequence[str]] = None, *, local_first_seed: Optional[str] = None
) -> int:
    args = _parse_args(argv)
    sku_map_path = args.sku_map or (args.config_dir / "sku_map.json")
    paths = _selected_paths(args)

    try:
        first_seed = (
            _first_seed_path(local_first_seed, paths)
            if local_first_seed is not None
            else None
        )
        if first_seed is not None and (
            args.config_dir.resolve() != (_repo_root() / "bench/config").resolve()
            or args.sku_map is not None
            or args.config_key is not None
            or args.allow_regressions
        ):
            raise BaselineError(
                "first-seed preparation requires the canonical full validation policy"
            )
        if args.max_regression_pct < 0:
            raise BaselineError("--max-regression-pct must be >= 0")
        if args.allow_regressions and not args.reject_regressions_from:
            raise BaselineError(
                "--allow-regressions requires --reject-regressions-from"
            )
        missing = [path for path in paths if not path.is_file()]
        if missing:
            raise BaselineError(f"operator JSON file(s) not found: {missing}")
        index = load_config_index(args.config_dir / "operators")
        sku_map = load_sku_map(sku_map_path, strict=True)
        validate_sku_map_cuda_coverage(sku_map)
        policies = load_sku_policies(sku_map_path, strict=True)
        stems = set(policies)
        config_key_filter = normalize_operator_names(args.config_key)
        if config_key_filter is not None:
            unknown = sorted(config_key_filter - set(index.refs_by_key))
            if unknown:
                raise BaselineError(f"unknown config key(s): {unknown}")

        errors: List[str] = []
        docs_by_path: Dict[Path, OperatorConfigFile] = {}
        for path in paths:
            doc = index.docs_by_path.get(path)
            if doc is None:
                selected = load_config_index(paths=[path])
                doc = selected.docs_by_path[path]
            docs_by_path[path] = doc
            prepare = first_seed is not None and path.resolve() == first_seed
            if prepare and (doc.benchmark != path.stem or not doc.configs):
                raise BaselineError(
                    "first-seed document must declare its own nonempty operator"
                )
            errors.extend(
                validate_baselines_in_document(
                    doc,
                    index,
                    stems,
                    sku_policies=policies,
                    config_key_filter=config_key_filter,
                    prepare_first_seed=prepare,
                )
            )
        if not errors and args.reject_regressions_from:
            if args.allow_regressions:
                verify_git_ref(args.reject_regressions_from)
            else:
                errors.extend(
                    _baseline_regression_errors_against_ref(
                        docs_by_path,
                        paths,
                        args.reject_regressions_from,
                        args.max_regression_pct,
                        config_key_filter,
                    )
                )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if errors:
        print("JSON baseline validation failed:", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)
        return 1

    if args.allow_regressions:
        print(
            "WARNING: reviewed same-key baseline regressions are allowed; "
            "schema, quality, and parity checks remain enforced.",
            file=sys.stderr,
        )

    scope = ", ".join(str(path) for path in paths)
    if first_seed is not None:
        print(
            f"PREPARATION ONLY: {first_seed}; absent baseline cases deferred. "
            "Not baseline completion: reference-SKU measurements and strict validation "
            "are still required. Other selected documents remain strict."
        )
    else:
        print(f"JSON baseline validation passed ({len(paths)} file(s)): {scope}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
