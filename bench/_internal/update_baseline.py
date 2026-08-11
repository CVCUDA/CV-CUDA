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

"""Validate and import run_bench.py JSON artifacts into committed baselines."""

from __future__ import annotations

import argparse
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
    BaselineUpdate,
    OperatorConfigFile,
    apply_updates_to_document,
    baseline_updates_from_jsons,
    load_config_index,
    sku_stems,
    validate_baselines_in_document,
    validate_baseline_updates_quality,
    write_json,
)


def _collect_jsons(paths: Sequence[Path]) -> List[Path]:
    jsons: List[Path] = []
    for path in paths:
        if not path.exists():
            raise BaselineError(f"--from path does not exist: {path}")
        if path.is_file():
            if path.suffix.lower() == ".json":
                jsons.append(path)
            else:
                raise BaselineError(f"--from file is not a .json file: {path}")
        else:
            found_jsons = sorted(path.glob("*.json"))
            if not found_jsons:
                raise BaselineError(f"--from directory has no *.json files: {path}")
            jsons.extend(found_jsons)
    if not jsons:
        raise BaselineError("no input files collected from --from arguments")

    seen: Set[Path] = set()
    deduped_jsons: List[Path] = []
    for path in jsons:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped_jsons.append(path)
    return deduped_jsons


def _normalize_operators(raw: Optional[Sequence[str]]) -> Optional[Set[str]]:
    if not raw:
        return None
    out: Set[str] = set()
    for item in raw:
        for token in str(item).split(","):
            token = token.strip()
            if token:
                out.add(token)
    return out or None


_OP_FILE_RE = re.compile(r"^src/cvcuda/priv/Op([A-Za-z0-9_]+)\.(?:cu|cpp|hpp|h)$")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_diff_ref(requested: Optional[str]) -> str:
    if requested:
        return requested
    for candidate in ("origin/main", "main"):
        rc = subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", candidate],
            cwd=_repo_root(),
            capture_output=True,
        ).returncode
        if rc == 0:
            return candidate
    raise BaselineError(
        "--from-diff could not find origin/main or main. Pass an explicit ref."
    )


def _ops_from_git_diff(ref: str, bench_names: Set[str]) -> Set[str]:
    proc = subprocess.run(
        ["git", "diff", "--name-only", f"{ref}...HEAD"],
        cwd=_repo_root(),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise BaselineError(f"`git diff` failed:\n{proc.stderr.strip()}")

    matched: Set[str] = set()
    seen_unmatched: List[str] = []
    for filename in [line.strip() for line in proc.stdout.splitlines() if line.strip()]:
        match = _OP_FILE_RE.match(filename)
        if not match:
            continue
        stem = match.group(1).lower()
        if stem in bench_names:
            matched.add(stem)
        else:
            seen_unmatched.append(f"{filename} -> {stem!r}")

    if not matched:
        message = (
            f"--from-diff (ref={ref}) found no operator files matching "
            "src/cvcuda/priv/Op*.{cu,cpp,hpp,h} whose stem is also present in "
            "the input benchmark JSON."
        )
        if seen_unmatched:
            message += "\n  Files inspected:\n    " + "\n    ".join(seen_unmatched)
        raise BaselineError(message)
    return matched


def _updates_for_operators(
    updates: Dict[Tuple[str, str, str], BaselineUpdate],
    operators: Optional[Set[str]],
) -> Dict[Tuple[str, str, str], BaselineUpdate]:
    if operators is None:
        return dict(updates)
    return {
        key: update for key, update in updates.items() if update.benchmark in operators
    }


def _existing_metrics(
    doc: OperatorConfigFile, update: BaselineUpdate, language: str
) -> Optional[float]:
    entry = doc.configs.get(update.config_key, {})
    baselines = entry.get("baselines", {})
    case_payload = (
        baselines.get(update.case_key, {}) if isinstance(baselines, dict) else {}
    )
    metric_payload = (
        case_payload.get(update.sku, {}) if isinstance(case_payload, dict) else {}
    )
    field_name = "gpu_time_us_cpp" if language == "cpp" else "gpu_time_us_python"
    value = metric_payload.get(field_name) if isinstance(metric_payload, dict) else None
    return float(value) if isinstance(value, (int, float)) else None


def _diff_summary(
    docs_by_path: Dict[Path, OperatorConfigFile],
    updates: Sequence[BaselineUpdate],
) -> List[dict]:
    rows: List[dict] = []
    for update in updates:
        doc = docs_by_path[
            next(
                path
                for path, doc in docs_by_path.items()
                if update.config_key in doc.configs
            )
        ]
        _, axes = update.case_key.split("[", 1)
        axes_tuple = tuple(
            tuple(part.rstrip("]").split("=", 1))
            for part in ("[" + axes).split("[")
            if part
        )
        for language, metric in update.language_metrics.items():
            prev_us = _existing_metrics(doc, update, language)
            pct = (
                (metric.gpu_time_us / prev_us - 1.0) * 100.0
                if prev_us is not None and prev_us > 0
                else None
            )
            rows.append(
                {
                    "benchmark": update.benchmark,
                    "config_key": update.config_key,
                    "language": language,
                    "sku": update.sku,
                    "axes": axes_tuple,
                    "prev_us": prev_us,
                    "new_us": metric.gpu_time_us,
                    "pct": pct,
                    "n_runs": update.n_runs,
                }
            )
    return rows


def _baseline_regression_errors(
    diffs: Sequence[dict],
    max_regression_pct: float,
) -> List[str]:
    errors: List[str] = []
    for item in diffs:
        pct = item["pct"]
        if pct is None or pct <= max_regression_pct:
            continue
        label = (
            f"{item['sku']} {item['config_key']} [{item['language']}] "
            f"({format_axes(item['axes'])})"
        )
        errors.append(
            f"{label}: {pct:+.2f}% "
            f"({item['prev_us']:.2f} -> {item['new_us']:.2f} us)"
        )
    return errors


def _format_regression_error(
    regressions: Sequence[str],
    max_regression_pct: float,
) -> str:
    shown = list(regressions[:80])
    omitted = len(regressions) - len(shown)
    message = (
        f"refusing to import same-key baseline regressions > "
        f"{max_regression_pct:.2f}%:\n  " + "\n  ".join(shown)
    )
    if omitted > 0:
        message += f"\n  ... {omitted} more row(s) omitted."
    message += (
        "\nUse --allow-regressions only for an intentional, reviewed baseline reset."
    )
    return message


def _print_plan(
    updates_by_path: Dict[Path, List[BaselineUpdate]],
    operators: Optional[Set[str]],
) -> None:
    print("--- update plan (dry run) ---")
    print(f"Operators: {sorted(operators) if operators is not None else '(all)'}")
    for path, updates in sorted(updates_by_path.items()):
        skus = sorted({update.sku for update in updates})
        print(f"  {path}: {len(updates)} case/SKU update(s), SKU(s) {skus}")
    print("--- end plan; re-run without --dry-run to write. ---")


def _print_summary(diffs: List[dict], operators: Optional[Set[str]]) -> None:
    if not diffs:
        return
    ops_label = sorted(operators) if operators is not None else "(all)"
    print()
    print(f"=== summary (operators: {ops_label}) ===")
    diffs_sorted = sorted(
        diffs,
        key=lambda item: (
            item["sku"],
            item["benchmark"],
            item["config_key"],
            item["language"],
        ),
    )
    for item in diffs_sorted[:80]:
        label = (
            f"{item['sku']} {item['config_key']} [{item['language']}] "
            f"({format_axes(item['axes'])})"
        )
        if item["prev_us"] is None:
            print(f"  NEW       {item['new_us']:>9.2f} us  {label}")
        else:
            print(
                f"  {item['pct']:+7.2f}%  {item['prev_us']:>9.2f} -> "
                f"{item['new_us']:>9.2f} us  {label}"
            )
    if len(diffs_sorted) > 80:
        print(f"  ... {len(diffs_sorted) - 80} more row(s) omitted.")


def _format_markdown_summary(diffs: List[dict], operators: Optional[Set[str]]) -> str:
    lines: List[str] = []
    ops_label = ", ".join(sorted(operators)) if operators is not None else "(all)"
    lines.append(f"# Baseline update -- operators: {ops_label}")
    lines.append("")
    lines.append("| SKU | Delta | Before (us) | After (us) | n_runs | Benchmark |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for item in diffs:
        axes_str = format_axes(item["axes"])
        label = f"`{item['config_key']}` [{item['language']}] ({axes_str})"
        if item["prev_us"] is None:
            delta = "**NEW**"
            before = "-"
        else:
            delta = f"{item['pct']:+.2f}%"
            before = f"{item['prev_us']:.2f}"
        lines.append(
            f"| {item['sku']} | {delta} | {before} | {item['new_us']:.2f} | "
            f"{item['n_runs']} | {label} |"
        )
    lines.append("")
    return "\n".join(lines)


def cmd_update(args: argparse.Namespace) -> int:
    config_dir = args.config_dir
    operators_dir = config_dir / "operators"
    sku_map_path = args.sku_map or (config_dir / "sku_map.json")
    if args.max_regression_pct < 0:
        raise BaselineError("--max-regression-pct must be >= 0")

    jsons = _collect_jsons(args.from_paths)
    index = load_config_index(operators_dir)
    all_updates = baseline_updates_from_jsons(
        jsons, index=index, sku_map_path=sku_map_path
    )

    bench_names = {update.benchmark for update in all_updates.values()}
    if args.from_diff is not None:
        ref = _resolve_diff_ref(args.from_diff)
        operators = _ops_from_git_diff(ref, bench_names)
        print(f"--from-diff: resolved {ref}...HEAD -> {sorted(operators)}")
    else:
        operators = _normalize_operators(args.operators)
        if operators is not None:
            missing = sorted(operators - bench_names)
            if missing:
                raise BaselineError(
                    f"--operator names not found in any input JSON: {missing}. "
                    f"Known benchmark names across inputs: {sorted(bench_names)}"
                )

    updates = _updates_for_operators(all_updates, operators)
    if not updates:
        print("WARN: no matching rows to update.")
        return 0

    quality_errors = validate_baseline_updates_quality(updates.values())
    if quality_errors:
        raise BaselineError(
            "refusing to import quality-violating JSON baselines:\n  "
            + "\n  ".join(quality_errors)
        )

    updates_by_path: Dict[Path, List[BaselineUpdate]] = {}
    for update in updates.values():
        path = index.require(update.config_key).path
        updates_by_path.setdefault(path, []).append(update)

    diffs = _diff_summary(index.docs_by_path, list(updates.values()))
    if not args.allow_regressions:
        regressions = _baseline_regression_errors(diffs, args.max_regression_pct)
        if regressions:
            raise BaselineError(
                _format_regression_error(regressions, args.max_regression_pct)
            )

    new_payloads: Dict[Path, dict] = {}
    for path, path_updates in updates_by_path.items():
        doc = index.docs_by_path[path]
        new_payloads[path] = apply_updates_to_document(doc, path_updates)

    stems = sku_stems(sku_map_path, strict=True)
    validation_errors: List[str] = []
    for path, payload in new_payloads.items():
        doc = OperatorConfigFile(
            path=path,
            benchmark=payload["benchmark"],
            configs=payload["configs"],
            raw=payload,
            new_shape=True,
        )
        validation_errors.extend(validate_baselines_in_document(doc, index, stems))
    if validation_errors:
        raise BaselineError(
            "refusing to write invalid JSON baselines:\n  "
            + "\n  ".join(validation_errors)
        )

    if args.dry_run:
        _print_plan(updates_by_path, operators)
        return 0

    for path, payload in sorted(new_payloads.items()):
        before = path.read_text() if path.exists() else ""
        after = __import__("json").dumps(payload, indent=4) + "\n"
        if before != after:
            write_json(path, payload)
            print(f"updated {path}")
        else:
            print(f"unchanged {path}")

    _print_summary(diffs, operators)
    if args.write_summary:
        args.write_summary.write_text(_format_markdown_summary(diffs, operators))
        print(f"wrote markdown summary to {args.write_summary}")

    return 0


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Update committed operator baselines from benchmark JSON artifacts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  bench/_internal/update_baseline.py --from bench_output.json --operator resize --dry-run
  bench/_internal/update_baseline.py --from bench_output.json --operator resize \\
    --write-summary baseline-update.md
  bench/_internal/update_baseline.py --from bench_diagnostics/ --from-diff origin/main

Output:
  Updates matching files under <config-dir>/operators/. Use --dry-run first.

Exit status:
  0  Validation passed and the requested update or dry run completed.
  1  Input, quality, parity, or regression validation failed.
  2  Command syntax is invalid.
""",
    )
    p.add_argument(
        "--from",
        dest="from_paths",
        action="append",
        metavar="JSON_OR_DIR",
        type=Path,
        required=True,
        help=(
            "Read a JSON artifact or all *.json files directly under a "
            "directory. Repeatable."
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

    ops_group = p.add_mutually_exclusive_group()
    ops_group.add_argument(
        "--operator",
        action="append",
        default=None,
        dest="operators",
        metavar="NAME",
        help="Update only these operator names; repeatable or comma-separated.",
    )
    ops_group.add_argument(
        "--from-diff",
        nargs="?",
        const="",
        default=None,
        metavar="REF",
        help=(
            "Update only operators changed between REF and HEAD. With no REF, "
            "use origin/main, then main."
        ),
    )

    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and show what would change without writing files.",
    )
    p.add_argument(
        "--write-summary",
        metavar="MD",
        type=Path,
        default=None,
        dest="write_summary",
        help=(
            "After updating, write a Markdown before/after table to MD "
            "(not written with --dry-run)."
        ),
    )
    p.add_argument(
        "--max-regression-pct",
        type=float,
        default=DEFAULT_MAX_BASELINE_REGRESSION_PCT,
        help=(
            "Maximum allowed same-key slowdown when replacing an existing "
            f"baseline metric. Default: {DEFAULT_MAX_BASELINE_REGRESSION_PCT}."
        ),
    )
    p.add_argument(
        "--allow-regressions",
        action="store_true",
        help=(
            "Permit slower replacement values after review. Schema, quality, "
            "and parity checks still apply."
        ),
    )
    p.add_argument("--allow-missing", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    try:
        return cmd_update(args)
    except BaselineError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    sys.exit(main())
