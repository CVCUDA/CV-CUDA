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

"""Compare a run_bench.py JSON artifact with committed per-GPU baselines."""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from _internal.axes import format_axes
from _internal.baselines import (
    DEFAULT_CONFIG_DIR,
    BaselineError,
    BaselineUpdate,
    ConfigIndex,
    ConfigRef,
    baseline_updates_from_jsons,
    load_config_index,
    parse_case_key,
    validate_baselines_in_document,
    validate_case_key_for_config,
    sku_stems,
)


@dataclass
class Thresholds:
    regression: float = 0.10
    improvement: float = 0.10


@dataclass
class RowResult:
    benchmark: str
    config_key: str
    language: str
    axes: Tuple[Tuple[str, str], ...]
    base_mean_us: float
    cur_gpu_us: float
    delta: float


@dataclass
class MissingRow:
    benchmark: str
    config_key: str
    language: str
    axes: Tuple[Tuple[str, str], ...]
    reason: str = ""


@dataclass
class CompareResult:
    regressions: List[RowResult] = field(default_factory=list)
    improvements: List[RowResult] = field(default_factory=list)
    missing_in_current: List[MissingRow] = field(default_factory=list)
    new_in_current: List[MissingRow] = field(default_factory=list)
    missing_sku: List[MissingRow] = field(default_factory=list)
    matched: int = 0
    matched_abs_deltas: List[float] = field(default_factory=list)

    @property
    def any_fail(self) -> bool:
        return bool(
            self.regressions
            or self.improvements
            or self.missing_in_current
            or self.new_in_current
            or self.missing_sku
        )


def _axes_from_case_key(case_key: str) -> Tuple[Tuple[str, str], ...]:
    _, axes = parse_case_key(case_key)
    return axes


def _baseline_rows_for_sku(
    index: ConfigIndex,
    sku: str,
    tiers: Optional[set],
    operators: Optional[set[str]] = None,
) -> Dict[Tuple[str, str, str], RowResult]:
    rows: Dict[Tuple[str, str, str], RowResult] = {}
    for ref in index.refs_by_key.values():
        if operators is not None and ref.benchmark not in operators:
            continue
        if tiers is not None and ref.tier not in tiers:
            continue
        baselines = ref.entry.get("baselines", {})
        if not isinstance(baselines, dict):
            continue
        for case_key, case_payload in baselines.items():
            if not isinstance(case_payload, dict) or sku not in case_payload:
                continue
            metrics = case_payload[sku]
            axes = validate_case_key_for_config(case_key, ref)
            for language, time_field in {
                "cpp": "gpu_time_us_cpp",
                "python": "gpu_time_us_python",
            }.items():
                if time_field not in metrics:
                    continue
                key = (ref.key, case_key, language)
                rows[key] = RowResult(
                    benchmark=ref.benchmark,
                    config_key=ref.key,
                    language=language,
                    axes=axes,
                    base_mean_us=float(metrics[time_field]),
                    cur_gpu_us=0.0,
                    delta=0.0,
                )
    return rows


def _current_rows(
    updates: Sequence[BaselineUpdate],
) -> Dict[Tuple[str, str, str], RowResult]:
    rows: Dict[Tuple[str, str, str], RowResult] = {}
    duplicates: List[str] = []
    for update in updates:
        axes = _axes_from_case_key(update.case_key)
        for language, metric in update.language_metrics.items():
            key = (update.config_key, update.case_key, language)
            if key in rows:
                duplicates.append(f"duplicate current JSON metric for {key}")
            rows[key] = RowResult(
                benchmark=update.benchmark,
                config_key=update.config_key,
                language=language,
                axes=axes,
                base_mean_us=0.0,
                cur_gpu_us=metric.gpu_time_us,
                delta=0.0,
            )
    if duplicates:
        raise BaselineError("duplicate current metrics:\n  " + "\n  ".join(duplicates))
    return rows


def _case_has_other_sku(ref: ConfigRef, case_key: str, sku: str) -> bool:
    baselines = ref.entry.get("baselines", {})
    case_payload = baselines.get(case_key) if isinstance(baselines, dict) else None
    return isinstance(case_payload, dict) and case_payload and sku not in case_payload


def compare_updates(
    updates: Sequence[BaselineUpdate],
    *,
    index: ConfigIndex,
    sku: str,
    thresholds: Thresholds = Thresholds(),
    operators: Optional[set[str]] = None,
) -> CompareResult:
    result = CompareResult()
    if not updates:
        return result

    if operators is not None:
        unknown = operators - set(index.refs_by_operator)
        if unknown:
            raise BaselineError(f"unknown benchmark operator(s): {sorted(unknown)}")
        unexpected = {update.benchmark for update in updates} - operators
        if unexpected:
            raise BaselineError(
                f"current JSON contains unselected operator(s): {sorted(unexpected)}"
            )

    observed_skus = {update.sku for update in updates}
    if observed_skus != {sku}:
        raise BaselineError(
            f"current JSON contains SKU(s) {sorted(observed_skus)}, "
            f"but compare resolved {sku!r}"
        )

    current_tiers = set()
    for update in updates:
        tier = index.require(update.config_key).tier
        if tier:
            current_tiers.add(tier)
    tiers = current_tiers or None
    baseline = _baseline_rows_for_sku(index, sku, tiers, operators)
    current = _current_rows(updates)

    baseline_keys = set(baseline)
    current_keys = set(current)

    for key in sorted(baseline_keys - current_keys):
        row = baseline[key]
        result.missing_in_current.append(
            MissingRow(row.benchmark, row.config_key, row.language, row.axes)
        )

    for key in sorted(current_keys - baseline_keys):
        config_key, case_key, language = key
        ref = index.require(config_key)
        axes = _axes_from_case_key(case_key)
        row = MissingRow(ref.benchmark, config_key, language, axes)
        if _case_has_other_sku(ref, case_key, sku):
            row.reason = f"case exists but SKU {sku!r} is missing"
            result.missing_sku.append(row)
        else:
            row.reason = "case key is not present in JSON baselines"
            result.new_in_current.append(row)

    for key in sorted(baseline_keys & current_keys):
        base_row = baseline[key]
        cur_row = current[key]
        base_mean = base_row.base_mean_us
        cur_us = cur_row.cur_gpu_us
        result.matched += 1
        if base_mean <= 0:
            continue

        delta = cur_us / base_mean - 1.0
        result.matched_abs_deltas.append(abs(delta))
        row = RowResult(
            benchmark=base_row.benchmark,
            config_key=base_row.config_key,
            language=base_row.language,
            axes=base_row.axes,
            base_mean_us=base_mean,
            cur_gpu_us=cur_us,
            delta=delta,
        )
        if delta > thresholds.regression:
            result.regressions.append(row)
        elif -delta > thresholds.improvement:
            result.improvements.append(row)

    return result


def _axes_str(axes: Iterable[Tuple[str, str]]) -> str:
    return format_axes(axes)


def _fmt_row(row: RowResult) -> str:
    label = f"{row.config_key} [{row.language}] ({_axes_str(row.axes)})"
    return (
        f"{label} -- {row.delta * 100:+.2f}% "
        f"({row.base_mean_us:.2f} -> {row.cur_gpu_us:.2f} us)"
    )


def _fmt_missing(row: MissingRow) -> str:
    suffix = f" -- {row.reason}" if row.reason else ""
    return f"{row.config_key} [{row.language}] ({_axes_str(row.axes)}){suffix}"


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    mid = len(sorted_vals) // 2
    if len(sorted_vals) % 2:
        return sorted_vals[mid]
    return 0.5 * (sorted_vals[mid - 1] + sorted_vals[mid])


def _summary_stats_lines(result: CompareResult) -> List[str]:
    if not result.matched_abs_deltas:
        return []
    lines = [
        f"- all-rows |Delta|: median {_median(result.matched_abs_deltas) * 100:.2f}%, "
        f"max {max(result.matched_abs_deltas) * 100:.2f}%"
    ]
    flagged_abs = [abs(r.delta) for r in result.regressions + result.improvements]
    if flagged_abs:
        lines.append(
            f"- flagged |Delta|: min {min(flagged_abs) * 100:.2f}%, "
            f"median {_median(flagged_abs) * 100:.2f}%, "
            f"max {max(flagged_abs) * 100:.2f}%"
        )
    return lines


def format_markdown(
    result: CompareResult,
    sku: str,
    baseline_label: Path | str,
    current_path: Path,
    thresholds: Thresholds,
) -> str:
    lines: List[str] = []
    lines.append(f"# Performance regression report -- {sku}")
    lines.append("")
    lines.append(f"- baseline source: `{baseline_label}`")
    lines.append(f"- current: `{current_path}`")
    lines.append(
        f"- thresholds: regression={thresholds.regression:.3g}, "
        f"improvement={thresholds.improvement:.3g}"
    )
    lines.append(
        f"- matched: {result.matched}, "
        f"regressions: {len(result.regressions)}, "
        f"improvements: {len(result.improvements)}, "
        f"missing: {len(result.missing_in_current)}, "
        f"new: {len(result.new_in_current)}, "
        f"missing SKU: {len(result.missing_sku)}"
    )
    lines.extend(_summary_stats_lines(result))
    lines.append("")

    def _row_section(title: str, rows: List[RowResult]) -> None:
        lines.append(f"## {title} ({len(rows)})")
        if not rows:
            lines.append("_none_")
        else:
            for row in rows:
                lines.append(f"- {_fmt_row(row)}")
        lines.append("")

    def _missing_section(title: str, rows: List[MissingRow]) -> None:
        lines.append(f"## {title} ({len(rows)})")
        if not rows:
            lines.append("_none_")
        else:
            for row in rows:
                lines.append(f"- {_fmt_missing(row)}")
        lines.append("")

    _row_section("Regressions", result.regressions)
    _row_section("Unexpected improvements", result.improvements)
    _missing_section("Missing in current", result.missing_in_current)
    _missing_section("New in current", result.new_in_current)
    _missing_section("Missing SKU", result.missing_sku)
    return "\n".join(lines)


def format_junit(result: CompareResult, sku: str) -> bytes:
    failures = (
        len(result.regressions)
        + len(result.improvements)
        + len(result.missing_in_current)
        + len(result.new_in_current)
        + len(result.missing_sku)
    )
    total = (
        result.matched
        + len(result.missing_in_current)
        + len(result.new_in_current)
        + len(result.missing_sku)
    )
    suite = ET.Element(
        "testsuite",
        {
            "name": f"cvcuda.perf_regression.{sku}",
            "tests": str(total),
            "failures": str(failures),
            "errors": "0",
        },
    )

    def _case(name: str, classname: str, message: str, text: str) -> None:
        case = ET.SubElement(
            suite,
            "testcase",
            {
                "classname": classname,
                "name": name,
            },
        )
        failure = ET.SubElement(
            case, "failure", {"type": classname, "message": message}
        )
        failure.text = text

    for row in result.regressions:
        _case(
            f"{row.config_key}[{row.language}]({_axes_str(row.axes)})",
            "cvcuda.perf.regression",
            f"GPU time increased by {row.delta * 100:+.2f}% vs baseline",
            _fmt_row(row),
        )
    for row in result.improvements:
        _case(
            f"{row.config_key}[{row.language}]({_axes_str(row.axes)})",
            "cvcuda.perf.unexpected_improvement",
            f"GPU time decreased by {row.delta * 100:+.2f}% vs baseline",
            _fmt_row(row),
        )
    for category, rows in (
        ("missing_in_current", result.missing_in_current),
        ("new_in_current", result.new_in_current),
        ("missing_sku", result.missing_sku),
    ):
        for row in rows:
            _case(
                f"{row.config_key}[{row.language}]({_axes_str(row.axes)})",
                f"cvcuda.perf.{category}",
                row.reason or category,
                _fmt_missing(row),
            )

    return ET.tostring(suite, encoding="utf-8", xml_declaration=True)


def _validate_json_baselines(index: ConfigIndex, sku_map_path: Path) -> None:
    stems = sku_stems(sku_map_path, strict=True)
    errors: List[str] = []
    for doc in index.docs_by_path.values():
        errors.extend(validate_baselines_in_document(doc, index, stems))
    if errors:
        raise BaselineError("invalid JSON baseline entries:\n  " + "\n  ".join(errors))


def _resolve_current_sku(
    updates: Sequence[BaselineUpdate], cli_sku: Optional[str]
) -> Tuple[str, str]:
    observed = sorted({update.sku for update in updates})
    if not observed:
        raise BaselineError("current JSON contains no baseline metrics")
    if cli_sku:
        if cli_sku not in observed:
            raise BaselineError(
                f"--sku {cli_sku!r} is not present in current JSON SKU(s) {observed}"
            )
        return cli_sku, "--sku"
    if len(observed) != 1:
        raise BaselineError(
            f"current JSON contains multiple SKU(s) {observed}; pass --sku"
        )
    return observed[0], "current JSON SKU key"


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compare one benchmark JSON artifact with the committed operator "
            "baselines."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  compare_to_baseline.py --current bench_output.json
  compare_to_baseline.py --current bench_output.json --operator resize \\
    --markdown comparison.md --junit comparison.xml

Exit status:
  0  All expected rows matched within both thresholds.
  1  A regression, unexpected improvement, or incompatible row was found.
  2  The input artifact or benchmark configuration is invalid.
""",
    )
    p.add_argument(
        "--current",
        metavar="JSON",
        type=Path,
        required=True,
        help="JSON artifact produced by run_bench.py --output.",
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
        "--sku",
        metavar="SKU",
        type=str,
        default=None,
        help="Compare this SKU key. Omit when the artifact contains one SKU.",
    )
    p.add_argument(
        "--operator",
        type=str,
        default=None,
        help=(
            "Compare only these operator names, comma-separated (default: all). "
            "The input JSON must contain only the selected operators."
        ),
    )
    p.add_argument(
        "--regression",
        metavar="FRACTION",
        type=float,
        default=0.10,
        help=(
            "Fail when a matched result is slower by more than this fraction "
            "(default: 0.10 = 10%%)."
        ),
    )
    p.add_argument(
        "--improvement",
        metavar="FRACTION",
        type=float,
        default=0.10,
        help=(
            "Fail when a matched result is faster by more than this fraction, "
            "which can indicate a stale baseline (default: 0.10 = 10%%)."
        ),
    )
    p.add_argument(
        "--junit",
        metavar="XML",
        type=Path,
        default=None,
        help="Write a JUnit XML report to XML.",
    )
    p.add_argument(
        "--markdown",
        metavar="MD",
        type=Path,
        default=None,
        help="Write the Markdown report to MD instead of printing it.",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    sku_map_path = args.sku_map or (args.config_dir / "sku_map.json")
    operators_dir = args.config_dir / "operators"

    try:
        index = load_config_index(operators_dir)
        _validate_json_baselines(index, sku_map_path)
        operators = None
        if args.operator is not None:
            operators = {
                item.strip() for item in args.operator.split(",") if item.strip()
            }
            if not operators:
                raise BaselineError("--operator must select at least one benchmark")
        updates = list(
            baseline_updates_from_jsons(
                [args.current], index=index, sku_map_path=sku_map_path
            ).values()
        )
        sku, source = _resolve_current_sku(updates, args.sku)
        updates = [update for update in updates if update.sku == sku]
        thresholds = Thresholds(args.regression, args.improvement)
        result = compare_updates(
            updates,
            index=index,
            sku=sku,
            thresholds=thresholds,
            operators=operators,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"Resolved SKU: {sku} ({source})")
    print(
        f"matched={result.matched} regressions={len(result.regressions)} "
        f"improvements={len(result.improvements)} "
        f"missing={len(result.missing_in_current)} "
        f"new={len(result.new_in_current)} missing_sku={len(result.missing_sku)}"
    )

    report = format_markdown(result, sku, operators_dir, args.current, thresholds)
    if args.markdown:
        args.markdown.write_text(report)
    else:
        print()
        print(report)

    if args.junit:
        args.junit.write_bytes(format_junit(result, sku))

    return 1 if result.any_fail else 0


if __name__ == "__main__":
    sys.exit(main())
