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
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

# Support direct source-tree invocation from any working directory.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _internal.axes import format_axes
from _internal import baseline_stability
from _internal.baselines import (
    DEFAULT_CONFIG_DIR,
    DEFAULT_MAX_BASELINE_REGRESSION_PCT,
    LANGUAGES,
    BaselineError,
    BaselineUpdate,
    BenchmarkArtifact,
    ConfigIndex,
    OperatorConfigFile,
    apply_updates_to_document,
    baseline_updates_from_artifacts,
    collect_artifact_jsons,
    committed_baseline_time_us,
    load_config_index,
    load_artifacts,
    load_sku_policies,
    merge_baseline_updates,
    normalize_operator_names,
    parse_case_key,
    validate_baselines_in_document,
    validate_baseline_updates_quality,
    write_json,
)
from _internal.quality import (
    BASELINE_REFRESH_INCOMPLETE_FIELD,
    BASELINE_MIN_CLUSTER_SIZE,
    BASELINE_RUNS_PER_SKU,
    MAX_BASELINE_RANGE_PCT,
    MIN_BASELINE_BIAS_PCT,
    exceeds_limit,
)

_OP_FILE_RE = re.compile(r"^src/cvcuda/priv/Op([A-Za-z0-9_]+)\.(?:cu|cpp|hpp|h)$")
_MAX_CONSOLE_ROWS = 80


def _row_label(item: dict) -> str:
    return (
        f"{item['sku']} {item['config_key']} [{item['language']}] "
        f"({format_axes(item['axes'])})"
    )


def _diff_sort_key(item: dict) -> Tuple[str, str, str, str, str]:
    return (
        item["sku"],
        item["benchmark"],
        item["config_key"],
        item["case_key"],
        item["language"],
    )


def _revision_arg(value: str) -> str:
    revision = baseline_stability.normalize_revision(value)
    if revision is None or len(revision) != 40:
        raise argparse.ArgumentTypeError(
            "--expect-revision must be a 40 digit hex revision"
        )
    return revision


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


def _stable_updates(
    updates: Dict[Tuple[str, str, str], BaselineUpdate],
    report: baseline_stability.StabilityReport,
    *,
    artifacts: Optional[Sequence[BenchmarkArtifact]] = None,
    index: Optional[ConfigIndex] = None,
    sku_map_path: Optional[Path] = None,
) -> Dict[Tuple[str, str, str], BaselineUpdate]:
    """Aggregate stable selections; quality and regression gates run afterward."""
    if report.issues:
        raise BaselineError(
            "refusing stable-only import due to invalid stability inputs:\n  "
            + "\n  ".join(report.issues)
        )
    stable_rows = {
        (row.config_key, row.case_key, row.sku, row.language)
        for row in report.rows
        if row.stable
    }
    stable = {
        key: update
        for key, update in updates.items()
        if all((*key, language) in stable_rows for language in LANGUAGES)
    }
    if report.minimum_cluster_runs == report.expected_runs:
        return stable

    if artifacts is None or index is None or sku_map_path is None:
        raise BaselineError("clustered stable-only import requires raw artifacts")
    return _updates_from_selected_sources(
        stable, report, artifacts, index, sku_map_path
    )


def _updates_from_selected_sources(
    stable: Dict[Tuple[str, str, str], BaselineUpdate],
    report: baseline_stability.StabilityReport,
    artifacts: Sequence[BenchmarkArtifact],
    index: ConfigIndex,
    sku_map_path: Path,
) -> Dict[Tuple[str, str, str], BaselineUpdate]:
    """Rebuild each aggregate from the selected runs, preserving language pairing."""
    sources_by_key: Dict[Tuple[str, str, str], Set[str]] = {}
    for row in report.rows:
        key = (row.config_key, row.case_key, row.sku)
        if row.stable:
            sources = set(row.sources)
            if key in sources_by_key:
                sources_by_key[key] &= sources
            else:
                sources_by_key[key] = sources

    selected: List[BaselineUpdate] = []
    for artifact in artifacts:
        source = str(artifact.path.resolve())
        artifact_updates = baseline_updates_from_artifacts(
            [artifact], index=index, sku_map_path=sku_map_path
        )
        for key, update in artifact_updates.items():
            if key in stable and source in sources_by_key.get(key, set()):
                selected.append(update)
    return merge_baseline_updates(selected)


def _diff_summary(
    index: ConfigIndex,
    updates: Sequence[BaselineUpdate],
) -> List[dict]:
    rows: List[dict] = []
    for update in updates:
        entry = index.require(update.config_key).entry
        _, axes_tuple = parse_case_key(update.case_key)
        for language, metric in update.language_metrics.items():
            prev_us = committed_baseline_time_us(
                entry,
                update.case_key,
                update.sku,
                language,
            )
            pct = (
                (metric.gpu_time_us / prev_us - 1.0) * 100.0
                if prev_us is not None and prev_us > 0
                else None
            )
            rows.append(
                {
                    "benchmark": update.benchmark,
                    "config_key": update.config_key,
                    "case_key": update.case_key,
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


def _regression_findings(
    diffs: Sequence[dict],
    max_regression_pct: float,
) -> Tuple[Set[Tuple[str, str, str]], List[str]]:
    withheld: Set[Tuple[str, str, str]] = set()
    errors: List[str] = []
    for item in diffs:
        pct = item["pct"]
        if pct is None or not exceeds_limit(pct, max_regression_pct):
            continue
        withheld.add((item["config_key"], item["case_key"], item["sku"]))
        errors.append(
            f"{_row_label(item)}: {pct:+.2f}% "
            f"({item['prev_us']:.2f} -> {item['new_us']:.2f} us)"
        )
    return withheld, errors


def _withheld_rows(diffs: Sequence[dict], *, kind: str, reason: str) -> List[dict]:
    return [
        {**item, "withheld_kind": kind, "withheld_reason": reason} for item in diffs
    ]


def _format_regression_error(
    regressions: Sequence[str],
    max_regression_pct: float,
) -> str:
    shown = list(regressions[:_MAX_CONSOLE_ROWS])
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
    partial_paths: Set[Path],
) -> None:
    print("--- update plan (dry run) ---")
    print(f"Operators: {sorted(operators) if operators is not None else '(all)'}")
    for path, updates in sorted(updates_by_path.items()):
        skus = sorted({update.sku for update in updates})
        print(f"  {path}: {len(updates)} case/SKU update(s), SKU(s) {skus}")
    for path in sorted(partial_paths - updates_by_path.keys()):
        print(f"  {path}: partial-refresh marker only")
    print("--- end plan; re-run without --dry-run to write. ---")


def _print_withheld(diffs: Sequence[dict]) -> None:
    if not diffs:
        return
    print("--- withheld entries ---")
    for item in diffs[:_MAX_CONSOLE_ROWS]:
        print(f"  {item['withheld_reason']}: {_row_label(item)}")
    if len(diffs) > _MAX_CONSOLE_ROWS:
        print(f"  ... {len(diffs) - _MAX_CONSOLE_ROWS} more row(s) omitted.")
    if any(item["withheld_kind"] == "regression" for item in diffs):
        print(
            "--- regression withholding is a temporary investigation snapshot; "
            "resolve every slowdown before treating the refresh as final. ---"
        )


def _print_summary(diffs: List[dict], operators: Optional[Set[str]]) -> None:
    if not diffs:
        return
    ops_label = sorted(operators) if operators is not None else "(all)"
    print()
    print(f"=== summary (operators: {ops_label}) ===")
    diffs_sorted = sorted(diffs, key=_diff_sort_key)
    for item in diffs_sorted[:_MAX_CONSOLE_ROWS]:
        label = _row_label(item)
        if item["prev_us"] is None:
            print(f"  NEW       {item['new_us']:>9.2f} us  {label}")
        elif item["pct"] is None:
            print(
                f"  N/A       {item['prev_us']:>9.2f} -> "
                f"{item['new_us']:>9.2f} us  {label}"
            )
        else:
            print(
                f"  {item['pct']:+7.2f}%  {item['prev_us']:>9.2f} -> "
                f"{item['new_us']:>9.2f} us  {label}"
            )
    if len(diffs_sorted) > _MAX_CONSOLE_ROWS:
        print(f"  ... {len(diffs_sorted) - _MAX_CONSOLE_ROWS} more row(s) omitted.")


def _format_markdown_summary(
    diffs: List[dict],
    operators: Optional[Set[str]],
    withheld_diffs: Sequence[dict] = (),
    coverage_issues: Sequence[str] = (),
    revision: Optional[str] = None,
) -> str:
    lines: List[str] = []
    ops_label = ", ".join(sorted(operators)) if operators is not None else "(all)"
    lines.append(f"# Baseline update -- operators: {ops_label}")
    lines.append("")
    if revision is not None:
        lines.append(f"- measured revision: `{revision}`")
        lines.append("")
    lines.append("| SKU | Delta | Before (us) | After (us) | n_runs | Benchmark |")
    lines.append("|---|---:|---:|---:|---:|---|")
    diffs_sorted = sorted(diffs, key=_diff_sort_key)
    for item in diffs_sorted[:_MAX_CONSOLE_ROWS]:
        axes_str = format_axes(item["axes"])
        label = f"`{item['config_key']}` [{item['language']}] ({axes_str})"
        if item["prev_us"] is None:
            delta = "**NEW**"
            before = "-"
        else:
            delta = f"{item['pct']:+.2f}%" if item["pct"] is not None else "N/A"
            before = f"{item['prev_us']:.2f}"
        lines.append(
            f"| {item['sku']} | {delta} | {before} | {item['new_us']:.2f} | "
            f"{item['n_runs']} | {label} |"
        )
    lines.append("")
    if len(diffs_sorted) > _MAX_CONSOLE_ROWS:
        lines.append(
            f"_{len(diffs_sorted) - _MAX_CONSOLE_ROWS} more updated row(s) omitted._"
        )
        lines.append("")
    if coverage_issues:
        lines.append("## Coverage issues")
        lines.append("")
        lines.extend(f"- {issue}" for issue in coverage_issues)
        lines.append("")
    if withheld_diffs:
        lines.append("## Withheld entries")
        lines.append("")
        lines.append("| Reason | SKU | Delta | Candidate (us) | Benchmark |")
        lines.append("|---|---|---:|---:|---|")
        for item in withheld_diffs:
            axes_str = format_axes(item["axes"])
            label = f"`{item['config_key']}` [{item['language']}] ({axes_str})"
            delta = f"{item['pct']:+.2f}%" if item["pct"] is not None else "N/A"
            lines.append(
                f"| {item['withheld_reason']} | {item['sku']} | {delta} | "
                f"{item['new_us']:.2f} | {label} |"
            )
        lines.append("")
        if any(item["withheld_kind"] == "regression" for item in withheld_diffs):
            lines.append(
                "Regression withholding is a temporary investigation snapshot; "
                "resolve every slowdown before treating the refresh as final."
            )
            lines.append("")
    return "\n".join(lines)


def cmd_update(args: argparse.Namespace) -> int:
    config_dir = args.config_dir
    operators_dir = config_dir / "operators"
    sku_map_path = args.sku_map or (config_dir / "sku_map.json")
    if args.max_regression_pct is None:
        args.max_regression_pct = (
            MIN_BASELINE_BIAS_PCT
            if args.stable_only
            else DEFAULT_MAX_BASELINE_REGRESSION_PCT
        )
    if args.max_regression_pct < 0:
        raise BaselineError("--max-regression-pct must be >= 0")
    if (
        (args.stable_only or args.withhold_regressions)
        and not args.dry_run
        and args.write_summary is None
    ):
        raise BaselineError(
            "baseline withholding requires --write-summary unless --dry-run is used"
        )

    jsons = collect_artifact_jsons(args.from_paths)
    index = load_config_index(operators_dir)
    artifacts = load_artifacts(jsons)
    all_updates = baseline_updates_from_artifacts(
        artifacts, index=index, sku_map_path=sku_map_path
    )
    policies = load_sku_policies(sku_map_path, strict=True)

    bench_names = {update.benchmark for update in all_updates.values()}
    if args.from_diff is not None:
        ref = _resolve_diff_ref(args.from_diff)
        operators = _ops_from_git_diff(ref, bench_names)
        print(f"--from-diff: resolved {ref}...HEAD -> {sorted(operators)}")
    else:
        operators = normalize_operator_names(args.operators)
        if operators is not None:
            missing = sorted(operators - bench_names)
            if missing:
                raise BaselineError(
                    f"--operator names not found in any input JSON: {missing}. "
                    f"Known benchmark names across inputs: {sorted(bench_names)}"
                )
        elif args.stable_only:
            missing = sorted(set(index.refs_by_operator) - bench_names)
            if missing:
                raise BaselineError(
                    "stable-only artifacts do not cover every configured operator: "
                    f"{missing}. Pass --operator for a scoped wave."
                )

    updates = _updates_for_operators(all_updates, operators)
    withheld_diffs: List[dict] = []
    coverage_issues: List[str] = []
    coverage_issue_keys: List[baseline_stability.RowKey] = []
    wave_revision: Optional[str] = None
    if args.stable_only:
        before_updates = dict(updates)
        stable_input_count = len(updates)
        report = baseline_stability.analyse_artifacts(
            artifacts,
            index=index,
            sku_map_path=sku_map_path,
            operators=operators,
            clustered=args.clustered,
            sku_scope=normalize_operator_names(args.sku),
        )
        updates = _stable_updates(
            updates,
            report,
            artifacts=artifacts,
            index=index,
            sku_map_path=sku_map_path,
        )
        expected_revision = args.expect_revision
        if report.revisions != [expected_revision]:
            raise BaselineError(
                f"stability wave revision {report.revisions or '(missing)'} does not "
                f"match --expect-revision {expected_revision}"
            )
        wave_revision = expected_revision
        coverage_issues = report.row_issues
        coverage_issue_keys = report.row_issue_keys
        for issue in coverage_issues:
            print(f"--stable-only coverage: {issue}")
        stable_withheld = [
            update for key, update in before_updates.items() if key not in updates
        ]
        stable_withheld_diffs = _diff_summary(index, stable_withheld)
        missing_committed = sorted(
            {
                _row_label(item)
                for item in stable_withheld_diffs
                if item["prev_us"] is None
            }
        )
        if missing_committed:
            raise BaselineError(
                "refusing to withhold unstable or incomplete entries without a "
                "committed baseline; new rows have no prior value to retain and "
                "must be remeasured until stable:\n  " + "\n  ".join(missing_committed)
            )
        withheld_diffs.extend(
            _withheld_rows(
                stable_withheld_diffs,
                kind="unstable",
                reason="unstable or incomplete wave",
            )
        )
        print(
            f"--stable-only: withholding {stable_input_count - len(updates)} "
            f"of {stable_input_count} entries"
        )

    diffs = _diff_summary(index, list(updates.values()))
    if args.withhold_regressions:
        withheld, regressions = _regression_findings(diffs, args.max_regression_pct)
        regression_input_count = len(updates)
        updates = {
            key: update for key, update in updates.items() if key not in withheld
        }
        withheld_diffs.extend(
            _withheld_rows(
                [
                    item
                    for item in diffs
                    if (item["config_key"], item["case_key"], item["sku"]) in withheld
                ],
                kind="regression",
                reason=f"candidate regression > {args.max_regression_pct:.2f}%",
            )
        )
        print(
            f"--withhold-regressions: withholding "
            f"{regression_input_count - len(updates)} "
            f"entries with {len(regressions)} metric regressions > "
            f"{args.max_regression_pct:.2f}%"
        )
        for regression in regressions:
            print(f"  {regression}")
        diffs = [
            item
            for item in diffs
            if (item["config_key"], item["case_key"], item["sku"]) not in withheld
        ]

    all_withheld = not updates and bool(withheld_diffs)
    if all_withheld:
        print("WARN: all matching rows were withheld.")
    elif not updates:
        print("WARN: no input row matched the selected operators.")
        return 1

    quality_errors = validate_baseline_updates_quality(
        updates.values(), policies=policies
    )
    if quality_errors:
        raise BaselineError(
            "refusing to import quality-violating JSON baselines:\n  "
            + "\n  ".join(quality_errors)
        )

    updates_by_path: Dict[Path, List[BaselineUpdate]] = {}
    for update in updates.values():
        path = index.require(update.config_key).path
        updates_by_path.setdefault(path, []).append(update)

    partial_paths = {index.require(item["config_key"]).path for item in withheld_diffs}
    partial_paths.update(index.require(key[0]).path for key in coverage_issue_keys)

    if not args.withhold_regressions and not args.allow_regressions:
        _, regressions = _regression_findings(diffs, args.max_regression_pct)
        if regressions:
            raise BaselineError(
                _format_regression_error(regressions, args.max_regression_pct)
            )

    new_payloads: Dict[Path, dict] = {}
    for path in updates_by_path.keys() | partial_paths:
        doc = index.docs_by_path[path]
        payload = apply_updates_to_document(doc, updates_by_path.get(path, []))
        # Only a guarded stable-only import may clear an existing partial marker.
        if args.stable_only:
            if path in partial_paths:
                payload[BASELINE_REFRESH_INCOMPLETE_FIELD] = True
            else:
                payload.pop(BASELINE_REFRESH_INCOMPLETE_FIELD, None)
        new_payloads[path] = payload

    stems = set(policies)
    validation_errors: List[str] = []
    for path, payload in sorted(new_payloads.items()):
        # Validating marker-only documents would surface pre-existing baseline
        # gaps that this import did not introduce.
        if path not in updates_by_path:
            continue
        doc = OperatorConfigFile(
            path=path,
            benchmark=payload["benchmark"],
            configs=payload["configs"],
            raw=payload,
            new_shape=True,
        )
        validation_errors.extend(
            validate_baselines_in_document(doc, index, stems, sku_policies=policies)
        )
    if validation_errors:
        raise BaselineError(
            "refusing to write invalid JSON baselines:\n  "
            + "\n  ".join(validation_errors)
        )

    if args.dry_run:
        _print_plan(updates_by_path, operators, partial_paths)
        _print_withheld(withheld_diffs)
        return 1 if all_withheld else 0

    summary_temp: Optional[Path] = None
    if args.write_summary:
        if args.write_summary.exists() and not args.write_summary.is_file():
            raise BaselineError(
                f"markdown summary path is not a file: {args.write_summary}"
            )
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=args.write_summary.parent,
                prefix=f".{args.write_summary.name}.",
                suffix=".tmp",
                delete=False,
            ) as summary:
                summary_temp = Path(summary.name)
                summary.write(
                    _format_markdown_summary(
                        diffs,
                        operators,
                        withheld_diffs,
                        coverage_issues,
                        wave_revision,
                    )
                )
        except Exception as exc:
            if summary_temp is not None:
                summary_temp.unlink(missing_ok=True)
            if not isinstance(exc, OSError):
                raise
            raise BaselineError(
                f"could not write markdown summary to {args.write_summary}: {exc}"
            ) from exc

    try:
        for path, payload in sorted(new_payloads.items()):
            current_text = path.read_text() if path.exists() else ""
            after = json.dumps(payload, indent=4) + "\n"
            if current_text != after:
                write_json(path, payload)
                print(f"updated {path}")
            else:
                print(f"unchanged {path}")
    except Exception:
        if summary_temp is not None:
            summary_temp.unlink(missing_ok=True)
        raise

    if summary_temp is not None:
        try:
            summary_temp.replace(args.write_summary)
        except OSError as exc:
            summary_temp.unlink(missing_ok=True)
            raise BaselineError(
                f"could not publish markdown summary to {args.write_summary}: {exc}"
            ) from exc

    _print_summary(diffs, operators)
    _print_withheld(withheld_diffs)
    if args.write_summary:
        print(f"wrote markdown summary to {args.write_summary}")

    return 1 if all_withheld else 0


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Update committed operator baselines from benchmark JSON artifacts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Single-artifact development check:
  bench/_internal/update_baseline.py --from bench_output.json --operator resize --dry-run
  # Reference-SKU guarded import:
  bench/_internal/update_baseline.py --from bench_diagnostics/ --operator resize \\
    --stable-only --clustered --expect-revision <measured-sha> \\
    --write-summary baseline-update.md

Output:
  Updates matching files under <config-dir>/operators/. Use --dry-run first.

Exit status:
  0  Validation passed and the requested update or dry run completed.
  1  Validation failed, every selected entry was withheld, or no entry matched.
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
        "--stable-only",
        action="store_true",
        help=(
            "Require a valid repeated-run stability wave and import only entries "
            "whose C++ and Python timings both have a range <= "
            f"{MAX_BASELINE_RANGE_PCT:g}%%."
        ),
    )
    p.add_argument(
        "--sku",
        action="append",
        metavar="STEM",
        help=(
            "Restrict the cohort to these SKU stems; repeatable or comma-separated. "
            "Use for a scoped regen that measures one SKU on purpose (jetson-burnin); "
            "every SKU still in scope owes its full run count."
        ),
    )
    p.add_argument(
        "--clustered",
        action="store_true",
        help=(
            f"With --stable-only, require {BASELINE_RUNS_PER_SKU} runs per SKU "
            f"and use the largest cluster of at least {BASELINE_MIN_CLUSTER_SIZE} "
            "matching runs stable in both languages. Quality and regression "
            "checks still apply. Without this flag, all five runs must be stable."
        ),
    )
    p.add_argument(
        "--expect-revision",
        type=_revision_arg,
        default=None,
        metavar="SHA",
        help=(
            "Required with --stable-only. Refuse artifacts not measured at "
            "this revision."
        ),
    )
    regression_group = p.add_mutually_exclusive_group()
    regression_group.add_argument(
        "--withhold-regressions",
        action="store_true",
        help=(
            "With --stable-only, retain existing entries when any language "
            "regresses by more than --max-regression-pct. Requiring a stable "
            "wave prevents noise-driven improvements from being accepted. "
            "This creates a temporary investigation snapshot; resolve every "
            "withheld entry before treating the refresh as final."
        ),
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
        default=None,
        help=(
            "Maximum allowed same-key slowdown when replacing an existing "
            "baseline metric. Default: "
            f"{MIN_BASELINE_BIAS_PCT:g} with --stable-only, otherwise "
            f"{DEFAULT_MAX_BASELINE_REGRESSION_PCT:g}."
        ),
    )
    regression_group.add_argument(
        "--allow-regressions",
        action="store_true",
        help=(
            "Permit slower replacement values after review. Schema, quality, "
            "and parity checks still apply."
        ),
    )
    args = p.parse_args(argv)
    if args.expect_revision is not None and not args.stable_only:
        p.error("--expect-revision requires --stable-only")
    if args.stable_only and args.expect_revision is None:
        p.error("--stable-only requires --expect-revision")
    if args.withhold_regressions and not args.stable_only:
        p.error("--withhold-regressions requires --stable-only")
    if args.clustered and not args.stable_only:
        p.error("--clustered requires --stable-only")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    try:
        return cmd_update(args)
    except BaselineError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    sys.exit(main())
