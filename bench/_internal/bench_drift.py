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

"""Report benchmark rows whose level has shifted, without changing anything.

The committed-baseline gate compares a run against a fixed +/-10% band. Measured
spread on the reference SKUs is far tighter than that -- around 0.6% across a
day and 1.5% within a stable stretch -- so a row can slide several percent and
still pass. This reads a window of nightly artifacts and reports rows whose
recent level sits outside their own historical spread, which is a far finer
instrument than the fixed band.

It is deliberately read-only. It writes no baseline and fails no gate; a shift
here is a prompt to look, not a verdict. That keeps a false positive cheap and
makes it impossible for this to corrupt the reference it measures.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Support direct source-tree invocation from any working directory.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _internal.baselines import (  # noqa: E402
    DEFAULT_CONFIG_DIR,
    BaselineError,
    baseline_update_samples_from_artifacts,
    load_artifacts,
    load_config_index,
)

# A stretch needs this many observations before its level means anything.
DEFAULT_MIN_SAMPLES = 8

# How far the recent level must sit from the earlier level, measured in the
# earlier stretch's own spread. Benchmarks are not normally distributed, so this
# is a robust-scale multiple rather than a true confidence level.
DEFAULT_SIGMA = 4.0

# A shift smaller than this is not worth anyone's attention no matter how tight
# the spread, because it is inside the run-to-run reality of a shared pool.
DEFAULT_MIN_SHIFT_PCT = 2.0

# Shifts at or above this are almost certainly a landed change rather than
# drift, and are reported separately so the quiet ones stay visible.
DEFAULT_LARGE_SHIFT_PCT = 8.0

RowKey = Tuple[str, str, str]


@dataclass(frozen=True)
class DriftFinding:
    """One row whose recent level sits away from where it used to sit."""

    config_key: str
    case_key: str
    sku: str
    baseline_us: float
    recent_us: float
    shift_pct: float
    spread_pct: float
    sigmas: float
    n_baseline: int
    n_recent: int

    @property
    def slower(self) -> bool:
        return self.recent_us > self.baseline_us

    @property
    def label(self) -> str:
        return f"{self.sku} {self.case_key}"


@dataclass
class DriftReport:
    findings: List[DriftFinding] = field(default_factory=list)
    rows_examined: int = 0
    rows_skipped: int = 0

    @property
    def drifted(self) -> List[DriftFinding]:
        return [f for f in self.findings if abs(f.shift_pct) < DEFAULT_LARGE_SHIFT_PCT]

    @property
    def stepped(self) -> List[DriftFinding]:
        return [f for f in self.findings if abs(f.shift_pct) >= DEFAULT_LARGE_SHIFT_PCT]


def robust_spread_pct(values: Sequence[float], centre: float) -> float:
    """Spread as a percentage of the centre, from the median absolute deviation.

    MAD rather than a standard deviation because a single outlying run -- a
    noisy node, a retried job -- should not widen the band enough to hide a
    real shift.
    """
    if centre <= 0 or len(values) < 2:
        return 0.0
    mad = statistics.median(abs(v - centre) for v in values)
    # 1.4826 rescales MAD to a standard deviation for normal data; benchmark
    # noise is not normal, but the constant keeps the sigma count comparable to
    # the intuition people already have.
    return mad * 1.4826 / centre * 100.0


def analyse_row(
    values: Sequence[float],
    *,
    split: int,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    sigma: float = DEFAULT_SIGMA,
    min_shift_pct: float = DEFAULT_MIN_SHIFT_PCT,
) -> Optional[Tuple[float, float, float, float, float]]:
    """Compare the tail of a series against its head.

    Returns ``(baseline, recent, shift_pct, spread_pct, sigmas)`` when the tail
    has moved far enough to report, else None.
    """
    head, tail = list(values[:split]), list(values[split:])
    if len(head) < min_samples or len(tail) < min_samples:
        return None

    baseline = statistics.median(head)
    recent = statistics.median(tail)
    if baseline <= 0:
        return None

    shift_pct = (recent / baseline - 1.0) * 100.0
    if abs(shift_pct) < min_shift_pct:
        return None

    spread_pct = robust_spread_pct(head, baseline)
    # A perfectly repeatable row would divide by zero; treat its spread as the
    # reporting floor so it cannot manufacture an unbounded sigma count.
    sigmas = abs(shift_pct) / max(spread_pct, min_shift_pct / sigma)
    if sigmas < sigma:
        return None
    return baseline, recent, shift_pct, spread_pct, sigmas


def analyse(
    samples_by_key: Dict[RowKey, List],
    *,
    recent_fraction: float = 0.3,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    sigma: float = DEFAULT_SIGMA,
    min_shift_pct: float = DEFAULT_MIN_SHIFT_PCT,
) -> DriftReport:
    """Split every row into an earlier and a recent stretch and compare them."""
    report = DriftReport()
    for key in sorted(samples_by_key):
        updates = samples_by_key[key]
        values = [
            u.language_metrics["cpp"].gpu_time_us
            for u in updates
            if "cpp" in u.language_metrics
        ]
        report.rows_examined += 1
        if len(values) < 2 * min_samples:
            report.rows_skipped += 1
            continue
        split = max(min_samples, int(len(values) * (1.0 - recent_fraction)))
        outcome = analyse_row(
            values,
            split=split,
            min_samples=min_samples,
            sigma=sigma,
            min_shift_pct=min_shift_pct,
        )
        if outcome is None:
            continue
        baseline, recent, shift_pct, spread_pct, sigmas = outcome
        config_key, case_key, sku = key
        report.findings.append(
            DriftFinding(
                config_key=config_key,
                case_key=case_key,
                sku=sku,
                baseline_us=baseline,
                recent_us=recent,
                shift_pct=shift_pct,
                spread_pct=spread_pct,
                sigmas=sigmas,
                n_baseline=split,
                n_recent=len(values) - split,
            )
        )
    report.findings.sort(key=lambda f: -abs(f.sigmas))
    return report


def format_report(report: DriftReport) -> str:
    lines = [
        "# Benchmark drift report",
        "",
        f"- rows examined: {report.rows_examined}",
        f"- rows with too little history: {report.rows_skipped}",
        f"- rows that moved: {len(report.findings)} "
        f"({len(report.stepped)} large, {len(report.drifted)} quiet)",
        "",
    ]
    if not report.findings:
        lines += ["No row moved beyond its own spread.", ""]
        return "\n".join(lines)

    for title, group in (
        ("Quiet drift (inside the committed-baseline band)", report.drifted),
        ("Large shifts (likely a landed change)", report.stepped),
    ):
        lines += [f"## {title} — {len(group)}", ""]
        if not group:
            lines += ["_none_", ""]
            continue
        lines += [
            "| Shift | Was (us) | Now (us) | Spread | Sigmas | SKU | Case |",
            "|---:|---:|---:|---:|---:|---|---|",
        ]
        for f in group:
            lines.append(
                f"| {f.shift_pct:+.2f}% | {f.baseline_us:.1f} | {f.recent_us:.1f} "
                f"| {f.spread_pct:.2f}% | {f.sigmas:.1f} | {f.sku} | `{f.case_key}` |"
            )
        lines.append("")
    return "\n".join(lines)


def validate_thresholds(args: argparse.Namespace) -> None:
    """Reject threshold values the analysis cannot act on.

    A zero sigma or floor divides by zero when scaling a shift against a row's
    spread, and a non-positive sample count asks for the median of an empty
    stretch. Both are better refused up front than surfaced as a traceback part
    way through a wave.
    """
    if args.sigma <= 0:
        raise BaselineError("--sigma must be > 0")
    if args.min_shift_pct <= 0:
        raise BaselineError("--min-shift-pct must be > 0")
    if args.min_samples < 1:
        raise BaselineError("--min-samples must be >= 1")
    if not 0.0 < args.recent_fraction < 1.0:
        raise BaselineError("--recent-fraction must be between 0 and 1, exclusive")


def cmd_report(args: argparse.Namespace) -> int:
    validate_thresholds(args)

    paths: List[Path] = []
    for path in args.from_paths:
        if not path.exists():
            raise BaselineError(f"--from path does not exist: {path}")
        paths.extend(sorted(path.glob("*.json")) if path.is_dir() else [path])
    if not paths:
        raise BaselineError("no benchmark artifacts collected from --from")

    artifacts = load_artifacts(paths)
    nightly = [a for a in artifacts if a.run_metadata.get("is_nightly")]
    if not args.any_run_context:
        skipped = len(artifacts) - len(nightly)
        if skipped:
            print(f"ignoring {skipped} artifact(s) that are not nightly runs")
        artifacts = nightly
    if len(artifacts) < 2 * args.min_samples:
        raise BaselineError(
            f"need at least {2 * args.min_samples} eligible artifacts, "
            f"got {len(artifacts)}"
        )

    index = load_config_index(args.config_dir / "operators")
    # A window of past artifacts always contains rows the current config no
    # longer declares, because declared axis values change over time. Those are
    # skipped and counted; failing on them would make the report unusable on
    # any window longer than the last config edit.
    skipped_rows: List[str] = []
    samples = baseline_update_samples_from_artifacts(
        artifacts,
        index=index,
        sku_map_path=args.config_dir / "sku_map.json",
        errors_out=skipped_rows,
    )
    if skipped_rows:
        stale = {line.split(": ", 2)[-1].split("[")[0] for line in skipped_rows}
        print(
            f"skipped {len(skipped_rows)} row(s) the current config no longer "
            f"declares, across {len(stale)} config(s)"
        )
    report = analyse(
        samples,
        recent_fraction=args.recent_fraction,
        min_samples=args.min_samples,
        sigma=args.sigma,
        min_shift_pct=args.min_shift_pct,
    )

    text = format_report(report)
    print(text)
    for target, payload in (
        (args.markdown, text),
        (
            args.json_out,
            json.dumps([f.__dict__ for f in report.findings], indent=2, default=str),
        ),
    ):
        if not target:
            continue
        # A CI job routinely writes into a directory it also creates.
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(payload)
        print(f"wrote {target}")
    # Always success: this reports, it does not gate.
    return 0


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Report benchmark rows whose level has shifted.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  bench/_internal/bench_drift.py --from nightly_wave
  bench/_internal/bench_drift.py --from nightly_wave --markdown drift.md

Exit status:
  0  The report was produced. A shift is never an error here.
  1  The inputs or configuration were unusable.
  2  Command syntax is invalid.
""",
    )
    p.add_argument(
        "--from",
        dest="from_paths",
        action="append",
        required=True,
        metavar="JSON_OR_DIR",
        type=Path,
        help="A benchmark artifact or a directory of them. Repeatable.",
    )
    p.add_argument(
        "--config-dir",
        type=Path,
        default=DEFAULT_CONFIG_DIR,
        metavar="DIR",
        help="Benchmark config directory (default: bench/config).",
    )
    p.add_argument(
        "--recent-fraction",
        type=float,
        default=0.3,
        help="Share of each series treated as recent (default: 0.3).",
    )
    p.add_argument(
        "--min-samples",
        type=int,
        default=DEFAULT_MIN_SAMPLES,
        help=f"Observations each stretch needs " f"(default: {DEFAULT_MIN_SAMPLES}).",
    )
    p.add_argument(
        "--sigma",
        type=float,
        default=DEFAULT_SIGMA,
        help=f"Robust-scale multiples a shift must clear "
        f"(default: {DEFAULT_SIGMA}).",
    )
    p.add_argument(
        "--min-shift-pct",
        type=float,
        default=DEFAULT_MIN_SHIFT_PCT,
        help=f"Smallest shift worth reporting " f"(default: {DEFAULT_MIN_SHIFT_PCT}).",
    )
    p.add_argument(
        "--any-run-context",
        action="store_true",
        help="Analyse non-nightly artifacts too. For local use; a "
        "merge request run measures its own branch.",
    )
    p.add_argument(
        "--markdown",
        type=Path,
        default=None,
        metavar="MD",
        help="Also write the report to MD.",
    )
    p.add_argument(
        "--json-out",
        type=Path,
        default=None,
        metavar="JSON",
        help="Also write the findings to JSON.",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    try:
        return cmd_report(args)
    except BaselineError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
