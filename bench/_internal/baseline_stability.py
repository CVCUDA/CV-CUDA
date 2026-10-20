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

"""Check repeated-run stability before importing benchmark baselines.

Stability gates imports; optimistic and under-sampled labels are audit signals.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import statistics
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _internal.baselines import (  # noqa: E402
    DEFAULT_CONFIG_DIR,
    LANGUAGES,
    BaselineError,
    BaselineUpdate,
    BenchmarkArtifact,
    ConfigIndex,
    baseline_updates_from_payload,
    collect_artifact_jsons,
    committed_baseline_metrics,
    committed_baseline_time_us,
    expected_case_keys_for_entry,
    load_artifacts,
    load_config_index,
    SkuBaselinePolicy,
    explicit_routing_stems,
    load_sku_map,
    load_sku_policies,
    normalize_operator_names,
    resolve_sku_from_metadata,
)
from _internal.quality import (  # noqa: E402
    BASELINE_MIN_CLUSTER_SIZE,
    BASELINE_RUNS_PER_SKU,
    MAX_BASELINE_RANGE_PCT,
    MIN_BASELINE_BIAS_PCT,
    exceeds_limit,
)

MAX_REPORT_ROWS = 100
REVISION_RE = re.compile(
    r"^bench-burnin-\d+-(?:cu\d+-)?[A-Za-z0-9_.-]+-\d+-([0-9a-fA-F]{8})\.json$"
)
METADATA_REVISION_RE = re.compile(r"^[0-9a-fA-F]{8,40}$")
RowKey = Tuple[str, str, str, str]
Sample = Tuple[float, BaselineUpdate, str]


def normalize_revision(value: str) -> Optional[str]:
    """Return a normalized revision while retaining its available provenance."""
    value = value.strip()
    return value.lower() if METADATA_REVISION_RE.fullmatch(value) else None


@dataclass(frozen=True)
class StabilityRow:
    config_key: str
    benchmark: str
    case_key: str
    sku: str
    language: str
    values_us: List[float]
    sources: List[str]
    minimum_us: float
    median_us: float
    mean_us: float
    maximum_us: float
    range_pct: float
    committed_us: Optional[float]
    committed_n_runs: Optional[int]
    candidate_delta_pct: Optional[float]
    stable: bool
    optimistic: bool
    under_sampled: bool


@dataclass
class StabilityReport:
    artifacts: int
    expected_runs: int
    minimum_cluster_runs: int
    artifact_counts_by_sku: Dict[str, int]
    provenance_by_sku: Dict[str, Dict[str, List[str]]]
    refs: List[str]
    revisions: List[str]
    rows: List[StabilityRow] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)
    row_issues: List[str] = field(default_factory=list)
    row_issue_keys: List[RowKey] = field(default_factory=list)

    @property
    def unstable_rows(self) -> List[StabilityRow]:
        return [row for row in self.rows if not row.stable]

    @property
    def optimistic_rows(self) -> List[StabilityRow]:
        return [row for row in self.rows if row.optimistic]

    @property
    def under_sampled_rows(self) -> List[StabilityRow]:
        """Return one row per atomic case/SKU baseline entry."""
        rows: Dict[Tuple[str, str, str], StabilityRow] = {}
        for row in self.rows:
            if row.under_sampled:
                rows.setdefault((row.config_key, row.case_key, row.sku), row)
        return list(rows.values())


def committed_metric(
    update: BaselineUpdate, language: str, index: ConfigIndex
) -> Tuple[Optional[float], Optional[int]]:
    entry = index.require(update.config_key).entry
    metrics = committed_baseline_metrics(entry, update.case_key, update.sku)
    n_runs = metrics.get("n_runs")
    return (
        committed_baseline_time_us(entry, update.case_key, update.sku, language),
        n_runs if isinstance(n_runs, int) and not isinstance(n_runs, bool) else None,
    )


def expected_rows(
    index: ConfigIndex,
    skus: Set[str],
    operators: Optional[Set[str]] = None,
    policies: Optional[Mapping[str, SkuBaselinePolicy]] = None,
) -> Set[RowKey]:
    rows: Set[RowKey] = set()
    for ref in index.refs_by_key.values():
        if operators is not None and ref.benchmark not in operators:
            continue
        tier = ref.entry.get("tier")
        for case_key in expected_case_keys_for_entry(ref.key, ref.entry):
            for sku in skus:
                policy = (policies or {}).get(sku)
                # A SKU restricted to some tiers owes rows for those alone, and only
                # for the languages it measures. Without this a Jetson wave reads as
                # missing thousands of advanced-tier and C++ rows it will never have.
                if policy is not None:
                    if isinstance(tier, str) and tier not in policy.tiers:
                        continue
                    languages = policy.languages
                else:
                    languages = LANGUAGES
                for language in languages:
                    rows.add((ref.key, case_key, sku, language))
    return rows


def artifact_revisions(
    path: Path, metadata: dict
) -> Tuple[bool, Optional[str], Optional[str]]:
    git_sha = str(metadata.get("git_sha") or "")
    has_metadata_revision = bool(git_sha.strip())
    metadata_revision = normalize_revision(git_sha)
    match = REVISION_RE.fullmatch(path.name)
    filename_revision = match.group(1).lower() if match else None
    return has_metadata_revision, metadata_revision, filename_revision


def row_stats(
    key: RowKey,
    samples: Sequence[Sample],
    index: ConfigIndex,
    *,
    minimum_cluster_runs: int = BASELINE_MIN_CLUSTER_SIZE,
    input_complete: bool = True,
) -> StabilityRow:
    config_key, case_key, sku, language = key
    values = [sample[0] for sample in samples]
    mean = statistics.fmean(values)
    median = statistics.median(values)
    range_pct = _range_pct(values)
    committed, committed_runs = committed_metric(samples[0][1], language, index)
    delta = (
        (mean / committed - 1.0) * 100.0
        if committed is not None and committed > 0
        else None
    )
    stable = (
        input_complete
        and len(values) >= minimum_cluster_runs
        and not exceeds_limit(range_pct, MAX_BASELINE_RANGE_PCT)
    )
    optimistic = bool(
        len(values) >= minimum_cluster_runs
        and committed is not None
        and committed < min(values)
        and delta is not None
        and exceeds_limit(delta, MIN_BASELINE_BIAS_PCT)
    )
    return StabilityRow(
        config_key=config_key,
        benchmark=samples[0][1].benchmark,
        case_key=case_key,
        sku=sku,
        language=language,
        values_us=values,
        sources=[sample[2] for sample in samples],
        minimum_us=min(values),
        median_us=median,
        mean_us=mean,
        maximum_us=max(values),
        range_pct=range_pct,
        committed_us=committed,
        committed_n_runs=committed_runs,
        candidate_delta_pct=delta,
        stable=stable,
        optimistic=optimistic,
        under_sampled=(
            committed is not None
            and (committed_runs is None or committed_runs < BASELINE_MIN_CLUSTER_SIZE)
        ),
    )


def _range_pct(values: Sequence[float]) -> float:
    return (max(values) - min(values)) / statistics.median(values) * 100.0


def _stable_cluster_sources(
    samples_by_language: Dict[str, Sequence[Sample]],
    *,
    expected_runs: int,
    minimum_cluster_runs: int,
) -> Optional[Set[str]]:
    """Select the largest source-aligned cluster meeting both timing-range limits.

    Equal-sized clusters are ranked by the sum of their language means, then
    by source names for deterministic ties. Noise, parity, and comparison with
    committed baselines are separate import checks.
    """
    by_language = {
        language: {sample[2]: sample[0] for sample in samples}
        for language, samples in samples_by_language.items()
    }
    if set(by_language) != LANGUAGES:
        return None
    source_sets = [set(values) for values in by_language.values()]
    if any(len(sources) != expected_runs for sources in source_sets):
        return None
    if any(sources != source_sets[0] for sources in source_sets[1:]):
        return None

    sources = sorted(source_sets[0])

    def stable(candidate: Sequence[str]) -> bool:
        return all(
            not exceeds_limit(
                _range_pct([by_source[source] for source in candidate]),
                MAX_BASELINE_RANGE_PCT,
            )
            for by_source in by_language.values()
        )

    def score(candidate: Tuple[str, ...]) -> Tuple[float, Tuple[str, ...]]:
        total_mean = sum(
            statistics.fmean([by_language[language][source] for source in candidate])
            for language in sorted(LANGUAGES)
        )
        return total_mean, candidate

    if stable(sources):
        return set(sources)
    if minimum_cluster_runs == expected_runs:
        return None

    for size in range(expected_runs - 1, minimum_cluster_runs - 1, -1):
        candidates = [
            candidate
            for candidate in itertools.combinations(sources, size)
            if stable(candidate)
        ]
        if candidates:
            return set(max(candidates, key=score))
    return None


def analyse_artifacts(
    artifacts: Sequence[BenchmarkArtifact],
    *,
    index: ConfigIndex,
    sku_map_path: Path,
    operators: Optional[Set[str]] = None,
    clustered: bool = False,
    sku_scope: Optional[Set[str]] = None,
) -> StabilityReport:
    expected_runs = BASELINE_RUNS_PER_SKU if clustered else BASELINE_MIN_CLUSTER_SIZE
    minimum_cluster_runs = BASELINE_MIN_CLUSTER_SIZE
    if operators is not None:
        unknown = sorted(operators - set(index.refs_by_operator))
        if unknown:
            raise BaselineError(f"unknown --operator value(s): {unknown}")
    sku_map = load_sku_map(sku_map_path, strict=True)
    # Explicit-routing stems (Jetson) are absent from the hardware-routed map by
    # design; they are still SKUs that owe baselines, so they belong in the cohort.
    explicit_skus = explicit_routing_stems(sku_map_path, strict=True)
    skus = set(sku_map.values()) | explicit_skus
    if sku_scope is not None:
        # A scoped regen -- jetson-burnin is the standing example -- measures one SKU
        # deliberately. Narrow the cohort to what was asked for rather than relaxing
        # the completeness rule: every SKU still in scope owes its full run count, so
        # an unscoped wave cannot be completed with a convenient subset.
        unknown = sorted(sku_scope - skus)
        if unknown:
            raise BaselineError(f"unknown --sku value(s): {unknown}")
        skus = set(sku_scope)
    grouped: Dict[RowKey, List[Sample]] = {}
    counts = {sku: 0 for sku in sorted(skus)}
    provenance = {
        sku: {"nodes": set(), "drivers": set(), "vbios": set()} for sku in skus
    }
    issues: List[str] = []
    row_issues: List[str] = []
    refs: Set[str] = set()
    revisions: Set[str] = set()
    digests: Dict[str, str] = {}
    validated_case_keys: Set[str] = set()

    required_metadata = (
        "git_ref",
        "run_context",
        "cuda_major",
        "gpu_name",
        "power_cap_w",
        "locked_sm_clock_mhz",
        "vbios_version",
        "driver_version",
        "k8s_node_name",
    )
    # An explicit-routing SKU is a named board, not a pod holding a discovered GPU:
    # nvidia-smi reports no power cap, locked clock or vBIOS there, and the run is not
    # a Kubernetes node. Requiring those would reject every such wave for lacking
    # values the platform never produces. Provenance that does exist -- the ref, the
    # run context, the CUDA major -- is still required, so an unlabelled wave is
    # still rejected.
    explicit_required_metadata = ("git_ref", "run_context", "cuda_major")
    for artifact in artifacts:
        canonical = json.dumps(
            artifact.baselines, sort_keys=True, separators=(",", ":")
        )
        digest = hashlib.sha256(canonical.encode()).hexdigest()
        if digest in digests:
            issues.append(f"{artifact.path}: duplicates artifact {digests[digest]}")
        else:
            digests[digest] = str(artifact.path)

        metadata = artifact.run_metadata
        artifact_skus = {
            sku
            for case_payload in artifact.baselines.values()
            if isinstance(case_payload, dict)
            for sku in case_payload
        }
        is_explicit = len(artifact_skus) == 1 and artifact_skus <= explicit_skus
        missing = [
            key
            for key in (
                explicit_required_metadata if is_explicit else required_metadata
            )
            if not metadata.get(key)
        ]
        if missing:
            issues.append(f"{artifact.path.name}: missing metadata {missing}")
        if not is_explicit and metadata.get("sm_clock_device_verified") is not True:
            # The device-code clock probe is a CUDA-GPU mechanism; an Orin locks its
            # clocks through nvpmodel/jetson_clocks and cannot report through it.
            issues.append(f"{artifact.path.name}: SM clock was not device-verified")
        if metadata.get("git_ref"):
            refs.add(str(metadata["git_ref"]))
        has_metadata_revision, metadata_revision, filename_revision = (
            artifact_revisions(artifact.path, metadata)
        )
        if has_metadata_revision and metadata_revision is None:
            issues.append(
                f"{artifact.path.name}: metadata git_sha is not an 8-40 digit "
                "hex revision"
            )
        if (
            metadata_revision
            and filename_revision
            and not metadata_revision.startswith(filename_revision)
        ):
            issues.append(
                f"{artifact.path.name}: metadata revision {metadata_revision} does not "
                f"match filename revision {filename_revision}"
            )
        revision = metadata_revision if has_metadata_revision else filename_revision
        if revision:
            revisions.add(revision)
        else:
            issues.append(f"{artifact.path.name}: missing revision identity")

        updates = baseline_updates_from_payload(
            artifact.baselines,
            index=index,
            sku_stem_set=skus,
            source=artifact.path,
            validated_case_keys=validated_case_keys,
        )
        payload_skus = {update.sku for update in updates}
        selected_updates = [
            update
            for update in updates
            if operators is None or update.benchmark in operators
        ]
        claimed_sku = resolve_sku_from_metadata(metadata, sku_map)
        if claimed_sku is None and len(payload_skus) == 1:
            # An explicit-routing SKU is named by the run, not discovered, so there
            # is no device metadata to confirm it against. Accept the declared stem
            # when it is one we know; anything else still falls through as a
            # mismatch, so an unrouted payload cannot invent a SKU.
            declared = next(iter(payload_skus))
            if declared in explicit_skus:
                claimed_sku = declared
        if len(payload_skus) != 1 or claimed_sku not in payload_skus:
            issues.append(
                f"{artifact.path.name}: payload SKU {sorted(payload_skus)} does not "
                f"match metadata SKU {claimed_sku!r}"
            )
        else:
            counts[claimed_sku] += 1
            for category, metadata_key in (
                ("nodes", "k8s_node_name"),
                ("drivers", "driver_version"),
                ("vbios", "vbios_version"),
            ):
                value = metadata.get(metadata_key)
                if value:
                    provenance[claimed_sku][category].add(str(value))

        non_raw = sorted(
            {update.n_runs for update in selected_updates if update.n_runs != 1}
        )
        if non_raw:
            issues.append(
                f"{artifact.path.name}: stability analysis requires raw per-run "
                f"artifacts with n_runs=1, found {non_raw}"
            )

        for update in selected_updates:
            for language, metric in update.language_metrics.items():
                key = (update.config_key, update.case_key, update.sku, language)
                grouped.setdefault(key, []).append(
                    (
                        metric.gpu_time_us,
                        update,
                        str(artifact.path.resolve()),
                    )
                )

    for sku, count in counts.items():
        if count != expected_runs:
            suffix = (
                "; remove superseded artifacts before analysis"
                if count > expected_runs
                else ""
            )
            issues.append(
                f"{sku}: expected {expected_runs} artifacts exactly, found "
                f"{count}{suffix}"
            )
    full_revisions = {revision for revision in revisions if len(revision) == 40}
    if len(full_revisions) == 1:
        full_revision = next(iter(full_revisions))
        if all(full_revision.startswith(revision) for revision in revisions):
            revisions = {full_revision}
    if len(revisions) != 1:
        issues.append(f"expected one revision identity, found {sorted(revisions)}")
    if len(refs) != 1:
        issues.append(f"expected one git ref, found {sorted(refs)}")

    sku_policies = load_sku_policies(sku_map_path, strict=True)
    missing_rows = expected_rows(index, skus, operators, sku_policies) - set(grouped)
    if missing_rows:
        missing_entries = sorted({key[:3] for key in missing_rows})
        examples = ", ".join(
            f"{case_key} [{sku}]" for _, case_key, sku in missing_entries[:3]
        )
        row_issues.append(
            f"missing declared case/SKU entries: {len(missing_entries)}; "
            f"first: {examples}"
        )

    selected_samples = dict(grouped)
    complete_entries: Set[Tuple[str, str, str]] = set()
    for entry_key in sorted({key[:3] for key in grouped}):
        per_language = {
            language: grouped.get((*entry_key, language), []) for language in LANGUAGES
        }
        source_sets = [
            {sample[2] for sample in samples} for samples in per_language.values()
        ]
        if all(len(sources) == expected_runs for sources in source_sets) and all(
            sources == source_sets[0] for sources in source_sets[1:]
        ):
            complete_entries.add(entry_key)
        selected_sources = _stable_cluster_sources(
            per_language,
            expected_runs=expected_runs,
            minimum_cluster_runs=minimum_cluster_runs,
        )
        if selected_sources is None:
            continue
        for language, samples in per_language.items():
            selected_samples[(*entry_key, language)] = [
                sample for sample in samples if sample[2] in selected_sources
            ]

    rows = [
        row_stats(
            key,
            selected_samples[key],
            index,
            minimum_cluster_runs=minimum_cluster_runs,
            input_complete=key[:3] in complete_entries,
        )
        for key in sorted(grouped)
    ]
    incomplete_rows = {
        key for key, samples in grouped.items() if len(samples) != expected_runs
    }
    if incomplete_rows:
        incomplete_entries = {key[:3] for key in incomplete_rows}
        row_issues.append(
            f"case/SKU entries without {expected_runs} raw runs: "
            f"{len(incomplete_entries)}"
        )

    return StabilityReport(
        artifacts=len(artifacts),
        expected_runs=expected_runs,
        minimum_cluster_runs=minimum_cluster_runs,
        artifact_counts_by_sku=counts,
        provenance_by_sku={
            sku: {name: sorted(values) for name, values in details.items()}
            for sku, details in sorted(provenance.items())
        },
        refs=sorted(refs),
        revisions=sorted(revisions),
        rows=rows,
        issues=issues,
        row_issues=row_issues,
        row_issue_keys=sorted(missing_rows | incomplete_rows),
    )


def analyse(
    paths: Sequence[Path],
    *,
    config_dir: Path = DEFAULT_CONFIG_DIR,
    sku_map_path: Optional[Path] = None,
    operators: Optional[Set[str]] = None,
    clustered: bool = False,
    sku_scope: Optional[Set[str]] = None,
) -> StabilityReport:
    return analyse_artifacts(
        load_artifacts(paths),
        index=load_config_index(config_dir / "operators"),
        sku_map_path=sku_map_path or (config_dir / "sku_map.json"),
        operators=operators,
        clustered=clustered,
        sku_scope=sku_scope,
    )


def format_rows(rows: Sequence[StabilityRow], *, label: str) -> List[str]:
    lines = [
        "| Old | Min | Median | Mean | Max | Runs (us) | Range | Delta | SKU | Lang | Case |",
        "|---:|---:|---:|---:|---:|---|---:|---:|---|---|---|",
    ]
    for row in rows[:MAX_REPORT_ROWS]:
        old = f"{row.committed_us:.3f}" if row.committed_us is not None else "new"
        delta = (
            f"{row.candidate_delta_pct:+.2f}%"
            if row.candidate_delta_pct is not None
            else "new"
        )
        lines.append(
            f"| {old} | {row.minimum_us:.3f} | {row.median_us:.3f} | "
            f"{row.mean_us:.3f} | {row.maximum_us:.3f} | "
            f"{', '.join(f'{value:.3f}' for value in row.values_us)} | "
            f"{row.range_pct:.2f}% | {delta} | {row.sku} | {row.language} | "
            f"`{row.case_key}` |"
        )
    if len(rows) > MAX_REPORT_ROWS:
        lines.append(f"\n_{len(rows) - MAX_REPORT_ROWS} more {label} rows omitted._")
    return lines


def format_under_sampled_rows(rows: Sequence[StabilityRow]) -> List[str]:
    lines = [
        "| n_runs | SKU | Config | Case |",
        "|---:|---|---|---|",
    ]
    for row in rows[:MAX_REPORT_ROWS]:
        n_runs = row.committed_n_runs if row.committed_n_runs is not None else "missing"
        lines.append(
            f"| {n_runs} | {row.sku} | `{row.config_key}` | `{row.case_key}` |"
        )
    if len(rows) > MAX_REPORT_ROWS:
        lines.append(
            f"\n_{len(rows) - MAX_REPORT_ROWS} more under-sampled rows omitted._"
        )
    return lines


def format_report(report: StabilityReport) -> str:
    unstable = sorted(report.unstable_rows, key=lambda row: -row.range_pct)
    optimistic = sorted(
        report.optimistic_rows, key=lambda row: -(row.candidate_delta_pct or 0.0)
    )
    under_sampled = sorted(
        report.under_sampled_rows,
        key=lambda row: (row.benchmark, row.case_key, row.sku, row.language),
    )
    lines = [
        "# Baseline stability report",
        "",
        f"- artifacts: {report.artifacts}",
        f"- accepted cluster: at least {report.minimum_cluster_runs} of "
        f"{report.expected_runs} runs",
        f"- rows: {len(report.rows)}",
        f"- unstable rows: {len(unstable)}",
        f"- under-sampled committed rows: {len(under_sampled)}",
        f"- optimistic committed baselines: {len(optimistic)}",
        f"- input issues: {len(report.issues)}",
        f"- coverage issues: {len(report.row_issues)}",
        f"- git ref: {', '.join(report.refs) or '(missing)'}",
        f"- revision: {', '.join(report.revisions) or '(missing)'}",
        "",
        "| SKU | Artifacts | Nodes | Drivers | vBIOS |",
        "|---|---:|---|---|---|",
    ]
    for sku, count in sorted(report.artifact_counts_by_sku.items()):
        provenance = report.provenance_by_sku[sku]
        nodes = provenance["nodes"]
        lines.append(
            f"| {sku} | {count} | {len(nodes)}: {', '.join(nodes) or '-'} | "
            f"{', '.join(provenance['drivers']) or '-'} | "
            f"{', '.join(provenance['vbios']) or '-'} |"
        )
    stable_python_entries = {
        (row.config_key, row.case_key, row.sku)
        for row in report.rows
        if row.language == "python" and row.stable
    }
    accepted = Counter(
        (row.sku, len(row.sources))
        for row in report.rows
        if row.language == "cpp"
        and row.stable
        and (row.config_key, row.case_key, row.sku) in stable_python_entries
    )
    if accepted:
        lines += [
            "",
            "## Stable selections",
            "",
            "These selections meet the timing-range limit; import quality and "
            "regression checks still apply.",
            "",
            "| SKU | Selected runs | Case/SKU entries |",
            "|---|---:|---:|",
        ]
        lines.extend(
            f"| {sku} | {runs} of {report.expected_runs} | {count} |"
            for (sku, runs), count in sorted(accepted.items())
        )
    if report.issues:
        lines += ["", "## Input issues", ""]
        lines.extend(f"- {issue}" for issue in report.issues)
    if report.row_issues:
        lines += ["", "## Coverage issues", ""]
        lines.extend(f"- {issue}" for issue in report.row_issues)
    lines += ["", "## Optimistic committed baselines", ""]
    lines += format_rows(optimistic, label="optimistic") if optimistic else ["None."]
    lines += ["", "## Under-sampled committed rows", ""]
    lines += format_under_sampled_rows(under_sampled) if under_sampled else ["None."]
    lines += ["", "## Unstable rows", ""]
    lines += format_rows(unstable, label="unstable") if unstable else ["None."]
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check repeated reference baseline runs before import.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  bench/_internal/baseline_stability.py --from artifacts/ --operator resize \\
    --clustered --markdown stability.md --json-out stability.json

Exit status:
  0  Inputs are valid and every selected row is stable.
  1  Inputs are invalid, coverage is incomplete, or a selected row is unstable.
  2  Command syntax is invalid.
""",
    )
    parser.add_argument(
        "--from",
        dest="from_paths",
        action="append",
        required=True,
        metavar="JSON_OR_DIR",
        type=Path,
        help="Read a JSON artifact or all *.json files in a directory; repeatable.",
    )
    parser.add_argument(
        "--config-dir",
        metavar="DIR",
        type=Path,
        default=DEFAULT_CONFIG_DIR,
        help="Benchmark config directory (default: bench/config).",
    )
    parser.add_argument(
        "--sku-map",
        metavar="JSON",
        type=Path,
        default=None,
        help="SKU map to use (default: <config-dir>/sku_map.json).",
    )
    parser.add_argument(
        "--operator",
        action="append",
        default=None,
        metavar="NAME",
        help="Check only these operator names; repeatable or comma-separated.",
    )
    parser.add_argument(
        "--clustered",
        action="store_true",
        help=(
            f"Require {BASELINE_RUNS_PER_SKU} runs per SKU and select the largest "
            f"cluster of at least {BASELINE_MIN_CLUSTER_SIZE} matching runs whose "
            f"C++ and Python ranges are each <= {MAX_BASELINE_RANGE_PCT:g}%% "
            "of their median. Without this flag, all five runs must be stable."
        ),
    )
    parser.add_argument(
        "--sku",
        action="append",
        metavar="STEM",
        help=(
            "Restrict the cohort to these SKU stems; repeatable or comma-separated. "
            "Use for a scoped regen that measures one SKU on purpose (jetson-burnin); "
            "every SKU still in scope owes its full run count."
        ),
    )
    parser.add_argument(
        "--markdown",
        metavar="MD",
        type=Path,
        help="Write the human-readable report to MD.",
    )
    parser.add_argument(
        "--json-out",
        metavar="JSON",
        type=Path,
        help="Write the machine-readable report to JSON.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        report = analyse(
            collect_artifact_jsons(args.from_paths),
            config_dir=args.config_dir,
            sku_map_path=args.sku_map,
            operators=normalize_operator_names(args.operator),
            clustered=args.clustered,
            sku_scope=normalize_operator_names(args.sku),
        )
        markdown = format_report(report)
        print(markdown)
        for path, payload in (
            (args.markdown, markdown),
            (args.json_out, json.dumps(asdict(report), indent=2) + "\n"),
        ):
            if path:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(payload)
        return 1 if report.issues or report.row_issues or report.unstable_rows else 0
    except BaselineError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
