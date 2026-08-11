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
"""Canonical parser, generator, and validator for optimization MR summaries.

This module owns the public, versioned Markdown contract. It intentionally has
no deployment-specific workflow policy: lifecycle consumers pass repository
evidence to ``validate_summary`` when they need numerical validation.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence


REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / "bench"
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

from _internal.baselines import (  # noqa: E402
    BaselineError,
    expected_case_keys_for_entry,
    fake_planar_pairing_issues,
    parse_case_key,
)


CHECKLIST_LABELS = (
    "Pixelwise equality to reference",
    "Memory-footprint checks",
    "Baselines updated",
    "Baseline validation",
    "Lead exhaustion",
    "Review/refactor gate",
)
SECTION_HEADINGS = (
    "Scope",
    "Impact",
    "Layout comparison",
    "Evidence checklist",
    "Top learnings",
)
START_TOKEN = "cvcuda-optimize-summary:v1"
END_MARKER = "<!-- /cvcuda-optimize-summary:v1 -->"
MAX_DESCRIPTION_BYTES = 2_000_000
MAX_BLOCK_BYTES = 400_000
MAX_OPTIMIZED_CASES = 10_000
MAX_SECONDARY_OPERATORS = 64
NA_STATS = "n/a — no matched equivalent configurations"
DEFAULT_LEARNING = "No optimization attempt has been accepted or rejected yet."
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_OPERATOR_RE = re.compile(r"^[A-Za-z][A-Za-z0-9]{0,127}$")
_START_RE = re.compile(r"^<!-- cvcuda-optimize-summary:v1 (\{.*\}) -->\s*$")
_FACTOR_RE = re.compile(r"^(\d{1,9}\.\d{2})x$")
_STATS_RE = re.compile(r"^(\d{1,9}\.\d{2})x / (\d{1,9}\.\d{2})x / (\d{1,9}\.\d{2})x$")
_DURATION_STATS_RE = re.compile(
    r"^(-?\d{1,9}\.\d{2}) µs / (-?\d{1,9}\.\d{2}) µs / " r"(-?\d{1,9}\.\d{2}) µs$"
)
_PLACEHOLDER_RE = re.compile(
    r"<[^>]+>|\b(?:todo|tbd|placeholder|add hard evidence|add one concise)\b",
    re.IGNORECASE,
)

# These exact checked-in config aliases predate unique-workload enforcement.
# Keep them readable without double-weighting summaries, but reject every new
# alias pair. Remove entries as the owning operator configs are migrated.
_LEGACY_DUPLICATE_CONFIG_ALIASES = {
    "bilateralfilter": {
        (
            "bilateralfilter_rgb_auto_fakeplanar_nchw_advanced",
            "bilateralfilter_uchar3_fakeplanar_nchw_advanced",
        ),
    },
    "brightnesscontrast": {
        (
            "brightnesscontrast_fakeplanar_nchw_float3_advanced",
            "brightnesscontrast_fakeplanar_rgb_f32_nchw_1080p_advanced",
        ),
        (
            "brightnesscontrast_fakeplanar_nchw_uchar3_advanced",
            "brightnesscontrast_fakeplanar_rgb_u8_nchw_1080p_advanced",
        ),
        (
            "brightnesscontrast_float3_varshape_advanced",
            "brightnesscontrast_rgb_f32_1080p_advanced",
        ),
        (
            "brightnesscontrast_planar_nchw_float3_varshape_advanced",
            "brightnesscontrast_planar_rgb_f32_nchw_1080p_advanced",
        ),
    },
    "centercrop": {
        (
            "centercrop_fakeplanar_nchw_float3_advanced",
            "centercrop_fakeplanar_nchw_rgb_f32_1080p_advanced",
        ),
    },
    "clahe": {
        ("clahe_clip20_advanced", "clahe_tiles8_varshape_1080p_advanced"),
    },
    "colortwist": {
        (
            "colortwist_fakeplanar_nchw_advanced",
            "colortwist_fakeplanar_nchw_rgb_u8_1080p_advanced",
        ),
        (
            "colortwist_fakeplanar_nchw_advanced",
            "colortwist_fakeplanar_nchw_rgba_u8_1080p_advanced",
        ),
    },
    "copymakeborder": {
        (
            "copymakeborder_fake_planar_rgb_u8_reflect101_1080p_advanced",
            "copymakeborder_fake_planar_uchar3_advanced",
        ),
    },
    "customcrop": {
        (
            "customcrop_fakeplanar_nchw_float3_advanced",
            "customcrop_fakeplanar_nchw_rgb_f32_full_1080p_advanced",
        ),
        (
            "customcrop_fakeplanar_nchw_rgb_u8_full_1080p_advanced",
            "customcrop_fakeplanar_nchw_uchar3_advanced",
        ),
    },
    "normalize": {
        (
            "normalize_fakeplanar_nchw_1080p_float3_advanced",
            "normalize_fakeplanar_rgb_f32_nchw_1080p_advanced",
        ),
        ("normalize_float3_advanced", "normalize_rgb_f32_1080p_advanced"),
        (
            "normalize_planar_nchw_1080p_float3_advanced",
            "normalize_planar_rgb_f32_nchw_1080p_advanced",
        ),
    },
    "pillowresize": {
        (
            "pillowresize_fakeplanar_nchw_contract_cubic_1080p_float3_advanced",
            "pillowresize_fakeplanar_nchw_contract_cubic_1080p_varshape_float3_advanced",
        ),
        (
            "pillowresize_fakeplanar_nchw_contract_cubic_1080p_float4_advanced",
            "pillowresize_fakeplanar_nchw_contract_cubic_1080p_varshape_float4_advanced",
        ),
        (
            "pillowresize_fakeplanar_nchw_contract_cubic_1080p_uchar3_advanced",
            "pillowresize_fakeplanar_nchw_contract_cubic_1080p_varshape_uchar3_advanced",
        ),
        (
            "pillowresize_fakeplanar_nchw_contract_cubic_1080p_uchar4_advanced",
            "pillowresize_fakeplanar_nchw_contract_cubic_1080p_varshape_uchar4_advanced",
        ),
        (
            "pillowresize_planar_nchw_contract_cubic_1080p_float4_advanced",
            "pillowresize_planar_nchw_contract_cubic_1080p_varshape_float4_advanced",
        ),
        (
            "pillowresize_planar_nchw_contract_cubic_1080p_uchar4_advanced",
            "pillowresize_planar_nchw_contract_cubic_1080p_varshape_uchar4_advanced",
        ),
    },
}


class SummaryError(ValueError):
    """The bounded optimization-summary contract is malformed."""


@dataclass(frozen=True)
class SummaryMetadata:
    operator: str
    state: str
    baseline_commit: str
    candidate_commit: str
    optimized_cases: tuple[str, ...]
    impact_metric: str = "cpp_time"
    secondary_operators: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "optimized_cases", tuple(self.optimized_cases))
        if isinstance(self.secondary_operators, (str, bytes)):
            raise SummaryError("metadata secondary_operators must be an array")
        object.__setattr__(self, "secondary_operators", tuple(self.secondary_operators))


@dataclass(frozen=True)
class ChecklistItem:
    label: str
    checked: bool
    evidence: str


@dataclass(frozen=True)
class ImpactRow:
    sku: str
    scope: str
    configurations: int
    minimum: Decimal
    median: Decimal
    maximum: Decimal


@dataclass(frozen=True)
class PythonOverheadImpactRow:
    sku: str
    scope: str
    configurations: int
    before: tuple[Decimal, Decimal, Decimal]
    after: tuple[Decimal, Decimal, Decimal]
    reduction: tuple[Decimal, Decimal, Decimal]


@dataclass(frozen=True)
class LayoutRow:
    sku: str
    comparison: str
    before: tuple[Decimal, Decimal, Decimal] | None
    after: tuple[Decimal, Decimal, Decimal] | None


@dataclass(frozen=True)
class OptimizationSummary:
    metadata: SummaryMetadata
    text: str
    block: str
    prefix: str
    suffix: str
    title_operator: str
    status_line: str
    warnings: tuple[str, ...]
    bottleneck: str
    profile_evidence: str
    optimized_count: int
    total_count: int
    categories: tuple[str, ...]
    impact_rows: tuple[ImpactRow | PythonOverheadImpactRow, ...]
    layout_rows: tuple[LayoutRow, ...]
    checklist: tuple[ChecklistItem, ...]
    learnings: tuple[str, ...]


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str


@dataclass(frozen=True)
class ValidationResult:
    errors: tuple[ValidationIssue, ...] = ()
    warnings: tuple[ValidationIssue, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass(frozen=True)
class _Case:
    key: str
    tier: str
    axes: tuple[tuple[str, str], ...]
    baselines: Mapping[str, Any]

    @property
    def axis_map(self) -> dict[str, str]:
        return dict(self.axes)


@dataclass(frozen=True)
class _ConfigSurface:
    benchmark: str
    cases: Mapping[str, _Case]
    skus: tuple[str, ...]


def parse_summary(description: str) -> OptimizationSummary:
    """Parse exactly one live v1 block from an untrusted MR description."""

    if not isinstance(description, str):
        raise SummaryError("MR description must be text")
    if len(description.encode("utf-8")) > MAX_DESCRIPTION_BYTES:
        raise SummaryError("MR description exceeds the parser size limit")

    lines = description.splitlines(keepends=True)
    live = _live_marker_lines(lines)
    starts = []
    for index in live:
        match = _START_RE.fullmatch(lines[index].rstrip("\r\n"))
        if match:
            starts.append((index, match))
    ends = [index for index in live if lines[index].rstrip("\r\n") == END_MARKER]
    if len(starts) != 1 or len(ends) != 1:
        raise SummaryError(
            "expected exactly one live start marker and one matching end marker"
        )
    start_index, start_match = starts[0]
    end_index = ends[0]
    if end_index <= start_index:
        raise SummaryError("optimization summary end marker precedes its start")
    if any(
        START_TOKEN in lines[index] and index not in {start_index, end_index}
        for index in live
    ):
        raise SummaryError("nested or duplicate optimization summary marker")

    prefix = "".join(lines[slice(None, start_index)])
    end_line = lines[end_index]
    end_text = end_line.rstrip("\r\n")
    end_eol = end_line[slice(len(end_text), None)]
    block = "".join(lines[slice(start_index, end_index)]) + end_text
    suffix = end_eol + "".join(lines[slice(end_index + 1, None)])
    if len(block.encode("utf-8")) > MAX_BLOCK_BYTES:
        raise SummaryError("optimization summary block exceeds the parser size limit")
    metadata = _parse_metadata(start_match.group(1))
    body = "".join(lines[slice(start_index + 1, end_index)]).strip("\r\n")
    parsed = _parse_body(body, metadata)
    return OptimizationSummary(
        metadata=metadata,
        text=description,
        block=block,
        prefix=prefix,
        suffix=suffix,
        **parsed,
    )


def validate_summary(
    summary_or_description: OptimizationSummary | str,
    *,
    baseline_config: Mapping[str, Any] | None = None,
    candidate_config: Mapping[str, Any] | None = None,
    sku_map: Mapping[str, Any] | None = None,
    expected_operator: str | None = None,
    expected_candidate_commit: str | None = None,
    evidence_results: Mapping[str, str] | None = None,
) -> ValidationResult:
    """Validate structure and, when supplied, all artifact-derived claims."""

    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []
    try:
        summary = (
            summary_or_description
            if isinstance(summary_or_description, OptimizationSummary)
            else parse_summary(summary_or_description)
        )
    except SummaryError as exc:
        return ValidationResult((ValidationIssue("parse", str(exc)),), ())

    meta = summary.metadata
    if expected_operator and meta.operator.casefold() != expected_operator.casefold():
        errors.append(
            ValidationIssue(
                "operator-mismatch",
                f"metadata operator {meta.operator!r} does not match {expected_operator!r}",
            )
        )
    if summary.title_operator != meta.operator:
        errors.append(
            ValidationIssue(
                "title-operator", "visible title does not match metadata operator"
            )
        )
    if expected_candidate_commit and meta.candidate_commit != expected_candidate_commit:
        errors.append(
            ValidationIssue(
                "candidate-commit",
                "metadata candidate_commit does not match the revision being validated",
            )
        )
    expected_status = (
        "Final reference-hardware evidence"
        if meta.state == "final"
        else "Provisional local evidence"
    )
    if expected_status not in summary.status_line:
        errors.append(
            ValidationIssue("state-status", f"status line must say {expected_status!r}")
        )
    for label, commit in (
        ("base", meta.baseline_commit),
        ("candidate", meta.candidate_commit),
    ):
        if commit[:12] not in summary.status_line:
            errors.append(
                ValidationIssue(
                    "status-commit", f"status line does not show the {label} commit"
                )
            )

    if not _profile_is_measured(summary.profile_evidence):
        issue = ValidationIssue(
            "profile-evidence",
            "primary bottleneck requires a measured profiler metric and named signal",
        )
        (errors if meta.state == "final" else warnings).append(issue)

    if summary.optimized_count != len(meta.optimized_cases):
        errors.append(
            ValidationIssue(
                "optimized-count",
                "visible optimized count does not match hidden exact case keys",
            )
        )
    _validate_visible_tables(summary, errors)
    _validate_checklist(summary, evidence_results, errors, warnings)
    _validate_learnings(summary, errors, warnings)

    supplied = [
        baseline_config is not None,
        candidate_config is not None,
        sku_map is not None,
    ]
    if any(supplied) and not all(supplied):
        errors.append(
            ValidationIssue(
                "artifact-inputs",
                "baseline config, candidate config, and SKU map must be supplied together",
            )
        )
    elif all(supplied):
        try:
            _validate_artifact_claims(
                summary,
                baseline_config or {},
                candidate_config or {},
                sku_map or {},
                errors,
                warnings,
            )
        except SummaryError as exc:
            errors.append(ValidationIssue("benchmark-surface", str(exc)))

    return ValidationResult(tuple(errors), tuple(warnings))


def generate_summary(
    metadata: SummaryMetadata,
    *,
    baseline_config: Mapping[str, Any],
    candidate_config: Mapping[str, Any],
    sku_map: Mapping[str, Any],
    bottleneck: str,
    profile_evidence: str,
    checklist: Mapping[str, Any] | Sequence[ChecklistItem] | None = None,
    learnings: Sequence[str] = (),
    prefix: str = "",
    suffix: str = "",
) -> str:
    """Render a summary, deriving every numerical field from benchmark data."""

    _validate_metadata(metadata)
    if bottleneck not in {"Memory-bound", "Compute-bound"}:
        raise SummaryError("bottleneck must be Memory-bound or Compute-bound")
    baseline = _config_surface(baseline_config, "baseline")
    candidate = _config_surface(candidate_config, "candidate")
    if candidate.benchmark.casefold() != metadata.operator.casefold():
        raise SummaryError(
            f"candidate benchmark {candidate.benchmark!r} does not match operator {metadata.operator!r}"
        )
    if baseline.benchmark.casefold() != candidate.benchmark.casefold():
        raise SummaryError("baseline and candidate benchmark identities differ")
    _assert_comparable_surfaces(baseline, candidate)
    optimized = _optimized_cases(metadata, candidate)
    references = _reference_skus(sku_map)
    skus = _complete_skus(baseline, candidate, metadata.impact_metric)
    if metadata.state == "final":
        missing = [stem for stem in references if stem not in skus]
        if missing:
            coverage = (
                "C++ timing"
                if metadata.impact_metric == "cpp_time"
                else "Python/C++ timing"
            )
            raise SummaryError(
                f"final summary lacks full before/after {coverage} coverage for reference SKU(s): "
                + ", ".join(missing)
            )
    if not skus:
        raise SummaryError("no SKU has full before/after C++ timing coverage")

    categories = _categories(optimized)
    impact_rows = _expected_impact_rows(
        baseline,
        candidate,
        optimized,
        skus,
        references,
        metadata.impact_metric,
    )
    layout_rows = _expected_layout_rows(baseline, candidate, skus, references)
    checklist_items = _normalize_checklist(checklist)
    learning_items = tuple(learnings) or (DEFAULT_LEARNING,)
    local_skus = [stem for stem in skus if stem not in references]

    metadata_payload = {
        "operator": metadata.operator,
        **(
            {"secondary_operators": list(metadata.secondary_operators)}
            if metadata.secondary_operators
            else {}
        ),
        "state": metadata.state,
        **(
            {"impact_metric": metadata.impact_metric}
            if metadata.impact_metric != "cpp_time"
            else {}
        ),
        "baseline_commit": metadata.baseline_commit,
        "candidate_commit": metadata.candidate_commit,
        "optimized_cases": list(metadata.optimized_cases),
    }
    metadata_json = json.dumps(
        metadata_payload,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    impact_explanation, impact_header, impact_separator = _impact_table_schema(
        metadata.impact_metric
    )
    status = (
        "Final reference-hardware evidence"
        if metadata.state == "final"
        else "Provisional local evidence"
    )
    lines = [
        f"<!-- {START_TOKEN} {metadata_json} -->",
        f"## {metadata.operator} optimization summary",
        "",
        f"> {status} · base `{metadata.baseline_commit[:12]}` · candidate `{metadata.candidate_commit[:12]}`",
    ]
    for stem in local_skus:
        required = (
            " and ".join(_sku_label(item) for item in references) or "reference-SKU"
        )
        lines.extend(
            [
                "",
                "> ⚠️ **Non-reference local GPU:** Results from "
                f"{_sku_label(stem)} are provisional and cannot satisfy final readiness; "
                f"final {required} statistics are still required.",
            ]
        )
    lines.extend(
        [
            "",
            f"**Primary bottleneck: {bottleneck}**",
            "",
            profile_evidence.strip(),
            "",
            "### Scope",
            "",
            f"**Configurations optimized:** {len(optimized)} / {len(candidate.cases)} total",
            "",
            "**Optimized categories:**",
            "",
            *[f"- `{category}`" for category in categories],
            "",
            "### Impact",
            "",
            impact_explanation,
            "",
            impact_header,
            impact_separator,
            *[_render_impact_row(row) for row in impact_rows],
            "",
            "### Layout comparison",
            "",
            "Timing ratio is `numerator / Planar`; values above `1.00x` mean Planar is faster.",
            "",
            "| SKU | Comparison | Before min / median / max | After min / median / max |",
            "|---|---|---:|---:|",
            *[_render_layout_row(row) for row in layout_rows],
            "",
            "### Evidence checklist",
            "",
            *[
                f"- [{'x' if item.checked else ' '}] **{item.label}** — {item.evidence}"
                for item in checklist_items
            ],
            "",
            "### Top learnings",
            "",
            *[f"- {item.strip()}" for item in learning_items],
            END_MARKER,
        ]
    )
    block = "\n".join(lines)
    rendered = _join_description(prefix, block, suffix)
    if metadata.state == "final":
        validation = validate_summary(
            rendered,
            baseline_config=baseline_config,
            candidate_config=candidate_config,
            sku_map=sku_map,
        )
        if not validation.ok:
            detail = "; ".join(
                f"{item.code}: {item.message}" for item in validation.errors[:6]
            )
            raise SummaryError("cannot render an invalid final summary: " + detail)
    return rendered


def refresh_summary(
    description: str,
    *,
    baseline_config: Mapping[str, Any],
    candidate_config: Mapping[str, Any],
    sku_map: Mapping[str, Any],
    state: str | None = None,
    candidate_commit: str | None = None,
    impact_metric: str | None = None,
    secondary_operators: Sequence[str] | None = None,
) -> str:
    """Regenerate derived fields while preserving bounded human-authored fields."""

    old = parse_summary(description)
    metadata = SummaryMetadata(
        operator=old.metadata.operator,
        state=state or old.metadata.state,
        baseline_commit=old.metadata.baseline_commit,
        candidate_commit=candidate_commit or old.metadata.candidate_commit,
        optimized_cases=old.metadata.optimized_cases,
        impact_metric=impact_metric or old.metadata.impact_metric,
        secondary_operators=(
            tuple(secondary_operators)
            if secondary_operators is not None
            else old.metadata.secondary_operators
        ),
    )
    return generate_summary(
        metadata,
        baseline_config=baseline_config,
        candidate_config=candidate_config,
        sku_map=sku_map,
        bottleneck=old.bottleneck,
        profile_evidence=old.profile_evidence,
        checklist=old.checklist,
        learnings=old.learnings,
        prefix=old.prefix,
        suffix=old.suffix,
    )


def _live_marker_lines(lines: Sequence[str]) -> list[int]:
    live: list[int] = []
    fence: str | None = None
    for index, line in enumerate(lines):
        stripped = line.lstrip()
        marker = re.match(r"^(`{3,}|~{3,})", stripped)
        if marker:
            token = marker.group(1)
            if fence is None:
                fence = token
            elif token[0] == fence[0] and len(token) >= len(fence):
                fence = None
            continue
        if fence is None and START_TOKEN in line:
            live.append(index)
    return live


def _parse_metadata(blob: str) -> SummaryMetadata:
    if len(blob.encode("utf-8")) > 300_000:
        raise SummaryError("metadata exceeds the parser size limit")
    try:
        data = json.loads(blob, object_pairs_hook=_unique_json_object)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise SummaryError(f"metadata is not valid JSON: {exc}") from exc
    required = {
        "operator",
        "state",
        "baseline_commit",
        "candidate_commit",
        "optimized_cases",
    }
    allowed = required | {"impact_metric", "secondary_operators"}
    if (
        not isinstance(data, dict)
        or not required.issubset(data)
        or not set(data).issubset(allowed)
    ):
        raise SummaryError(
            "metadata must contain exactly operator, state, baseline_commit, "
            "candidate_commit, and optimized_cases, with optional impact_metric "
            "and secondary_operators"
        )
    if not isinstance(data["optimized_cases"], list):
        raise SummaryError("metadata optimized_cases must be an array")
    if not isinstance(data.get("secondary_operators", []), list):
        raise SummaryError("metadata secondary_operators must be an array")
    metadata = SummaryMetadata(
        operator=data["operator"],
        state=data["state"],
        baseline_commit=data["baseline_commit"],
        candidate_commit=data["candidate_commit"],
        optimized_cases=tuple(data["optimized_cases"]),
        impact_metric=data.get("impact_metric", "cpp_time"),
        secondary_operators=tuple(data.get("secondary_operators", [])),
    )
    _validate_metadata(metadata)
    return metadata


def _unique_json_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise SummaryError(f"metadata JSON repeats field {key!r}")
        result[key] = value
    return result


def _validate_metadata(metadata: SummaryMetadata) -> None:
    if not isinstance(metadata.operator, str) or not _OPERATOR_RE.fullmatch(
        metadata.operator
    ):
        raise SummaryError("metadata operator is invalid")
    if not isinstance(metadata.state, str) or metadata.state not in {
        "provisional",
        "final",
    }:
        raise SummaryError("metadata state must be provisional or final")
    if not isinstance(metadata.impact_metric, str) or metadata.impact_metric not in {
        "cpp_time",
        "python_overhead",
    }:
        raise SummaryError("metadata impact_metric must be cpp_time or python_overhead")
    secondary = metadata.secondary_operators
    if len(secondary) > MAX_SECONDARY_OPERATORS:
        raise SummaryError(
            "metadata secondary_operators may contain at most "
            f"{MAX_SECONDARY_OPERATORS} names"
        )
    if any(
        not isinstance(operator, str) or not _OPERATOR_RE.fullmatch(operator)
        for operator in secondary
    ):
        raise SummaryError("metadata secondary_operators contains an invalid operator")
    normalized_secondary = [operator.casefold() for operator in secondary]
    if len(set(normalized_secondary)) != len(normalized_secondary):
        raise SummaryError("metadata secondary_operators contains duplicate operators")
    if metadata.operator.casefold() in normalized_secondary:
        raise SummaryError(
            "metadata secondary_operators must not repeat the primary operator"
        )
    for field, value in (
        ("baseline_commit", metadata.baseline_commit),
        ("candidate_commit", metadata.candidate_commit),
    ):
        if not isinstance(value, str) or not _SHA_RE.fullmatch(value):
            raise SummaryError(
                f"metadata {field} must be a full lowercase 40-character commit SHA"
            )
    cases = metadata.optimized_cases
    if not cases or len(cases) > MAX_OPTIMIZED_CASES:
        raise SummaryError(
            f"metadata optimized_cases must contain 1..{MAX_OPTIMIZED_CASES} keys"
        )
    if any(not isinstance(case, str) or not case or len(case) > 4096 for case in cases):
        raise SummaryError("metadata contains an invalid optimized case key")
    if len(set(cases)) != len(cases):
        raise SummaryError("metadata optimized_cases contains duplicate keys")


def _parse_body(body: str, metadata: SummaryMetadata) -> dict[str, Any]:
    headings = [
        (match.group(1), match.group(2), match.start(), match.end())
        for match in re.finditer(r"^(#{2,3})\s+(.+?)\s*$", body, re.MULTILINE)
    ]
    expected = [f"{metadata.operator} optimization summary", *SECTION_HEADINGS]
    actual = [text for _, text, _, _ in headings]
    if actual != expected:
        raise SummaryError(
            "visible headings must appear exactly once in canonical order: "
            + " -> ".join(expected)
        )
    if headings[0][0] != "##" or any(item[0] != "###" for item in headings[1:]):
        raise SummaryError("summary title must be ## and section headings must be ###")

    preamble = body[slice(headings[0][3], headings[1][2])].strip()
    scope = body[slice(headings[1][3], headings[2][2])].strip()
    impact = body[slice(headings[2][3], headings[3][2])].strip()
    layout = body[slice(headings[3][3], headings[4][2])].strip()
    checklist_text = body[slice(headings[4][3], headings[5][2])].strip()
    learnings_text = body[slice(headings[5][3], None)].strip()

    status_lines = [
        line.strip() for line in preamble.splitlines() if line.startswith("> ")
    ]
    ordinary_status = [
        line for line in status_lines if "Non-reference local GPU" not in line
    ]
    if len(ordinary_status) != 1:
        raise SummaryError("summary preamble must contain exactly one status line")
    bottleneck_matches = list(
        re.finditer(
            r"^\*\*Primary bottleneck: (Memory-bound|Compute-bound)\*\*$",
            preamble,
            re.MULTILINE,
        )
    )
    if len(bottleneck_matches) != 1:
        raise SummaryError("summary requires one exact Primary bottleneck line")
    bottleneck_match = bottleneck_matches[0]
    profile = preamble[slice(bottleneck_match.end(), None)].strip()
    if not profile:
        raise SummaryError("Primary bottleneck must be followed by profiler evidence")
    warnings = tuple(line for line in status_lines if "Non-reference local GPU" in line)

    count_match = re.search(
        r"^\*\*Configurations optimized:\*\* (\d{1,9}) / (\d{1,9}) total$",
        scope,
        re.MULTILINE,
    )
    if not count_match:
        raise SummaryError("Scope has no canonical Configurations optimized count")
    categories = tuple(re.findall(r"^- `([^`\r\n]+)`\s*$", scope, re.MULTILINE))
    if not categories or len(categories) != len(set(categories)):
        raise SummaryError("Scope requires unique Container Layout Type categories")
    scope_lines = [line.strip() for line in scope.splitlines() if line.strip()]
    expected_scope_lines = [
        count_match.group(0),
        "**Optimized categories:**",
        *[f"- `{category}`" for category in categories],
    ]
    if scope_lines != expected_scope_lines:
        raise SummaryError(
            "Scope may contain only the canonical count and category list"
        )

    impact_rows = _parse_impact_table(impact, metadata.impact_metric)
    layout_rows = _parse_layout_table(layout)
    checklist = _parse_checklist(checklist_text)
    learnings = tuple(
        match.group(1).strip()
        for match in re.finditer(r"^- (.+?)\s*$", learnings_text, re.MULTILINE)
    )
    if not 1 <= len(learnings) <= 5:
        raise SummaryError("Top learnings must contain one to five bullets")
    nonblank = [line for line in learnings_text.splitlines() if line.strip()]
    if len(nonblank) != len(learnings):
        raise SummaryError("Top learnings may contain only bullet lines")

    return {
        "title_operator": metadata.operator,
        "status_line": ordinary_status[0],
        "warnings": warnings,
        "bottleneck": bottleneck_match.group(1),
        "profile_evidence": profile,
        "optimized_count": int(count_match.group(1)),
        "total_count": int(count_match.group(2)),
        "categories": categories,
        "impact_rows": impact_rows,
        "layout_rows": layout_rows,
        "checklist": checklist,
        "learnings": learnings,
    }


def _impact_table_schema(impact_metric: str) -> tuple[str, str, str]:
    if impact_metric == "cpp_time":
        return (
            "Speedup is `before / after`; `2.00x` means twice as fast.",
            "| SKU | Configuration scope | Configurations | Min | Median | Max |",
            "|---|---|---:|---:|---:|---:|",
        )
    if impact_metric == "python_overhead":
        return (
            "Python overhead is `gpu_time_us_python - gpu_time_us_cpp`; "
            "reduction is `before - after`, so positive values mean less overhead.",
            "| SKU | Configuration scope | Configurations | Before min / median / max | "
            "After min / median / max | Reduction min / median / max |",
            "|---|---|---:|---:|---:|---:|",
        )
    raise SummaryError(f"unsupported impact metric {impact_metric!r}")


def _parse_impact_table(
    section: str, impact_metric: str
) -> tuple[ImpactRow | PythonOverheadImpactRow, ...]:
    explanation, header, _ = _impact_table_schema(impact_metric)
    _require_table_only_section(section, explanation)
    rows = _table_rows(section)
    expected_header = tuple(
        cell.strip() for cell in header.strip().strip("|").split("|")
    )
    if not rows or rows[0] != expected_header:
        raise SummaryError("Impact table header does not match the v1 schema")
    parsed = []
    for cells in rows[1:]:
        if len(cells) != 6 or cells[1] not in {"Optimized", "Full operator"}:
            raise SummaryError("Impact table contains a malformed row")
        try:
            if impact_metric == "cpp_time":
                parsed.append(
                    ImpactRow(
                        cells[0],
                        cells[1],
                        int(cells[2]),
                        _parse_factor(cells[3]),
                        _parse_factor(cells[4]),
                        _parse_factor(cells[5]),
                    )
                )
            else:
                parsed.append(
                    PythonOverheadImpactRow(
                        cells[0],
                        cells[1],
                        int(cells[2]),
                        _parse_duration_stats(cells[3]),
                        _parse_duration_stats(cells[4]),
                        _parse_duration_stats(cells[5]),
                    )
                )
        except ValueError as exc:
            raise SummaryError(
                f"Impact table contains a malformed value: {exc}"
            ) from exc
    if not parsed:
        raise SummaryError("Impact table must contain at least one SKU")
    return tuple(parsed)


def _parse_layout_table(section: str) -> tuple[LayoutRow, ...]:
    _require_table_only_section(
        section,
        "Timing ratio is `numerator / Planar`; values above `1.00x` mean Planar is faster.",
    )
    rows = _table_rows(section)
    expected_header = (
        "SKU",
        "Comparison",
        "Before min / median / max",
        "After min / median / max",
    )
    if not rows or rows[0] != expected_header:
        raise SummaryError(
            "Layout comparison table header does not match the v1 schema"
        )
    parsed = []
    for cells in rows[1:]:
        if len(cells) != 4 or cells[1] not in {
            "Interleaved / Planar",
            "FakePlanar / Planar",
        }:
            raise SummaryError("Layout comparison table contains a malformed row")
        before = _parse_stats(cells[2])
        after = _parse_stats(cells[3])
        if (before is None) != (after is None):
            raise SummaryError("layout n/a must appear in both before and after cells")
        parsed.append(LayoutRow(cells[0], cells[1], before, after))
    if not parsed:
        raise SummaryError("Layout comparison table must contain at least one SKU")
    return tuple(parsed)


def _table_rows(section: str) -> list[tuple[str, ...]]:
    table_lines = [
        line for line in section.splitlines() if line.strip().startswith("|")
    ]
    if len(table_lines) < 3:
        return []
    separator = tuple(
        cell.strip() for cell in table_lines[1].strip().strip("|").split("|")
    )
    if not all(re.fullmatch(r":?-{3,}:?", cell) for cell in separator):
        raise SummaryError("Markdown table separator is malformed")
    rows = []
    for line in [table_lines[0], *table_lines[2:]]:
        rows.append(tuple(cell.strip() for cell in line.strip().strip("|").split("|")))
    return rows


def _require_table_only_section(section: str, explanation: str) -> None:
    lines = [line.strip() for line in section.splitlines() if line.strip()]
    if (
        not lines
        or lines[0] != explanation
        or any(not line.startswith("|") for line in lines[1:])
    ):
        raise SummaryError(
            "table section may contain only its canonical explanation and table"
        )


def _parse_checklist(section: str) -> tuple[ChecklistItem, ...]:
    pattern = re.compile(
        r"^- \[([ xX])\] \*\*([^*\r\n]+)\*\* — (.+?)\s*$", re.MULTILINE
    )
    matches = list(pattern.finditer(section))
    if [match.group(2) for match in matches] != list(CHECKLIST_LABELS):
        raise SummaryError(
            "Evidence checklist labels must appear exactly once in canonical order"
        )
    nonblank = [line for line in section.splitlines() if line.strip()]
    if len(nonblank) != len(matches):
        raise SummaryError(
            "Evidence checklist may contain only the six canonical item lines"
        )
    return tuple(
        ChecklistItem(
            match.group(2), match.group(1).lower() == "x", match.group(3).strip()
        )
        for match in matches
    )


def _parse_factor(value: str) -> Decimal:
    match = _FACTOR_RE.fullmatch(value)
    if not match:
        raise ValueError(f"expected a factor such as 1.23x, got {value!r}")
    return Decimal(match.group(1))


def _parse_stats(value: str) -> tuple[Decimal, Decimal, Decimal] | None:
    if value == NA_STATS:
        return None
    match = _STATS_RE.fullmatch(value)
    if not match:
        raise SummaryError(f"invalid min / median / max cell {value!r}")
    return tuple(Decimal(item) for item in match.groups())  # type: ignore[return-value]


def _parse_duration_stats(value: str) -> tuple[Decimal, Decimal, Decimal]:
    match = _DURATION_STATS_RE.fullmatch(value)
    if not match:
        raise ValueError(
            f"expected microsecond min / median / max values, got {value!r}"
        )
    return tuple(Decimal(item) for item in match.groups())  # type: ignore[return-value]


def _config_surface(config: Mapping[str, Any], label: str) -> _ConfigSurface:
    if not isinstance(config, Mapping):
        raise SummaryError(f"{label} benchmark config must be an object")
    benchmark = config.get("benchmark")
    entries = config.get("configs")
    if not isinstance(benchmark, str) or not benchmark:
        raise SummaryError(f"{label} benchmark config has no benchmark identity")
    if not isinstance(entries, Mapping) or not entries:
        raise SummaryError(f"{label} benchmark config has no configs")
    cases: dict[str, _Case] = {}
    workload_identities: dict[
        tuple[str, tuple[tuple[str, str], ...]], tuple[str, str]
    ] = {}
    for config_key, raw_entry in entries.items():
        if not isinstance(config_key, str) or not isinstance(raw_entry, Mapping):
            raise SummaryError(f"{label} config entries must be named objects")
        entry = dict(raw_entry)
        tier = entry.get("tier")
        if tier not in {"basic", "advanced"}:
            raise SummaryError(
                f"{label} config {config_key!r} has invalid or missing tier {tier!r}"
            )
        try:
            expected = expected_case_keys_for_entry(config_key, entry)
        except BaselineError as exc:
            raise SummaryError(f"{label} config {config_key!r}: {exc}") from exc
        baselines = entry.get("baselines")
        if not isinstance(baselines, Mapping):
            raise SummaryError(
                f"{label} config {config_key!r} baselines must be an object"
            )
        expected_set = set(expected)
        actual_set = set(baselines)
        if expected_set != actual_set:
            missing = sorted(expected_set - actual_set)
            extra = sorted(actual_set - expected_set)
            detail = []
            if missing:
                detail.append(
                    f"missing {len(missing)} expanded case(s), e.g. {missing[0]}"
                )
            if extra:
                detail.append(f"has {len(extra)} undeclared case(s), e.g. {extra[0]}")
            raise SummaryError(f"{label} config {config_key!r} " + "; ".join(detail))
        for key in expected:
            if key in cases:
                raise SummaryError(f"{label} expanded case key is duplicated: {key}")
            payload = baselines[key]
            if not isinstance(payload, Mapping):
                raise SummaryError(
                    f"{label} case {key!r} SKU payload must be an object"
                )
            try:
                _, axes = parse_case_key(key)
            except BaselineError as exc:
                raise SummaryError(f"{label} case {key!r}: {exc}") from exc
            workload_identity = (tier, tuple(sorted(axes)))
            previous = workload_identities.get(workload_identity)
            if previous is not None:
                previous_config, previous_key = previous
                alias_pair = tuple(sorted((previous_config, config_key)))
                if alias_pair not in _LEGACY_DUPLICATE_CONFIG_ALIASES.get(
                    benchmark, set()
                ):
                    raise SummaryError(
                        f"{label} duplicate expanded workload identity "
                        f"(tier, complete raw axes): {previous_key!r} and {key!r}"
                    )
                # This is a transitional compatibility path only. Keeping one
                # canonical case prevents the legacy alias from double-weighting
                # Full operator statistics while its config cleanup is pending.
                if previous_config == alias_pair[0]:
                    continue
                cases.pop(previous_key)
            workload_identities[workload_identity] = (config_key, key)
            cases[key] = _Case(key, tier, axes, payload)
    skus = {str(stem) for case in cases.values() for stem in case.baselines}
    for axis in ("layout", "inputKind"):
        presence = {axis in case.axis_map for case in cases.values()}
        if len(presence) > 1:
            raise SummaryError(
                f"{label} benchmark mixes cases with and without the {axis} axis"
            )
    try:
        pairing_issues = fake_planar_pairing_issues(entries)
    except BaselineError as exc:
        raise SummaryError(
            f"{label} FakePlanar pairing cannot be evaluated: {exc}"
        ) from exc
    if pairing_issues:
        raise SummaryError(
            f"{label} benchmark has {len(pairing_issues)} unmatched FakePlanar case "
            "signature(s); every FakePlanar case must be advanced Tensor and have "
            "exactly one same-tier otherwise-identical native NCHW/CHW case; "
            + pairing_issues[0]
        )
    return _ConfigSurface(benchmark, cases, tuple(sorted(skus)))


def _optimized_cases(
    metadata: SummaryMetadata, candidate: _ConfigSurface
) -> tuple[_Case, ...]:
    unknown = [key for key in metadata.optimized_cases if key not in candidate.cases]
    if unknown:
        raise SummaryError(
            f"optimized_cases contains {len(unknown)} unknown key(s), e.g. {unknown[0]}"
        )
    optimized = tuple(candidate.cases[key] for key in metadata.optimized_cases)
    fake = [
        case.key for case in optimized if case.axis_map.get("layout") == "NCHW_FAKE"
    ]
    if fake:
        raise SummaryError(
            "layout-conversion reference cases cannot be optimization targets, e.g. "
            + fake[0]
        )
    return optimized


def _reference_skus(sku_map: Mapping[str, Any]) -> tuple[str, ...]:
    entries = sku_map.get("entries") if isinstance(sku_map, Mapping) else None
    if not isinstance(entries, list):
        raise SummaryError("SKU map must contain an entries array")
    stems = []
    for entry in entries:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("stem"), str):
            raise SummaryError("SKU map entries must contain string stems")
        stems.append(entry["stem"])
    if len(stems) != len(set(stems)):
        raise SummaryError("SKU map contains duplicate stems")
    return tuple(sorted(stems, key=_sku_label))


def _complete_skus(
    baseline: _ConfigSurface,
    candidate: _ConfigSurface,
    impact_metric: str = "cpp_time",
) -> tuple[str, ...]:
    timing_fields = (
        ("gpu_time_us_cpp",)
        if impact_metric == "cpp_time"
        else ("gpu_time_us_cpp", "gpu_time_us_python")
    )
    complete = []
    for stem in sorted(set(baseline.skus) | set(candidate.skus)):
        missing = [
            key
            for key, case in candidate.cases.items()
            if key not in baseline.cases
            or any(
                _timing(baseline.cases[key], stem, field) is None
                or _timing(case, stem, field) is None
                for field in timing_fields
            )
        ]
        if missing:
            fields = " and ".join(timing_fields)
            raise SummaryError(
                f"SKU {stem!r} lacks full before/after {fields} coverage "
                f"for {len(missing)} expanded case(s), e.g. {missing[0]}"
            )
        complete.append(stem)
    return tuple(complete)


def _assert_comparable_surfaces(
    baseline: _ConfigSurface, candidate: _ConfigSurface
) -> None:
    baseline_keys = set(baseline.cases)
    candidate_keys = set(candidate.cases)
    if baseline_keys != candidate_keys:
        missing = sorted(baseline_keys - candidate_keys)
        added = sorted(candidate_keys - baseline_keys)
        detail = []
        if missing:
            detail.append(
                f"candidate omits {len(missing)} baseline case(s), e.g. {missing[0]}"
            )
        if added:
            detail.append(
                f"candidate adds {len(added)} case(s) without before data, e.g. {added[0]}"
            )
        raise SummaryError(
            "baseline and candidate expanded benchmark surfaces differ: "
            + "; ".join(detail)
        )
    tier_changes = [
        key
        for key in candidate_keys
        if baseline.cases[key].tier != candidate.cases[key].tier
    ]
    if tier_changes:
        raise SummaryError(
            "baseline and candidate tier assignments differ, e.g. " + tier_changes[0]
        )


def _timing(case: _Case, stem: str, field: str = "gpu_time_us_cpp") -> Decimal | None:
    payload = case.baselines.get(stem)
    if not isinstance(payload, Mapping):
        return None
    value = payload.get(field)
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    if not result.is_finite() or result <= 0:
        return None
    return result


def _categories(cases: Sequence[_Case]) -> tuple[str, ...]:
    return tuple(sorted({_category(case) for case in cases}))


def _category(case: _Case) -> str:
    axes = case.axis_map
    container = axes.get("inputKind", "Tensor")
    if container not in {"Tensor", "TensorBatch", "VarShape"}:
        raise SummaryError(f"unsupported inputKind {container!r} in {case.key}")
    layout = {
        "NHWC": "Interleaved",
        "HWC": "Interleaved",
        "NCHW": "Planar",
        "CHW": "Planar",
        "NCHW_FAKE": "FakePlanar",
        "NW": "NoLayout",
        "NWC": "NoLayout",
        None: "NoLayout",
    }.get(axes.get("layout"))
    if layout is None:
        raise SummaryError(f"unsupported layout {axes.get('layout')!r} in {case.key}")
    if not axes.get("InOutDataType"):
        raise SummaryError(f"case has no InOutDataType axis: {case.key}")
    return f"{container} {layout} {_logical_type(axes)}"


def _logical_type(axes: Mapping[str, str]) -> str:
    dtype = axes["InOutDataType"]
    code = axes.get("code")
    if code:
        return _color_conversion(code, dtype)

    channels = _vector_width(dtype)
    num_channels = axes.get("numChannels")
    if num_channels:
        channels = int(num_channels)
    source = _friendly_type_with_width(dtype, channels)

    output_dtype = axes.get("outDataType")
    if output_dtype:
        output = _friendly_type_with_width(output_dtype, channels)
        return f"{source}→{output}"

    output_channels = axes.get("outChannels")
    if output_channels:
        output = _friendly_type_with_width(dtype, int(output_channels))
        return source if output == source else f"{source}→{output}"
    return source


def _friendly_type(dtype: str) -> str:
    names = {
        "uchar3": "RGB8",
        "uchar4": "RGBA8",
        "float3": "RGBF32",
        "float4": "RGBAF32",
        "uint8": "U8",
        "uint16": "U16",
        "uint32": "U32",
        "int16": "S16",
        "int32": "S32",
        "float32": "F32",
        "short2": "2S16",
        "short4": "4S16",
    }
    if dtype in names:
        return names[dtype]
    if "->" in dtype:
        return "→".join(_friendly_type(item.strip()) for item in dtype.split("->"))
    return dtype.upper()


def _vector_width(dtype: str) -> int:
    match = re.fullmatch(r"(?:uchar|float|short)([234])", dtype)
    return int(match.group(1)) if match else 1


def _friendly_type_with_width(dtype: str, width: int) -> str:
    scalar = {
        "uint8": "U8",
        "uchar3": "U8",
        "uchar4": "U8",
        "uint16": "U16",
        "uint32": "U32",
        "int16": "S16",
        "int32": "S32",
        "short2": "S16",
        "short4": "S16",
        "float32": "F32",
        "float3": "F32",
        "float4": "F32",
    }.get(dtype, _friendly_type(dtype))
    if width == 3 and scalar == "U8":
        return "RGB8"
    if width == 4 and scalar == "U8":
        return "RGBA8"
    if width == 3 and scalar == "F32":
        return "RGBF32"
    if width == 4 and scalar == "F32":
        return "RGBAF32"
    return scalar if width == 1 else f"{width}{scalar}"


def _color_conversion(code: str, dtype: str) -> str:
    if "2" not in code:
        raise SummaryError(f"unsupported color conversion code {code!r}")
    source, target = code.split("2", 1)
    nv_format = None
    for suffix in ("_NV12", "_NV21"):
        if target.endswith(suffix):
            target = target[: -len(suffix)]
            nv_format = suffix[1:]
            break
    width = _vector_width(dtype)
    if nv_format and source == "YUV":
        source = nv_format
    elif nv_format and target == "YUV":
        target = nv_format

    def endpoint(name: str, *, vector_endpoint: bool) -> str:
        if name in {"NV12", "NV21"}:
            return name
        if name in {"RGB", "BGR"} and vector_endpoint and width == 4:
            name += "A"
        return name + "8"

    return (
        f"{endpoint(source, vector_endpoint=not source.startswith('NV'))}→"
        f"{endpoint(target, vector_endpoint=not target.startswith('NV'))}"
    )


def _expected_impact_rows(
    baseline: _ConfigSurface,
    candidate: _ConfigSurface,
    optimized: Sequence[_Case],
    skus: Sequence[str],
    references: Sequence[str],
    impact_metric: str = "cpp_time",
) -> tuple[ImpactRow | PythonOverheadImpactRow, ...]:
    rows = []
    ordered_skus = _ordered_skus(skus, references)
    full = tuple(candidate.cases.values())
    for stem in ordered_skus:
        for scope, cases in (("Optimized", optimized), ("Full operator", full)):
            if impact_metric == "cpp_time":
                stats = _stats(_speedups(baseline, cases, stem))
                rows.append(
                    ImpactRow(
                        _sku_label(stem),
                        scope,
                        len(cases),
                        stats[0],
                        stats[1],
                        stats[2],
                    )
                )
            else:
                before, after, reduction = _python_overhead_values(
                    baseline, cases, stem
                )
                rows.append(
                    PythonOverheadImpactRow(
                        _sku_label(stem),
                        scope,
                        len(cases),
                        _stats(before),
                        _stats(after),
                        _stats(reduction),
                    )
                )
    return tuple(rows)


def _speedups(
    baseline: _ConfigSurface, cases: Sequence[_Case], stem: str
) -> tuple[Decimal, ...]:
    values = []
    for candidate_case in cases:
        baseline_case = baseline.cases.get(candidate_case.key)
        before = _timing(baseline_case, stem) if baseline_case else None
        after = _timing(candidate_case, stem)
        if before is None or after is None:
            raise SummaryError(
                f"missing positive gpu_time_us_cpp for {candidate_case.key} on {stem}"
            )
        values.append(before / after)
    return tuple(values)


def _python_overhead_values(
    baseline: _ConfigSurface, cases: Sequence[_Case], stem: str
) -> tuple[tuple[Decimal, ...], tuple[Decimal, ...], tuple[Decimal, ...]]:
    before_values = []
    after_values = []
    reductions = []
    for candidate_case in cases:
        baseline_case = baseline.cases.get(candidate_case.key)
        timings = (
            _timing(baseline_case, stem, "gpu_time_us_cpp") if baseline_case else None,
            _timing(baseline_case, stem, "gpu_time_us_python")
            if baseline_case
            else None,
            _timing(candidate_case, stem, "gpu_time_us_cpp"),
            _timing(candidate_case, stem, "gpu_time_us_python"),
        )
        if any(value is None for value in timings):
            raise SummaryError(
                "missing positive gpu_time_us_cpp or gpu_time_us_python for "
                f"{candidate_case.key} on {stem}"
            )
        before_cpp, before_python, after_cpp, after_python = timings
        before_gap = before_python - before_cpp
        after_gap = after_python - after_cpp
        before_values.append(before_gap)
        after_values.append(after_gap)
        reductions.append(before_gap - after_gap)
    return tuple(before_values), tuple(after_values), tuple(reductions)


def _expected_layout_rows(
    baseline: _ConfigSurface,
    candidate: _ConfigSurface,
    skus: Sequence[str],
    references: Sequence[str],
) -> tuple[LayoutRow, ...]:
    rows = []
    pairs = {
        "Interleaved / Planar": _layout_pairs(candidate, "Interleaved"),
        "FakePlanar / Planar": _layout_pairs(candidate, "FakePlanar"),
    }
    for stem in _ordered_skus(skus, references):
        for comparison in ("Interleaved / Planar", "FakePlanar / Planar"):
            matched = pairs[comparison]
            if not matched:
                before = after = None
            else:
                before_values = []
                after_values = []
                for numerator_key, planar_key in matched:
                    before_values.append(
                        _ratio_for_pair(baseline, numerator_key, planar_key, stem)
                    )
                    after_values.append(
                        _ratio_for_pair(candidate, numerator_key, planar_key, stem)
                    )
                before = _stats(before_values)
                after = _stats(after_values)
            rows.append(LayoutRow(_sku_label(stem), comparison, before, after))
    return tuple(rows)


def _layout_pairs(
    surface: _ConfigSurface, numerator: str
) -> tuple[tuple[str, str], ...]:
    groups: dict[tuple[Any, ...], dict[str, list[str]]] = {}
    layout_names = {
        "NHWC": "Interleaved",
        "HWC": "Interleaved",
        "NCHW": "Planar",
        "CHW": "Planar",
        "NCHW_FAKE": "FakePlanar",
    }
    for case in surface.cases.values():
        axes = case.axis_map
        kind = layout_names.get(axes.get("layout"))
        if kind is None:
            continue
        identity = (
            case.tier,
            tuple(
                sorted((name, value) for name, value in case.axes if name != "layout")
            ),
        )
        groups.setdefault(identity, {}).setdefault(kind, []).append(case.key)
    pairs = []
    for identity, layouts in groups.items():
        numerator_keys = layouts.get(numerator, [])
        planar_keys = layouts.get("Planar", [])
        if not numerator_keys or not planar_keys:
            continue
        duplicates = {
            kind: layouts[kind]
            for kind in (numerator, "Planar")
            if len(layouts.get(kind, [])) > 1
        }
        if duplicates:
            sample = next(iter(duplicates.values()))
            raise SummaryError(
                "ambiguous layout comparison signature "
                f"{identity!r}; multiple same-layout cases include {sample[:2]}"
            )
        if len(numerator_keys) == 1 and len(planar_keys) == 1:
            pairs.append((numerator_keys[0], planar_keys[0]))
    return tuple(sorted(pairs))


def _ratio_for_pair(
    surface: _ConfigSurface, numerator_key: str, planar_key: str, stem: str
) -> Decimal:
    numerator = surface.cases.get(numerator_key)
    planar = surface.cases.get(planar_key)
    if numerator is None or planar is None:
        raise SummaryError(
            f"baseline/candidate layout surfaces differ for pair {numerator_key!r} / {planar_key!r}"
        )
    numerator_time = _timing(numerator, stem)
    planar_time = _timing(planar, stem)
    if numerator_time is None or planar_time is None:
        raise SummaryError(f"layout pair lacks gpu_time_us_cpp on {stem}")
    return numerator_time / planar_time


def _stats(values: Iterable[Decimal]) -> tuple[Decimal, Decimal, Decimal]:
    ordered = sorted(values)
    if not ordered:
        raise SummaryError("cannot calculate statistics for an empty case set")
    return (_round(ordered[0]), _round(median(ordered)), _round(ordered[-1]))


def _round(value: Decimal) -> Decimal:
    rounded = value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return Decimal("0.00") if rounded == 0 else rounded


def _sku_label(stem: str) -> str:
    match = re.match(r"^((?:A|H)\d+)(?:_|$)", stem, re.IGNORECASE)
    return match.group(1).upper() if match else stem.replace("_", " ")


def _ordered_skus(skus: Sequence[str], references: Sequence[str]) -> tuple[str, ...]:
    present = set(skus)
    ordered = tuple(
        [stem for stem in references if stem in present]
        + sorted(present - set(references), key=_sku_label)
    )
    labels = [_sku_label(stem) for stem in ordered]
    if len(labels) != len(set(labels)):
        raise SummaryError(
            "reported SKU stems do not have unique human-readable labels"
        )
    return ordered


def _render_impact_row(row: ImpactRow | PythonOverheadImpactRow) -> str:
    if isinstance(row, PythonOverheadImpactRow):
        return (
            f"| {row.sku} | {row.scope} | {row.configurations} | "
            f"{_duration_stats_cell(row.before)} | {_duration_stats_cell(row.after)} | "
            f"{_duration_stats_cell(row.reduction)} |"
        )
    return (
        f"| {row.sku} | {row.scope} | {row.configurations} | "
        f"{_factor(row.minimum)} | {_factor(row.median)} | {_factor(row.maximum)} |"
    )


def _render_layout_row(row: LayoutRow) -> str:
    return (
        f"| {row.sku} | {row.comparison} | {_stats_cell(row.before)} | "
        f"{_stats_cell(row.after)} |"
    )


def _factor(value: Decimal) -> str:
    return f"{value:.2f}x"


def _stats_cell(values: tuple[Decimal, Decimal, Decimal] | None) -> str:
    if values is None:
        return NA_STATS
    return " / ".join(_factor(value) for value in values)


def _duration_stats_cell(values: tuple[Decimal, Decimal, Decimal]) -> str:
    return " / ".join(f"{value:.2f} µs" for value in values)


def _normalize_checklist(
    checklist: Mapping[str, Any] | Sequence[ChecklistItem] | None,
) -> tuple[ChecklistItem, ...]:
    if checklist is None:
        return tuple(
            ChecklistItem(label, False, "<add hard evidence before checking>")
            for label in CHECKLIST_LABELS
        )
    if isinstance(checklist, Mapping):
        items = []
        for label in CHECKLIST_LABELS:
            value = checklist.get(label)
            if isinstance(value, ChecklistItem):
                items.append(value)
            elif isinstance(value, (tuple, list)) and len(value) == 2:
                items.append(ChecklistItem(label, bool(value[0]), str(value[1])))
            elif isinstance(value, str):
                items.append(ChecklistItem(label, False, value))
            else:
                raise SummaryError(f"checklist is missing {label!r}")
        return tuple(items)
    items = tuple(checklist)
    if [item.label for item in items] != list(CHECKLIST_LABELS):
        raise SummaryError("checklist labels are incomplete or out of order")
    return items


def _join_description(prefix: str, block: str, suffix: str) -> str:
    result = prefix
    if result and not result.endswith("\n"):
        result += "\n"
    result += block
    if suffix:
        if not result.endswith("\n") and not suffix.startswith("\n"):
            result += "\n"
        result += suffix
    return result


def _profile_is_measured(evidence: str) -> bool:
    return bool(
        re.search(r"\d+(?:\.\d+)?\s*%", evidence)
        and re.search(
            r"\b(?:SOL|BWUtil|DRAM|compute|memory|stall|scoreboard|throughput|occupancy)\b",
            evidence,
            re.IGNORECASE,
        )
    )


def _validate_visible_tables(
    summary: OptimizationSummary, errors: list[ValidationIssue]
) -> None:
    impact_by_sku: dict[str, list[ImpactRow | PythonOverheadImpactRow]] = {}
    for row in summary.impact_rows:
        impact_by_sku.setdefault(row.sku, []).append(row)
        expected_count = (
            summary.optimized_count if row.scope == "Optimized" else summary.total_count
        )
        if row.configurations != expected_count:
            errors.append(
                ValidationIssue(
                    "impact-count",
                    f"{row.sku} {row.scope} count does not match Scope",
                )
            )
        statistics = (
            (("speedup", (row.minimum, row.median, row.maximum)),)
            if isinstance(row, ImpactRow)
            else (
                ("before", row.before),
                ("after", row.after),
                ("reduction", row.reduction),
            )
        )
        for label, values in statistics:
            if not values[0] <= values[1] <= values[2]:
                errors.append(
                    ValidationIssue(
                        "impact-order",
                        f"{row.sku} {row.scope} {label} statistics are not ordered",
                    )
                )
    for sku, rows in impact_by_sku.items():
        if [row.scope for row in rows] != ["Optimized", "Full operator"]:
            errors.append(
                ValidationIssue(
                    "impact-rows",
                    f"{sku} must have exactly Optimized then Full operator rows",
                )
            )

    layout_by_sku: dict[str, list[LayoutRow]] = {}
    for row in summary.layout_rows:
        layout_by_sku.setdefault(row.sku, []).append(row)
        for label, values in (("before", row.before), ("after", row.after)):
            if values is not None and not values[0] <= values[1] <= values[2]:
                errors.append(
                    ValidationIssue(
                        "layout-order",
                        f"{row.sku} {row.comparison} {label} statistics are not ordered",
                    )
                )
    for sku, rows in layout_by_sku.items():
        if [row.comparison for row in rows] != [
            "Interleaved / Planar",
            "FakePlanar / Planar",
        ]:
            errors.append(
                ValidationIssue(
                    "layout-rows",
                    f"{sku} must have exactly both canonical layout-comparison rows",
                )
            )
    if list(impact_by_sku) != list(layout_by_sku):
        errors.append(
            ValidationIssue(
                "sku-rows", "Impact and Layout comparison must report the same SKUs"
            )
        )


def _validate_checklist(
    summary: OptimizationSummary,
    evidence_results: Mapping[str, str] | None,
    errors: list[ValidationIssue],
    warnings: list[ValidationIssue],
) -> None:
    for item in summary.checklist:
        if summary.metadata.state == "final" and not item.checked:
            errors.append(
                ValidationIssue("checklist-unchecked", f"{item.label} is unchecked")
            )
        if item.checked and not _hard_evidence(item.evidence):
            errors.append(
                ValidationIssue(
                    "checklist-evidence",
                    f"{item.label} is checked without concrete hard evidence",
                )
            )
        if evidence_results is not None and item.checked:
            status = evidence_results.get(item.label)
            if status != "PASS":
                errors.append(
                    ValidationIssue(
                        "checklist-gate",
                        f"{item.label} is checked but its hard gate is {status or 'missing'}",
                    )
                )
        if not item.checked and _PLACEHOLDER_RE.search(item.evidence):
            target = errors if summary.metadata.state == "final" else warnings
            target.append(
                ValidationIssue(
                    "checklist-placeholder",
                    f"{item.label} still contains a placeholder",
                )
            )


def _hard_evidence(evidence: str) -> bool:
    text = evidence.strip()
    if len(text) < 20 or _PLACEHOLDER_RE.search(text):
        return False
    if text.casefold() in {"pass", "passed", "yes", "done", "n/a"}:
        return False
    named = bool(
        re.search(
            r"(?:\b\w+\.py\b|\b\w+_test\w*\b|\bODO-\d+\b|`[^`]+`|"
            r"\b(?:baseline|artifact|benchmark|run_bench|EXPECT_(?:EQ|NEAR)|"
            r"ncu|nsys|commit|SHA)\b)",
            text,
            re.IGNORECASE,
        )
    )
    result = bool(
        re.search(
            r"\b(?:pass(?:ed)?|green|zero regressions?|no regressions?|\d+\s*/\s*\d+|"
            r"at[- ]?ridge|strikes?|recommendations?|updated|imported|committed)\b",
            text,
            re.IGNORECASE,
        )
    )
    return named and result


def _validate_learnings(
    summary: OptimizationSummary,
    errors: list[ValidationIssue],
    warnings: list[ValidationIssue],
) -> None:
    for learning in summary.learnings:
        if learning == DEFAULT_LEARNING:
            target = errors if summary.metadata.state == "final" else warnings
            target.append(
                ValidationIssue(
                    "learning-placeholder",
                    "Top learnings still contains the provisional default",
                )
            )
            continue
        if len(learning) > 500 or _PLACEHOLDER_RE.search(learning):
            errors.append(
                ValidationIssue(
                    "learning-placeholder",
                    "Top learnings contains a placeholder or oversized item",
                )
            )
            continue
        if not re.search(r"[.!?]$", learning):
            errors.append(
                ValidationIssue(
                    "learning-sentence", "Each top learning must be one sentence"
                )
            )
        if re.search(r"[!?]\s+\S|\.\s+[A-Z]", learning[:-1]):
            errors.append(
                ValidationIssue(
                    "learning-sentence", "Each top learning must be one sentence"
                )
            )


def _validate_artifact_claims(
    summary: OptimizationSummary,
    baseline_config: Mapping[str, Any],
    candidate_config: Mapping[str, Any],
    sku_map: Mapping[str, Any],
    errors: list[ValidationIssue],
    warnings: list[ValidationIssue],
) -> None:
    baseline = _config_surface(baseline_config, "baseline")
    candidate = _config_surface(candidate_config, "candidate")
    if baseline.benchmark.casefold() != candidate.benchmark.casefold():
        raise SummaryError("baseline and candidate benchmark identities differ")
    _assert_comparable_surfaces(baseline, candidate)
    if candidate.benchmark.casefold() != summary.metadata.operator.casefold():
        raise SummaryError(
            "metadata operator does not match candidate benchmark identity"
        )
    optimized = _optimized_cases(summary.metadata, candidate)
    if summary.total_count != len(candidate.cases):
        errors.append(
            ValidationIssue(
                "full-count",
                "visible total is not the full expanded basic+advanced operator case set",
            )
        )
    expected_categories = _categories(optimized)
    if summary.categories != expected_categories:
        errors.append(
            ValidationIssue(
                "categories",
                "optimized categories do not match exact Container Layout Type derivation",
            )
        )

    references = _reference_skus(sku_map)
    complete_skus = _complete_skus(baseline, candidate, summary.metadata.impact_metric)
    missing_references = [stem for stem in references if stem not in complete_skus]
    if summary.metadata.state == "final" and missing_references:
        errors.append(
            ValidationIssue(
                "reference-coverage",
                "final summary lacks full basic+advanced timing coverage for reference SKU(s): "
                + ", ".join(missing_references),
            )
        )
    if not complete_skus:
        raise SummaryError("no SKU has timings for every expanded candidate case")

    expected_impact = _expected_impact_rows(
        baseline,
        candidate,
        optimized,
        complete_skus,
        references,
        summary.metadata.impact_metric,
    )
    if summary.impact_rows != expected_impact:
        errors.append(
            ValidationIssue(
                "impact-statistics",
                "Impact rows/counts/min/median/max do not match the selected impact metric",
            )
        )
    expected_layout = _expected_layout_rows(
        baseline, candidate, complete_skus, references
    )
    if summary.layout_rows != expected_layout:
        errors.append(
            ValidationIssue(
                "layout-statistics",
                "Layout rows or before/after min/median/max do not match full matched pairs",
            )
        )

    local_labels = {
        _sku_label(stem) for stem in complete_skus if stem not in set(references)
    }
    warning_text = "\n".join(summary.warnings)
    for label in sorted(local_labels):
        if label not in warning_text:
            errors.append(
                ValidationIssue(
                    "local-sku-warning",
                    f"non-reference SKU {label} is reported without the required warning",
                )
            )
    if not local_labels and summary.warnings:
        errors.append(
            ValidationIssue(
                "local-sku-warning",
                "non-reference warning is present without a local SKU row",
            )
        )
