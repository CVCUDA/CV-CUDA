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
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Mapping, Sequence

import optimization_evidence as evidence_mod

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
    "Generalization review",
    "Evidence checklist",
    "Top learnings",
)
GENERALIZATION_AREAS = (
    "Layouts / input kinds",
    "Dtypes / channels",
    "Shapes / parameters",
    "Shared code / sibling operators",
)
SUMMARY_TITLE = "Optimization summary"
# Fixed by the contract, not by whoever writes the description: a tunable
# threshold is a knob for making a change look better than it measured.
MOVED_PCT = 5.0
ALL_SCOPE = "All configurations"
IMPROVED_SCOPE = "Improved"
REGRESSED_SCOPE = "Regressed"
IMPACT_SCOPES = (ALL_SCOPE, IMPROVED_SCOPE, REGRESSED_SCOPE)
LAYOUT_COMPARISONS = ("Interleaved / Planar", "FakePlanar / Planar")
START_TOKEN = "cvcuda-optimize-summary:v3"
END_MARKER = "<!-- /cvcuda-optimize-summary:v3 -->"
MAX_DESCRIPTION_BYTES = 2_000_000
MAX_BLOCK_BYTES = 400_000
MAX_OPERATORS = 64
NA_STATS = "n/a — no matched equivalent configurations"
AMBIGUOUS_STATS = "n/a — ambiguous layout signatures"
LAYOUT_NOTES = (NA_STATS, AMBIGUOUS_STATS)
DEFAULT_LEARNING = "No optimization attempt has been recorded yet."
PROFILE_PENDING = "No Nsight Compute profile has been recorded yet."
BOTTLENECK_PENDING = "Not yet profiled"
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_OPERATOR_RE = re.compile(r"^[A-Za-z][A-Za-z0-9]{0,127}$")
_START_RE = re.compile(r"^<!-- " + re.escape(START_TOKEN) + r" (\{.*\}) -->\s*$")
_FACTOR_RE = re.compile(r"^(\d{1,9}\.\d{2})x$")
_STATS_RE = re.compile(r"^(\d{1,9}\.\d{2})x / (\d{1,9}\.\d{2})x / (\d{1,9}\.\d{2})x$")
_DURATION_STATS_RE = re.compile(
    r"^(-?\d{1,9}\.\d{2}) µs / (-?\d{1,9}\.\d{2}) µs / " r"(-?\d{1,9}\.\d{2}) µs$"
)
# An identifier immediately before "<" makes it C++ template syntax, not an
# unfilled "<...>" placeholder.
_PLACEHOLDER_RE = re.compile(
    r"(?<![\w>])<[^>]+>|\b(?:todo|tbd|placeholder|add hard evidence|add one concise)\b",
    re.IGNORECASE,
)


class SummaryError(ValueError):
    """The bounded optimization-summary contract is malformed."""


@dataclass(frozen=True)
class SummaryMetadata:
    operators: tuple[str, ...]
    state: str
    baseline_commit: str
    candidate_commit: str
    impact_metric: str = "cpp_time"
    baseline_retention: str | None = None
    evidence: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.operators, (str, bytes)):
            raise SummaryError("metadata operators must be an array")
        object.__setattr__(self, "operators", tuple(self.operators))
        if self.evidence is None:
            object.__setattr__(self, "evidence", evidence_mod.empty())

    @property
    def primary(self) -> str:
        """The operator that scope authorization and the evidence gates key on."""
        return self.operators[0]


@dataclass(frozen=True)
class ChecklistItem:
    label: str
    checked: bool
    evidence: str


@dataclass(frozen=True)
class ScopeRow:
    operator: str
    configurations: int
    improved: int
    regressed: int


@dataclass(frozen=True)
class ImpactRow:
    operator: str
    sku: str
    scope: str
    configurations: int
    minimum: Decimal
    median: Decimal
    maximum: Decimal


@dataclass(frozen=True)
class PythonOverheadImpactRow:
    operator: str
    sku: str
    scope: str
    configurations: int
    before: tuple[Decimal, Decimal, Decimal]
    after: tuple[Decimal, Decimal, Decimal]
    reduction: tuple[Decimal, Decimal, Decimal]


@dataclass(frozen=True)
class LayoutRow:
    operator: str
    sku: str
    comparison: str
    before: tuple[Decimal, Decimal, Decimal] | None
    after: tuple[Decimal, Decimal, Decimal] | None
    note: str = NA_STATS


@dataclass(frozen=True)
class GeneralizationRow:
    area: str
    assessment: str
    decision: str


@dataclass(frozen=True)
class OptimizationSummary:
    metadata: SummaryMetadata
    text: str
    block: str
    prefix: str
    suffix: str
    status_line: str
    warnings: tuple[str, ...]
    bottleneck: str
    profile_evidence: str
    scope_rows: tuple[ScopeRow, ...]
    impact_rows: tuple[ImpactRow | PythonOverheadImpactRow, ...]
    layout_rows: tuple[LayoutRow, ...]
    generalization_rows: tuple[GeneralizationRow, ...]
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


def parse_summary(description: str) -> OptimizationSummary:
    """Parse exactly one live summary block from an untrusted MR description."""

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
    scope_rows: Sequence[ScopeRow] | None = None,
    impact_rows: Sequence[ImpactRow | PythonOverheadImpactRow] | None = None,
    layout_rows: Sequence[LayoutRow] | None = None,
    reference_skus: Sequence[str] | None = None,
    expected_operators: Sequence[str] | None = None,
    expected_candidate_commit: str | None = None,
    retention_detail: str | None = None,
) -> ValidationResult:
    """Validate structure, recorded evidence, and, when rows are supplied, figures."""

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
    if expected_operators is not None:
        declared = [operator.casefold() for operator in meta.operators]
        if declared != [operator.casefold() for operator in expected_operators]:
            errors.append(
                ValidationIssue(
                    "operator-mismatch",
                    f"metadata operators {list(meta.operators)} do not match "
                    f"{list(expected_operators)}",
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

    _validate_rendered_evidence(summary, retention_detail, errors, warnings)
    _validate_visible_tables(summary, errors)
    (errors if meta.state == "final" else warnings).extend(
        validate_generalization_rows(summary.generalization_rows, meta.evidence)
    )

    for label, reported, expected in (
        ("scope", summary.scope_rows, scope_rows),
        ("impact", summary.impact_rows, impact_rows),
        ("layout", summary.layout_rows, layout_rows),
    ):
        if expected is not None and tuple(reported) != tuple(expected):
            errors.append(
                ValidationIssue(
                    f"{label}-statistics",
                    f"{label} rows do not match the committed baselines",
                )
            )

    if reference_skus is not None and meta.state == "final":
        reported_skus = {row.sku for row in summary.impact_rows}
        missing = [sku for sku in reference_skus if sku not in reported_skus]
        if missing:
            errors.append(
                ValidationIssue(
                    "reference-coverage",
                    "final summary lacks reference SKU(s): " + ", ".join(missing),
                )
            )
        local = sorted(reported_skus - set(reference_skus))
        warning_text = "\n".join(summary.warnings)
        for sku in local:
            if sku not in warning_text:
                errors.append(
                    ValidationIssue(
                        "local-sku-warning",
                        f"non-reference SKU {sku} is reported without the required "
                        "warning",
                    )
                )

    return ValidationResult(tuple(errors), tuple(warnings))


def lead_exhaustion(
    metadata: SummaryMetadata, generalization_rows: Sequence[GeneralizationRow]
) -> "evidence_mod.GateResult":
    """Recorded ridge/strikes, or reviewed deferral of every remaining opportunity."""

    evidence = metadata.evidence
    result = evidence_mod.lead_exhaustion(
        evidence.get("attempts", ()), evidence.get("profile")
    )
    if result.status == evidence_mod.PASS:
        return result
    if remaining_generalizations_deferred(generalization_rows, evidence):
        return evidence_mod.GateResult(
            evidence_mod.PASS,
            "Remaining generalizations deferred; see Generalization review.",
        )
    return result


def rendered_fields(
    metadata: SummaryMetadata,
    *,
    generalization_rows: Sequence[GeneralizationRow] = (),
    candidate: str | None = None,
    retention_detail: str | None = None,
) -> dict[str, Any]:
    """Every evidence-bearing field, derived only from recorded evidence."""

    evidence = metadata.evidence
    profile = evidence.get("profile")
    rows = evidence_mod.checklist(
        evidence,
        checklist_labels(metadata),
        candidate=candidate or metadata.candidate_commit,
        retention_detail=retention_detail,
        exhaustion=lead_exhaustion(metadata, generalization_rows),
    )
    return {
        "bottleneck": (
            evidence_mod.derive_bottleneck(profile) if profile else BOTTLENECK_PENDING
        ),
        "profile_evidence": (
            evidence_mod.render_profile(profile) if profile else PROFILE_PENDING
        ),
        "checklist": tuple(ChecklistItem(*row) for row in rows),
        "learnings": evidence_mod.draft_learnings(evidence.get("attempts", ()))
        or (DEFAULT_LEARNING,),
        "memory": (
            evidence_mod.render_memory(evidence["memory"])
            if "memory" in evidence
            else None
        ),
    }


def generate_summary(
    metadata: SummaryMetadata,
    *,
    scope_rows: Sequence[ScopeRow],
    impact_rows: Sequence[ImpactRow | PythonOverheadImpactRow],
    layout_rows: Sequence[LayoutRow],
    local_skus: Sequence[str] = (),
    generalization_rows: Sequence[GeneralizationRow] | None = None,
    retention_detail: str | None = None,
    prefix: str = "",
    suffix: str = "",
) -> str:
    """Render a summary from measured rows and recorded evidence.

    This module owns the block's shape, not its arithmetic. Callers supply rows
    derived from committed baselines, and every evidence line is rendered from
    ``metadata.evidence``, so no figure or claim here can originate with
    whoever is writing the description.
    """

    _validate_metadata(metadata)
    _require_rows_cover_operators(metadata, scope_rows, impact_rows, layout_rows)

    review_rows = (
        tuple(generalization_rows)
        if generalization_rows is not None
        else tuple(
            GeneralizationRow(
                area,
                "<opportunity or reason none applies; cite code/config/profile>",
                "<Applied: validation; Deferred: scope reason; Not applicable: reason>",
            )
            for area in GENERALIZATION_AREAS
        )
    )
    fields = rendered_fields(
        metadata, generalization_rows=review_rows, retention_detail=retention_detail
    )
    metadata_payload = {
        "operators": list(metadata.operators),
        "state": metadata.state,
        **(
            {"baseline_retention": metadata.baseline_retention}
            if metadata.baseline_retention
            else {}
        ),
        **(
            {"impact_metric": metadata.impact_metric}
            if metadata.impact_metric != "cpp_time"
            else {}
        ),
        "baseline_commit": metadata.baseline_commit,
        "candidate_commit": metadata.candidate_commit,
        "evidence": metadata.evidence,
    }
    metadata_json = json.dumps(
        metadata_payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    status = (
        "Final reference-hardware evidence"
        if metadata.state == "final"
        else "Provisional local evidence"
    )
    repair = metadata.evidence.get("reference_repair")
    if repair:
        status += " · " + repair["status_qualification"]
    lines = [
        f"<!-- {START_TOKEN} {metadata_json} -->",
        f"## {SUMMARY_TITLE}",
        "",
        f"> {status} · base `{metadata.baseline_commit[:12]}` · candidate "
        f"`{metadata.candidate_commit[:12]}`",
    ]
    for sku in local_skus:
        lines.extend(
            [
                "",
                "> ⚠️ **Non-reference local GPU:** Results from "
                f"{sku} are provisional and cannot satisfy final readiness; "
                "final reference-SKU statistics are still required.",
            ]
        )
    lines.extend(
        [
            "",
            f"**Primary bottleneck: {fields['bottleneck']}**",
            "",
            fields["profile_evidence"],
            "",
            "### Scope",
            "",
            *render_scope_table(scope_rows),
            "",
            "### Impact",
            "",
            *render_impact_table(impact_rows, metadata.impact_metric),
            "",
            "### Layout comparison",
            "",
            *render_layout_table(layout_rows),
            "",
            "### Generalization review",
            "",
            "| Area | Assessment | Decision |",
            "|---|---|---|",
            *[
                f"| {row.area} | {row.assessment} | {row.decision} |"
                for row in review_rows
            ],
            "",
            "### Evidence checklist",
            "",
            *[
                f"- [{'x' if item.checked else ' '}] **{item.label}** — {item.evidence}"
                for item in fields["checklist"]
            ],
            "",
            "### Top learnings",
            "",
            *[f"- {item}" for item in fields["learnings"]],
            END_MARKER,
        ]
    )
    suffix = _replace_memory_section(suffix, fields["memory"])
    rendered = _join_description(prefix, "\n".join(lines), suffix)
    validation = validate_summary(
        rendered,
        scope_rows=scope_rows,
        impact_rows=impact_rows,
        layout_rows=layout_rows,
        retention_detail=retention_detail,
    )
    if not validation.ok and metadata.state == "final":
        detail = "; ".join(
            f"{item.code}: {item.message}" for item in validation.errors[:6]
        )
        raise SummaryError("cannot render an invalid final summary: " + detail)
    return rendered


_MEMORY_SECTION_RE = re.compile(
    r"(?:^|\n)## Memory footprint\n(?:[^\n#][^\n]*\n?)*", re.MULTILINE
)


def _replace_memory_section(suffix: str, rendered: str | None) -> str:
    """Keep exactly one rendered memory section after the bounded block."""

    body = _MEMORY_SECTION_RE.sub("\n", suffix).strip("\n")
    if rendered is None:
        return ("\n" + body + "\n") if body else suffix
    return "\n\n" + rendered + "\n" + (("\n" + body + "\n") if body else "")


def memory_sections(description: str) -> list[str]:
    return [
        match.group(0).strip("\n") for match in _MEMORY_SECTION_RE.finditer(description)
    ]


def _require_rows_cover_operators(
    metadata: SummaryMetadata,
    scope_rows: Sequence[ScopeRow],
    impact_rows: Sequence[Any],
    layout_rows: Sequence[LayoutRow],
) -> None:
    """Every declared operator must appear in every table, in declaration order."""

    declared = list(metadata.operators)
    for label, rows in (
        ("Scope", scope_rows),
        ("Impact", impact_rows),
        ("Layout comparison", layout_rows),
    ):
        seen = _operators_reported(rows)
        if seen != declared:
            raise SummaryError(
                f"{label} rows must cover every declared operator exactly once, "
                f"in declaration order; declared {declared}, reported {seen}"
            )


def _operators_reported(rows: Sequence[Any]) -> list[str]:
    """Operator names a table reports, in first-seen order."""

    seen: list[str] = []
    for row in rows:
        if row.operator not in seen:
            seen.append(row.operator)
    return seen


def refresh_summary(
    description: str,
    *,
    scope_rows: Sequence[ScopeRow],
    impact_rows: Sequence[ImpactRow | PythonOverheadImpactRow],
    layout_rows: Sequence[LayoutRow],
    state: str | None = None,
    candidate_commit: str | None = None,
    local_skus: Sequence[str] = (),
    evidence: Mapping[str, Any] | None = None,
    retention_detail: str | None = None,
) -> str:
    """Regenerate every derived field; only the generalization review is authored."""

    old = parse_summary(description)
    expected_row = (
        PythonOverheadImpactRow
        if old.metadata.impact_metric == "python_overhead"
        else ImpactRow
    )
    if any(not isinstance(row, expected_row) for row in impact_rows):
        raise SummaryError(
            f"impact rows do not match the {old.metadata.impact_metric!r} "
            "impact metric this summary declares"
        )
    metadata = SummaryMetadata(
        operators=old.metadata.operators,
        state=state or old.metadata.state,
        baseline_commit=old.metadata.baseline_commit,
        candidate_commit=candidate_commit or old.metadata.candidate_commit,
        impact_metric=old.metadata.impact_metric,
        baseline_retention=old.metadata.baseline_retention,
        evidence=evidence if evidence is not None else old.metadata.evidence,
    )
    return generate_summary(
        metadata,
        scope_rows=scope_rows,
        impact_rows=impact_rows,
        layout_rows=layout_rows,
        local_skus=local_skus,
        # Missing or empty reviews need the template to make incomplete evidence visible.
        generalization_rows=old.generalization_rows or None,
        retention_detail=retention_detail,
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
    required = {"operators", "state", "baseline_commit", "candidate_commit", "evidence"}
    allowed = required | {"impact_metric", "baseline_retention"}
    if (
        not isinstance(data, dict)
        or not required.issubset(data)
        or not set(data).issubset(allowed)
    ):
        raise SummaryError(
            "metadata must contain exactly operators, state, baseline_commit, "
            "candidate_commit, and evidence, with optional impact_metric and "
            "baseline_retention"
        )
    if not isinstance(data["operators"], list):
        raise SummaryError("metadata operators must be an array")
    metadata = SummaryMetadata(
        operators=tuple(data["operators"]),
        state=data["state"],
        baseline_commit=data["baseline_commit"],
        candidate_commit=data["candidate_commit"],
        impact_metric=data.get("impact_metric", "cpp_time"),
        baseline_retention=data.get("baseline_retention"),
        evidence=data["evidence"],
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
    if metadata.baseline_retention is not None and (
        not isinstance(metadata.baseline_retention, str)
        or not re.fullmatch(r"[a-z][a-z0-9-]{0,95}", metadata.baseline_retention)
    ):
        raise SummaryError("metadata baseline_retention must name a reviewed policy")
    operators = metadata.operators
    if not 1 <= len(operators) <= MAX_OPERATORS:
        raise SummaryError(f"metadata operators must contain 1..{MAX_OPERATORS} names")
    if any(
        not isinstance(operator, str) or not _OPERATOR_RE.fullmatch(operator)
        for operator in operators
    ):
        raise SummaryError("metadata operators contains an invalid operator")
    normalized = [operator.casefold() for operator in operators]
    if len(set(normalized)) != len(normalized):
        raise SummaryError("metadata operators contains duplicate operators")
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
    for field, value in (
        ("baseline_commit", metadata.baseline_commit),
        ("candidate_commit", metadata.candidate_commit),
    ):
        if not isinstance(value, str) or not _SHA_RE.fullmatch(value):
            raise SummaryError(
                f"metadata {field} must be a full lowercase 40-character commit SHA"
            )
    if metadata.baseline_commit == metadata.candidate_commit:
        raise SummaryError("metadata baseline_commit must precede candidate_commit")
    try:
        evidence_mod.validate(metadata.evidence)
    except evidence_mod.EvidenceError as exc:
        raise SummaryError(f"metadata evidence is invalid: {exc}") from exc


def _parse_body(body: str, metadata: SummaryMetadata) -> dict[str, Any]:
    headings = [
        (match.group(1), match.group(2), match.start(), match.end())
        for match in re.finditer(r"^(#{2,3})\s+(.+?)\s*$", body, re.MULTILINE)
    ]
    expected = [SUMMARY_TITLE, *SECTION_HEADINGS]
    if not any(text == "Generalization review" for _, text, _, _ in headings):
        expected.remove("Generalization review")
    if [text for _, text, _, _ in headings] != expected:
        raise SummaryError(
            "visible headings must appear exactly once in canonical order: "
            + " -> ".join(expected)
        )
    if headings[0][0] != "##" or any(item[0] != "###" for item in headings[1:]):
        raise SummaryError("summary title must be ## and section headings must be ###")

    bounds = [item[2] for item in headings[1:]] + [len(body)]
    sections = {
        name: body[slice(headings[index][3], bounds[index])].strip()
        for index, name in enumerate(expected)
    }

    preamble = sections[SUMMARY_TITLE]
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
            r"^\*\*Primary bottleneck: "
            r"(Memory-bound|Compute-bound|Not yet profiled)\*\*$",
            preamble,
            re.MULTILINE,
        )
    )
    if len(bottleneck_matches) != 1:
        raise SummaryError("summary requires one exact Primary bottleneck line")
    profile = preamble[slice(bottleneck_matches[0].end(), None)].strip()
    if not profile:
        raise SummaryError("Primary bottleneck must be followed by profiler evidence")

    learnings_text = sections["Top learnings"]
    learnings = tuple(
        match.group(1).strip()
        for match in re.finditer(r"^- (.+?)\s*$", learnings_text, re.MULTILINE)
    )
    if not 1 <= len(learnings) <= 5:
        raise SummaryError("Top learnings must contain one to five bullets")
    if len([line for line in learnings_text.splitlines() if line.strip()]) != len(
        learnings
    ):
        raise SummaryError("Top learnings may contain only bullet lines")

    return {
        "status_line": ordinary_status[0],
        "warnings": tuple(
            line for line in status_lines if "Non-reference local GPU" in line
        ),
        "bottleneck": bottleneck_matches[0].group(1),
        "profile_evidence": profile,
        "scope_rows": _parse_scope_table(sections["Scope"]),
        "impact_rows": _parse_impact_table(sections["Impact"], metadata.impact_metric),
        "layout_rows": _parse_layout_table(sections["Layout comparison"]),
        "generalization_rows": (
            parse_generalization_review(sections["Generalization review"])
            if "Generalization review" in sections
            else ()
        ),
        "checklist": _parse_checklist(sections["Evidence checklist"], metadata),
        "learnings": learnings,
    }


def _scope_table_schema() -> tuple[str, str, str]:
    return (
        "Counted per configuration, not per measurement; Improved and Regressed "
        f"count configurations that moved at least {MOVED_PCT:g}% on any "
        "reported SKU.",
        "| Operator | Configurations | Improved | Regressed |",
        "|---|---:|---:|---:|",
    )


def _parse_scope_table(section: str) -> tuple[ScopeRow, ...]:
    rows = _validated_table_rows(section, _scope_table_schema(), "Scope")
    parsed = []
    for cells in rows[1:]:
        if len(cells) != 4:
            raise SummaryError("Scope table contains a malformed row")
        try:
            parsed.append(
                ScopeRow(cells[0], int(cells[1]), int(cells[2]), int(cells[3]))
            )
        except ValueError as exc:
            raise SummaryError(
                f"Scope table contains a malformed value: {exc}"
            ) from exc
    if not parsed:
        raise SummaryError("Scope table must contain at least one operator")
    return tuple(parsed)


def _header_cells(header: str) -> tuple[str, ...]:
    return tuple(cell.strip() for cell in header.strip().strip("|").split("|"))


def _validated_table_rows(
    section: str, schema: tuple[str, str, str], label: str
) -> list[tuple[str, ...]]:
    """Check a section carries only its canonical explanation and header."""

    explanation, header, _ = schema
    _require_table_only_section(section, explanation)
    rows = _table_rows(section)
    if not rows or rows[0] != _header_cells(header):
        raise SummaryError(f"{label} table header does not match the v2 schema")
    return rows


def _impact_table_schema(impact_metric: str) -> tuple[str, str, str]:
    if impact_metric == "cpp_time":
        return (
            "Speedup is `before / after`; `2.00x` means twice as fast.",
            "| Operator | SKU | Scope | Configurations | Min | Median | Max |",
            "|---|---|---|---:|---:|---:|---:|",
        )
    if impact_metric == "python_overhead":
        return (
            "Python overhead is `gpu_time_us_python - gpu_time_us_cpp`; "
            "reduction is `before - after`, so positive values mean less overhead.",
            "| Operator | SKU | Scope | Configurations | Before min / median / max | "
            "After min / median / max | Reduction min / median / max |",
            "|---|---|---|---:|---:|---:|---:|",
        )
    raise SummaryError(f"unsupported impact metric {impact_metric!r}")


def _parse_impact_table(
    section: str, impact_metric: str
) -> tuple[ImpactRow | PythonOverheadImpactRow, ...]:
    rows = _validated_table_rows(section, _impact_table_schema(impact_metric), "Impact")
    parsed = []
    for cells in rows[1:]:
        if len(cells) != len(rows[0]) or cells[2] not in IMPACT_SCOPES:
            raise SummaryError("Impact table contains a malformed row")
        try:
            if impact_metric == "cpp_time":
                parsed.append(
                    ImpactRow(
                        cells[0],
                        cells[1],
                        cells[2],
                        int(cells[3]),
                        _parse_factor(cells[4]),
                        _parse_factor(cells[5]),
                        _parse_factor(cells[6]),
                    )
                )
            else:
                parsed.append(
                    PythonOverheadImpactRow(
                        cells[0],
                        cells[1],
                        cells[2],
                        int(cells[3]),
                        _parse_duration_stats(cells[4]),
                        _parse_duration_stats(cells[5]),
                        _parse_duration_stats(cells[6]),
                    )
                )
        except ValueError as exc:
            raise SummaryError(
                f"Impact table contains a malformed value: {exc}"
            ) from exc
    if not parsed:
        raise SummaryError("Impact table must contain at least one operator")
    return tuple(parsed)


def _layout_table_schema() -> tuple[str, str, str]:
    return (
        "Timing ratio is `numerator / Planar`; values above `1.00x` mean Planar "
        "is faster.",
        "| Operator | SKU | Comparison | Before min / median / max | "
        "After min / median / max |",
        "|---|---|---|---:|---:|",
    )


def _parse_layout_table(section: str) -> tuple[LayoutRow, ...]:
    rows = _validated_table_rows(section, _layout_table_schema(), "Layout comparison")
    parsed = []
    for cells in rows[1:]:
        if len(cells) != 5 or cells[2] not in LAYOUT_COMPARISONS:
            raise SummaryError("Layout comparison table contains a malformed row")
        before = _parse_stats(cells[3])
        after = _parse_stats(cells[4])
        if (before is None) != (after is None):
            raise SummaryError("layout n/a must appear in both before and after cells")
        if before is None and cells[3] != cells[4]:
            raise SummaryError(
                "layout n/a note must be identical in the before and after cells"
            )
        note = cells[3] if before is None else NA_STATS
        parsed.append(LayoutRow(cells[0], cells[1], cells[2], before, after, note))
    if not parsed:
        raise SummaryError("Layout comparison table must contain at least one operator")
    return tuple(parsed)


def parse_generalization_review(section: str) -> tuple[GeneralizationRow, ...]:
    """Read the canonical review table, with an optional section heading."""

    lines = [line.strip() for line in section.splitlines() if line.strip()]
    if lines and lines[0] == "### Generalization review":
        lines = lines[1:]
    if any(not line.startswith("|") for line in lines):
        raise SummaryError("Generalization review may contain only its table")
    rows = _table_rows("\n".join(lines))
    if not rows or rows[0] != ("Area", "Assessment", "Decision"):
        raise SummaryError(
            "Generalization review table header must be Area | Assessment | Decision"
        )
    if len(_header_cells(lines[1])) != 3:
        raise SummaryError(
            "Generalization review table separator must have three columns"
        )
    if any(len(cells) != 3 for cells in rows[1:]):
        raise SummaryError("Generalization review table contains a malformed row")
    return tuple(GeneralizationRow(*cells) for cells in rows[1:])


def _table_rows(section: str) -> list[tuple[str, ...]]:
    table_lines = [
        line for line in section.splitlines() if line.strip().startswith("|")
    ]
    if len(table_lines) < 2:
        return []
    separator = tuple(
        cell.strip() for cell in table_lines[1].strip().strip("|").split("|")
    )
    if not all(re.fullmatch(r":?-{3,}:?", cell) for cell in separator):
        raise SummaryError("Markdown table separator is malformed")
    rows = []
    for line in [table_lines[0], *table_lines[2:]]:
        rows.append(
            tuple(
                cell.strip() for cell in re.split(r"(?<!\\)\|", line.strip().strip("|"))
            )
        )
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


def checklist_labels(metadata: SummaryMetadata) -> tuple[str, ...]:
    return tuple(
        (
            "Baselines verified unchanged"
            if metadata.baseline_retention and label == "Baselines updated"
            else label
        )
        for label in CHECKLIST_LABELS
    )


def _parse_checklist(
    section: str, metadata: SummaryMetadata
) -> tuple[ChecklistItem, ...]:
    pattern = re.compile(
        r"^- \[([ xX])\] \*\*([^*\r\n]+)\*\* — (.+?)\s*$", re.MULTILINE
    )
    matches = list(pattern.finditer(section))
    if [match.group(2) for match in matches] != list(checklist_labels(metadata)):
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
    if value in LAYOUT_NOTES:
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


def render_scope_table(rows: Sequence[ScopeRow]) -> list[str]:
    """Render the Scope section body: explanation, header, and rows."""
    explanation, header, separator = _scope_table_schema()
    return [explanation, "", header, separator, *[_render_scope_row(r) for r in rows]]


def render_impact_table(
    rows: Sequence[ImpactRow | PythonOverheadImpactRow],
    impact_metric: str = "cpp_time",
) -> list[str]:
    """Render the Impact section body: explanation, header, and rows."""
    explanation, header, separator = _impact_table_schema(impact_metric)
    return [explanation, "", header, separator, *[_render_impact_row(r) for r in rows]]


def render_layout_table(rows: Sequence[LayoutRow]) -> list[str]:
    """Render the Layout comparison section body: explanation, header, and rows."""
    explanation, header, separator = _layout_table_schema()
    return [explanation, "", header, separator, *[_render_layout_row(r) for r in rows]]


def _render_scope_row(row: ScopeRow) -> str:
    return (
        f"| {row.operator} | {row.configurations} | {row.improved} | {row.regressed} |"
    )


def _render_impact_row(row: ImpactRow | PythonOverheadImpactRow) -> str:
    if isinstance(row, PythonOverheadImpactRow):
        return (
            f"| {row.operator} | {row.sku} | {row.scope} | {row.configurations} | "
            f"{_duration_stats_cell(row.before)} | {_duration_stats_cell(row.after)} | "
            f"{_duration_stats_cell(row.reduction)} |"
        )
    return (
        f"| {row.operator} | {row.sku} | {row.scope} | {row.configurations} | "
        f"{_factor(row.minimum)} | {_factor(row.median)} | {_factor(row.maximum)} |"
    )


def _render_layout_row(row: LayoutRow) -> str:
    return (
        f"| {row.operator} | {row.sku} | {row.comparison} | "
        f"{_stats_cell(row.before, row.note)} | {_stats_cell(row.after, row.note)} |"
    )


def _factor(value: Decimal) -> str:
    return f"{value:.2f}x"


def _stats_cell(
    values: tuple[Decimal, Decimal, Decimal] | None, note: str = NA_STATS
) -> str:
    if values is None:
        return note
    return " / ".join(_factor(value) for value in values)


def _duration_stats_cell(values: tuple[Decimal, Decimal, Decimal]) -> str:
    return " / ".join(f"{value:.2f} µs" for value in values)


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


def _validate_visible_tables(
    summary: OptimizationSummary, errors: list[ValidationIssue]
) -> None:
    """Check the tables against each other and against the declared operators."""

    declared = list(summary.metadata.operators)
    for label, rows in (
        ("scope", summary.scope_rows),
        ("impact", summary.impact_rows),
        ("layout", summary.layout_rows),
    ):
        if _operators_reported(rows) != declared:
            errors.append(
                ValidationIssue(
                    f"{label}-operators",
                    f"{label} table must report every declared operator exactly "
                    "once, in declaration order",
                )
            )

    scope_by_operator = {row.operator: row for row in summary.scope_rows}
    for row in summary.impact_rows:
        scope = scope_by_operator.get(row.operator)
        if scope is None:
            continue
        ceiling = {
            ALL_SCOPE: scope.configurations,
            IMPROVED_SCOPE: scope.improved,
            REGRESSED_SCOPE: scope.regressed,
        }[row.scope]
        if row.configurations > ceiling:
            errors.append(
                ValidationIssue(
                    "impact-count",
                    f"{row.operator} {row.sku} {row.scope} counts "
                    f"{row.configurations} configurations, more than the "
                    f"{ceiling} Scope reports for the operator",
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
        for name, values in statistics:
            if not values[0] <= values[1] <= values[2]:
                errors.append(
                    ValidationIssue(
                        "impact-order",
                        f"{row.operator} {row.sku} {row.scope} {name} statistics "
                        "are not ordered",
                    )
                )

    for row in summary.scope_rows:
        if row.improved + row.regressed > row.configurations:
            errors.append(
                ValidationIssue(
                    "scope-count",
                    f"{row.operator} reports more moved configurations than it has",
                )
            )

    layout_seen: dict[tuple[str, str], list[str]] = {}
    for row in summary.layout_rows:
        layout_seen.setdefault((row.operator, row.sku), []).append(row.comparison)
        for name, values in (("before", row.before), ("after", row.after)):
            if values is not None and not values[0] <= values[1] <= values[2]:
                errors.append(
                    ValidationIssue(
                        "layout-order",
                        f"{row.operator} {row.sku} {row.comparison} {name} "
                        "statistics are not ordered",
                    )
                )
    for (operator, sku), comparisons in layout_seen.items():
        if comparisons != list(LAYOUT_COMPARISONS):
            errors.append(
                ValidationIssue(
                    "layout-rows",
                    f"{operator} {sku} must have exactly both canonical "
                    "layout-comparison rows",
                )
            )

    for label, rows, key in (
        ("impact", summary.impact_rows, lambda r: (r.operator, r.sku, r.scope)),
        ("layout", summary.layout_rows, lambda r: (r.operator, r.sku, r.comparison)),
        ("scope", summary.scope_rows, lambda r: (r.operator,)),
    ):
        keys = [key(row) for row in rows]
        if len(set(keys)) != len(keys):
            errors.append(
                ValidationIssue(
                    f"{label}-duplicate-rows",
                    f"{label} table repeats a row for the same operator and scope",
                )
            )

    impact_pairs = {(row.operator, row.sku) for row in summary.impact_rows}
    if impact_pairs != set(layout_seen):
        errors.append(
            ValidationIssue(
                "sku-rows",
                "Impact and Layout comparison must report the same operator/SKU pairs",
            )
        )


def _validate_rendered_evidence(
    summary: OptimizationSummary,
    retention_detail: str | None,
    errors: list[ValidationIssue],
    warnings: list[ValidationIssue],
) -> None:
    """The visible evidence must be exactly what the records render to."""

    meta = summary.metadata
    fields = rendered_fields(
        meta,
        generalization_rows=summary.generalization_rows,
        retention_detail=retention_detail,
    )
    final = meta.state == "final"
    for code, visible, expected in (
        ("bottleneck", summary.bottleneck, fields["bottleneck"]),
        ("profile-evidence", summary.profile_evidence, fields["profile_evidence"]),
        ("checklist", summary.checklist, fields["checklist"]),
        ("learnings", summary.learnings, fields["learnings"]),
    ):
        if visible != expected:
            errors.append(
                ValidationIssue(
                    f"{code}-rendering",
                    f"visible {code} differs from its rendering of the embedded "
                    "evidence; regenerate the summary instead of editing it",
                )
            )
    sections = memory_sections(summary.suffix)
    expected_memory = [fields["memory"]] if fields["memory"] else []
    if sections != expected_memory:
        errors.append(
            ValidationIssue(
                "memory-rendering",
                "Memory footprint section differs from its rendering of the "
                "embedded memory record",
            )
        )
    if meta.evidence.get("profile") is None:
        (errors if final else warnings).append(
            ValidationIssue("profile-missing", "no Nsight Compute profile is recorded")
        )
    if final:
        for item in summary.checklist:
            if not item.checked:
                errors.append(
                    ValidationIssue(
                        "checklist-unchecked", f"{item.label}: {item.evidence}"
                    )
                )
        if summary.learnings == (DEFAULT_LEARNING,):
            errors.append(
                ValidationIssue("learnings-missing", "no attempt has been recorded")
            )


def applied_decision(evidence: Mapping[str, Any]) -> str | None:
    """The only accepted Applied disposition: the recorded passing test run."""

    tests = evidence.get("tests")
    if not tests:
        return None
    return f"Applied: validated by the recorded test run at `{tests['commit'][:12]}`."


def validate_generalization_rows(
    rows: Sequence[GeneralizationRow],
    evidence: Mapping[str, Any] | None = None,
) -> tuple[ValidationIssue, ...]:
    """Require an evidenced assessment and disposition for every review area."""

    issues = []
    present = {row.area for row in rows}
    for area in GENERALIZATION_AREAS:
        if area not in present:
            issues.append(
                ValidationIssue(
                    "generalization-area", f"Generalization review is missing {area}"
                )
            )
    for row in rows:
        if row.area not in GENERALIZATION_AREAS:
            issues.append(
                ValidationIssue(
                    "generalization-area",
                    f"Unknown generalization review area: {row.area}",
                )
            )
        if _PLACEHOLDER_RE.search(row.assessment) or not re.search(
            r"`[^`\r\n]+`|\b[\w/-]+\.(?:cu|cuh|cpp|hpp|h|py|json)\b|"
            r"\b(?:ncu|nsys)\b[^\n]*\d+(?:\.\d+)?\s*(?:%|(?:us|µs|ms|x)\b)",
            row.assessment,
            re.IGNORECASE,
        ):
            issues.append(
                ValidationIssue(
                    "generalization-assessment",
                    f"{row.area} requires a concrete assessment citing code, "
                    "configuration, or profile evidence",
                )
            )
        decision = re.fullmatch(
            r"(Applied|Deferred|Not applicable):\s*(.+)", row.decision
        )
        if (
            not decision
            or not decision.group(2).strip()
            or _PLACEHOLDER_RE.search(row.decision)
            or decision.group(2).strip().rstrip(".!;").casefold()
            in {"none", "n/a", "not applicable", "nothing to do", "no opportunities"}
        ):
            issues.append(
                ValidationIssue(
                    "generalization-decision",
                    f"{row.area} requires Applied: validation, Deferred: scope reason, "
                    "or Not applicable: reason",
                )
            )
        elif decision.group(1) == "Applied" and (
            evidence is None or row.decision != applied_decision(evidence)
        ):
            issues.append(
                ValidationIssue(
                    "generalization-validation",
                    f"{row.area} Applied decision must cite the recorded test run: "
                    f"{applied_decision(evidence or {}) or 'no tests recorded'}",
                )
            )
    return tuple(issues)


def remaining_generalizations_deferred(
    rows: Sequence[GeneralizationRow], evidence: Mapping[str, Any]
) -> bool:
    """A completed review that explicitly defers at least one opportunity."""

    return (
        bool(rows)
        and not validate_generalization_rows(rows, evidence)
        and any(row.decision.startswith("Deferred:") for row in rows)
    )
