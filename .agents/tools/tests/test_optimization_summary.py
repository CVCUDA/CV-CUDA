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

from __future__ import annotations

import json
import re
import sys
from dataclasses import replace
from decimal import Decimal as D
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / ".agents" / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import evidence_fixtures as fixtures  # noqa: E402

from optimization_summary import (  # noqa: E402
    ALL_SCOPE,
    AMBIGUOUS_STATS,
    GENERALIZATION_AREAS,
    IMPROVED_SCOPE,
    NA_STATS,
    REGRESSED_SCOPE,
    GeneralizationRow,
    ImpactRow,
    LayoutRow,
    PythonOverheadImpactRow,
    ScopeRow,
    SummaryError,
    SummaryMetadata,
    _PLACEHOLDER_RE,
    applied_decision,
    generate_summary,
    parse_summary,
    parse_generalization_review,
    refresh_summary,
    remaining_generalizations_deferred,
    validate_generalization_rows,
    validate_summary,
)

BASE = "1" * 40
CANDIDATE = "2" * 40


def test_retention_metadata_requires_truthful_checklist_label():
    original = _render()
    retained = original.replace(
        '"state":"final"', '"state":"final","baseline_retention":"reviewed-example"'
    )
    retained = retained.replace(
        "**Baselines updated**", "**Baselines verified unchanged**"
    )
    summary = parse_summary(retained)
    assert summary.metadata.baseline_retention == "reviewed-example"
    assert summary.checklist[2].label == "Baselines verified unchanged"
    with pytest.raises(SummaryError):
        parse_summary(
            retained.replace(
                "**Baselines verified unchanged**", "**Baselines updated**"
            )
        )
    with pytest.raises(SummaryError):
        parse_summary(
            original.replace(
                "**Baselines updated**", "**Baselines verified unchanged**"
            )
        )


def _metadata(
    *, operators=("Example",), state="final", impact_metric="cpp_time", evidence=None
):
    return SummaryMetadata(
        operators=operators,
        state=state,
        baseline_commit=BASE,
        candidate_commit=CANDIDATE,
        impact_metric=impact_metric,
        evidence=fixtures.complete() if evidence is None else evidence,
    )


def _scope(operators=("Example",)):
    return tuple(ScopeRow(name, 20, 5, 1) for name in operators)


def _impact(operators=("Example",), skus=("A100",)):
    rows = []
    for name in operators:
        for sku in skus:
            rows.extend(
                [
                    ImpactRow(
                        name, sku, ALL_SCOPE, 20, D("0.90"), D("1.20"), D("2.40")
                    ),
                    ImpactRow(
                        name, sku, IMPROVED_SCOPE, 5, D("1.10"), D("1.60"), D("2.40")
                    ),
                    ImpactRow(
                        name, sku, REGRESSED_SCOPE, 1, D("0.90"), D("0.90"), D("0.90")
                    ),
                ]
            )
    return tuple(rows)


def _layout(operators=("Example",), skus=("A100",)):
    rows = []
    for name in operators:
        for sku in skus:
            rows.append(
                LayoutRow(
                    name,
                    sku,
                    "Interleaved / Planar",
                    (D("0.80"), D("0.95"), D("1.10")),
                    (D("1.10"), D("1.40"), D("1.70")),
                )
            )
            rows.append(LayoutRow(name, sku, "FakePlanar / Planar", None, None))
    return tuple(rows)


def _generalization():
    return (
        GeneralizationRow(
            GENERALIZATION_AREAS[0],
            "Vector loads in `TensorKernel` can extend to the `VarShape` path.",
            "Deferred: VarShape dispatch changes are outside this MR.",
        ),
        GeneralizationRow(
            GENERALIZATION_AREAS[1],
            "The `LoadStore<T>` helper already handles U8 and F32 channels.",
            "Not applicable: All supported dtypes use the same specialization.",
        ),
        GeneralizationRow(
            GENERALIZATION_AREAS[2],
            "The large-shape branch in `LaunchKernel` now uses the same vector loads.",
            applied_decision(fixtures.complete()),
        ),
        GeneralizationRow(
            GENERALIZATION_AREAS[3],
            "The staging loop in SiblingOp.cu could use the same vector loads.",
            "Deferred: The sibling operator requires a separate MR.",
        ),
    )


def _render(*, metadata=None, scope=None, impact=None, layout=None, **kwargs):
    metadata = metadata or _metadata()
    operators = metadata.operators
    return generate_summary(
        metadata,
        scope_rows=scope if scope is not None else _scope(operators),
        impact_rows=impact if impact is not None else _impact(operators),
        layout_rows=layout if layout is not None else _layout(operators),
        generalization_rows=kwargs.pop("generalization_rows", _generalization()),
        **kwargs,
    )


# --------------------------------------------------------------------- metadata


def test_metadata_round_trips_every_declared_operator():
    text = _render(
        metadata=_metadata(operators=("Example", "Sibling")),
        scope=_scope(("Example", "Sibling")),
        impact=_impact(("Example", "Sibling")),
        layout=_layout(("Example", "Sibling")),
    )
    parsed = parse_summary(text)

    assert parsed.metadata.operators == ("Example", "Sibling")
    assert parsed.metadata.primary == "Example"


def test_single_operator_metadata_omits_the_default_impact_metric():
    block = parse_summary(_render()).block
    payload = json.loads(block.splitlines()[0].split(" ", 2)[2].rsplit(" -->", 1)[0])

    assert payload == {
        "operators": ["Example"],
        "state": "final",
        "baseline_commit": BASE,
        "candidate_commit": CANDIDATE,
        "evidence": fixtures.complete(),
    }


@pytest.mark.parametrize(
    ("operators", "match"),
    [
        ((), "1..64"),
        (("Example", "example"), "duplicate"),
        (("Not An Operator",), "invalid operator"),
    ],
)
def test_invalid_operator_lists_are_rejected(operators, match):
    metadata = SummaryMetadata(
        operators=operators,
        state="final",
        baseline_commit=BASE,
        candidate_commit=CANDIDATE,
    )
    with pytest.raises(SummaryError, match=match):
        _render(metadata=metadata)


def test_a_baseline_equal_to_the_candidate_is_rejected():
    with pytest.raises(SummaryError, match="must precede"):
        _render(
            metadata=SummaryMetadata(
                operators=("Example",),
                state="final",
                baseline_commit=BASE,
                candidate_commit=BASE,
            )
        )


# ------------------------------------------------------------------ round trip


def test_every_table_is_keyed_by_operator():
    text = _render()

    assert "| Operator | Configurations | Improved | Regressed |" in text
    assert "| Operator | SKU | Scope | Configurations | Min | Median | Max |" in text
    assert "| Operator | SKU | Comparison |" in text


def test_a_clean_summary_validates_against_the_rows_it_was_built_from():
    scope, impact, layout = _scope(), _impact(), _layout()
    text = _render(scope=scope, impact=impact, layout=layout)

    result = validate_summary(
        text,
        scope_rows=scope,
        impact_rows=impact,
        layout_rows=layout,
        reference_skus=["A100"],
        expected_operators=("Example",),
    )
    assert result.ok, result.errors


def test_a_regressed_scope_survives_the_round_trip():
    parsed = parse_summary(_render())
    regressed = [row for row in parsed.impact_rows if row.scope == REGRESSED_SCOPE]

    assert len(regressed) == 1
    assert regressed[0].median == D("0.90")


# ------------------------------------------------------------------- tampering


@pytest.mark.parametrize(
    ("old", "new", "code"),
    [
        ("| Example | 20 | 5 | 1 |", "| Example | 20 | 9 | 1 |", "scope-statistics"),
        (
            "| Example | A100 | Improved | 5 | 1.10x | 1.60x | 2.40x |",
            "| Example | A100 | Improved | 5 | 1.10x | 2.30x | 2.40x |",
            "impact-statistics",
        ),
        (
            "| Example | A100 | Interleaved / Planar | 0.80x / 0.95x / 1.10x "
            "| 1.10x / 1.40x / 1.70x |",
            "| Example | A100 | Interleaved / Planar | 0.80x / 0.95x / 1.10x "
            "| 1.10x / 1.65x / 1.70x |",
            "layout-statistics",
        ),
    ],
)
def test_a_figure_the_baselines_do_not_support_is_rejected(old, new, code):
    scope, impact, layout = _scope(), _impact(), _layout()
    text = _render(scope=scope, impact=impact, layout=layout)
    tampered = text.replace(old, new)
    assert tampered != text, old

    result = validate_summary(
        tampered, scope_rows=scope, impact_rows=impact, layout_rows=layout
    )
    assert not result.ok
    assert any(item.code == code for item in result.errors), result.errors


def test_an_impact_count_that_contradicts_scope_is_rejected():
    text = _render().replace(
        "| Example | A100 | Improved | 5 |", "| Example | A100 | Improved | 6 |"
    )
    result = validate_summary(text)

    assert any(item.code == "impact-count" for item in result.errors), result.errors


def test_unordered_statistics_are_rejected():
    text = _render().replace(
        "| Example | A100 | Improved | 5 | 1.10x | 1.60x | 2.40x |",
        "| Example | A100 | Improved | 5 | 1.10x | 2.60x | 2.40x |",
    )
    result = validate_summary(text)

    assert any(item.code == "impact-order" for item in result.errors), result.errors


# ------------------------------------------------------- operator completeness


def test_a_declared_operator_missing_from_a_table_is_refused_at_render():
    with pytest.raises(SummaryError, match="declaration order"):
        _render(
            metadata=_metadata(operators=("Example", "Sibling")),
            scope=_scope(("Example", "Sibling")),
            impact=_impact(("Example",)),
            layout=_layout(("Example", "Sibling")),
        )


def test_tables_must_agree_on_the_operator_and_sku_pairs_they_report():
    scope = _scope(("Example",))
    impact = _impact(("Example",), skus=("A100", "H100"))
    layout = _layout(("Example",), skus=("A100",))

    with pytest.raises(SummaryError, match="same operator/SKU pairs"):
        _render(scope=scope, impact=impact, layout=layout)


def test_layout_requires_both_canonical_comparisons():
    layout = tuple(row for row in _layout() if row.comparison != "FakePlanar / Planar")
    with pytest.raises(SummaryError, match="both canonical"):
        _render(layout=layout)


def test_layout_notes_survive_the_round_trip():
    layout = (
        _layout()[0],
        LayoutRow(
            "Example", "A100", "FakePlanar / Planar", None, None, AMBIGUOUS_STATS
        ),
    )
    parsed = parse_summary(_render(layout=layout))

    assert parsed.layout_rows[1].note == AMBIGUOUS_STATS
    assert parsed.layout_rows[1].before is None
    assert AMBIGUOUS_STATS != NA_STATS


# --------------------------------------------------------------- sku coverage


def test_a_final_summary_missing_a_reference_sku_is_rejected():
    result = validate_summary(_render(), reference_skus=["A100", "H100"])

    assert any(item.code == "reference-coverage" for item in result.errors)


def test_a_non_reference_sku_must_carry_its_warning():
    text = _render()
    result = validate_summary(text, reference_skus=["H100"])

    assert any(item.code == "local-sku-warning" for item in result.errors)


def test_a_declared_local_sku_warning_satisfies_the_check():
    text = _render(local_skus=["A100"])
    result = validate_summary(text, reference_skus=["H100"])

    assert not any(item.code == "local-sku-warning" for item in result.errors)


# ----------------------------------------------------------------- other state


def test_a_provisional_summary_says_so():
    text = _render(metadata=_metadata(state="provisional"))

    assert "Provisional local evidence" in text
    assert validate_summary(text).ok


def test_python_overhead_uses_its_own_impact_schema():
    metadata = _metadata(impact_metric="python_overhead")
    triple = (D("1.00"), D("2.00"), D("3.00"))
    impact = tuple(
        PythonOverheadImpactRow("Example", "A100", scope, count, triple, triple, triple)
        for scope, count in ((ALL_SCOPE, 20), (IMPROVED_SCOPE, 5), (REGRESSED_SCOPE, 1))
    )
    text = _render(metadata=metadata, impact=impact)

    assert "Reduction min / median / max" in text
    assert parse_summary(text).metadata.impact_metric == "python_overhead"


def test_refresh_preserves_authored_prose_and_replaces_the_numbers():
    text = _render()
    moved = tuple(
        ImpactRow(
            row.operator,
            row.sku,
            row.scope,
            row.configurations,
            D("1.00"),
            D("1.00"),
            D("1.00"),
        )
        for row in _impact()
    )
    refreshed = refresh_summary(
        text, scope_rows=_scope(), impact_rows=moved, layout_rows=_layout()
    )
    parsed = parse_summary(refreshed)

    assert parsed.learnings == parse_summary(text).learnings
    assert parsed.bottleneck == "Memory-bound"
    assert parsed.generalization_rows == parse_summary(text).generalization_rows
    assert all(row.median == D("1.00") for row in parsed.impact_rows)


# ------------------------------------------------------------------- evidence


def test_cpp_template_syntax_is_not_read_as_an_unfilled_placeholder():
    for text in (
        "static_cast<float>(v)",
        "cuda::MakeType<unsigned int, Bytes / 4>",
        "std::is_floating_point_v<T> excludes __half",
    ):
        assert not _PLACEHOLDER_RE.search(text), text

    for text in ("<fill in the measured speedup>", "see <path/to/report>", "TODO"):
        assert _PLACEHOLDER_RE.search(text), text


def test_refresh_refuses_rows_that_contradict_the_declared_metric():
    """Both tables have seven columns, so a mismatch renders rather than crashes."""

    overhead = _metadata(impact_metric="python_overhead")
    triple = (D("1.00"), D("2.00"), D("3.00"))
    text = _render(
        metadata=overhead,
        impact=tuple(
            PythonOverheadImpactRow(
                "Example", "A100", scope, count, triple, triple, triple
            )
            for scope, count in (
                (ALL_SCOPE, 20),
                (IMPROVED_SCOPE, 5),
                (REGRESSED_SCOPE, 1),
            )
        ),
    )

    with pytest.raises(SummaryError, match="do not match the 'python_overhead'"):
        refresh_summary(
            text,
            scope_rows=_scope(),
            impact_rows=_impact(),
            layout_rows=_layout(),
        )


@pytest.mark.parametrize(
    ("prefix", "code"),
    [
        ("| Example | A100 | Improved |", "impact-duplicate-rows"),
        ("| Example | A100 | Interleaved / Planar |", "layout-duplicate-rows"),
        ("| Example | 20 |", "scope-duplicate-rows"),
    ],
)
def test_a_repeated_row_is_refused_without_recomputing_the_numbers(prefix, code):
    """CI recomputation caught these; a local gate run did not."""

    text = _render()
    row = next(line for line in text.splitlines() if line.startswith(prefix))
    tampered = text.replace(row, row + "\n" + row)
    assert tampered != text

    result = validate_summary(tampered)

    assert not result.ok
    assert any(item.code == code for item in result.errors), result.errors


def _without_generalization(text):
    return re.sub(
        r"### Generalization review\n.*?(?=### Evidence checklist)",
        "",
        text,
        flags=re.DOTALL,
    )


def test_final_summary_requires_generalization_review():
    result = validate_summary(_without_generalization(_render()))

    assert not result.ok
    assert {issue.code for issue in result.errors} == {"generalization-area"}


def test_provisional_missing_review_refreshes_to_all_four_template_rows():
    text = _without_generalization(_render(metadata=_metadata(state="provisional")))
    result = validate_summary(text)
    assert result.ok
    assert any(issue.code == "generalization-area" for issue in result.warnings)

    refreshed = refresh_summary(
        text, scope_rows=_scope(), impact_rows=_impact(), layout_rows=_layout()
    )
    rows = parse_summary(refreshed).generalization_rows
    assert tuple(row.area for row in rows) == GENERALIZATION_AREAS
    assert all(_PLACEHOLDER_RE.search(row.assessment) for row in rows)


def test_generalization_template_warns_in_draft_and_blocks_final():
    text = _render(metadata=_metadata(state="provisional"), generalization_rows=None)
    result = validate_summary(text)
    assert result.ok
    assert any(issue.code == "generalization-assessment" for issue in result.warnings)

    with pytest.raises(SummaryError, match="generalization-assessment"):
        _render(generalization_rows=None)


def test_generalization_requires_every_area_but_allows_multiple_opportunities():
    rows = _generalization()
    assert not validate_generalization_rows(
        evidence=fixtures.complete(),
        rows=(
            *rows,
            replace(
                rows[0],
                assessment="The tail in `TensorKernel` can also use scalar loads.",
            ),
        ),
    )
    for index, area in enumerate(GENERALIZATION_AREAS):
        issues = validate_generalization_rows(
            rows[:index] + rows[slice(index + 1, None)]
        )
        assert any(
            issue.code == "generalization-area" and area in issue.message
            for issue in issues
        )
    issues = validate_generalization_rows((*rows, replace(rows[0], area="Anything")))
    assert any(issue.code == "generalization-area" for issue in issues)


def test_generalization_accepts_concise_assessments_and_scope_reasons():
    rows = (
        replace(
            _generalization()[0],
            assessment="`K` can vectorize.",
            decision="Deferred: F32 scope.",
        ),
        *_generalization()[1:],
    )
    assert not validate_generalization_rows(rows, fixtures.complete())


@pytest.mark.parametrize(
    "assessment",
    [
        "",
        "None",
        "No generalization opportunities were found.",
        "<cite evidence>",
        "TODO: inspect `TensorKernel`",
    ],
)
def test_generalization_rejects_unsubstantiated_assessments(assessment):
    rows = (
        replace(_generalization()[0], assessment=assessment),
        *_generalization()[1:],
    )
    assert any(
        issue.code == "generalization-assessment"
        for issue in validate_generalization_rows(rows)
    )


@pytest.mark.parametrize(
    "decision",
    [
        "",
        "Deferred",
        "Deferred:",
        "Deferred:   ",
        "Deferred: none",
        "Deferred: TODO",
        "Deferred: nothing to do",
        "Not applicable: no opportunities",
        "Not applicable: Not applicable.",
        "Applied: everything is fine now",
    ],
)
def test_generalization_rejects_unsupported_decisions(decision):
    rows = (replace(_generalization()[0], decision=decision), *_generalization()[1:])
    assert any(
        issue.code.startswith("generalization-")
        for issue in validate_generalization_rows(rows)
    )


def test_generalization_table_parser_accepts_the_cli_section_format():
    text = _render()
    section = text.split("### Generalization review\n", 1)[1].split(
        "### Evidence checklist", 1
    )[0]
    assert parse_generalization_review(section) == _generalization()
    assert (
        parse_generalization_review("### Generalization review\n" + section)
        == _generalization()
    )


def test_generalization_preserves_escaped_pipes_through_refresh():
    rows = (
        replace(
            _generalization()[0],
            assessment=r"`flags \| MASK` in Kernel.cu also covers planar inputs.",
            decision=r"Deferred: `run_bench \| tee` scope is a follow-up.",
        ),
        *_generalization()[1:],
    )
    text = _render(generalization_rows=rows)
    assert parse_summary(text).generalization_rows == rows
    refreshed = refresh_summary(
        text, scope_rows=_scope(), impact_rows=_impact(), layout_rows=_layout()
    )
    assert parse_summary(refreshed).generalization_rows == rows
    assert validate_summary(refreshed).ok


def test_empty_generalization_table_is_incomplete_evidence():
    assert (
        parse_generalization_review("| Area | Assessment | Decision |\n|---|---|---|")
        == ()
    )
    assert validate_generalization_rows(())


@pytest.mark.parametrize(
    "section",
    [
        "| Area | Assessment | Verdict |\n|---|---|---|\n| A | B | C |",
        "| Area | Assessment | Decision |\n|---|---|---|\n| A | B |",
        "| Area | Assessment | Decision |\n|---|---|\n| A | B | C |",
        "Review completed.\n| Area | Assessment | Decision |\n|---|---|---|",
    ],
)
def test_generalization_table_rejects_malformed_structure(section):
    with pytest.raises(SummaryError, match="Generalization review"):
        parse_generalization_review(section)


def test_generalization_table_rejects_malformed_separator():
    with pytest.raises(SummaryError, match="separator"):
        parse_generalization_review(
            "| Area | Assessment | Decision |\n| nope |---|---|\n| A | B | C |"
        )


def test_generalization_assessment_accepts_measured_profile_evidence():
    rows = (
        replace(
            _generalization()[0],
            assessment="ncu showed 92% Memory SOL across both input paths.",
        ),
        *_generalization()[1:],
    )
    assert not validate_generalization_rows(rows, fixtures.complete())


def _unexhausted():
    evidence = fixtures.complete()
    evidence["profile"]["memory_pct"] = 40.0
    return evidence


def test_lead_exhaustion_accepts_a_completed_review_with_a_deferral():
    evidence = _unexhausted()
    assert remaining_generalizations_deferred(_generalization(), evidence)
    text = _render(metadata=_metadata(evidence=evidence))
    item = parse_summary(text).checklist[4]
    assert item.checked and "deferred" in item.evidence
    assert validate_summary(text).ok


def test_lead_deferral_requires_a_completed_review_and_explicit_scope():
    evidence = _unexhausted()
    assert not remaining_generalizations_deferred(_generalization()[:1], evidence)
    rows = tuple(
        replace(
            row,
            decision="Not applicable: Every supported path already uses this specialization.",
        )
        for row in _generalization()
    )
    assert not remaining_generalizations_deferred(rows, evidence)
    with pytest.raises(SummaryError, match="checklist-unchecked"):
        _render(metadata=_metadata(evidence=evidence), generalization_rows=rows)


# --------------------------------------------------------------- v3 evidence


def test_v2_blocks_are_rejected():
    text = _render().replace("summary:v3", "summary:v2")
    with pytest.raises(SummaryError, match="start marker"):
        parse_summary(text)


def test_metadata_without_evidence_is_rejected():
    text = _render()
    start = text.index('"evidence":')
    end = text.index(',"operators"') + 1
    with pytest.raises(SummaryError, match="evidence"):
        parse_summary(text[:start] + text[end:])


def test_malformed_evidence_is_rejected():
    text = _render().replace('"exit_code":0', '"exit_code":"0"', 1)
    with pytest.raises(SummaryError, match="evidence is invalid"):
        parse_summary(text)


@pytest.mark.parametrize(
    ("old", "new", "code"),
    [
        (
            "**Primary bottleneck: Memory-bound**",
            "**Primary bottleneck: Compute-bound**",
            "bottleneck-rendering",
        ),
        ("at 93.0% Memory SOL", "at 99.0% Memory SOL", "profile-evidence-rendering"),
        (
            "attributable peak increase 0 B",
            "no memory growth at all",
            "checklist-rendering",
        ),
        (
            "- [x] **Review/refactor gate**",
            "- [ ] **Review/refactor gate**",
            "checklist-rendering",
        ),
        ("was kept at 1.50x", "was kept at 3.00x", "learnings-rendering"),
        (
            "Peak attributable increase: 0 B",
            "Peak attributable increase: 1 B",
            "memory-rendering",
        ),
    ],
)
def test_hand_edited_evidence_lines_are_rejected(old, new, code):
    text = _render()
    assert old in text
    result = validate_summary(text.replace(old, new))
    assert code in {issue.code for issue in result.errors}


def test_bottleneck_is_derived_from_the_profile_record():
    evidence = fixtures.complete()
    evidence["profile"].update(memory_pct=30.0, compute_pct=95.0)
    parsed = parse_summary(_render(metadata=_metadata(evidence=evidence)))
    assert parsed.bottleneck == "Compute-bound"


def test_missing_profile_renders_a_pending_bottleneck():
    evidence = fixtures.complete()
    del evidence["profile"]
    metadata = _metadata(state="provisional", evidence=evidence)
    parsed = parse_summary(_render(metadata=metadata))
    assert parsed.bottleneck == "Not yet profiled"


def test_learnings_are_drafted_from_attempts_kept_first_by_speedup():
    evidence = fixtures.complete()
    evidence["attempts"] = [
        fixtures.attempt("struck", 0.9, "Wider blocks"),
        fixtures.attempt("kept", 1.2, "Hoisted index math"),
        fixtures.attempt("kept", 2.4, "Byte carriers"),
    ]
    learnings = parse_summary(_render(metadata=_metadata(evidence=evidence))).learnings
    assert learnings == (
        "Byte carriers was kept at 2.40x median (2.40x to 2.40x over 4 cases).",
        "Hoisted index math was kept at 1.20x median (1.20x to 1.20x over 4 cases).",
        "Wider blocks was struck at 0.90x median over 4 cases.",
    )


def test_a_final_summary_with_an_unmet_gate_cannot_render():
    evidence = fixtures.complete()
    evidence["tests"]["cases"][0]["outcome"] = "failed"
    with pytest.raises(SummaryError, match="checklist-unchecked"):
        _render(metadata=_metadata(evidence=evidence))
    draft = _render(metadata=_metadata(state="provisional", evidence=evidence))
    assert not parse_summary(draft).checklist[0].checked


def test_refresh_replaces_evidence_and_rerenders_every_derived_line():
    draft = _render(
        metadata=_metadata(
            state="provisional",
            evidence=fixtures.with_(fixtures.complete(), memory=None, attempts=None),
        )
    )
    assert "## Memory footprint" not in draft
    refreshed = refresh_summary(
        draft,
        scope_rows=_scope(),
        impact_rows=_impact(),
        layout_rows=_layout(),
        state="final",
        evidence=fixtures.complete(),
    )
    assert validate_summary(refreshed).ok
    assert refreshed.count("## Memory footprint") == 1
