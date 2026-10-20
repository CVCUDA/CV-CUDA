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
"""Contract/smoke tests for the optimization DoD checker (tools/optimize_op.py).

Guards the harness's promises — determinism, the GAP/MANUAL exit contract, correct
changed-implementation attribution (incl. the cross-op CamelCase-prefix trap), and
results-summary format enforcement — without coupling to live per-op state. Stdlib-only;
runs in the bench Python unit-test CI step alongside tools/tests/test_review_op.py.
"""

import importlib.util
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / ".agents" / "tools" / "tests")
)
import evidence_fixtures as fixtures  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
TOOL = REPO / "tools" / "optimize_op.py"
VALID = {"PASS", "GAP", "N-A", "MANUAL"}


def run(*args):
    return subprocess.run(
        [sys.executable, str(TOOL), *args],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=60,
    )


def _module():
    spec = importlib.util.spec_from_file_location("optimize_op", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _synthetic_reviewed_scope(mr_iid, primary, secondary):
    return (
        str(mr_iid),
        str(primary).casefold(),
        tuple(sorted(operator.casefold() for operator in secondary)),
    ) == ("123", "osd", ("bndbox",))


# ---- CLI contract -----------------------------------------------------------------------
def test_preflight_runs_and_reports():
    r = run("Resize", "--phase", "preflight", "--format", "json")
    data = json.loads(r.stdout)
    assert data["phase"] == "preflight" and data["findings"]
    assert all(f["status"] in VALID for f in data["findings"])


def test_phases_are_deterministic():
    for phase in ("preflight", "evidence"):
        a = run("Resize", "--phase", phase, "--format", "json")
        b = run("Resize", "--phase", phase, "--format", "json")
        assert a.stdout == b.stdout, f"{phase} not deterministic"


def test_exit_code_matches_blocking_contract():
    r = run("Resize", "--phase", "evidence", "--format", "json")
    findings = json.loads(r.stdout)["findings"]
    assert all(f["status"] in VALID for f in findings)
    has_blocker = any(f["status"] in {"GAP", "MANUAL"} for f in findings)
    assert r.returncode in (0, 1)
    assert (r.returncode == 1) == has_blocker


def test_evidence_exit_blocks_unresolved_manual(monkeypatch):
    m = _module()
    monkeypatch.setattr(
        m,
        "check_evidence",
        lambda *_args: [m.Finding("ODO-X", "evidence", m.MANUAL, "needs proof")],
    )
    assert m.main(["Resize", "--phase", "evidence"]) == 1


def _mock_summary_checkout(monkeypatch, module, event_type, target_available=True):
    source = "a" * 40
    candidate = "b" * 40 if event_type in {"merged_result", "merge_train"} else source
    target = "c" * 40
    diff_base = "d" * 40
    for name in (
        "CI_MERGE_REQUEST_EVENT_TYPE",
        "CI_COMMIT_SHA",
        "CI_MERGE_REQUEST_SOURCE_BRANCH_SHA",
        "CI_MERGE_REQUEST_TARGET_BRANCH_SHA",
        "CI_MERGE_REQUEST_DIFF_BASE_SHA",
    ):
        monkeypatch.delenv(name, raising=False)
    if event_type:
        monkeypatch.setenv("CI_MERGE_REQUEST_EVENT_TYPE", event_type)
        monkeypatch.setenv("CI_COMMIT_SHA", candidate)
        monkeypatch.setenv("CI_MERGE_REQUEST_SOURCE_BRANCH_SHA", source)
        monkeypatch.setenv("CI_MERGE_REQUEST_DIFF_BASE_SHA", diff_base)
        if target_available:
            monkeypatch.setenv("CI_MERGE_REQUEST_TARGET_BRANCH_SHA", target)
    monkeypatch.setattr(
        module,
        "resolve_commit",
        lambda ref: (
            candidate
            if ref == "HEAD"
            else ref if ref in {source, candidate, target, diff_base} else ""
        ),
    )
    monkeypatch.setattr(module, "git_success", lambda *_args: True)
    monkeypatch.setattr(
        module,
        "git",
        lambda *args: (
            diff_base if args == ("merge-base", "origin/main", "HEAD") else ""
        ),
    )
    expected_target = target if candidate != source and target_available else diff_base
    return source, candidate, expected_target


@pytest.mark.parametrize("authorized", (True, False))
@pytest.mark.parametrize("stale", (False, True))
@pytest.mark.parametrize(
    ("event_type", "target_available"),
    [
        (None, True),
        ("detached", True),
        ("merged_result", True),
        ("merge_train", True),
        ("merged_result", False),
    ],
)
@pytest.mark.parametrize("invalid_retention", [False, True])
def test_odo2_reference_exception_resolves_only_its_checklist_gate(
    monkeypatch, authorized, stale, event_type, target_available, invalid_retention
):
    module = _module()
    source, candidate, target = _mock_summary_checkout(
        monkeypatch, module, event_type, target_available
    )
    paths = module.resolve_op("CopyMakeBorder")
    monkeypatch.setattr(module, "all_operator_stems", lambda: {"copymakeborder"})
    monkeypatch.setattr(
        module, "changed_paths", lambda _base: ["src/cvcuda/priv/OpCopyMakeBorder.cu"]
    )
    monkeypatch.setattr(module, "read_results", lambda _path: "summary")
    monkeypatch.setattr(
        module,
        "parse_summary",
        lambda _text: SimpleNamespace(
            metadata=SimpleNamespace(
                operators=("CopyMakeBorder",),
                candidate_commit="e" * 40 if stale else source,
                baseline_retention="invalid" if invalid_retention else None,
                evidence={},
            )
        ),
    )
    if invalid_retention:

        def rejected(*_args):
            raise ValueError("invalid retention proof")

        monkeypatch.setattr(module, "_retention_evidence", rejected)
    policy_calls = []

    def reviewed(summary, **kwargs):
        policy_calls.append(kwargs)
        return authorized

    monkeypatch.setattr(module, "is_reviewed_reference_exception", reviewed)
    monkeypatch.setattr(module, "review_op_findings", lambda *_args: [])
    for function, finding_id in (
        ("_leads_exhausted", "ODO-4"),
        ("_api_abi", "ODO-5"),
        ("_perf_hygiene", "ODO-6"),
        ("_baseline_regression_gate", "ODO-7"),
        ("_review_refactor_gate", "ODO-8"),
        ("_memory_footprint_gate", "ODO-9"),
        ("_operator_scope_gate", "ODO-10"),
        ("_fake_planar_coverage_gate", "ODO-11"),
        ("_binding_impact_gate", "ODO-12"),
    ):
        monkeypatch.setattr(
            module,
            function,
            lambda *_args, finding_id=finding_id: module.Finding(
                finding_id, "evidence", module.PASS, "ok"
            ),
        )

    def results_format(_results, guide, _retention):
        return module.Finding(
            "ODO-3", "evidence", module.PASS, "format", guideline=guide
        )

    monkeypatch.setattr(module, "_results_format", results_format)
    findings = module.check_evidence(
        paths, "origin/main", "summary.md", mr_iid="reviewed"
    )
    assert next(item for item in findings if item.id == "ODO-2").status == (
        module.NA if authorized and not stale and not invalid_retention else module.GAP
    )
    assert next(item for item in findings if item.id == "ODO-1").status == module.GAP
    if stale or invalid_retention:
        assert not policy_calls
        return
    assert len(policy_calls) == 1
    assert policy_calls[0]["mr_iid"] == "reviewed"
    assert policy_calls[0]["source_commit"] == source
    assert policy_calls[0]["candidate_commit"] == candidate
    assert policy_calls[0]["diff_base_commit"] == target


@pytest.mark.parametrize(
    "event_type", [None, "detached", "merged_result", "merge_train"]
)
@pytest.mark.parametrize("stale", [False, True])
def test_retention_evidence_binds_source_and_merge_context(
    monkeypatch, event_type, stale
):
    module = _module()
    source, candidate, _target = _mock_summary_checkout(monkeypatch, module, event_type)
    summary = SimpleNamespace(
        metadata=SimpleNamespace(
            operators=("BndBox", "OSD"),
            baseline_retention="reviewed",
            baseline_commit="d" * 40,
            candidate_commit="e" * 40 if stale else source,
        )
    )
    import baseline_retention_policy

    calls = []

    def validate(**kwargs):
        calls.append(kwargs)
        return "reviewed proof"

    monkeypatch.setattr(baseline_retention_policy, "validate_retention", validate)
    if stale:
        with pytest.raises(ValueError, match="match source HEAD"):
            module._retention_evidence(summary, "2003")
        assert not calls
    else:
        assert module._retention_evidence(summary, "2003") == "reviewed proof"
        assert calls[0]["source_commit"] == source
        assert calls[0]["candidate_commit"] == candidate
        assert calls[0]["operators"] == ("BndBox", "OSD")


@pytest.mark.parametrize(
    "event_type", (None, "detached", "merged_result", "merge_train")
)
@pytest.mark.parametrize("stale", (False, True))
def test_results_format_binds_summary_to_source_context(monkeypatch, event_type, stale):
    module = _module()
    source, _candidate, _target = _mock_summary_checkout(
        monkeypatch, module, event_type
    )
    summary = SimpleNamespace(
        metadata=SimpleNamespace(
            operators=("CopyMakeBorder",),
            baseline_commit="d" * 40,
            candidate_commit="e" * 40 if stale else source,
        ),
        scope_rows=(),
    )
    monkeypatch.setattr(module, "parse_summary", lambda _text: summary)

    def validate(parsed, *, expected_candidate_commit, retention_detail):
        ok = parsed.metadata.candidate_commit == expected_candidate_commit
        return SimpleNamespace(
            ok=ok,
            errors=(
                ()
                if ok
                else (
                    SimpleNamespace(code="candidate-commit", message="stale summary"),
                )
            ),
            warnings=(),
        )

    monkeypatch.setattr(module, "validate_summary", validate)
    finding = module._results_format("summary", "guide", None)
    assert finding.status == (module.GAP if stale else module.PASS)


@pytest.mark.parametrize(
    "fault",
    (
        "missing_source",
        "missing_target",
        "wrong_checkout",
        "unknown_source",
        "unknown_target",
        "source_not_ancestor",
        "target_not_ancestor",
    ),
)
def test_results_format_rejects_unbound_merge_context(monkeypatch, fault):
    module = _module()
    source, _candidate, target = _mock_summary_checkout(
        monkeypatch, module, "merged_result"
    )
    summary = SimpleNamespace(
        metadata=SimpleNamespace(
            operators=("CopyMakeBorder",),
            baseline_commit="d" * 40,
            candidate_commit=source,
        ),
        scope_rows=(),
    )
    monkeypatch.setattr(module, "parse_summary", lambda _text: summary)
    monkeypatch.setattr(
        module,
        "validate_summary",
        lambda *_args, **_kwargs: SimpleNamespace(ok=True, errors=(), warnings=()),
    )
    if fault == "missing_source":
        monkeypatch.delenv("CI_MERGE_REQUEST_SOURCE_BRANCH_SHA")
    elif fault == "missing_target":
        monkeypatch.delenv("CI_MERGE_REQUEST_TARGET_BRANCH_SHA")
        monkeypatch.delenv("CI_MERGE_REQUEST_DIFF_BASE_SHA")
    elif fault == "wrong_checkout":
        monkeypatch.setenv("CI_COMMIT_SHA", "f" * 40)
    elif fault == "unknown_source":
        monkeypatch.setenv("CI_MERGE_REQUEST_SOURCE_BRANCH_SHA", "f" * 40)
    elif fault == "unknown_target":
        monkeypatch.setenv("CI_MERGE_REQUEST_TARGET_BRANCH_SHA", "f" * 40)
    else:
        rejected = source if fault == "source_not_ancestor" else target
        monkeypatch.setattr(module, "git_success", lambda *args: args[2] != rejected)
    assert module._results_format("summary", "guide", {}).status == module.GAP


def test_reference_exception_is_denied_without_internal_module(monkeypatch):
    original = Path.is_file
    monkeypatch.setattr(
        Path,
        "is_file",
        lambda path: (
            False if path.name == "optimization_reference_policy.py" else original(path)
        ),
    )
    module = _module()
    assert not module.is_reviewed_reference_exception(object(), mr_iid="reviewed")


def test_phase_is_required_and_validated():
    assert run("Resize").returncode == 2  # missing --phase
    assert run("Resize", "--phase", "bogus").returncode == 2  # invalid choice


def test_preflight_ben6_gap_blocks_pre2(monkeypatch):
    m = _module()
    paths = m.resolve_op("Invert")

    def findings(_paths, domain):
        if domain == "bench":
            return [
                {
                    "id": "BEN-6",
                    "status": "GAP",
                    "summary": "1 unmatched FakePlanar case",
                }
            ]
        return []

    monkeypatch.setattr(m, "review_op_findings", findings)
    monkeypatch.setattr(m, "read", lambda _path: '{"baselines":{"case":{}}}')

    pre2 = next(
        item for item in m.check_preflight(paths, "origin/main") if item.id == "PRE-2"
    )

    assert pre2.status == "GAP"
    assert "BEN-6" in pre2.evidence


def test_final_evidence_reruns_ben6_and_blocks_orphan_fake_planar(monkeypatch):
    m = _module()
    paths = m.resolve_op("Invert")

    monkeypatch.setattr(m, "all_operator_stems", lambda: {"invert"})
    monkeypatch.setattr(m, "changed_paths", lambda _base: [])
    monkeypatch.setattr(m, "read_results", lambda _path: "")
    monkeypatch.setattr(
        m,
        "review_op_findings",
        lambda _paths, domain: (
            [
                {
                    "id": "BEN-6",
                    "status": "GAP",
                    "summary": "1 unmatched FakePlanar case",
                }
            ]
            if domain == "bench"
            else []
        ),
    )
    monkeypatch.setattr(
        m,
        "_leads_exhausted",
        lambda *_args: m.Finding("ODO-4", "evidence", m.PASS, "ok"),
    )
    monkeypatch.setattr(
        m,
        "_api_abi",
        lambda *_args: m.Finding("ODO-5", "evidence", m.PASS, "ok"),
    )
    monkeypatch.setattr(
        m,
        "_perf_hygiene",
        lambda *_args: m.Finding("ODO-6", "evidence", m.PASS, "ok"),
    )
    monkeypatch.setattr(
        m,
        "_baseline_regression_gate",
        lambda *_args: m.Finding("ODO-7", "evidence", m.PASS, "ok"),
    )
    monkeypatch.setattr(
        m,
        "_review_refactor_gate",
        lambda *_args: m.Finding("ODO-8", "evidence", m.PASS, "ok"),
    )
    monkeypatch.setattr(
        m,
        "_memory_footprint_gate",
        lambda *_args: m.Finding("ODO-9", "evidence", m.PASS, "ok"),
    )
    monkeypatch.setattr(m, "_changed_perf_operators", lambda *_args: {"invert"})
    monkeypatch.setattr(
        m,
        "_results_format",
        lambda *_args: m.Finding("ODO-3", "evidence", m.PASS, "ok"),
    )

    evidence = m.check_evidence(paths, "origin/main", None)
    ben6 = next(item for item in evidence if item.id == "ODO-11")

    assert ben6.status == "GAP"
    assert "BEN-6" in ben6.evidence


def test_final_fake_planar_gate_does_not_autopass_unresolved_status(monkeypatch):
    m = _module()
    paths = m.resolve_op("Invert")

    for status in (m.MANUAL, "UNEXPECTED"):
        monkeypatch.setattr(
            m,
            "review_op_findings",
            lambda *_args, status=status: [
                {
                    "id": "BEN-6",
                    "status": status,
                    "summary": "coverage result needs review",
                }
            ],
        )

        finding = m._fake_planar_coverage_gate(paths, "guide")

        assert finding.status == m.MANUAL
        assert repr(status) in finding.evidence
        assert "coverage result needs review" in finding.evidence


# ---- changed-implementation attribution (deterministic, no git) -------------------------
def test_kernel_attribution_longest_prefix():
    m = _module()
    ops = m.all_operator_stems()
    resize = m.resolve_op("Resize")
    hq = m.resolve_op("HQResize")
    assert m._is_op_kernel("src/cvcuda/priv/OpResize.cu", resize, ops) is True
    # cross-op trap: a longer operator name must not be attributed to the shorter prefix
    assert (
        m._is_op_kernel("src/cvcuda/priv/OpResizeCropConvertReformat.cu", resize, ops)
        is False
    )
    assert m._is_op_kernel("src/cvcuda/priv/OpHQResizeKernel.cuh", resize, ops) is False
    assert m._is_op_kernel("src/cvcuda/priv/OpHQResizeKernel.cuh", hq, ops) is True
    # non-priv paths never count
    assert m._is_op_kernel("bench/config/operators/resize.json", resize, ops) is False


def test_odo10_accepts_synthetic_reviewed_secondary_scope(monkeypatch):
    m = _module()
    osd = m.resolve_op("OSD")
    monkeypatch.setattr(m, "is_reviewed_secondary_scope", _synthetic_reviewed_scope)
    monkeypatch.setattr(
        m,
        "parse_summary",
        lambda _results: SimpleNamespace(
            metadata=SimpleNamespace(primary="OSD", operators=("OSD", "BndBox")),
            impact_rows=(
                SimpleNamespace(operator="OSD"),
                SimpleNamespace(operator="BndBox"),
            ),
        ),
    )

    finding = m._operator_scope_gate(osd, {"osd", "bndbox"}, "summary", "123", "guide")

    assert finding.status == m.PASS
    assert "code-reviewed secondary" in finding.summary


@pytest.mark.parametrize("proof", [True, False])
@pytest.mark.parametrize("foreign", [True, False])
def test_odo10_retained_shared_scope_requires_verified_proof(
    monkeypatch, proof, foreign
):
    m = _module()
    paths = m.resolve_op("OSD")
    monkeypatch.setattr(m, "is_reviewed_secondary_scope", _synthetic_reviewed_scope)
    monkeypatch.setattr(
        m,
        "parse_summary",
        lambda _: SimpleNamespace(
            metadata=SimpleNamespace(
                primary="OSD",
                operators=("OSD", "BndBox"),
                baseline_retention="reviewed",
            ),
            impact_rows=(
                SimpleNamespace(operator="OSD"),
                SimpleNamespace(operator="BndBox"),
            ),
        ),
    )
    monkeypatch.setattr(m, "_retention_evidence", lambda *_: "proof" if proof else None)
    finding = m._operator_scope_gate(
        paths, {"other"} if foreign else set(), "summary", "123", "guide"
    )
    assert finding.status == (m.PASS if proof and not foreign else m.GAP)


def test_odo10_rejects_a_summary_for_a_different_operator(monkeypatch):
    """ODO-3 used to carry this cross-check via expected_operator."""

    m = _module()
    osd = m.resolve_op("OSD")
    monkeypatch.setattr(
        m,
        "parse_summary",
        lambda _results: SimpleNamespace(
            metadata=SimpleNamespace(primary="BndBox", operators=("BndBox",)),
            impact_rows=(SimpleNamespace(operator="BndBox"),),
        ),
    )

    finding = m._operator_scope_gate(osd, {"osd"}, "summary", None, "guide")

    assert finding.status == m.GAP
    assert "different operator" in finding.summary
    assert "BndBox" in finding.evidence


def test_odo10_requires_measured_impact_for_every_declared_secondary(monkeypatch):
    m = _module()
    osd = m.resolve_op("OSD")
    monkeypatch.setattr(m, "is_reviewed_secondary_scope", _synthetic_reviewed_scope)
    monkeypatch.setattr(
        m,
        "parse_summary",
        lambda _results: SimpleNamespace(
            metadata=SimpleNamespace(primary="OSD", operators=("OSD", "BndBox")),
            impact_rows=(SimpleNamespace(operator="OSD"),),
        ),
    )

    finding = m._operator_scope_gate(osd, {"osd", "bndbox"}, "summary", "123", "guide")

    assert finding.status == m.GAP
    assert "Impact" in finding.evidence


def test_odo10_rejects_partial_declared_secondary_coverage(monkeypatch):
    m = _module()
    osd = m.resolve_op("OSD")
    monkeypatch.setattr(m, "is_reviewed_secondary_scope", _synthetic_reviewed_scope)
    monkeypatch.setattr(
        m,
        "parse_summary",
        lambda _results: SimpleNamespace(
            metadata=SimpleNamespace(
                primary="OSD", operators=("OSD", "BndBox", "Warp")
            ),
            impact_rows=(
                SimpleNamespace(operator="OSD"),
                SimpleNamespace(operator="BndBox"),
            ),
        ),
    )

    finding = m._operator_scope_gate(
        osd, {"osd", "bndbox", "warp"}, "summary", "123", "guide"
    )

    assert finding.status == m.GAP
    assert "Impact" in finding.evidence


def test_odo10_secondary_scope_fails_closed(monkeypatch):
    m = _module()
    osd = m.resolve_op("OSD")
    monkeypatch.setattr(m, "is_reviewed_secondary_scope", _synthetic_reviewed_scope)

    cases = (
        ("122", "OSD", ("BndBox",), {"osd", "bndbox"}),
        ("123", "BndBox", ("OSD",), {"osd", "bndbox"}),
        ("123", "OSD", ("BndBox",), {"osd"}),
        ("123", "OSD", ("BndBox",), {"osd", "bndbox", "warp"}),
        ("123", "OSD", ("Warp",), {"osd", "warp"}),
    )
    for mr_iid, metadata_operator, secondary, touched in cases:
        monkeypatch.setattr(
            m,
            "parse_summary",
            lambda _results, metadata_operator=metadata_operator, secondary=secondary: (
                SimpleNamespace(
                    metadata=SimpleNamespace(
                        primary=metadata_operator,
                        operators=(metadata_operator, *secondary),
                    ),
                    impact_rows=tuple(
                        SimpleNamespace(operator=name)
                        for name in (metadata_operator, *secondary)
                    ),
                )
            ),
        )

        finding = m._operator_scope_gate(osd, touched, "summary", mr_iid, "guide")

        assert finding.status == m.GAP, (mr_iid, metadata_operator, secondary, touched)


def test_odo10_default_scope_remains_strict(monkeypatch):
    m = _module()
    osd = m.resolve_op("OSD")
    monkeypatch.setattr(
        m,
        "parse_summary",
        lambda _results: SimpleNamespace(
            metadata=SimpleNamespace(primary="OSD", operators=("OSD",)),
            impact_rows=(SimpleNamespace(operator="OSD"),),
        ),
    )

    assert (
        m._operator_scope_gate(osd, {"osd"}, "summary", None, "guide").status == m.PASS
    )
    assert (
        m._operator_scope_gate(osd, {"osd", "bndbox"}, "summary", None, "guide").status
        == m.GAP
    )


def test_implementation_attribution_includes_only_the_matching_binding():
    m = _module()
    ops = m.all_operator_stems()
    resize = m.resolve_op("Resize")
    hq = m.resolve_op("HQResize")

    assert m._is_op_implementation(
        "python/mod_cvcuda/operators/OpHQResize.cpp", hq, ops
    )
    assert not m._is_op_implementation(
        "python/mod_cvcuda/operators/OpHQResize.cpp", resize, ops
    )
    assert m._is_op_implementation("src/cvcuda/priv/OpHQResizeKernel.cuh", hq, ops)


def test_kernel_attribution_explicit_shared_sources():
    """A shared source is attributed to every consumer even when its name derives none of them.

    The fixtures used to be legacy/warp.cu and legacy/filter.cu; both operators migrated, so the
    shared kernels now live in priv headers. The property under test is unchanged: attribution
    comes from the source map, not from the filename.
    """
    m = _module()
    ops = m.all_operator_stems()
    warp_affine = m.resolve_op("WarpAffine")
    warp_perspective = m.resolve_op("WarpPerspective")
    gaussian = m.resolve_op("Gaussian")
    average_blur = m.resolve_op("AverageBlur")

    shared_warp = "src/cvcuda/priv/WarpCommon.cuh"
    assert m._is_op_kernel(shared_warp, warp_affine, ops) is True
    assert m._is_op_kernel(shared_warp, warp_perspective, ops) is True

    shared_filter = "src/cvcuda/priv/FilterCommon.cuh"
    assert m._is_op_kernel(shared_filter, gaussian, ops) is True
    assert m._is_op_kernel(shared_filter, average_blur, ops) is True


@pytest.mark.parametrize("operator", ["Split", "Merge"])
@pytest.mark.parametrize("suffix", ["cu", "hpp"])
def test_channel_copy_implementation_attribution(operator, suffix):
    module = _module()
    operators = module.all_operator_stems()
    paths = module.resolve_op(operator)
    source = f"src/cvcuda/priv/ChannelCopy.{suffix}"

    assert module._is_op_kernel(source, paths, operators)
    assert module._is_op_implementation(source, paths, operators)
    assert not module._is_op_implementation(
        source, module.resolve_op("Resize"), operators
    )


def test_multi_operator_perf_attribution_uses_configs_and_longest_priv_prefix():
    m = _module()
    ops = m.all_operator_stems()
    touched = m._changed_perf_operators(
        [
            "bench/config/operators/resize.json",
            "bench/config/operators/warpaffine.json",
            "src/cvcuda/priv/OpHQResizeKernel.cuh",
            "python/mod_cvcuda/operators/OpPillowResize.cpp",
            "docs/sphinx/api/python.rst",
        ],
        ops,
    )
    assert touched == {"resize", "warpaffine", "hqresize", "pillowresize"}


@pytest.mark.parametrize("retention", [False, True])
def test_binding_only_change_triggers_implementation_evidence_gates(
    monkeypatch, retention
):
    m = _module()
    hq = m.resolve_op("HQResize")
    binding = "python/mod_cvcuda/operators/OpHQResize.cpp"
    reviewed = []

    monkeypatch.setattr(m, "all_operator_stems", lambda: {"hqresize"})
    monkeypatch.setattr(m, "changed_paths", lambda _base: [binding])
    evidence = fixtures.complete()
    monkeypatch.setattr(m, "read_results", lambda _path: "summary")
    monkeypatch.setattr(
        m,
        "parse_summary",
        lambda _text: SimpleNamespace(
            metadata=SimpleNamespace(
                evidence=evidence,
                candidate_commit=fixtures.CANDIDATE,
                primary="HQResize",
                operators=("HQResize",),
                baseline_retention=None,
            ),
            impact_rows=(),
        ),
    )
    monkeypatch.setattr(
        m,
        "_binding_impact_gate",
        lambda *_args: m.Finding("ODO-12", "evidence", m.PASS, "ok"),
    )
    monkeypatch.setattr(
        m,
        "_imported_baselines",
        lambda ev: m.evidence_mod.GateResult(m.PASS, "imported"),
    )
    monkeypatch.setattr(m, "review_op_findings", lambda *_args: [])
    monkeypatch.setattr(
        m,
        "git",
        lambda *args: (
            '+                "gpu_gap_stddev_us": 4.25'
            if args[0] == "diff" and not retention
            else ""
        ),
    )
    monkeypatch.setattr(
        m,
        "_retention_evidence",
        lambda *_args: "reviewed proof" if retention else None,
        raising=False,
    )
    monkeypatch.setattr(
        m,
        "_leads_exhausted",
        lambda *_args: m.Finding("ODO-4", "evidence", m.PASS, "ok"),
    )
    monkeypatch.setattr(
        m,
        "_api_abi",
        lambda *_args: m.Finding("ODO-5", "evidence", m.PASS, "ok"),
    )
    monkeypatch.setattr(
        m,
        "_perf_hygiene",
        lambda *_args: m.Finding("ODO-6", "evidence", m.PASS, "ok"),
    )
    monkeypatch.setattr(
        m,
        "_baseline_regression_gate",
        lambda *_args: m.Finding("ODO-7", "evidence", m.PASS, "ok"),
    )

    def review_gate(_paths, _results, changed, _guide):
        reviewed.extend(changed)
        return m.Finding("ODO-8", "evidence", m.PASS, "ok")

    monkeypatch.setattr(m, "_review_refactor_gate", review_gate)
    monkeypatch.setattr(m, "_changed_perf_operators", lambda *_args: {"hqresize"})
    monkeypatch.setattr(
        m,
        "_results_format",
        lambda *_args: m.Finding("ODO-3", "evidence", m.PASS, "ok"),
    )
    monkeypatch.setattr(
        m,
        "_fake_planar_coverage_gate",
        lambda *_args: m.Finding("ODO-11", "evidence", m.PASS, "ok"),
    )

    findings = m.check_evidence(hq, "origin/main", None)

    assert not any(item.id == "ODO-0" for item in findings)
    assert next(item for item in findings if item.id == "ODO-1").status == "PASS"
    assert next(item for item in findings if item.id == "ODO-2").status == "PASS"
    assert next(item for item in findings if item.id == "ODO-9").status == "PASS"
    assert reviewed == [binding]


def test_perf_hygiene_treats_binding_changes_as_implementation(monkeypatch):
    m = _module()
    hq = m.resolve_op("HQResize")
    binding = "python/mod_cvcuda/operators/OpHQResize.cpp"
    sha = "1" * 40
    merge_base = "2" * 40

    for subject, expected in (
        ("fix(hqresize): cache requirements", "GAP"),
        ("perf(hqresize): cache requirements", "PASS"),
    ):

        def fake_git(*args, subject=subject):
            if args[0] == "merge-base":
                return merge_base + "\n"
            if args[0] == "log":
                assert args[1] == f"{merge_base}..HEAD"
                assert "--no-merges" in args
                return f"{sha}\x1f{subject}\n"
            if args[0] == "show":
                return binding + "\n"
            return ""

        monkeypatch.setattr(m, "git", fake_git)
        result = m._perf_hygiene(hq, "origin/main", {"hqresize"}, "guide")
        assert result.status == expected, subject


def test_perf_hygiene_excludes_target_only_commits_from_diverged_base(monkeypatch):
    m = _module()
    hq = m.resolve_op("HQResize")
    branch_sha = "1" * 40
    target_sha = "2" * 40
    merge_base = "3" * 40
    binding = "python/mod_cvcuda/operators/OpHQResize.cpp"

    def fake_git(*args):
        if args[0] == "merge-base":
            return merge_base + "\n"
        if args[0] == "log":
            assert args[1] == f"{merge_base}..HEAD"
            assert "--no-merges" in args
            return f"{branch_sha}\x1fperf(hqresize): cache requirements\n"
        if args[0] == "show":
            assert args[-1] != target_sha
            return binding + "\n"
        return ""

    monkeypatch.setattr(m, "git", fake_git)

    result = m._perf_hygiene(hq, target_sha, {"hqresize"}, "guide")

    assert result.status == "PASS"


def _binding_impact_config(
    case_keys,
    *,
    gap_us,
    gap_stddev_us=2.0,
    noise_us=2.0,
    n_runs=5,
    include_gap_stddev=True,
):
    configs = {}
    for case_key in case_keys:
        config_key = case_key.split("[", 1)[0]
        baselines = {}
        for sku in ("A100", "H100"):
            metrics = {
                "n_runs": n_runs,
                "gpu_time_us_cpp": 1000.0,
                "gpu_time_us_python": 1000.0 + gap_us,
                "gpu_noise_us_cpp": noise_us,
                "gpu_noise_us_python": noise_us,
            }
            if include_gap_stddev:
                metrics["gpu_gap_stddev_us"] = gap_stddev_us
            baselines[sku] = metrics
        configs[config_key] = {
            "tier": "advanced",
            "baselines": {
                case_key: baselines,
            },
        }
    return {"benchmark": "hqresize", "configs": configs}


def _binding_impact_inputs(
    monkeypatch,
    module,
    *,
    reduction_us,
    gap_stddev_us=2.0,
    noise_us=2.0,
    n_runs=5,
    include_gap_stddev=True,
    reviewed_deferral=False,
):
    from optimization_summary import GENERALIZATION_AREAS, GeneralizationRow

    cases = (
        "target_nhwc[layout=NHWC][inputKind=TensorBatch]",
        "target_nchw[layout=NCHW][inputKind=TensorBatch]",
    )
    metadata = SimpleNamespace(
        impact_metric="python_overhead",
        baseline_commit="a" * 40,
        optimized_cases=cases,
    )
    baseline = _binding_impact_config(
        cases,
        gap_us=40.0,
        gap_stddev_us=gap_stddev_us,
        noise_us=noise_us,
        n_runs=n_runs,
        include_gap_stddev=include_gap_stddev,
    )
    candidate = _binding_impact_config(
        cases,
        gap_us=40.0 - reduction_us,
        gap_stddev_us=gap_stddev_us,
        noise_us=noise_us,
        n_runs=n_runs,
        include_gap_stddev=include_gap_stddev,
    )
    sku_map = {"entries": [{"stem": "A100"}, {"stem": "H100"}]}

    monkeypatch.setattr(
        module,
        "parse_summary",
        lambda _results: SimpleNamespace(
            metadata=metadata,
            generalization_rows=(
                tuple(
                    GeneralizationRow(
                        area,
                        "The dispatch in `HQResize` could apply the Tensor fast path to NCHW.",
                        "Deferred: keep this MR scoped to the NHWC binding path.",
                    )
                    for area in GENERALIZATION_AREAS
                )
                if reviewed_deferral
                else ()
            ),
        ),
    )
    monkeypatch.setattr(module, "git_json", lambda *_args: baseline)
    monkeypatch.setattr(
        module,
        "_load_json",
        lambda path: sku_map if str(path).endswith("sku_map.json") else candidate,
    )
    return cases


def test_binding_impact_gate_requires_every_target_to_clear_combined_error(monkeypatch):
    m = _module()
    hq = m.resolve_op("HQResize")
    cases = _binding_impact_inputs(monkeypatch, m, reduction_us=10.0, noise_us=100.0)

    finding = m._binding_impact_gate(
        hq,
        "summary",
        ["python/mod_cvcuda/operators/OpHQResize.cpp"],
        "guide",
    )

    expected_error = math.sqrt(2 * (2.0**2) / 5)
    assert finding.status == "PASS"
    assert f"{expected_error:.2f} us" in finding.evidence
    assert "paired-gap combined SE" in finding.evidence
    assert all(case.split("[", 1)[0] in finding.evidence for case in cases)


def test_binding_impact_gate_rejects_claim_within_paired_gap_error(monkeypatch):
    m = _module()
    hq = m.resolve_op("HQResize")
    _binding_impact_inputs(monkeypatch, m, reduction_us=1.0)

    finding = m._binding_impact_gate(
        hq,
        "summary",
        ["python/mod_cvcuda/operators/OpHQResize.cpp"],
        "guide",
    )

    assert finding.status == "GAP"
    assert "does not clear" in finding.evidence


def test_binding_sibling_deferral_cannot_waive_unchanged_layout(monkeypatch):
    m = _module()
    from optimization_summary import (
        GENERALIZATION_AREAS,
        GeneralizationRow,
        validate_generalization_rows,
    )

    cases = _binding_impact_inputs(monkeypatch, m, reduction_us=10.0)
    summary = m.parse_summary("summary")
    summary.generalization_rows = tuple(
        GeneralizationRow(
            area,
            (
                "`SiblingOp.cpp` could reuse this dispatch."
                if area == "Shared code / sibling operators"
                else "`HQResize` uses the same binding dispatch across this area."
            ),
            (
                "Deferred: keep sibling operators in a separate MR."
                if area == "Shared code / sibling operators"
                else "Not applicable: these paths already use the same dispatch."
            ),
        )
        for area in GENERALIZATION_AREAS
    )
    assert not validate_generalization_rows(summary.generalization_rows)
    monkeypatch.setattr(m, "parse_summary", lambda _: summary)
    candidate = m._load_json(m.resolve_op("HQResize").bench_cfg)
    for metrics in candidate["configs"]["target_nchw"]["baselines"][cases[1]].values():
        metrics["gpu_time_us_python"] = 1040.0

    finding = m._binding_impact_gate(
        m.resolve_op("HQResize"),
        "summary",
        ["python/mod_cvcuda/operators/OpHQResize.cpp"],
        "guide",
    )
    assert finding.status == "GAP", finding.evidence
    assert "target_nchw" in finding.evidence
    assert "does not clear standard error" in finding.evidence


@pytest.mark.parametrize(
    "deferred_reduction, dispersion, complete_review, expected",
    [
        (10.0, 2.0, True, "PASS"),
        (0.0, 2.0, True, "GAP"),
        (0.5, 2.0, True, "GAP"),
        (-0.5, 2.0, True, "GAP"),
        (-3.0, 2.0, True, "GAP"),
        (3.0, 10.0, True, "GAP"),
        (0.0, 2.0, False, "GAP"),
    ],
)
def test_binding_impact_requires_each_reviewed_case_to_clear_error(
    monkeypatch, deferred_reduction, dispersion, complete_review, expected
):
    m = _module()
    cases = _binding_impact_inputs(
        monkeypatch, m, reduction_us=10.0, reviewed_deferral=True
    )
    summary = m.parse_summary("summary")
    if not complete_review:
        summary.generalization_rows = summary.generalization_rows[:-1]
    monkeypatch.setattr(m, "parse_summary", lambda _: summary)
    candidate = m._load_json(m.resolve_op("HQResize").bench_cfg)
    for metrics in candidate["configs"]["target_nchw"]["baselines"][cases[1]].values():
        metrics["gpu_time_us_python"] = 1040.0 - deferred_reduction
        metrics["gpu_gap_stddev_us"] = dispersion

    finding = m._binding_impact_gate(
        m.resolve_op("HQResize"),
        "summary",
        ["python/mod_cvcuda/operators/OpHQResize.cpp"],
        "guide",
    )
    assert finding.status == expected, finding.evidence


@pytest.mark.parametrize(
    "fault", ["missing_candidate", "missing_before", "no_win", "no_cases", "parity"]
)
def test_binding_deferral_retains_full_coverage_proven_impact_and_parity(
    monkeypatch, fault
):
    m = _module()
    cases = _binding_impact_inputs(
        monkeypatch, m, reduction_us=10.0, reviewed_deferral=True
    )
    candidate = m._load_json(m.resolve_op("HQResize").bench_cfg)
    if fault == "missing_candidate":
        del candidate["configs"]["target_nchw"]
    elif fault == "missing_before":
        del m.git_json()["configs"]["target_nchw"]["baselines"][cases[1]]["H100"]
    elif fault == "no_cases":
        candidate["configs"].clear()
        m.git_json()["configs"].clear()
    else:
        for entry in candidate["configs"].values():
            for by_sku in entry["baselines"].values():
                for metrics in by_sku.values():
                    metrics["gpu_time_us_python"] = (
                        1040.0 if fault == "no_win" else 1070.0
                    )

    finding = m._binding_impact_gate(
        m.resolve_op("HQResize"),
        "summary",
        ["python/mod_cvcuda/operators/OpHQResize.cpp"],
        "guide",
    )
    assert finding.status == "GAP", finding.evidence


def test_binding_impact_gate_requires_paired_gap_dispersion(monkeypatch):
    m = _module()
    hq = m.resolve_op("HQResize")
    _binding_impact_inputs(
        monkeypatch,
        m,
        reduction_us=10.0,
        include_gap_stddev=False,
    )

    finding = m._binding_impact_gate(
        hq,
        "summary",
        ["python/mod_cvcuda/operators/OpHQResize.cpp"],
        "guide",
    )

    assert finding.status == "GAP"
    assert "paired-gap-dispersion" in finding.evidence


@pytest.mark.parametrize("n_runs", [1, 5])
@pytest.mark.parametrize("reduction_us, expected", [(1.0, "PASS"), (0.5, "GAP")])
def test_binding_impact_gate_uses_committed_timings_without_historical_dispersion(
    monkeypatch, n_runs, reduction_us, expected
):
    m = _module()
    cases = _binding_impact_inputs(monkeypatch, m, reduction_us=reduction_us)
    baseline = _binding_impact_config(
        cases, gap_us=40.0, n_runs=n_runs, include_gap_stddev=False
    )
    monkeypatch.setattr(m, "git_json", lambda *_args: baseline)

    finding = m._binding_impact_gate(
        m.resolve_op("HQResize"),
        "summary",
        ["python/mod_cvcuda/operators/OpHQResize.cpp"],
        "guide",
    )

    assert finding.status == expected
    assert "candidate SE vs fixed committed reference" in finding.evidence
    assert f"{2.0 / math.sqrt(5):.2f} us" in finding.evidence


@pytest.mark.parametrize("reduction_us, expected", [(1.0, "GAP"), (2.0, "PASS")])
def test_binding_impact_gate_retains_available_historical_dispersion(
    monkeypatch, reduction_us, expected
):
    m = _module()
    cases = _binding_impact_inputs(monkeypatch, m, reduction_us=reduction_us)
    baseline = _binding_impact_config(cases, gap_us=40.0)
    for entry in baseline["configs"].values():
        for by_sku in entry["baselines"].values():
            del by_sku["A100"]["gpu_gap_stddev_us"]
    monkeypatch.setattr(m, "git_json", lambda *_args: baseline)

    finding = m._binding_impact_gate(
        m.resolve_op("HQResize"),
        "summary",
        ["python/mod_cvcuda/operators/OpHQResize.cpp"],
        "guide",
    )

    assert finding.status == expected, finding.evidence
    for case in cases:
        short = case.split("[", 1)[0]
        measurement = (
            f"{short}: gap 40.00->{40.0 - reduction_us:.2f} us; "
            f"reduction {reduction_us:.2f} us; "
        )
        assert (
            f"H100 {measurement}paired-gap combined SE {math.sqrt(2 * 2.0**2 / 5):.2f} us"
            in finding.evidence
        )
        if expected == "PASS":
            assert (
                f"A100 {measurement}candidate SE vs fixed committed reference "
                f"{2.0 / math.sqrt(5):.2f} us" in finding.evidence
            )
    if expected == "GAP":
        assert "does not clear" in finding.evidence
        assert "A100" not in finding.evidence


def test_binding_impact_gate_requires_repeated_artifacts(monkeypatch):
    m = _module()
    hq = m.resolve_op("HQResize")
    _binding_impact_inputs(monkeypatch, m, reduction_us=10.0, n_runs=1)

    finding = m._binding_impact_gate(
        hq,
        "summary",
        ["python/mod_cvcuda/operators/OpHQResize.cpp"],
        "guide",
    )

    assert finding.status == "GAP"
    assert "paired-gap-dispersion" in finding.evidence


def test_binding_impact_gate_accepts_legacy_summary_metric(monkeypatch):
    m = _module()
    hq = m.resolve_op("HQResize")
    cases = _binding_impact_inputs(monkeypatch, m, reduction_us=10.0)
    monkeypatch.setattr(
        m,
        "parse_summary",
        lambda _results: SimpleNamespace(
            metadata=SimpleNamespace(
                impact_metric="cpp_time",
                baseline_commit="a" * 40,
                optimized_cases=cases,
            ),
            generalization_rows=(),
        ),
    )

    finding = m._binding_impact_gate(
        hq,
        "summary",
        ["python/mod_cvcuda/operators/OpHQResize.cpp"],
        "guide",
    )

    assert finding.status == "PASS"
    assert "paired-gap standard error" in finding.summary


def _binding_source(signature=None, registration=None, body=None, aliases=""):
    if signature is None:
        signature = "Tensor TensorHQResize(Tensor &src, std::optional<Stream> stream)"
    if registration is None:
        registration = """    m.def("hq_resize", &TensorHQResize, "src"_a, py::kw_only(),
          "stream"_a = nullptr);"""
    if body is None:
        body = "    return src;"
    return f"""namespace cvcudapy {{
namespace {{

{aliases}
{signature}
{{
{body}
}}

}} // namespace

void ExportOpHQResize(py::module &m)
{{
{registration}
}}

}} // namespace cvcudapy
"""


def test_binding_api_snapshot_recognizes_direct_and_nvtx_wrapped_callables():
    m = _module()
    signature = "TensorBatch TensorBatchHQResize(TensorBatch &src)"

    for callable_arg in (
        "&TensorBatchHQResize",
        'NvtxTrace("cvcuda.hq_resize", &TensorBatchHQResize)',
    ):
        source = _binding_source(
            signature=signature,
            registration=f'    m.def("hq_resize", {callable_arg}, "src"_a);',
        )

        snapshot = m._binding_api_snapshot(source, "HQResize")

        assert snapshot is not None, callable_arg
        assert snapshot[1] == (("TensorBatchHQResize", signature),)


def test_api_abi_referenced_transitive_using_alias_change_is_a_gap(monkeypatch):
    m = _module()
    hq = m.resolve_op("HQResize")
    signature = "Tensor TensorHQResize(Tensor &src, Rois rois)"
    baseline = _binding_source(
        signature=signature,
        aliases="using Roi = pybind11::tuple;\nusing Rois = std::vector<Roi>;",
    )
    candidate = baseline.replace(
        "using Roi = pybind11::tuple;", "using Roi = pybind11::list;"
    )
    _mock_api_abi_sources(
        monkeypatch,
        m,
        baseline,
        candidate,
        "-using Roi = pybind11::tuple;\n+using Roi = pybind11::list;",
    )

    result = m._api_abi(hq, "origin/main", "guide")

    assert result.status == "GAP"
    assert "type alias" in result.evidence


def test_api_abi_referenced_typedef_change_is_a_gap(monkeypatch):
    m = _module()
    hq = m.resolve_op("HQResize")
    signature = "Tensor TensorHQResize(Tensor &src, Roi roi)"
    baseline = _binding_source(
        signature=signature, aliases="typedef pybind11::tuple Roi;"
    )
    candidate = baseline.replace("pybind11::tuple", "pybind11::list")
    _mock_api_abi_sources(
        monkeypatch,
        m,
        baseline,
        candidate,
        "-typedef pybind11::tuple Roi;\n+typedef pybind11::list Roi;",
    )

    result = m._api_abi(hq, "origin/main", "guide")

    assert result.status == "GAP"
    assert "type alias" in result.evidence


def test_api_abi_unused_private_alias_change_is_implementation_only(monkeypatch):
    m = _module()
    hq = m.resolve_op("HQResize")
    baseline = _binding_source(aliases="using Scratch = pybind11::tuple;")
    candidate = baseline.replace("pybind11::tuple", "pybind11::list")
    _mock_api_abi_sources(
        monkeypatch,
        m,
        baseline,
        candidate,
        "-using Scratch = pybind11::tuple;\n+using Scratch = pybind11::list;",
    )

    result = m._api_abi(hq, "origin/main", "guide")

    assert result.status == "PASS"
    assert "binding implementation only" in result.summary


def test_api_abi_ambiguous_referenced_alias_requires_manual_review(monkeypatch):
    m = _module()
    hq = m.resolve_op("HQResize")
    aliases = "using Roi = pybind11::tuple;\nusing Roi = pybind11::list;"
    baseline = _binding_source(
        signature="Tensor TensorHQResize(Tensor &src, Roi roi)", aliases=aliases
    )
    candidate = baseline.replace("return src;", "return Tensor(src);")
    _mock_api_abi_sources(
        monkeypatch,
        m,
        baseline,
        candidate,
        "+    return Tensor(src);",
    )

    result = m._api_abi(hq, "origin/main", "guide")

    assert result.status == "MANUAL"
    assert "could not be parsed safely" in result.summary


def test_api_abi_unparseable_referenced_typedef_requires_manual_review(monkeypatch):
    m = _module()
    hq = m.resolve_op("HQResize")
    baseline = _binding_source(
        signature="Tensor TensorHQResize(Tensor &src, Roi roi)",
        aliases="typedef void (*Roi)(int);",
    )
    candidate = baseline.replace("return src;", "return Tensor(src);")
    _mock_api_abi_sources(
        monkeypatch,
        m,
        baseline,
        candidate,
        "+    return Tensor(src);",
    )

    result = m._api_abi(hq, "origin/main", "guide")

    assert result.status == "MANUAL"
    assert "could not be parsed safely" in result.summary


def test_api_abi_unsupported_referenced_typedefs_require_manual_review(monkeypatch):
    m = _module()
    hq = m.resolve_op("HQResize")

    for typedef in (
        "typedef pybind11::tuple Roi[2];",
        "typedef pybind11::tuple Roi(int);",
        "typedef pybind11::tuple (Owner::*Roi)(int);",
        "typedef pybind11::tuple Roi[2], Other;",
        "typedef struct { int value; } Roi;",
    ):
        baseline = _binding_source(
            signature="Tensor TensorHQResize(Tensor &src, Roi roi)",
            aliases=typedef,
        )
        candidate = baseline.replace("return src;", "return Tensor(src);")
        _mock_api_abi_sources(
            monkeypatch,
            m,
            baseline,
            candidate,
            "+    return Tensor(src);",
        )

        result = m._api_abi(hq, "origin/main", "guide")

        assert result.status == "MANUAL", typedef
        assert "could not be parsed safely" in result.summary


def test_api_abi_unsupported_private_typedef_respects_name_reachability(monkeypatch):
    m = _module()
    hq = m.resolve_op("HQResize")

    for typedef, status, summary in (
        (
            "typedef pybind11::tuple Scratch[2];",
            "PASS",
            "binding implementation only",
        ),
        (
            "typedef struct { int value; } Scratch;",
            "MANUAL",
            "could not be parsed safely",
        ),
    ):
        baseline = _binding_source(aliases=typedef)
        candidate = baseline.replace("return src;", "return Tensor(src);")
        _mock_api_abi_sources(
            monkeypatch,
            m,
            baseline,
            candidate,
            "+    return Tensor(src);",
        )

        result = m._api_abi(hq, "origin/main", "guide")

        assert result.status == status, typedef
        assert summary in result.summary


def _mock_api_abi_sources(monkeypatch, module, baseline, candidate, diff_line):
    merge_base = "a" * 40

    def fake_git(*args):
        if args[0] == "diff":
            return diff_line if args[-1].endswith("OpHQResize.cpp") else ""
        if args[0] == "merge-base":
            return merge_base + "\n"
        if args[0] == "show" and args[1].startswith(merge_base + ":"):
            return baseline
        if args[0] == "show" and args[1].startswith("HEAD:"):
            return candidate
        return ""

    monkeypatch.setattr(module, "git", fake_git)


def test_api_abi_binding_implementation_only_is_automatic_pass(monkeypatch):
    m = _module()
    hq = m.resolve_op("HQResize")
    baseline = _binding_source()
    candidate = _binding_source(body="    RequirementsCache cache;\n    return src;")
    _mock_api_abi_sources(
        monkeypatch,
        m,
        baseline,
        candidate,
        "+    RequirementsCache cache;",
    )

    result = m._api_abi(hq, "origin/main", "guide")

    assert result.status == "PASS"
    assert "binding implementation only" in result.summary


def test_api_abi_binding_export_change_remains_gap(monkeypatch):
    m = _module()
    hq = m.resolve_op("HQResize")
    baseline = _binding_source()
    candidate = _binding_source().replace('m.def("hq_resize"', 'm.def("hq_resize_new"')
    _mock_api_abi_sources(
        monkeypatch,
        m,
        baseline,
        candidate,
        '-    m.def("hq_resize", &TensorHQResize);\n'
        '+    m.def("hq_resize_new", &TensorHQResize);',
    )

    result = m._api_abi(hq, "origin/main", "guide")

    assert result.status == "GAP"
    assert "m.def registration" in result.evidence


def test_api_abi_binding_multiline_argument_change_remains_gap(monkeypatch):
    m = _module()
    hq = m.resolve_op("HQResize")
    baseline = _binding_source()
    candidate = baseline.replace('"stream"_a = nullptr', '"priority"_a = 0')
    _mock_api_abi_sources(
        monkeypatch,
        m,
        baseline,
        candidate,
        '-          "stream"_a = nullptr);\n+          "priority"_a = 0);',
    )

    result = m._api_abi(hq, "origin/main", "guide")

    assert result.status == "GAP"
    assert "m.def registration" in result.evidence


def test_api_abi_bound_callable_signature_change_remains_gap(monkeypatch):
    m = _module()
    hq = m.resolve_op("HQResize")
    baseline = _binding_source()
    candidate = _binding_source(
        signature="Tensor TensorHQResize(const Tensor &src, std::optional<Stream> stream)"
    )
    _mock_api_abi_sources(
        monkeypatch,
        m,
        baseline,
        candidate,
        "-Tensor TensorHQResize(Tensor &src, std::optional<Stream> stream)\n"
        "+Tensor TensorHQResize(const Tensor &src, std::optional<Stream> stream)",
    )

    result = m._api_abi(hq, "origin/main", "guide")

    assert result.status == "GAP"
    assert "bound callable signature" in result.evidence


def test_api_abi_unparseable_binding_surface_requires_manual_review(monkeypatch):
    m = _module()
    hq = m.resolve_op("HQResize")
    baseline = _binding_source()
    candidate = baseline.replace("ExportOpHQResize", "RegisterHQResize")
    _mock_api_abi_sources(
        monkeypatch,
        m,
        baseline,
        candidate,
        "-void ExportOpHQResize(py::module &m)\n+void RegisterHQResize(py::module &m)",
    )

    result = m._api_abi(hq, "origin/main", "guide")

    assert result.status == "MANUAL"
    assert "could not be parsed safely" in result.summary


# ---- ODO-6 commit hygiene --------------------------------------------------------------
def _init_git_history(repo):
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "optimize-op-test@nvidia.com"],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Optimize Op Test"],
        cwd=repo,
        check=True,
    )
    _commit(repo, "chore: establish test base", {"README.md": "base\n"})
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _commit(repo, subject, files):
    for name, contents in files.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)
    subprocess.run(["git", "add", "--all"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", subject], cwd=repo, check=True)


def test_perf_hygiene_accepts_only_proven_cpp_formatting(monkeypatch, tmp_path):
    m = _module()
    op = m.resolve_op("BoxBlur")
    repo = tmp_path / "formatting"
    _init_git_history(repo)
    source = "src/cvcuda/priv/legacy/box_blur.cu"
    before = "if constexpr (B\n              == CONSTANT) { run(); }\n"
    after = "if constexpr (B == CONSTANT) { run(); }\n"
    _commit(repo, "perf(boxblur): establish implementation", {source: before})
    base = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    _commit(repo, "style(boxblur): format dispatch", {source: after})
    monkeypatch.setattr(m, "REPO", repo)

    result = m._perf_hygiene(op, base, ["boxblur"], "guide")

    assert result.status == "PASS"
    assert "formatting-only" in result.evidence
    _commit(
        repo, "style(boxblur): change dispatch", {source: after.replace("==", "!=")}
    )
    assert m._perf_hygiene(op, base, ["boxblur"], "guide").status == "GAP"


def test_cpp_formatting_comparison_preserves_sensitive_whitespace():
    m = _module()
    assert m._cpp_formatting_only("if (a\n&& b) {}", "if (a && b) {}")
    for before, after in (
        ("int value;", "intvalue;"),
        ('const char *s = "a b";', 'const char *s = "ab";'),
        ('auto s = R"tag(a b)tag";', 'auto s = R"tag(ab)tag";'),
        ('auto s = "a"_tag;', 'auto s = "a" _tag;'),
        ("a ++ b;", "a + + b;"),
        ("float x = 1e+2;", "float x = 1e +2;"),
        ("a / * b;", "a /* b;"),
        ("a; // b\nc;", "a; // bc;"),
        ("#define F(x) x\n", "#define F (x) x\n"),
        ("STR(a + b)", "STR(a+b)"),
        ("#define STR(x) #x\nSTR(a+b)\n", "#define STR(x) #x\nSTR(a + b)\n"),
        ("#include <a>\n#include <b>\n", "#include <b>\n#include <a>\n"),
        ("a;\\\nb;", "a; b;"),
        ("__LINE__;\nx;", "__LINE__; x;"),
        ("// comment\na;", "//comment\na;"),
    ):
        assert not m._cpp_formatting_only(before, after), (before, after)


def test_cpp_formatting_comparison_preserves_indented_directives():
    m = _module()
    for prefix in ("", "int earlier;\n", "int earlier;\n\n"):
        for directive in ("  #define F(x) #x\n", "\t%:define F(x) #x\n"):
            before = prefix + directive + "int  value;\n"
            after = prefix + directive + "int value;\n"
            assert m._cpp_formatting_only(before, after), (prefix, directive)
            assert not m._cpp_formatting_only(before, after.replace("F(x)", "F (x)"))


def test_cpp_formatting_comparison_preserves_commented_macro_arguments():
    m = _module()
    for comment in ("/*same*/", "// same\n", "/*first*/ /*second*/"):
        before = f"#define STR(x) #x\nSTR {comment} (a+b)\n"
        after = f"#define STR(x) #x\nSTR {comment} (a + b)\n"
        assert not m._cpp_formatting_only(before, after), comment
        assert m._cpp_formatting_only(before, before)


def test_formatting_proof_rejects_missing_added_non_cpp_and_mode_changes(
    monkeypatch, tmp_path
):
    m = _module()
    repo = tmp_path / "proof-limits"
    _init_git_history(repo)
    monkeypatch.setattr(m, "REPO", repo)
    source = "kernel.cu"
    script = "kernel.py"
    _commit(repo, "style: add source", {source: "int x;\n", script: "pass\n"})
    assert not m._formatting_only_implementation("HEAD", [source])
    assert not m._formatting_only_implementation("HEAD", [script])
    assert not m._formatting_only_implementation("missing", [source])
    assert not m._formatting_only_implementation("HEAD", [])
    (repo / source).chmod(0o755)
    _commit(repo, "style: change mode", {source: "int  x;\n"})
    assert not m._formatting_only_implementation("HEAD", [source])
    (repo / source).chmod(0o644)
    _commit(repo, "style: restore mode", {source: "int x;\n"})
    (repo / source).write_bytes(b"\xff\n")
    subprocess.run(["git", "add", source], cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "style: binary bytes"], cwd=repo, check=True
    )
    assert not m._formatting_only_implementation("HEAD", [source])


def test_perf_hygiene_accepts_scoped_regression_backed_fix(monkeypatch, tmp_path):
    m = _module()
    boxblur = m.resolve_op("BoxBlur")
    repo = tmp_path / "paired"
    base = _init_git_history(repo)
    _commit(
        repo,
        "test(boxblur): cover signed negative pixels",
        {"tests/cvcuda/system/TestOpBoxBlur.cpp": "regression\n"},
    )
    _commit(
        repo,
        "fix(boxblur): honor signed 8-bit pixels",
        {"src/cvcuda/priv/legacy/box_blur.cu": "fix\n"},
    )
    monkeypatch.setattr(m, "REPO", repo)

    result = m._perf_hygiene(boxblur, base, ["boxblur"], "guide")

    assert result.status == "PASS"
    assert "regression-backed fix" in result.evidence


def test_perf_hygiene_rejects_unpaired_or_disguised_fix(monkeypatch, tmp_path):
    cases = (
        (
            "blanket",
            "test(boxblur): cover signed negative pixels",
            {"tests/cvcuda/system/TestOpBoxBlur.cpp": "regression\n"},
            "fix: honor signed 8-bit pixels",
        ),
        (
            "unpaired",
            None,
            {},
            "fix(boxblur): honor signed 8-bit pixels",
        ),
        (
            "mismatched-scope",
            "test(resize): cover signed negative pixels",
            {"tests/cvcuda/system/TestOpResize.cpp": "regression\n"},
            "fix(boxblur): honor signed 8-bit pixels",
        ),
        (
            "wrong-operator-test",
            "test(boxblur): cover signed negative pixels",
            {"tests/cvcuda/system/TestOpResize.cpp": "regression\n"},
            "fix(boxblur): honor signed 8-bit pixels",
        ),
        (
            "not-test-only",
            "test(boxblur): cover signed negative pixels",
            {
                "tests/cvcuda/system/TestOpBoxBlur.cpp": "regression\n",
                "bench/config/operators/boxblur.json": "{}\n",
            },
            "fix(boxblur): optimize staging",
        ),
    )
    for name, test_subject, test_files, fix_subject in cases:
        m = _module()
        boxblur = m.resolve_op("BoxBlur")
        repo = tmp_path / name
        base = _init_git_history(repo)
        if test_subject:
            _commit(repo, test_subject, test_files)
        _commit(
            repo,
            fix_subject,
            {"src/cvcuda/priv/legacy/box_blur.cu": "implementation\n"},
        )
        monkeypatch.setattr(m, "REPO", repo)

        result = m._perf_hygiene(boxblur, base, ["boxblur", "resize"], "guide")

        assert result.status == "GAP", name
        assert "fix" in result.evidence, name


def _compiler_trailers(parent):
    return (
        f"Compiler-Failure: {parent} https://ci.example/project/-/jobs/123\n"
        "Compiler-Verification: parent-fail/candidate-pass/codegen "
        + "a" * 64
        + " /absolute/artifacts/compiler-proof.md"
    )


def test_perf_hygiene_compiler_evidence_requires_manual_review(monkeypatch, tmp_path):
    m = _module()
    op = m.resolve_op("BoxBlur")
    repo = tmp_path / "compiler"
    _init_git_history(repo)
    source = "src/cvcuda/priv/OpBoxBlur.cu"
    _commit(repo, "perf(boxblur): establish implementation", {source: "old();\n"})
    base = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    _commit(
        repo,
        "build(boxblur): preserve dependent lookup\n\n"
        + _compiler_trailers(base)
        .replace(" https://", "\n https://")
        .replace(" " + "a" * 64, "\n " + "a" * 64)
        .replace(" /absolute/", "\n /absolute/"),
        {source: "compiler_compatible();\n"},
    )
    monkeypatch.setattr(m, "REPO", repo)

    result = m._perf_hygiene(op, base, ["boxblur"], "guide")

    assert (
        result.status == "MANUAL"
    )  # Structured claims never establish compiler or semantic proof.
    assert base in result.evidence
    assert "https://ci.example/project/-/jobs/123" in result.evidence
    assert "/absolute/artifacts/compiler-proof.md" in result.evidence
    assert "a" * 64 in result.evidence

    _commit(
        repo, "chore: unrelated implementation edit", {source: "another_change();\n"}
    )
    assert m._perf_hygiene(op, base, ["boxblur"], "guide").status == "GAP"


def test_perf_hygiene_rejects_unqualified_compiler_claims(monkeypatch, tmp_path):
    source = "src/cvcuda/priv/OpBoxBlur.cu"
    cases = (
        ("unscoped", "build: compiler compatibility", None, {}),
        ("wrong-scope", "build(resize): compiler compatibility", None, {}),
        ("missing-failure", None, lambda t: t.split("\n")[1], {}),
        ("missing-verification", None, lambda t: t.split("\n")[0], {}),
        ("wrong-parent", None, lambda t: t.replace(t.split()[1], "0" * 40), {}),
        (
            "abbreviated-parent",
            None,
            lambda t: t.replace(t.split()[1], t.split()[1][:8]),
            {},
        ),
        ("not-a-job", None, lambda t: t.replace("/-/jobs/123", "/-/pipelines/123"), {}),
        ("missing-codegen", None, lambda t: t.replace("/codegen", ""), {}),
        ("bad-report-hash", None, lambda t: t.replace("a" * 64, "a" * 8), {}),
        ("duplicate", None, lambda t: t + "\n" + t, {}),
        ("body-not-trailers", None, lambda t: t + "\n\nNot a trailer block.", {}),
        (
            "public-header",
            None,
            None,
            {"src/cvcuda/include/cvcuda/OpBoxBlur.h": "changed\n"},
        ),
        ("other-operator", None, None, {"src/cvcuda/priv/OpResize.cu": "changed\n"}),
        (
            "shared-header",
            None,
            None,
            {"src/cvcuda/priv/FilterCommon.cuh": "changed\n"},
        ),
        ("checker-test", None, None, {"tools/tests/test_optimize_op.py": "changed\n"}),
        ("new-tu", None, None, {"src/cvcuda/priv/OpBoxBlurExtra.cu": "new\n"}),
    )
    for name, subject, rewrite, extra in cases:
        m = _module()
        op = m.resolve_op("BoxBlur")
        repo = tmp_path / name
        _init_git_history(repo)
        initial = {source: "old();\n"}
        initial.update({path: "old\n" for path in extra if name != "new-tu"})
        _commit(repo, "perf(boxblur): establish implementation", initial)
        base = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip()
        trailers = _compiler_trailers(base)
        if rewrite:
            trailers = rewrite(trailers)
        _commit(
            repo,
            (subject or "build(boxblur): compiler compatibility") + "\n\n" + trailers,
            {source: "compiler_compatible();\n", **extra},
        )
        monkeypatch.setattr(m, "REPO", repo)
        assert (
            m._perf_hygiene(op, base, ["boxblur", "resize"], "guide").status == "GAP"
        ), name


def test_compiler_compatibility_requires_single_parent(monkeypatch):
    m = _module()
    op = m.resolve_op("BoxBlur")
    monkeypatch.setattr(m, "git", lambda *args: "a" * 40 + " " + "b" * 40)

    accepted, reason = m._compiler_compatibility_build(
        op, ["boxblur"], "HEAD", "build(boxblur): compiler fix"
    )

    assert not accepted
    assert "exactly one" in reason


# ---- results-summary format enforcement (base-independent) ------------------------------
def test_results_missing_sections_is_gap(tmp_path):
    rf = tmp_path / "res.md"
    rf.write_text("## Overall table\nspeedup\n## Pixelwise-equality\nEXPECT_EQ\n")
    r = run("Resize", "--phase", "evidence", "--results", str(rf), "--format", "json")
    odo3 = [f for f in json.loads(r.stdout)["findings"] if f["id"] == "ODO-3"][0]
    assert odo3["status"] == "GAP"


def test_results_absent_is_gap():
    r = run("Resize", "--phase", "evidence", "--format", "json")
    odo3 = [f for f in json.loads(r.stdout)["findings"] if f["id"] == "ODO-3"][0]
    assert odo3["status"] == "GAP"


# ---- record-backed gates (ODO-2/4/8/9); gate semantics live in test_optimization_evidence
def _summary(evidence, rows=()):
    return SimpleNamespace(
        metadata=SimpleNamespace(
            evidence=evidence, candidate_commit=fixtures.CANDIDATE
        ),
        generalization_rows=rows,
    )


def test_record_gates_pass_on_complete_evidence():
    m = _module()
    evidence = fixtures.complete()
    assert m._leads_exhausted(_summary(evidence), "g").status == m.PASS
    assert (
        m._review_refactor_gate(m.resolve_op("Resize"), evidence, ["x.cu"], "g").status
        == m.PASS
    )
    assert m._memory_footprint_gate(_summary(evidence), "g").status == m.PASS


def test_record_gates_are_gaps_without_records():
    m = _module()
    evidence = {"schema": 1}
    assert m._leads_exhausted(_summary(evidence), "g").status == m.GAP
    assert m._leads_exhausted(None, "g").status == m.GAP
    review = m._review_refactor_gate(m.resolve_op("Resize"), evidence, ["x.cu"], "g")
    assert review.status == m.GAP and "Resize refactor review" in review.fix
    assert m._memory_footprint_gate(_summary(evidence), "g").status == m.GAP
    assert m._memory_footprint_gate(None, "g").status == m.GAP


def test_review_gate_is_na_without_implementation_change():
    m = _module()
    assert m._review_refactor_gate(m.resolve_op("Resize"), {}, [], "g").status == m.NA


def test_memory_gate_escalates_to_manual_from_the_record():
    m = _module()
    evidence = fixtures.complete()
    evidence["memory"]["sites"][0]["after_peak_bytes"] += 20_000_000
    assert m._memory_footprint_gate(_summary(evidence), "g").status == m.MANUAL


def test_baseline_import_must_match_committed_configs(monkeypatch):
    m = _module()
    evidence = fixtures.complete()
    ((path, _digest),) = evidence["baselines"]["import"]["files"].items()
    monkeypatch.setattr(m, "read", lambda _p: "{}")
    assert m._imported_baselines(evidence).status == m.GAP
    import hashlib

    evidence["baselines"]["import"]["files"][path] = hashlib.sha256(b"{}").hexdigest()
    assert m._imported_baselines(evidence).status == m.PASS
    assert m._imported_baselines({}).status == m.GAP


# ---- ODO-7 comparable committed-baseline regression gate -------------------------------
def test_baseline_regression_gate_passes_when_validator_passes(monkeypatch):
    m = _module()
    resize = m.resolve_op("Resize")
    calls = []

    class Result:
        returncode = 0
        stdout = "JSON baseline validation passed"
        stderr = ""

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return Result()

    monkeypatch.setattr(m.subprocess, "run", fake_run)

    result = m._baseline_regression_gate(resize, "origin/main", "guide")

    assert result.status == "PASS"
    assert "--reject-regressions-from" in calls[0]
    assert "origin/main" in calls[0]
    assert "JSON baseline validation passed" in result.evidence


def test_baseline_regression_gate_blocks_validator_failure(monkeypatch):
    m = _module()
    resize = m.resolve_op("Resize")

    class Result:
        returncode = 1
        stdout = ""
        stderr = "same-key baseline regressed +41.02%"

    monkeypatch.setattr(m.subprocess, "run", lambda cmd, **kwargs: Result())

    result = m._baseline_regression_gate(resize, "origin/main", "guide")

    assert result.status == "GAP"
    assert "same-key baseline regressed" in result.evidence
    assert "revert the regressed baseline" in result.fix


@pytest.mark.parametrize("n_runs, partial", [(1, False), (5, False), (5, True)])
def test_preflight_reuses_committed_baselines_without_refresh_history(
    tmp_path, monkeypatch, n_runs, partial
):
    m = _module()
    paths = m.resolve_op("Gaussian")
    paths.bench_cfg = tmp_path / "gaussian.json"
    document = {
        "benchmark": "gaussian",
        "configs": {
            "gaussian_basic": {"baselines": {"case": {"A100": {"n_runs": n_runs}}}}
        },
    }
    if partial:
        document["baseline_refresh_incomplete"] = True
    paths.bench_cfg.write_text(json.dumps(document))
    monkeypatch.setattr(m, "review_op_findings", lambda *_args: [])
    monkeypatch.setattr(m.shutil, "which", lambda name: "/usr/bin/" + name)

    def unexpected_history(*_args, **_kwargs):
        raise AssertionError("committed baselines do not require refresh history")

    monkeypatch.setattr(m, "git", unexpected_history)
    monkeypatch.setattr(m.subprocess, "run", unexpected_history)

    findings = m.check_preflight(paths, "unavailable-history")

    assert not [finding for finding in findings if finding.status == m.GAP]
    assert next(f for f in findings if f.id == "PRE-3").status == m.PASS
    assert not any("baseline-regen" in finding.fix for finding in findings)


@pytest.mark.parametrize("tier", ["basic", "advanced"])
def test_preflight_still_blocks_missing_baseline_coverage(monkeypatch, tier):
    m = _module()
    paths = m.resolve_op("Gaussian")

    def review(_paths, domain):
        if domain == "bench":
            return [{"id": "BEN-7", "status": m.GAP, "summary": f"missing {tier} case"}]
        return []

    monkeypatch.setattr(m, "review_op_findings", review)
    findings = m.check_preflight(paths, "origin/main")
    coverage = next(f for f in findings if f.id == "PRE-2")
    assert coverage.status == m.GAP
    assert f"missing {tier} case" in coverage.evidence


def test_git_json_distinguishes_missing_and_malformed(monkeypatch):
    m = _module()
    monkeypatch.setattr(
        m.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout=""),
    )
    assert m.git_json("HEAD", "bench.json") is m._MALFORMED_GIT_JSON

    monkeypatch.setattr(
        m.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=1, stdout=""),
    )
    assert m.git_json("HEAD", "bench.json") is None


def _github_slug(heading):
    slug = re.sub(r"[^\w\- ]", "", heading.strip().lower())
    return slug.replace(" ", "-")


def test_every_gate_routes_to_an_existing_section():
    module = _module()
    source = TOOL.read_text(encoding="utf-8")
    emitted = set(re.findall(r'"((?:PRE|ODO)-\d+)"', source))
    assert emitted <= set(module.GATE_GUIDELINES), emitted - set(module.GATE_GUIDELINES)
    for gate, ref in module.GATE_GUIDELINES.items():
        path, _, anchor = ref.partition("#")
        text = (REPO / path).read_text(encoding="utf-8")
        # Headings inside fenced examples do not render as anchors.
        prose = re.sub(r"^```.*?^```", "", text, flags=re.M | re.S)
        slugs = {
            _github_slug(line.lstrip("#"))
            for line in prose.splitlines()
            if re.match(r"#{1,6} ", line)
        }
        assert anchor in slugs, f"{gate}: {ref} has no matching heading"
