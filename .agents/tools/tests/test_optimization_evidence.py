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
"""Gate and derivation tests for the machine-readable optimization evidence."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import evidence_fixtures as fx  # noqa: E402
import optimization_evidence as ev  # noqa: E402


def test_complete_fixture_passes_every_gate():
    evidence = ev.validate(fx.complete())
    rows = ev.checklist(
        evidence,
        [
            "Pixelwise equality to reference",
            "Memory-footprint checks",
            "Baselines updated",
            "Baseline validation",
            "Lead exhaustion",
            "Review/refactor gate",
        ],
        candidate=fx.CANDIDATE,
    )
    assert all(checked for _, checked, _ in rows), rows


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda e: e.update(schema=2), "schema"),
        (lambda e: e.update(extra={}), "unknown sections"),
        (lambda e: e["tests"].update(commit="abc"), "tests.commit"),
        (lambda e: e["attempts"][0].update(median_speedup=0), "median_speedup"),
        (
            lambda e: e["memory"]["sites"].append(dict(e["memory"]["sites"][0])),
            "repeats",
        ),
        (
            lambda e: e["baselines"]["import"]["files"].update({"x.json": fx.DIGEST}),
            "invalid path",
        ),
        (lambda e: e["review"].update(applied=True), "review.verify"),
        *(
            (lambda e, iid=iid: e["baselines"].update(mr_iid=iid), "mr_iid")
            for iid in (2059, "!2059", "", "²")
        ),
    ],
)
def test_schema_rejects_malformed_records(mutate, match):
    evidence = fx.complete()
    mutate(evidence)
    with pytest.raises(ev.EvidenceError, match=match):
        ev.validate(evidence)


# ---------------------------------------------------------------- ODO-1


def test_pixelwise_requires_the_candidate_commit():
    result = ev.pixelwise_gate(fx.tests(commit="3" * 40), fx.CANDIDATE)
    assert result.status == ev.GAP and "not candidate" in result.detail


def test_pixelwise_rejects_a_failing_test():
    tests = fx.tests()
    tests["cases"][0]["outcome"] = "failed"
    assert ev.pixelwise_gate(tests, fx.CANDIDATE).status == ev.GAP


def test_pixelwise_rejects_a_nonzero_test_run():
    result = ev.pixelwise_gate(fx.tests(gtest=fx.run(1)), fx.CANDIDATE)
    assert result.status == ev.GAP and "gtest exited 1" in result.detail


@pytest.mark.parametrize(
    "case_change",
    [
        {"reference": None},
        {"assertions": ["EXPECT_TRUE"]},
        {"categories": [["VarShape", "NHWC", "u8"]]},
        {"outcome": "skipped"},
    ],
)
def test_pixelwise_requires_every_category_covered_by_a_referenced_equality_test(
    case_change,
):
    tests = fx.tests()
    tests["cases"][0].update(case_change)
    result = ev.pixelwise_gate(tests, fx.CANDIDATE)
    assert result.status == ev.GAP and "Tensor/NHWC/u8" in result.detail


# ---------------------------------------------------------------- ODO-4


def test_ridge_on_the_profile_exhausts_leads():
    assert (
        ev.lead_exhaustion([], {"memory_pct": 91.0, "compute_pct": 5.0}).status
        == ev.PASS
    )


def test_three_strikes_after_the_last_kept_win_exhaust_leads():
    attempts = [fx.attempt("struck"), fx.attempt("kept")] + [fx.attempt("struck")] * 3
    assert (
        ev.lead_exhaustion(attempts, {"memory_pct": 40.0, "compute_pct": 5.0}).status
        == ev.PASS
    )


def test_strikes_before_a_later_win_do_not_count():
    attempts = [fx.attempt("struck")] * 3 + [fx.attempt("kept"), fx.attempt("struck")]
    result = ev.lead_exhaustion(attempts, {"memory_pct": 40.0, "compute_pct": 5.0})
    assert result.status == ev.GAP and result.detail.startswith("1 strike")


def test_learnings_cap_at_five_and_order_kept_before_struck():
    attempts = [fx.attempt("struck", 0.5 + i / 10, f"S{i}") for i in range(4)] + [
        fx.attempt("kept", 1 + i, f"K{i}") for i in range(3)
    ]
    lines = ev.draft_learnings(attempts)
    assert len(lines) == 5
    assert [line.split()[0] for line in lines] == ["K2", "K1", "K0", "S0", "S1"]


# ---------------------------------------------------------------- ODO-9


def _memory(sites, unattributed=0):
    return {
        "workload": "w",
        "before_report_sha256": fx.DIGEST,
        "after_report_sha256": fx.DIGEST,
        "sites": sites,
        "before_unattributed_bytes": 0,
        "after_unattributed_bytes": unattributed,
    }


def _site(name, before, after):
    return {"site": name, "before_peak_bytes": before, "after_peak_bytes": after}


def test_growth_at_one_site_is_not_netted_against_a_decrease_elsewhere():
    memory = _memory([_site("a", 100, 10_000_100), _site("b", 20_000_000, 0)])
    assert ev.memory_totals(memory)["increase"] == 10_000_000


def test_memory_at_zero_and_exactly_the_limit_passes():
    for after in (0, ev.MEMORY_LIMIT_BYTES):
        assert ev.memory_gate(_memory([_site("a", 0, after)])).status == ev.PASS


def test_memory_over_the_limit_needs_human_review():
    memory = _memory([_site("a", 0, ev.MEMORY_LIMIT_BYTES + 1)])
    result = ev.memory_gate(memory)
    assert result.status == ev.MANUAL and "exceeds" in result.detail


def test_a_new_allocation_site_needs_human_review():
    memory = _memory(
        [
            _site("a", 10, 10),
            {"site": "new", "before_peak_bytes": None, "after_peak_bytes": 64},
        ]
    )
    result = ev.memory_gate(memory)
    assert result.status == ev.MANUAL and "new allocation sites: new" in result.detail
    assert "New runtime CUDA allocations/frees: yes" in ev.render_memory(memory)


def test_unattributed_allocations_need_human_review():
    result = ev.memory_gate(_memory([_site("a", 1, 1)], unattributed=4096))
    assert result.status == ev.MANUAL and "without a CV-CUDA call site" in result.detail


# ---------------------------------------------------------------- ODO-8


def _applied_review(**changes):
    review = {
        "commit": fx.CANDIDATE,
        "assess": fx.run(0),
        "assess_verdict": "RECOMMENDATION",
        "applied": True,
        "verify": fx.run(0),
        "verify_verdict": "PASS",
        "tests": fx.tests(),
        "benchmark": {"artifact_sha256": fx.DIGEST, "regressions": 0, "cases": 6},
    }
    review.update(changes)
    return review


def test_an_applied_refactor_with_proof_passes():
    assert (
        ev.review_gate(
            ev.validate({"schema": 1, "review": _applied_review()})["review"]
        ).status
        == ev.PASS
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"verify_verdict": "GAP"},
        {"tests": fx.tests(commit="3" * 40)},
        {"benchmark": {"artifact_sha256": fx.DIGEST, "regressions": 2, "cases": 6}},
    ],
)
def test_an_applied_refactor_without_proof_fails(changes):
    assert ev.review_gate(_applied_review(**changes)).status == ev.GAP


# ---------------------------------------------------------------- ODO-2/7


def test_a_failed_baseline_import_is_a_gap():
    baselines = fx.complete()["baselines"]
    baselines["import"]["run"]["exit_code"] = 1
    assert ev.baseline_import_gate(baselines).status == ev.GAP


def test_a_failed_baseline_validation_is_a_gap():
    baselines = fx.complete()["baselines"]
    baselines["validation"]["exit_code"] = 2
    assert ev.baseline_validation_gate(baselines).status == ev.GAP


def test_every_recorded_commit_is_exposed_for_ancestry_checks():
    fields = dict(ev.commits(fx.complete()))
    assert set(fields) == {
        "tests.commit",
        "attempts[0].commit",
        "review.commit",
        "baselines.import.measured_commit",
    }
