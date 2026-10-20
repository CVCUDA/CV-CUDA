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
"""Machine-readable evidence for the optimization summary contract.

Every evidence-bearing line of an optimization summary (the bottleneck, the
evidence checklist, the top learnings, and the memory-footprint section) is a
pure function of one validated evidence object. Builders record tool output
into that object; nothing here accepts an authored claim, so a gate verdict
and the text a reviewer reads cannot disagree.

The object has these sections, each optional until the gate that needs it runs:

``tests``      executed test identities with outcome, reference, assertion, and
               the (container, layout, dtype) categories they cover, plus the
               categories the benchmark surface requires.
``profile``    the hot kernel and its memory/compute throughput from Nsight Compute.
``attempts``   ordered lead attempts with benchmark-derived speedups.
``memory``     per-call-site peak live bytes at baseline and candidate.
``review``     refactor assess/verify verdicts and post-refactor proof.
``baselines``  guarded baseline import and committed-baseline validation runs.
``reference_repair``  the disclosure text of a code-reviewed reference-snapshot
               repair; the deployment's policy module verifies it.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

SCHEMA_VERSION = 1
SECTIONS = (
    "tests",
    "profile",
    "attempts",
    "memory",
    "review",
    "baselines",
    "reference_repair",
)
PASS, GAP, MANUAL, NA = "PASS", "GAP", "MANUAL", "N-A"

RIDGE_PCT = 90.0
STRIKES_TO_EXHAUST = 3
MEMORY_LIMIT_BYTES = 10_000_000  # 10 MB in decimal bytes, as the guidelines state
MAX_LEARNINGS = 5
EQUALITY_ASSERTIONS = frozenset(
    {
        "EXPECT_EQ",
        "ASSERT_EQ",
        "EXPECT_NEAR",
        "ASSERT_NEAR",
        "assert_array_equal",
        "assert_allclose",
        "assert_equal",
    }
)

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")


class EvidenceError(ValueError):
    """The evidence object is malformed."""


@dataclass(frozen=True)
class GateResult:
    status: str
    detail: str


# ------------------------------------------------------------------ validation


def _require(cond: bool, message: str) -> None:
    if not cond:
        raise EvidenceError(message)


def _sha(value: Any, field: str) -> str:
    _require(
        isinstance(value, str) and bool(_SHA_RE.fullmatch(value)),
        f"{field} must be a full lowercase 40-character commit SHA",
    )
    return value


def _digest(value: Any, field: str) -> str:
    _require(
        isinstance(value, str) and bool(_HASH_RE.fullmatch(value)),
        f"{field} must be a lowercase SHA-256 hex digest",
    )
    return value


def _number(value: Any, field: str) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{field} must be a number",
    )
    return float(value)


def _int(value: Any, field: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0,
        f"{field} must be a non-negative integer",
    )
    return value


def _str(value: Any, field: str) -> str:
    _require(isinstance(value, str) and value.strip() != "", f"{field} must be text")
    return value


def _category(value: Any, field: str) -> tuple[str, str, str]:
    _require(
        isinstance(value, list)
        and len(value) == 3
        and all(isinstance(v, str) and v for v in value),
        f"{field} must be a [container, layout, dtype] triple",
    )
    return tuple(value)  # type: ignore[return-value]


def _run(value: Any, field: str) -> None:
    _require(isinstance(value, dict), f"{field} must be an object")
    _require(
        isinstance(value.get("command"), list)
        and all(isinstance(v, str) for v in value["command"])
        and value["command"],
        f"{field}.command must be a non-empty argv list",
    )
    _int(value.get("exit_code"), f"{field}.exit_code")
    _digest(value.get("output_sha256"), f"{field}.output_sha256")


def _validate_tests(section: Any) -> None:
    _require(isinstance(section, dict), "tests must be an object")
    _sha(section.get("commit"), "tests.commit")
    required = section.get("required_categories")
    _require(isinstance(required, list) and required, "tests.required_categories")
    for index, item in enumerate(required):
        _category(item, f"tests.required_categories[{index}]")
    cases = section.get("cases")
    _require(isinstance(cases, list) and cases, "tests.cases must be a non-empty list")
    for index, case in enumerate(cases):
        field = f"tests.cases[{index}]"
        _require(isinstance(case, dict), f"{field} must be an object")
        _str(case.get("id"), f"{field}.id")
        _require(case.get("framework") in {"gtest", "pytest"}, f"{field}.framework")
        _require(
            case.get("outcome") in {"passed", "failed", "skipped"}, f"{field}.outcome"
        )
        reference = case.get("reference")
        _require(reference is None or isinstance(reference, str), f"{field}.reference")
        assertions = case.get("assertions")
        _require(
            isinstance(assertions, list)
            and all(isinstance(a, str) for a in assertions),
            f"{field}.assertions",
        )
        categories = case.get("categories")
        _require(isinstance(categories, list), f"{field}.categories")
        for c_index, item in enumerate(categories):
            _category(item, f"{field}.categories[{c_index}]")
    for key in ("gtest", "pytest"):
        if key in section:
            _run(section[key], f"tests.{key}")


def _validate_profile(section: Any) -> None:
    _require(isinstance(section, dict), "profile must be an object")
    _str(section.get("hot_kernel"), "profile.hot_kernel")
    for key in ("hot_share_pct", "memory_pct", "compute_pct"):
        value = _number(section.get(key), f"profile.{key}")
        _require(0.0 <= value <= 100.0, f"profile.{key} must be a percentage")
    _digest(section.get("report_sha256"), "profile.report_sha256")
    findings = section.get("findings", [])
    _require(isinstance(findings, list), "profile.findings must be a list")
    for index, item in enumerate(findings):
        _require(
            isinstance(item, dict)
            and isinstance(item.get("rule"), str)
            and isinstance(item.get("speedup_pct"), (int, float)),
            f"profile.findings[{index}]",
        )


def _validate_attempts(section: Any) -> None:
    _require(isinstance(section, list), "attempts must be a list")
    for index, item in enumerate(section):
        field = f"attempts[{index}]"
        _require(isinstance(item, dict), f"{field} must be an object")
        _str(item.get("lead"), f"{field}.lead")
        _require(item.get("outcome") in {"kept", "struck"}, f"{field}.outcome")
        _sha(item.get("commit"), f"{field}.commit")
        for key in ("before_artifact_sha256", "after_artifact_sha256"):
            _digest(item.get(key), f"{field}.{key}")
        _int(item.get("cases"), f"{field}.cases")
        _require(item["cases"] > 0, f"{field}.cases must compare at least one case")
        for key in ("median_speedup", "min_speedup", "max_speedup"):
            _require(_number(item.get(key), f"{field}.{key}") > 0, f"{field}.{key}")
        profile = item.get("profile_delta")
        if profile is not None:
            _require(isinstance(profile, dict), f"{field}.profile_delta")
            _str(profile.get("metric"), f"{field}.profile_delta.metric")
            _number(profile.get("before"), f"{field}.profile_delta.before")
            _number(profile.get("after"), f"{field}.profile_delta.after")


def _validate_memory(section: Any) -> None:
    _require(isinstance(section, dict), "memory must be an object")
    for key in ("before_report_sha256", "after_report_sha256"):
        _digest(section.get(key), f"memory.{key}")
    _str(section.get("workload"), "memory.workload")
    sites = section.get("sites")
    _require(isinstance(sites, list), "memory.sites must be a list")
    names = set()
    for index, site in enumerate(sites):
        field = f"memory.sites[{index}]"
        _require(isinstance(site, dict), f"{field} must be an object")
        name = _str(site.get("site"), f"{field}.site")
        _require(name not in names, f"{field}.site repeats {name!r}")
        names.add(name)
        for key in ("before_peak_bytes", "after_peak_bytes"):
            if site.get(key) is not None:
                _int(site[key], f"{field}.{key}")
        _require(
            site.get("before_peak_bytes") is not None
            or site.get("after_peak_bytes") is not None,
            f"{field} must be observed at baseline or candidate",
        )
    for key in ("before_unattributed_bytes", "after_unattributed_bytes"):
        _int(section.get(key), f"memory.{key}")


def _validate_review(section: Any) -> None:
    _require(isinstance(section, dict), "review must be an object")
    _sha(section.get("commit"), "review.commit")
    _run(section.get("assess"), "review.assess")
    _require(
        section.get("assess_verdict") in {"PASS", "RECOMMENDATION", "MANUAL", "GAP"},
        "review.assess_verdict",
    )
    _require(isinstance(section.get("applied"), bool), "review.applied must be bool")
    if section["applied"]:
        _run(section.get("verify"), "review.verify")
        _require(
            section.get("verify_verdict") in {"PASS", "GAP", "MANUAL"},
            "review.verify_verdict",
        )
        _validate_tests(section.get("tests"))
        bench = section.get("benchmark")
        _require(isinstance(bench, dict), "review.benchmark must be an object")
        _digest(bench.get("artifact_sha256"), "review.benchmark.artifact_sha256")
        _int(bench.get("regressions"), "review.benchmark.regressions")
        _int(bench.get("cases"), "review.benchmark.cases")


def _validate_baselines(section: Any) -> None:
    _require(isinstance(section, dict), "baselines must be an object")
    if section.get("import") is not None:
        imported = section["import"]
        _require(isinstance(imported, dict), "baselines.import must be an object")
        _run(imported.get("dry_run"), "baselines.import.dry_run")
        _run(imported.get("run"), "baselines.import.run")
        _sha(imported.get("measured_commit"), "baselines.import.measured_commit")
        _digest(imported.get("summary_sha256"), "baselines.import.summary_sha256")
        files = imported.get("files")
        _require(
            isinstance(files, dict) and files,
            "baselines.import.files must be non-empty",
        )
        for path, digest in files.items():
            _require(
                isinstance(path, str)
                and path.startswith("bench/config/operators/")
                and path.endswith(".json"),
                f"baselines.import.files has an invalid path {path!r}",
            )
            _digest(digest, f"baselines.import.files[{path!r}]")
    _run(section.get("validation"), "baselines.validation")
    _str(section.get("base"), "baselines.base")
    if "mr_iid" in section:
        _require(
            isinstance(section["mr_iid"], str)
            and re.fullmatch(r"[0-9]+", section["mr_iid"]) is not None,
            "baselines.mr_iid must be a decimal MR IID string",
        )


def _validate_reference_repair(section: Any) -> None:
    _require(isinstance(section, dict), "reference_repair must be an object")
    _require(
        set(section) == {"status_qualification", "baseline_evidence"},
        "reference_repair must contain status_qualification and baseline_evidence",
    )
    for key in section:
        _str(section[key], f"reference_repair.{key}")


_VALIDATORS = {
    "tests": _validate_tests,
    "profile": _validate_profile,
    "attempts": _validate_attempts,
    "memory": _validate_memory,
    "review": _validate_review,
    "baselines": _validate_baselines,
    "reference_repair": _validate_reference_repair,
}


def validate(evidence: Any) -> dict:
    """Return ``evidence`` if it matches the schema, else raise EvidenceError."""

    _require(isinstance(evidence, dict), "evidence must be an object")
    _require(
        evidence.get("schema") == SCHEMA_VERSION,
        f"evidence.schema must be {SCHEMA_VERSION}",
    )
    unknown = set(evidence) - {"schema", *SECTIONS}
    _require(not unknown, f"evidence has unknown sections: {sorted(unknown)}")
    for name in SECTIONS:
        if name in evidence:
            _VALIDATORS[name](evidence[name])
    return evidence


def empty() -> dict:
    return {"schema": SCHEMA_VERSION}


def dumps(evidence: Mapping[str, Any]) -> str:
    """Canonical JSON so the embedded record is byte-stable across refreshes."""

    return json.dumps(evidence, sort_keys=True, ensure_ascii=False, indent=2) + "\n"


def commits(evidence: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Every recorded revision as ``(field, sha)``, for ancestry checks."""

    found = []
    if "tests" in evidence:
        found.append(("tests.commit", evidence["tests"]["commit"]))
    for index, item in enumerate(evidence.get("attempts", ())):
        found.append((f"attempts[{index}].commit", item["commit"]))
    if "review" in evidence:
        found.append(("review.commit", evidence["review"]["commit"]))
        if evidence["review"]["applied"]:
            found.append(("review.tests.commit", evidence["review"]["tests"]["commit"]))
    imported = evidence.get("baselines", {}).get("import")
    if imported:
        found.append(("baselines.import.measured_commit", imported["measured_commit"]))
    return found


# ------------------------------------------------------------------ derivations


def derive_bottleneck(profile: Mapping[str, Any]) -> str:
    return (
        "Memory-bound"
        if profile["memory_pct"] >= profile["compute_pct"]
        else "Compute-bound"
    )


def render_profile(profile: Mapping[str, Any]) -> str:
    text = (
        f"Nsight Compute: `{profile['hot_kernel']}` takes "
        f"{profile['hot_share_pct']:.1f}% of profiled time at "
        f"{profile['memory_pct']:.1f}% Memory SOL versus "
        f"{profile['compute_pct']:.1f}% Compute SOL."
    )
    top = profile.get("findings") or []
    if top:
        text += (
            f" Top finding: {top[0]['rule']} "
            f"(estimated {float(top[0]['speedup_pct']):.0f}% speedup)."
        )
    return text


def _hot_sol(profile: Mapping[str, Any] | None) -> float:
    if not profile:
        return 0.0
    return max(profile["memory_pct"], profile["compute_pct"])


def lead_exhaustion(
    attempts: Sequence[Mapping[str, Any]], profile: Mapping[str, Any] | None
) -> GateResult:
    """Ridge on the current profile, or three measured strikes after the last win."""

    if _hot_sol(profile) >= RIDGE_PCT:
        return GateResult(
            PASS,
            f"hot kernel at ridge: {_hot_sol(profile):.1f}% SOL >= {RIDGE_PCT:g}%",
        )
    last_kept = max(
        (i for i, item in enumerate(attempts) if item["outcome"] == "kept"),
        default=-1,
    )
    after = last_kept + 1
    strikes = sum(1 for item in attempts[after:] if item["outcome"] == "struck")
    if strikes >= STRIKES_TO_EXHAUST:
        return GateResult(PASS, f"{strikes} measured strikes since the last kept win")
    return GateResult(
        GAP,
        f"{strikes} strike(s) since the last kept win and hot-kernel SOL "
        f"{_hot_sol(profile):.1f}% < {RIDGE_PCT:g}%",
    )


def draft_learnings(attempts: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    kept = sorted(
        (a for a in attempts if a["outcome"] == "kept"),
        key=lambda a: (-a["median_speedup"], a["lead"]),
    )
    struck = sorted(
        (a for a in attempts if a["outcome"] == "struck"),
        key=lambda a: (a["median_speedup"], a["lead"]),
    )
    lines = []
    for item in kept:
        lines.append(
            f"{item['lead']} was kept at {item['median_speedup']:.2f}x median "
            f"({item['min_speedup']:.2f}x to {item['max_speedup']:.2f}x over "
            f"{item['cases']} cases)."
        )
    for item in struck:
        lines.append(
            f"{item['lead']} was struck at {item['median_speedup']:.2f}x median "
            f"over {item['cases']} cases."
        )
    return tuple(lines[:MAX_LEARNINGS])


def memory_totals(memory: Mapping[str, Any]) -> dict:
    increase = 0
    new_sites = []
    for site in memory["sites"]:
        before, after = site.get("before_peak_bytes"), site.get("after_peak_bytes")
        if before is None and after:
            new_sites.append(site["site"])
        # Per-site so one path's decrease can never hide another path's growth.
        increase += max(0, (after or 0) - (before or 0))
    return {
        "increase": increase,
        "new_sites": sorted(new_sites),
        "unattributed": memory["after_unattributed_bytes"] > 0
        or memory["before_unattributed_bytes"] > 0,
    }


def memory_gate(memory: Mapping[str, Any] | None) -> GateResult:
    if not memory:
        return GateResult(GAP, "no memory record")
    totals = memory_totals(memory)
    reasons = []
    if totals["unattributed"]:
        reasons.append("allocations without a CV-CUDA call site")
    if totals["new_sites"]:
        reasons.append("new allocation sites: " + ", ".join(totals["new_sites"]))
    if totals["increase"] > MEMORY_LIMIT_BYTES:
        reasons.append(f"{totals['increase']} B exceeds {MEMORY_LIMIT_BYTES} B")
    if reasons:
        return GateResult(MANUAL, "human review required: " + "; ".join(reasons))
    return GateResult(PASS, f"attributable peak increase {totals['increase']} B")


def render_memory(memory: Mapping[str, Any]) -> str:
    totals = memory_totals(memory)
    return "\n".join(
        [
            "## Memory footprint",
            f"Peak attributable increase: {totals['increase']} B",
            "New runtime CUDA allocations/frees: "
            + ("yes" if totals["new_sites"] or totals["unattributed"] else "no"),
            f"Evidence: nsys per-site peak on `{memory['workload']}` "
            f"(baseline {memory['before_report_sha256'][:12]}, "
            f"candidate {memory['after_report_sha256'][:12]}).",
        ]
    )


def pixelwise_gate(
    tests: Mapping[str, Any] | None, candidate: str | None
) -> GateResult:
    if not tests:
        return GateResult(GAP, "no tests record")
    if candidate and tests["commit"] != candidate:
        return GateResult(
            GAP, f"tests ran at {tests['commit'][:12]}, not candidate {candidate[:12]}"
        )
    for key in ("gtest", "pytest"):
        if key in tests and tests[key]["exit_code"]:
            return GateResult(GAP, f"{key} exited {tests[key]['exit_code']}")
    failed = [c["id"] for c in tests["cases"] if c["outcome"] == "failed"]
    if failed:
        return GateResult(GAP, f"{len(failed)} failed test(s): {', '.join(failed[:3])}")
    covered, proving = set(), 0
    for case in tests["cases"]:
        if (
            case["outcome"] == "passed"
            and case["reference"]
            and EQUALITY_ASSERTIONS.intersection(case["assertions"])
        ):
            proving += 1
            covered.update(tuple(c) for c in case["categories"])
    missing = [
        "/".join(c)
        for c in (tuple(x) for x in tests["required_categories"])
        if c not in covered
    ]
    if missing:
        return GateResult(
            GAP, "categories without referenced passing tests: " + ", ".join(missing)
        )
    return GateResult(
        PASS,
        f"{proving} passing referenced "
        f"tests cover all {len(tests['required_categories'])} benchmarked categories",
    )


def review_gate(review: Mapping[str, Any] | None) -> GateResult:
    if not review:
        return GateResult(GAP, "no review record")
    if review["assess"]["exit_code"] not in (0, 1) or review["assess_verdict"] == "GAP":
        return GateResult(GAP, "refactor assess did not complete cleanly")
    if not review["applied"]:
        return GateResult(
            PASS, f"refactor assess {review['assess_verdict']}; no refactor applied"
        )
    if review["verify_verdict"] != "PASS" or review["verify"]["exit_code"] != 0:
        return GateResult(GAP, "applied refactor did not pass --phase verify")
    tests = pixelwise_gate(review["tests"], review["commit"])
    if tests.status != PASS:
        return GateResult(GAP, "post-refactor tests: " + tests.detail)
    bench = review["benchmark"]
    if bench["regressions"]:
        return GateResult(
            GAP, f"post-refactor benchmark has {bench['regressions']} regression(s)"
        )
    return GateResult(
        PASS,
        f"refactor applied; verify PASS; tests pass; {bench['cases']} benchmark cases "
        "without regression",
    )


def baseline_import_gate(
    baselines: Mapping[str, Any] | None, repair: Mapping[str, Any] | None = None
) -> GateResult:
    if repair:
        return GateResult(PASS, repair["baseline_evidence"])
    imported = (baselines or {}).get("import")
    if not imported:
        return GateResult(GAP, "no guarded baseline import record")
    if imported["dry_run"]["exit_code"] or imported["run"]["exit_code"]:
        return GateResult(GAP, "baseline import did not exit cleanly")
    return GateResult(
        PASS,
        f"imported {len(imported['files'])} baseline file(s) measured at "
        f"{imported['measured_commit'][:12]}"
        + (
            f" with reviewed reset for !{baselines.get('mr_iid', '?')}"
            if "--allow-regressions" in imported["run"]["command"]
            else ""
        ),
    )


def baseline_files_mismatch(
    evidence: Mapping[str, Any], read_text: Callable[[str], str | None]
) -> str | None:
    """The first imported baseline file whose current bytes differ from the record."""

    imported = (evidence.get("baselines") or {}).get("import") or {}
    for path, digest in imported.get("files", {}).items():
        data = read_text(path)
        if data is None or hashlib.sha256(data.encode("utf-8")).hexdigest() != digest:
            return path
    return None


def baseline_validation_gate(baselines: Mapping[str, Any] | None) -> GateResult:
    run = (baselines or {}).get("validation")
    if not run:
        return GateResult(GAP, "no baseline validation record")
    if run["exit_code"]:
        return GateResult(GAP, f"validate_baselines exited {run['exit_code']}")
    detail = f"validate_baselines --reject-regressions-from {baselines['base']} passed"
    if "--allow-regressions" in run["command"]:
        detail += f" with reviewed reset for !{baselines.get('mr_iid', '?')}"
    return GateResult(PASS, detail)


# ------------------------------------------------------------------ checklist


def checklist(
    evidence: Mapping[str, Any],
    labels: Sequence[str],
    *,
    candidate: str | None = None,
    retention_detail: str | None = None,
    exhaustion: GateResult | None = None,
) -> tuple[tuple[str, bool, str], ...]:
    """Checklist rows ``(label, checked, text)``, each derived from one gate."""

    results = {
        "Pixelwise equality to reference": pixelwise_gate(
            evidence.get("tests"), candidate
        ),
        "Memory-footprint checks": memory_gate(evidence.get("memory")),
        "Baselines updated": baseline_import_gate(
            evidence.get("baselines"), evidence.get("reference_repair")
        ),
        "Baselines verified unchanged": GateResult(
            PASS if retention_detail else GAP,
            retention_detail or "no reviewed retention policy",
        ),
        "Baseline validation": baseline_validation_gate(evidence.get("baselines")),
        "Lead exhaustion": exhaustion
        or lead_exhaustion(evidence.get("attempts", ()), evidence.get("profile")),
        "Review/refactor gate": review_gate(evidence.get("review")),
    }
    return tuple(
        (label, results[label].status == PASS, results[label].detail)
        for label in labels
    )
