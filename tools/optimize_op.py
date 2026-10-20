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
"""Deterministic definition-of-done checker for an optimization campaign.

Gates the *edges* of an optimization campaign per .agents/guidance/OPTIMIZATION_GUIDELINES.md — it does not
choose or implement optimizations (that is the agent/GPU-driven loop), and it does not
write the MR summary (an internal generator derives that from committed baselines).

  --phase preflight : readiness BEFORE optimizing (correctness + bench coverage + baseline
                      + profiling availability). Reuses tools/review_op.py where present.
  --phase evidence  : definition-of-done AFTER optimizing, decided from the evidence
                      records embedded in the summary (recorded reference tests per
                      benchmarked category, baseline import, results-summary format,
                      generalization review, leads exhausted or deferred, memory
                      footprint, review/refactor), plus API/ABI and perf: commit hygiene.

  python3 tools/optimize_op.py <Operator> --phase preflight|evidence
          [--base <ref>] [--results <file>] [--mr-iid <iid>]
          [--format md|json] [--out PATH]
"""

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GUIDE = ".agents/guidance/OPTIMIZATION_GUIDELINES.md"
_SUMMARY_REF = ".agents/skills/op-tools/reference/optimize-op-summary.md"
_PROFILING_REF = ".agents/skills/op-tools/reference/optimize-op-profiling.md"
# The one section an agent must read to resolve each gate; the full guide is never required.
GATE_GUIDELINES = {
    "PRE-1": GUIDE + "#2-establish-coverage-and-baseline",
    "PRE-2": GUIDE + "#2-establish-coverage-and-baseline",
    "PRE-3": GUIDE + "#2-establish-coverage-and-baseline",
    "PRE-4": _PROFILING_REF + "#profiling",
    "PRE-5": GUIDE + "#1-survey-and-scope",
    "ODO-0": GUIDE + "#golden-rules",
    "ODO-1": GUIDE + "#golden-rules",
    "ODO-2": GUIDE + "#6-finish-the-mr",
    "ODO-3": _SUMMARY_REF + "#results-summary-format",
    "ODO-4": GUIDE + "#accept-drop-and-stop-criteria",
    "ODO-5": GUIDE + "#1-survey-and-scope",
    "ODO-6": GUIDE + "#commit-hygiene-odo-6",
    "ODO-7": GUIDE + "#6-finish-the-mr",
    "ODO-8": GUIDE + "#4-run-the-optimization-loop",
    "ODO-9": _SUMMARY_REF + "#memory-footprint",
    "ODO-10": _SUMMARY_REF + "#metadata-contract",
    "ODO-11": _SUMMARY_REF + "#results-summary-format",
    "ODO-12": _SUMMARY_REF + "#results-summary-format",
}


AGENT_TOOLS = REPO / ".agents" / "tools"
if str(AGENT_TOOLS) not in sys.path:
    sys.path.insert(0, str(AGENT_TOOLS))
BENCH_TOOLS = REPO / "bench"
if str(BENCH_TOOLS) not in sys.path:
    sys.path.insert(0, str(BENCH_TOOLS))
CI_TOOLS = REPO / "ci"
if str(CI_TOOLS) not in sys.path:
    sys.path.insert(0, str(CI_TOOLS))

import optimization_evidence as evidence_mod  # noqa: E402
from optimization_summary import (  # noqa: E402
    SummaryError,
    lead_exhaustion,
    parse_summary,
    validate_summary,
)

try:
    if not (CI_TOOLS / "optimization_secondary_scope_policy.py").is_file():
        raise ImportError("internal secondary-scope policy is unavailable")
    from optimization_secondary_scope_policy import (  # noqa: E402
        is_reviewed_secondary_scope,
    )
except ImportError:
    # The internal policy file is intentionally absent from the OSS mirror.
    # Missing policy must deny every multi-operator exception.
    def is_reviewed_secondary_scope(*_args):
        return False


try:
    if not (CI_TOOLS / "optimization_reference_policy.py").is_file():
        raise ImportError("internal reference policy is unavailable")
    from optimization_reference_policy import (  # noqa: E402
        is_reviewed_reference_exception,
    )
except ImportError:

    def is_reviewed_reference_exception(*_args, **_kwargs):
        return False


from binding_api import binding_api_snapshot as _binding_api_snapshot  # noqa: E402
from operator_source_map import SHARED_KERNEL_SOURCES  # noqa: E402
from _internal.quality import DEFAULT_BENCHMARK_QUALITY  # noqa: E402

PASS, GAP, NA, MANUAL = "PASS", "GAP", "N-A", "MANUAL"
_MALFORMED_GIT_JSON = object()


@dataclass
class Finding:
    id: str
    phase: str
    status: str
    summary: str
    evidence: str = ""
    guideline: str = ""
    fix: str = ""

    def __post_init__(self):
        self.guideline = GATE_GUIDELINES.get(self.id, self.guideline)


# --------------------------------------------------------------------------- io / git helpers
def read(p: Path):
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except (OSError, AttributeError):
        return None


def read_results(path):
    """Read an MR description from a path or stdin (``-``)."""
    if not path:
        return None
    if path == "-":
        return sys.stdin.read()
    return read(Path(path))


def rel(p: Path) -> str:
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


def git(*args):
    try:
        r = subprocess.run(
            ["git", *args], cwd=str(REPO), capture_output=True, text=True
        )
        return r.stdout if r.returncode == 0 else ""
    except OSError:
        return ""


def git_success(*args):
    """Return whether a read-only git command completed successfully."""
    try:
        return (
            subprocess.run(
                ["git", *args], cwd=str(REPO), capture_output=True, text=True
            ).returncode
            == 0
        )
    except OSError:
        return False


def git_json(ref, path):
    """Load JSON from git; distinguish absent blobs from malformed content."""
    try:
        result = subprocess.run(
            ["git", "show", f"{ref}:{path}"],
            cwd=str(REPO),
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return _MALFORMED_GIT_JSON


def resolve_commit(ref):
    return git("rev-parse", "--verify", f"{ref}^{{commit}}").strip()


def summary_context():
    candidate = resolve_commit("HEAD")
    if not candidate:
        raise SummaryError("Cannot resolve the checkout commit")
    source = candidate
    target = None
    if os.environ.get("CI_MERGE_REQUEST_EVENT_TYPE") in {
        "merged_result",
        "merge_train",
    }:
        source = os.environ.get("CI_MERGE_REQUEST_SOURCE_BRANCH_SHA", "")
        target = os.environ.get("CI_MERGE_REQUEST_TARGET_BRANCH_SHA") or os.environ.get(
            "CI_MERGE_REQUEST_DIFF_BASE_SHA", ""
        )
        if (
            os.environ.get("CI_COMMIT_SHA") != candidate
            or not source
            or not target
            or resolve_commit(source) != source
            or resolve_commit(target) != target
            or not git_success("merge-base", "--is-ancestor", source, candidate)
            or not git_success("merge-base", "--is-ancestor", target, candidate)
        ):
            raise SummaryError(
                "Merge-context commits are missing or do not match the checkout"
            )
    return source, candidate, target


def grep(pattern, text, flags=0):
    if not text:
        return []
    rx = re.compile(pattern, flags)
    return [(i + 1, ln) for i, ln in enumerate(text.splitlines()) if rx.search(ln)]


# --------------------------------------------------------------------- operator resolution
@dataclass
class OpPaths:
    op: str
    Op: str
    header: Path
    hpp: Path
    pybind: Path
    bench_cfg: Path


def resolve_op(arg):
    stem = arg.strip()
    op = stem.lower()
    hdr_dir = REPO / "src/cvcuda/include/cvcuda"
    header, Op = None, stem
    if hdr_dir.is_dir():
        for h in sorted(hdr_dir.glob("Op*.h")):
            if h.stem[2:].lower() == op:
                header, Op = h, h.stem[2:]
                break
    if header is None:
        header = hdr_dir / f"Op{stem}.h"
    return OpPaths(
        op=op,
        Op=Op,
        header=header,
        hpp=hdr_dir / f"Op{Op}.hpp",
        pybind=REPO / f"python/mod_cvcuda/operators/Op{Op}.cpp",
        bench_cfg=REPO / f"bench/config/operators/{op}.json",
    )


def review_op_findings(P, domain):
    """Run tools/review_op.py for a domain and return its findings (or None if unavailable)."""
    tool = REPO / "tools" / "review_op.py"
    if not tool.exists():
        return None
    try:
        r = subprocess.run(
            [sys.executable, str(tool), P.Op, "--domain", domain, "--format", "json"],
            cwd=str(REPO),
            capture_output=True,
            text=True,
        )
        return json.loads(r.stdout)["findings"]
    except (OSError, ValueError, KeyError):
        return None


def changed_paths(base, head="HEAD"):
    # --no-renames exposes both sides of a rename; without it a renamed kernel reports only its
    # destination and the source operator drops out of attribution.
    out = git("diff", "--name-only", "--no-renames", f"{base}...{head}")
    return [ln.strip() for ln in out.splitlines() if ln.strip()]


def all_operator_stems():
    """Operator stems from the bench manifest, for longest-prefix file attribution."""
    manifest = _load_json(REPO / "bench/config/bench_params.json") or {}
    return sorted((manifest.get("operators") or {}).keys())


def _is_op_kernel(path, P, all_ops):
    """A changed priv file belongs to <op> iff <op> is the LONGEST operator-name prefix of
    its filename body. Avoids both under-match (OpHQResizeKernel.cuh) and cross-op over-match
    (OpResizeCropConvertReformat.cu must not count as 'resize')."""
    pl = path.replace("\\", "/")
    if not pl.startswith("src/cvcuda/priv/"):
        return False
    rel = pl.removeprefix("src/cvcuda/priv/")
    if rel in SHARED_KERNEL_SOURCES.get(P.op, []):
        return True
    stem = Path(pl).stem.lower()
    body = stem[2:] if (stem.startswith("op") and "/legacy/" not in pl) else stem
    body = re.sub(r"[^a-z0-9]", "", body)
    candidates = [o for o in all_ops if body.startswith(o)] or (
        [P.op] if body.startswith(P.op) else []
    )
    best = max(candidates, key=len, default=None)
    return best == P.op


def _is_op_implementation(path, P, all_ops):
    """Return whether a changed path implements this operator's timed execution path."""
    normalized = path.replace("\\", "/")
    return normalized == rel(P.pybind).replace("\\", "/") or _is_op_kernel(
        normalized, P, all_ops
    )


def check_preflight(P, base):
    g = ".agents/guidance/OPTIMIZATION_GUIDELINES.md#establish-coverage-and-baseline"
    out = []

    # PRE-1/PRE-2: reuse review_op test + bench coverage
    for pid, domain, key_ids, label in [
        ("PRE-1", "test", {"TST-1", "TST-2"}, "Correctness/regression coverage"),
        (
            "PRE-2",
            "bench",
            {"BEN-4", "BEN-5", "BEN-6", "BEN-7"},
            "Benchmark coverage + baselines",
        ),
    ]:
        f = review_op_findings(P, domain)
        if f is None:
            out.append(
                Finding(
                    pid,
                    "preflight",
                    MANUAL,
                    f"{label}: run review_op.py (tools/review_op.py not present here)",
                    "reuse: python3 tools/review_op.py %s --domain %s" % (P.Op, domain),
                    g,
                )
            )
            continue
        gaps = [x for x in f if x["id"] in key_ids and x["status"] == "GAP"]
        if gaps:
            out.append(
                Finding(
                    pid,
                    "preflight",
                    GAP,
                    f"{label} incomplete",
                    "; ".join(f"{x['id']}:{x['summary']}" for x in gaps),
                    g,
                    "Add the missing coverage before optimizing (separate, non-perf commit).",
                )
            )
        else:
            out.append(
                Finding(
                    pid,
                    "preflight",
                    PASS,
                    f"{label} present (review_op {domain})",
                    "review_op %s domain: no blocking GAP in %s"
                    % (domain, sorted(key_ids)),
                    g,
                )
            )

    # PRE-3: committed baseline available
    cfg = read(P.bench_cfg)
    has_baseline = bool(cfg and re.search(r'"baselines"\s*:\s*\{[^}]*[A-Za-z]', cfg))
    out.append(
        Finding(
            "PRE-3",
            "preflight",
            PASS if has_baseline else GAP,
            (
                "Committed baselines available for the surface"
                if has_baseline
                else "No baselines in the op config"
            ),
            rel(P.bench_cfg),
            g,
            (
                ""
                if has_baseline
                else "Seed missing reference-SKU baseline rows before optimizing; "
                "reuse existing committed numbers."
            ),
        )
    )

    # PRE-4: profiling tools
    have = [t for t in ("ncu", "nsys") if shutil.which(t)]
    out.append(
        Finding(
            "PRE-4",
            "preflight",
            PASS if have else MANUAL,
            (
                "Profiler(s) available: " + ", ".join(have)
                if have
                else "ncu/nsys not on PATH"
            ),
            "PATH probe",
            ".agents/guidance/OPTIMIZATION_GUIDELINES.md#profiling",
            (
                ""
                if have
                else "Install/enable ncu or nsys, or record the unavailability in the MR per the guidelines."
            ),
        )
    )

    # PRE-5: API/ABI snapshot pointer (verified at evidence phase)
    out.append(
        Finding(
            "PRE-5",
            "preflight",
            MANUAL,
            "API/ABI must stay unchanged; verified at --phase evidence (ODO-5)",
            "public headers: %s, %s; binding: %s"
            % (rel(P.header), rel(P.hpp), rel(P.pybind)),
            ".agents/guidance/OPTIMIZATION_GUIDELINES.md#survey-and-scope",
        )
    )

    return out


# ============================================================================ EVIDENCE
def check_evidence(P, base, results_path, mr_iid=None):
    g = ".agents/guidance/OPTIMIZATION_GUIDELINES.md"
    out = []
    all_ops = all_operator_stems()
    changed = changed_paths(base)
    implementation_changed = [
        c for c in changed if _is_op_implementation(c, P, all_ops)
    ]

    if not implementation_changed:
        out.append(
            Finding(
                "ODO-0",
                "evidence",
                NA,
                "No optimization detected: no %s implementation change vs %s"
                % (P.Op, base),
                "changed implementation files: none",
                g,
            )
        )
        # still run the format/hygiene checks below where meaningful

    results = read_results(results_path)
    summary = _parse(results)
    evidence = summary.metadata.evidence if summary else {}

    # ODO-1: regression test + recorded pixelwise-equality run per benchmarked category
    test_f = review_op_findings(P, "test")
    tests_ok = test_f is not None and not [
        x for x in test_f if x["id"] in {"TST-1", "TST-2"} and x["status"] == "GAP"
    ]
    pixel = evidence_mod.pixelwise_gate(
        evidence.get("tests"), summary.metadata.candidate_commit if summary else None
    )
    if test_f is None:
        out.append(
            Finding(
                "ODO-1",
                "evidence",
                MANUAL,
                "Regression coverage: verify via review_op",
                "recorded tests: " + pixel.detail,
                g,
            )
        )
    elif implementation_changed and (not tests_ok or pixel.status != PASS):
        out.append(
            Finding(
                "ODO-1",
                "evidence",
                GAP,
                "Changed implementation lacks regression tests or a passing recorded reference run",
                "regression tests ok=%s; recorded tests: %s" % (tests_ok, pixel.detail),
                g,
                "Record the %s tests run at the candidate and refresh the summary."
                % P.Op,
            )
        )
    else:
        out.append(
            Finding(
                "ODO-1",
                "evidence",
                PASS,
                "Regression tests present + recorded pixelwise-equality run",
                pixel.detail,
                g,
            )
        )

    # ODO-2: baselines updated for changed configs
    base_diff = git(
        "diff", f"{base}...HEAD", "--", f"bench/config/operators/{P.op}.json"
    )
    baselines_touched = bool(
        re.search(r'gpu_time_us|gpu_gap_stddev_us|"baselines"|n_runs', base_diff)
    )
    retention, retention_error = None, None
    try:
        retention = _retention_evidence(summary, mr_iid)
    except (ValueError, OSError) as exc:
        retention_error = str(exc)
    reference_exception = False
    if (
        not retention_error
        and not retention
        and implementation_changed
        and not baselines_touched
        and summary
    ):
        try:
            source, candidate, target = summary_context()
            reference_exception = (
                not summary.metadata.baseline_retention
                and tuple(summary.metadata.operators) == (P.Op,)
                and summary.metadata.candidate_commit == source
                and is_reviewed_reference_exception(
                    summary,
                    repo=REPO,
                    mr_iid=mr_iid,
                    source_commit=source,
                    candidate_commit=candidate,
                    diff_base_commit=target or git("merge-base", base, "HEAD").strip(),
                )
            )
        except SummaryError:
            pass
    if retention_error:
        out.append(
            Finding(
                "ODO-2",
                "evidence",
                GAP,
                "Baseline retention proof rejected",
                retention_error,
                g,
            )
        )
    elif retention:
        out.append(
            Finding(
                "ODO-2", "evidence", PASS, "Baselines verified unchanged", retention, g
            )
        )
    elif not implementation_changed:
        out.append(
            Finding(
                "ODO-2",
                "evidence",
                NA,
                "No implementation change -> baseline update N-A",
                "",
                g,
            )
        )
    elif reference_exception:
        out.append(
            Finding(
                "ODO-2",
                "evidence",
                NA,
                "Baseline update N-A: pinned reference-snapshot exception validated",
                "ci/optimization_reference_policy.py passed; no fresh reference measurements",
                g,
            )
        )
    elif baselines_touched and _imported_baselines(evidence).status == PASS:
        out.append(
            Finding(
                "ODO-2",
                "evidence",
                PASS,
                "Baselines updated for the operator",
                _imported_baselines(evidence).detail,
                g,
            )
        )
    elif baselines_touched:
        out.append(
            Finding(
                "ODO-2",
                "evidence",
                GAP,
                "Baseline change has no matching guarded import record",
                _imported_baselines(evidence).detail,
                g,
                "Record the guarded %s baseline import and refresh the summary." % P.Op,
            )
        )
    else:
        out.append(
            Finding(
                "ODO-2",
                "evidence",
                GAP,
                "Implementation changed but baselines not updated [requires CI regen]",
                "no baseline delta in %s" % rel(P.bench_cfg),
                g,
                "Regenerate via the CI fan-out, then follow bench/README.md "
                "'Import, Validate, and Recheck' for the guarded %s import." % P.op,
            )
        )

    # ODO-4: lead exhaustion or reviewed deferrals
    out.append(_leads_exhausted(summary, g))

    # ODO-5: API/ABI unchanged
    out.append(_api_abi(P, base, g))

    # ODO-6: perf: commit hygiene
    out.append(_perf_hygiene(P, base, all_ops, g))

    # ODO-7: regression-surface deltas and comparable committed-baseline guard.
    out.append(_baseline_regression_gate(P, base, g))

    # ODO-8: review/refactor lock-in gate
    out.append(_review_refactor_gate(P, evidence, implementation_changed, g))

    # ODO-9: memory-footprint growth gate
    out.append(_memory_footprint_gate(summary, g))

    touched_ops = _changed_perf_operators(changed, all_ops)
    out.append(_operator_scope_gate(P, touched_ops, results, mr_iid, g))

    # ODO-11: rerun the exact FakePlanar pairing contract at the final gate.
    # Benchmark coverage can drift after preflight as configs and baselines are
    # edited, so a preflight-only BEN-6 result is not durable evidence.
    out.append(_fake_planar_coverage_gate(P, g))

    # ODO-12: C++ timing alone cannot prove a Python binding improvement.
    out.append(_binding_impact_gate(P, results, implementation_changed, g))

    odo3 = _results_format(results, g, retention)
    insert_at = next((i for i, item in enumerate(out) if item.id == "ODO-4"), len(out))
    out.insert(insert_at, odo3)
    return out


def _parse(results):
    try:
        return parse_summary(results or "")
    except SummaryError:
        return None


def _imported_baselines(evidence):
    """The import record, and whether the committed configs still match it."""

    result = evidence_mod.baseline_import_gate(evidence.get("baselines"))
    if result.status != PASS:
        return result
    path = evidence_mod.baseline_files_mismatch(evidence, lambda p: read(REPO / p))
    if path:
        return evidence_mod.GateResult(
            GAP, f"{path} differs from the recorded baseline import"
        )
    return result


def _retention_evidence(summary, mr_iid):
    if summary is None:
        return None
    metadata = summary.metadata
    if not metadata.baseline_retention:
        return None
    if not (CI_TOOLS / "baseline_retention_policy.py").is_file():
        raise ValueError("reviewed baseline retention policy unavailable")
    from baseline_retention_policy import validate_retention

    source, candidate, _ = summary_context()
    if metadata.candidate_commit != source:
        raise ValueError("retention summary candidate must match source HEAD")
    return validate_retention(
        repo=REPO,
        policy_id=metadata.baseline_retention,
        mr_iid=mr_iid,
        operators=metadata.operators,
        baseline_commit=metadata.baseline_commit,
        source_commit=source,
        candidate_commit=candidate,
    )


def _fake_planar_coverage_gate(P, g):
    findings = review_op_findings(P, "bench")
    if findings is None:
        return Finding(
            "ODO-11",
            "evidence",
            MANUAL,
            "Final FakePlanar coverage could not be rerun",
            f"python3 tools/review_op.py {P.Op} --domain bench",
            g,
        )
    ben6 = next((item for item in findings if item["id"] == "BEN-6"), None)
    if ben6 is None:
        return Finding(
            "ODO-11",
            "evidence",
            MANUAL,
            "Final FakePlanar coverage result is missing",
            "review_op bench output has no BEN-6 finding",
            g,
        )
    status = ben6.get("status")
    summary = ben6.get("summary", "")
    if status == GAP:
        return Finding(
            "ODO-11",
            "evidence",
            GAP,
            "Final FakePlanar coverage is incomplete",
            f"BEN-6:{summary}",
            g,
            "Add the exact advanced same-tier native/FakePlanar pairs, then rerun evidence.",
        )
    if status == NA:
        return Finding(
            "ODO-11",
            "evidence",
            NA,
            "FakePlanar comparison is not applicable",
            f"BEN-6:{summary}",
            g,
        )
    if status == PASS:
        return Finding(
            "ODO-11",
            "evidence",
            PASS,
            "Final FakePlanar coverage is exact",
            f"BEN-6:{summary}",
            g,
        )
    return Finding(
        "ODO-11",
        "evidence",
        MANUAL,
        "Final FakePlanar coverage status is unresolved",
        f"BEN-6 status={status!r}: {summary}",
        g,
    )


def _binding_impact_gate(P, results, implementation_changed, g):
    gid = "ODO-12"
    binding = rel(P.pybind).replace("\\", "/")
    changed = {path.replace("\\", "/") for path in implementation_changed}
    if changed != {binding}:
        return Finding(
            gid,
            "evidence",
            NA,
            "Python-overhead significance gate applies to binding-only optimizations",
            "changed implementation files=" + ", ".join(sorted(changed or {"none"})),
            g,
        )

    try:
        summary = parse_summary(results or "")
    except SummaryError as exc:
        return Finding(
            gid,
            "evidence",
            GAP,
            "Binding-only impact cannot be validated from the optimization summary",
            str(exc),
            g,
            "Generate a valid optimization summary.",
        )

    metadata = summary.metadata
    baseline = git_json(metadata.baseline_commit, rel(P.bench_cfg))
    candidate = _load_json(P.bench_cfg)
    sku_map = _load_json(REPO / "bench/config/sku_map.json")
    if not isinstance(baseline, dict) or not isinstance(candidate, dict):
        return Finding(
            gid,
            "evidence",
            GAP,
            "Binding-only impact benchmark data is unavailable",
            f"baseline={metadata.baseline_commit}; candidate={rel(P.bench_cfg)}",
            g,
            "Restore the declared baseline/candidate benchmark artifacts and rerun.",
        )
    entries = sku_map.get("entries") if isinstance(sku_map, dict) else None
    references = [
        item.get("stem")
        for item in (entries or [])
        if isinstance(item, dict) and isinstance(item.get("stem"), str)
    ]
    if not references:
        return Finding(
            gid,
            "evidence",
            GAP,
            "Binding-only impact has no configured reference SKUs",
            "bench/config/sku_map.json contains no valid stems",
            g,
        )

    def payload(config, case_key, sku):
        config_key = case_key.split("[", 1)[0]
        try:
            value = config["configs"][config_key]["baselines"][case_key][sku]
        except (KeyError, TypeError):
            return None
        return value if isinstance(value, dict) else None

    def metrics(value, *, reference=False):
        fields = ("gpu_time_us_cpp", "gpu_time_us_python", "n_runs")
        try:
            result = tuple(float(value[field]) for field in fields)
        except (KeyError, TypeError, ValueError):
            return None
        cpp, python, runs = result
        if (
            not all(math.isfinite(item) for item in result)
            or cpp <= 0
            or python <= 0
            or runs < 1
            or not runs.is_integer()
        ):
            return None
        if reference and "gpu_gap_stddev_us" not in value:
            return cpp, python, None, runs
        try:
            gap_stddev = float(value["gpu_gap_stddev_us"])
        except (KeyError, TypeError, ValueError):
            return None
        if not math.isfinite(gap_stddev) or gap_stddev < 0 or runs < 2:
            return None
        return cpp, python, gap_stddev, runs

    evidence = []
    failures = []
    for case_key in sorted(
        {
            key
            for config in (baseline, candidate)
            for entry in (config.get("configs") or {}).values()
            for key in (entry.get("baselines") or {})
        }
    ):
        short = case_key.split("[", 1)[0]
        for sku in references:
            before = metrics(payload(baseline, case_key, sku), reference=True)
            after = metrics(payload(candidate, case_key, sku))
            if before is None or after is None:
                failures.append(
                    f"{sku} {short}: missing timing/paired-gap-dispersion/run metrics"
                )
                continue
            b_cpp, b_python, b_gap_stddev, b_runs = before
            a_cpp, a_python, a_gap_stddev, a_runs = after
            before_gap = b_python - b_cpp
            after_gap = a_python - a_cpp
            reduction = before_gap - after_gap
            variance = a_gap_stddev**2 / a_runs
            error_label = "paired-gap combined SE"
            if b_gap_stddev is None:
                error_label = "candidate SE vs fixed committed reference"
            else:
                variance += b_gap_stddev**2 / b_runs
            standard_error = math.sqrt(variance)
            reasons = []
            if reduction <= standard_error:
                reasons.append("reduction does not clear standard error")
            if DEFAULT_BENCHMARK_QUALITY.absolute_parity_exceeds_limit(after_gap):
                reasons.append(
                    "candidate gap exceeds "
                    f"{DEFAULT_BENCHMARK_QUALITY.max_perf_diff_us:.0f} us"
                )
            line = (
                f"{sku} {short}: gap {before_gap:.2f}->{after_gap:.2f} us; "
                f"reduction {reduction:.2f} us; {error_label} "
                f"{standard_error:.2f} us"
            )
            evidence.append(line)
            if reasons:
                failures.append(line + " (" + "; ".join(reasons) + ")")
    if not evidence and not failures:
        failures.append("No benchmark cases provide binding-overhead measurements")

    if failures:
        return Finding(
            gid,
            "evidence",
            GAP,
            "Binding-only Python-overhead improvement is not proven",
            _clip_output("; ".join(failures)),
            g,
            "Use the committed reference timings. Collect repeated paired C++/Python "
            "candidate artifacts and prove every case's overhead reduction clears "
            "standard error on every reference SKU and passes parity. "
            "Generalization deferrals do not waive this gate.",
        )
    return Finding(
        gid,
        "evidence",
        PASS,
        "Binding-only Python-overhead improvement clears paired-gap standard error",
        _clip_output("; ".join(evidence)),
        g,
    )


def _results_format(results, g, retention_detail):
    gid = "ODO-3"
    if not results:
        return Finding(
            gid,
            "evidence",
            GAP,
            "Results summary not provided (pass --results <file> or the MR description)",
            "expected one bounded cvcuda-optimize-summary:v3 block",
            g,
            "Generate the summary with the mr-perf-summary generator and put it in the MR description.",
        )
    try:
        summary = parse_summary(results)
        source, candidate, _target = summary_context()
    except SummaryError as exc:
        return Finding(
            gid,
            "evidence",
            GAP,
            "Results summary or checkout context is missing or malformed",
            str(exc),
            g,
            "Regenerate the bounded block with the mr-perf-summary generator.",
        )

    metadata = summary.metadata
    if metadata.baseline_commit == metadata.candidate_commit:
        return Finding(
            gid,
            "evidence",
            GAP,
            "Summary baseline and candidate revisions are identical",
            f"commit={metadata.candidate_commit}",
            g,
            "Choose the committed pre-optimization benchmark revision and refresh the summary.",
        )
    if not git_success("merge-base", "--is-ancestor", metadata.baseline_commit, source):
        return Finding(
            gid,
            "evidence",
            GAP,
            "Summary baseline is not an ancestor of the source candidate",
            f"baseline={metadata.baseline_commit}; source={source}; candidate={candidate}",
            g,
            "Choose the committed pre-optimization benchmark revision and refresh the summary.",
        )
    result = validate_summary(
        summary,
        expected_candidate_commit=source,
        retention_detail=retention_detail,
    )
    if not result.ok:
        errors = [
            f"{getattr(item, 'code', 'summary')}: {getattr(item, 'message', item)}"
            for item in result.errors
        ]
        return Finding(
            gid,
            "evidence",
            GAP,
            "Versioned optimization summary failed deterministic validation",
            "; ".join(errors[:8]) + (" ..." if len(errors) > 8 else ""),
            g,
            "Record the missing evidence, refresh the summary, and rerun the gate.",
        )
    warnings = [
        f"{getattr(item, 'code', 'summary')}: {getattr(item, 'message', item)}"
        for item in result.warnings
    ]
    return Finding(
        gid,
        "evidence",
        PASS,
        "Versioned optimization summary is structurally and numerically valid",
        "operators=%s; improved=%d; regressed=%d%s"
        % (
            ", ".join(metadata.operators),
            sum(row.improved for row in summary.scope_rows),
            sum(row.regressed for row in summary.scope_rows),
            "; warnings: " + "; ".join(warnings) if warnings else "",
        ),
        g,
    )


def _leads_exhausted(summary, g):
    gid = "ODO-4"
    if summary is None:
        return Finding(
            gid, "evidence", GAP, "Lead exhaustion: no parsable summary", "", g
        )
    result = lead_exhaustion(summary.metadata, summary.generalization_rows)
    if result.status != PASS:
        return Finding(
            gid,
            "evidence",
            GAP,
            "Lead exhaustion is not established by the recorded attempts or review",
            result.detail,
            g,
            "Record attempts until the profile is at ridge or three post-win strikes, "
            "or defer remaining generalizations in the completed review.",
        )
    return Finding(
        gid,
        "evidence",
        PASS,
        "Lead exhaustion established by the records",
        result.detail,
        g,
    )


def _review_refactor_gate(P, evidence, implementation_changed, g):
    gid = "ODO-8"
    if not implementation_changed:
        return Finding(
            gid,
            "evidence",
            NA,
            "No implementation change -> review/refactor lock-in gate N-A",
            "",
            g,
        )
    result = evidence_mod.review_gate(evidence.get("review"))
    if result.status != PASS:
        return Finding(
            gid,
            "evidence",
            GAP,
            "Review/refactor lock-in gate is not established by the review record",
            result.detail,
            g,
            "Record the %s refactor review (and its verify run after a refactor) "
            "and refresh the summary." % P.Op,
        )
    return Finding(
        gid, "evidence", PASS, "Review/refactor record passes", result.detail, g
    )


def _memory_footprint_gate(summary, g):
    gid = "ODO-9"
    memory = summary.metadata.evidence.get("memory") if summary else None
    result = evidence_mod.memory_gate(memory)
    summaries = {
        PASS: "Memory footprint change is within the 10 MB limit",
        MANUAL: "Memory footprint needs human review",
        GAP: "Memory footprint record missing",
    }
    return Finding(
        gid,
        "evidence",
        result.status,
        summaries[result.status],
        result.detail,
        g,
        (
            ""
            if result.status == PASS
            else "Record the before/after memory profile and refresh the summary."
        ),
    )


def _finding_status(findings, finding_id):
    return next((item.status for item in findings if item.id == finding_id), MANUAL)


def _case_count(config):
    if not isinstance(config, dict):
        return 0
    return sum(
        len(item.get("baselines", {}))
        for item in config.get("configs", {}).values()
        if isinstance(item, dict) and isinstance(item.get("baselines", {}), dict)
    )


def _changed_perf_operators(paths, all_ops):
    """Attribute deterministic per-operator config, binding, and priv deltas.

    Deliberately narrower than operator_source_map.operators_for_paths, which asks "whose
    benchmarks can this change move" and therefore fans a shared kernel out to every consumer.
    ODO-10 asks the different question "is this MR authored against one operator", so it counts
    only files whose own name identifies the operator. Widening it here would make every
    Gaussian campaign look multi-operator, since Gaussian's kernel lives in the shared
    legacy/filter.cu alongside AverageBlur and Laplacian.
    """
    touched = set()
    known = set(all_ops)
    for path in paths:
        match = re.fullmatch(r"bench/config/operators/([a-z0-9]+)\.json", path)
        if match and match.group(1) in known:
            touched.add(match.group(1))
            continue
        normalized = path.replace("\\", "/")
        match = re.fullmatch(
            r"python/mod_cvcuda/operators/Op([A-Za-z0-9]+)\.cpp", normalized
        )
        if match:
            body = re.sub(r"[^a-z0-9]", "", match.group(1).lower())
            if body in known:
                touched.add(body)
            continue
        if not normalized.startswith("src/cvcuda/priv/Op"):
            continue
        body = re.sub(r"[^a-z0-9]", "", Path(normalized).stem[2:].lower())
        candidates = [op for op in all_ops if body.startswith(op)]
        if candidates:
            touched.add(max(candidates, key=len))
    return touched


def _operator_scope_gate(P, touched_ops, results, mr_iid, guide):
    """Enforce one operator unless one exact secondary scope was reviewed."""

    primary = P.op.casefold()
    actual = {operator.casefold() for operator in touched_ops}
    declared = ()
    metadata_operator = P.Op
    documented = False
    retained = False
    try:
        summary = parse_summary(results or "")
        metadata_operator = summary.metadata.primary
        declared = tuple(summary.metadata.operators[1:])
        if getattr(summary.metadata, "baseline_retention", None):
            try:
                retained = bool(_retention_evidence(summary, mr_iid))
            except (ValueError, OSError):
                retained = False
        reported = []
        for row in summary.impact_rows:
            operator = str(row.operator).casefold()
            if operator not in reported:
                reported.append(operator)
        documented = reported == [
            operator.casefold() for operator in summary.metadata.operators
        ]
    except SummaryError:
        pass

    if not declared:
        foreign = sorted(actual - {primary})
        mismatched = metadata_operator.casefold() != primary
        if mismatched:
            return Finding(
                "ODO-10",
                "evidence",
                GAP,
                "Summary declares a different operator than the one being gated",
                f"summary operator={metadata_operator}; gating={P.Op}",
                guide,
                "Run the gate against the operator the summary declares, or "
                "regenerate the summary for this operator.",
            )
        return Finding(
            "ODO-10",
            "evidence",
            GAP if foreign else PASS,
            (
                "Optimization MR spans multiple operators"
                if foreign
                else "Optimization MR is scoped to one operator"
            ),
            "attributed operators=" + ", ".join(sorted(actual or {primary})),
            guide,
            (
                "Split the work into one perf MR and one summary per operator."
                if foreign
                else ""
            ),
        )

    normalized_declared = {operator.casefold() for operator in declared}
    expected = {primary, *normalized_declared}
    reviewed = metadata_operator.casefold() == primary and is_reviewed_secondary_scope(
        mr_iid, metadata_operator, declared
    )
    exact_changes = actual == expected or (retained and actual <= expected)
    if reviewed and exact_changes and documented:
        return Finding(
            "ODO-10",
            "evidence",
            PASS,
            "Optimization MR uses a code-reviewed secondary baseline scope",
            "MR !%s; %s operators=%s"
            % (
                mr_iid,
                "verified retained" if retained else "attributed",
                ", ".join(sorted(expected if retained else actual)),
            ),
            guide,
        )

    reasons = []
    if not reviewed:
        reasons.append("declaration is not allowlisted for this exact MR/primary pair")
    if not exact_changes:
        reasons.append(
            "declared operators=%s but attributed operators=%s"
            % (
                ", ".join(sorted(expected)),
                ", ".join(sorted(actual)) or "none",
            )
        )
    if not documented:
        reasons.append(
            "summary carries no %s table for the declared operators" % "Impact"
        )
    return Finding(
        "ODO-10",
        "evidence",
        GAP,
        "Optimization MR secondary operator scope is not authorized",
        "; ".join(reasons),
        guide,
        "Use one operator, or obtain a code-reviewed exact MR/primary/secondary "
        "policy entry, make the changed baseline configs match the declaration, and "
        "regenerate the summary so every declared operator "
        "carries its own measured impact row.",
    )


def _api_abi(P, base, g):
    gid = "ODO-5"
    paths = [
        f"src/cvcuda/include/cvcuda/Op{P.Op}.h",
        f"src/cvcuda/include/cvcuda/Op{P.Op}.hpp",
        f"python/mod_cvcuda/operators/Op{P.Op}.cpp",
    ]
    public_diff = git("diff", f"{base}...HEAD", "--", *paths[:2])
    binding_diff = git("diff", f"{base}...HEAD", "--", paths[2])
    sig = [
        ln
        for ln in public_diff.splitlines()
        if ln[:1] in "+-"
        and ln[1:2] != ln[:1]
        and re.search(r"CVCUDA_PUBLIC|operator\(\)|Submit\s*\(", ln)
    ]
    if not public_diff.strip() and not binding_diff.strip():
        return Finding(
            gid,
            "evidence",
            PASS,
            "API/ABI surface unchanged (no diff in public headers/binding)",
            "no changes in " + ", ".join(Path(p).name for p in paths),
            g,
        )
    if sig:
        return Finding(
            gid,
            "evidence",
            GAP,
            "Public API/ABI signature(s) changed — that is feature work, not optimization",
            "; ".join(s.strip()[:80] for s in sig[:3]),
            g,
            "Revert the signature change or split it into a separate feature MR.",
        )

    if binding_diff.strip():
        merge_base = git("merge-base", base, "HEAD").strip()
        baseline_source = git("show", f"{merge_base}:{paths[2]}") if merge_base else ""
        candidate_source = git("show", f"HEAD:{paths[2]}")
        baseline_surface = _binding_api_snapshot(baseline_source, P.Op)
        candidate_surface = _binding_api_snapshot(candidate_source, P.Op)
        if baseline_surface is None or candidate_surface is None:
            return Finding(
                gid,
                "evidence",
                MANUAL,
                "Python binding surface could not be parsed safely",
                "unable to compare ExportOp%s registrations and bound callable signatures"
                % P.Op,
                g,
            )
        if baseline_surface != candidate_surface:
            changed_parts = []
            if baseline_surface[0] != candidate_surface[0]:
                changed_parts.append("m.def registration/arguments/defaults")
            if baseline_surface[1] != candidate_surface[1]:
                changed_parts.append("bound callable signature(s)")
            if baseline_surface[2] != candidate_surface[2]:
                changed_parts.append("bound callable type alias(es)")
            return Finding(
                gid,
                "evidence",
                GAP,
                "Public Python API signature(s) changed — that is feature work, not optimization",
                "; ".join(changed_parts),
                g,
                "Revert the signature change or split it into a separate feature MR.",
            )

    if not public_diff.strip() and binding_diff.strip():
        return Finding(
            gid,
            "evidence",
            PASS,
            "API/ABI surface unchanged (binding implementation only)",
            "Op%s.cpp changed with identical m.def registrations and bound callable signatures"
            % P.Op,
            g,
        )
    return Finding(
        gid,
        "evidence",
        MANUAL,
        "Public headers changed but no signature lines detected — verify API/ABI stability",
        "diff in " + ", ".join(Path(p).name for p in paths[:2]),
        g,
    )


def _baseline_regression_gate(P, base, g):
    gid = "ODO-7"
    tool = REPO / "bench" / "_internal" / "validate_baselines.py"
    if not tool.exists():
        return Finding(
            gid,
            "evidence",
            GAP,
            "Committed-baseline regression gate could not run",
            rel(tool) + " not found",
            g,
            "Restore bench/_internal/validate_baselines.py and rerun the evidence gate.",
        )

    cmd = [
        sys.executable,
        str(tool),
        "--config-dir",
        str(REPO / "bench" / "config"),
        "--reject-regressions-from",
        base,
    ]
    try:
        r = subprocess.run(
            cmd, cwd=str(REPO), capture_output=True, text=True, timeout=120
        )
    except subprocess.TimeoutExpired as exc:
        return Finding(
            gid,
            "evidence",
            GAP,
            "Committed-baseline regression gate timed out",
            _clip_output(str(exc)),
            g,
            "Fix the baseline validation hang or run the validator separately, "
            "then rerun the evidence gate.",
        )
    except OSError as exc:
        return Finding(
            gid,
            "evidence",
            GAP,
            "Committed-baseline regression gate could not run",
            str(exc),
            g,
            "Fix the local Python/script environment and rerun the evidence gate.",
        )

    command = " ".join(
        [
            "python3",
            "bench/_internal/validate_baselines.py",
            "--reject-regressions-from",
            base,
        ]
    )
    evidence = _clip_output("\n".join(x for x in (r.stdout, r.stderr) if x.strip()))
    if r.returncode == 0:
        return Finding(
            gid,
            "evidence",
            PASS,
            "Comparable committed baselines do not regress vs %s" % base,
            command + ("\n" + evidence if evidence else ""),
            ".agents/guidance/OPTIMIZATION_GUIDELINES.md#golden-rules",
        )

    return Finding(
        gid,
        "evidence",
        GAP,
        "Comparable committed baseline regression(s) detected or validation failed",
        command + f"\nexit={r.returncode}" + ("\n" + evidence if evidence else ""),
        ".agents/guidance/OPTIMIZATION_GUIDELINES.md#golden-rules",
        "Fix the performance regression or revert the regressed baseline row(s). "
        "Only new/non-comparable baseline rows may bypass this check.",
    )


def _cpp_formatting_tokens(source):
    """Conservative lexical evidence, not a preprocessor or device-code proof.

    Keep literals, comments, directives and possible macro invocations verbatim.
    Unknown syntax fails closed instead of guessing whether whitespace matters.
    """
    if "\\\n" in source or "\\\r\n" in source or "__LINE__" in source or "??" in source:
        return None
    # Separate newlines so whitespace cannot swallow the next directive's indentation.
    token = re.compile(
        r"(?P<directive>^[ \t]*(?:#|%:)[^\n]*)"
        r"|(?P<space>[^\S\n]+|\n)"
        r"|(?P<comment>//[^\n]*|/\*[\s\S]*?\*/)"
        r'|(?P<raw>(?:u8|u|U|L)?R"(?P<delimiter>[^ ()\\\t\r\n]{0,16})'
        r'\([\s\S]*?\)(?P=delimiter)"[a-zA-Z_0-9]*)'
        r'|(?P<literal>(?:u8|u|U|L)?(?:"(?:\\[^\r\n]|[^"\\\r\n])*"'
        r"|'(?:\\[^\r\n]|[^'\\\r\n])*')[a-zA-Z_0-9]*)"
        r"|(?P<identifier>[a-zA-Z_][a-zA-Z_0-9]*)"
        r"|(?P<number>(?:[0-9]|\.[0-9])(?:[eEpP][+-]|[a-zA-Z_0-9.]|\'[a-zA-Z_0-9])*)"
        r"|(?P<punct>%:%:|<<<=|>>>=|<<<|>>>|<=>|->\*|<<=|>>=|\.\.\."
        r"|::|->|\.\*|\+\+|--|<<|>>|<=|>=|==|!=|&&|\|\||\+=|-=|\*=|/=|%=|&=|\^=|\|=|##"
        r"|<:|:>|<%|%>|%:|[{}\[\]().,;:?~!+*/%<>=&^|\-])",
        re.MULTILINE,
    )
    tokens = []
    pos = 0
    while pos < len(source):
        match = token.match(source, pos)
        if match is None or (
            source.startswith("/*", pos) and match.lastgroup != "comment"
        ):
            return None
        if match.lastgroup != "space":
            tokens.append((match.group(), match.start(), match.end()))
        pos = match.end()
    # A macro may stringify its arguments: identical tokens alone are insufficient.
    keywords = {
        "if",
        "constexpr",
        "for",
        "while",
        "switch",
        "catch",
        "sizeof",
        "alignof",
        "decltype",
        "noexcept",
        "typeid",
        "requires",
        "static_assert",
    }
    opaque = []
    for i, (value, start, _) in enumerate(tokens[:-1]):
        if value in keywords or not re.fullmatch(r"[a-zA-Z_][a-zA-Z_0-9]*", value):
            continue
        # Preprocessing replaces comments with whitespace before invoking macros.
        arguments_start = i + 1
        while arguments_start < len(tokens) and tokens[arguments_start][0].startswith(
            ("/*", "//")
        ):
            arguments_start += 1
        if arguments_start == len(tokens) or tokens[arguments_start][0] != "(":
            continue
        depth = 0
        for value, _, end in tokens[arguments_start:]:
            depth += (value == "(") - (value == ")")
            if depth == 0:
                opaque.append(source[start:end])
                break
        else:
            return None
    return [value for value, _, _ in tokens], opaque


def _cpp_formatting_only(before, after):
    before_tokens = _cpp_formatting_tokens(before)
    return before_tokens is not None and before_tokens == _cpp_formatting_tokens(after)


def _formatting_only_implementation(sha, files):
    """Require existing regular C++ files and inspect checked parent/commit blobs."""
    if not files or any(
        Path(f).suffix not in {".cu", ".cuh", ".cpp", ".cc", ".cxx", ".h", ".hpp"}
        for f in files
    ):
        return False
    for path in files:
        raw = subprocess.run(
            ["git", "diff-tree", "--no-commit-id", "--raw", "-r", sha, "--", path],
            cwd=REPO,
            capture_output=True,
            text=True,
        )
        fields = raw.stdout.split()
        if (
            raw.returncode
            or len(fields) != 6
            or fields[0] != ":100644"
            or fields[1] != "100644"
            or fields[4] != "M"
        ):
            return False
        sources = []
        for revision in (sha + "^", sha):
            blob = subprocess.run(
                ["git", "show", revision + ":" + path],
                cwd=REPO,
                capture_output=True,
            )
            if blob.returncode:
                return False
            try:
                sources.append(blob.stdout.decode("utf-8"))
            except UnicodeDecodeError:
                return False
        if not _cpp_formatting_only(*sources):
            return False
    return True


def _perf_hygiene(P, base, all_ops, g):
    gid = "ODO-6"
    merge_base = git("merge-base", base, "HEAD").strip()
    if not merge_base:
        return Finding(
            gid,
            "evidence",
            MANUAL,
            "Unable to resolve the changed commit range",
            f"git merge-base {base} HEAD returned no commit",
            g,
        )
    log = git("log", f"{merge_base}..HEAD", "--no-merges", "--format=%H%x1f%s")
    commits = [ln.split("\x1f") for ln in log.splitlines() if "\x1f" in ln]
    if not commits:
        return Finding(gid, "evidence", MANUAL, "No commits vs base to check", "", g)
    bad = []
    regression_fixes = []
    formatting_only = []
    compiler_builds = []
    for sha, subj in commits:
        files = git("show", "--name-only", "--format=", sha).splitlines()
        implementation_files = [
            f.strip()
            for f in files
            if f.strip() and _is_op_implementation(f.strip(), P, all_ops)
        ]
        if not implementation_files or re.match(r"^perf(?:\([^)]+\))?:\s+\S", subj):
            continue
        if _formatting_only_implementation(sha, implementation_files):
            formatting_only.append(sha[:8])
            continue
        if re.match(r"^build(?:\([^)]*\))?:", subj):
            accepted, detail = _compiler_compatibility_build(P, all_ops, sha, subj)
            if accepted:
                compiler_builds.append("%s: %s" % (sha[:8], detail))
            else:
                bad.append("%s %s (%s)" % (sha[:8], subj[:60], detail))
            continue
        accepted, detail = _regression_backed_fix(P, all_ops, sha, subj)
        if accepted:
            regression_fixes.append("%s -> %s" % (detail[:8], sha[:8]))
        else:
            bad.append("%s %s (%s)" % (sha[:8], subj[:60], detail))
    if bad:
        return Finding(
            gid,
            "evidence",
            GAP,
            "Implementation-changing commit(s) lack perf: or a scoped regression-backed fix",
            "; ".join(bad[:3]),
            g,
            "Use perf: for kept optimization commits. A kernel bug fix must be a scoped "
            "fix(<operator>): commit immediately after its scoped, operator-specific, "
            "tests-only test(<operator>): regression commit.",
        )
    detail = "%d commit(s) checked" % len(commits)
    if regression_fixes:
        detail += "; regression-backed fix pair(s): " + ", ".join(regression_fixes)
    if formatting_only:
        detail += "; formatting-only implementation changes: " + ", ".join(
            formatting_only
        )
    if compiler_builds:
        return Finding(
            gid,
            "evidence",
            MANUAL,
            "Scoped compiler-compatibility build(s) require evidence review",
            detail + "; " + "; ".join(compiler_builds),
            g,
            "Verify the linked failing-parent CI job and the report hash, supported-toolchain "
            "parent-fail/candidate-pass results, and semantic/device-code equivalence. "
            "Trailer claims are not proof and do not automatically pass this gate.",
        )
    return Finding(
        gid,
        "evidence",
        PASS,
        "Implementation-changing commits use perf: or a scoped regression-backed fix pair",
        detail,
        g,
    )


def _compiler_compatibility_build(P, all_ops, sha, subject):
    """Route narrowly scoped compiler evidence to human review, never automatic PASS."""
    build = re.fullmatch(r"build\(([^()]+)\):\s+\S.*", subject)
    if not build or build.group(1).casefold() != P.op.casefold():
        return False, "compiler compatibility requires build(<exact operator>):"
    parents = git("show", "-s", "--format=%P", sha).split()
    if len(parents) != 1:
        return False, "compiler compatibility must have exactly one immediate parent"

    changes = git(
        "diff-tree", "--no-commit-id", "--raw", "--no-abbrev", "--no-renames", "-r", sha
    ).splitlines()
    if not changes:
        return False, "compiler compatibility has no translation-unit changes"
    for change in changes:
        fields = change.split("\t", 1)
        if len(fields) != 2:
            return False, "cannot parse compiler-compatibility changed files"
        metadata, path = fields
        modes = metadata.split()
        if (
            len(modes) != 5
            or modes[0:2] != [":100644", "100644"]
            or modes[-1] != "M"
            or Path(path).suffix not in {".cu", ".cpp", ".cc", ".cxx"}
            or not _is_op_kernel(path, P, all_ops)
            or path.removeprefix("src/cvcuda/priv/")
            in SHARED_KERNEL_SOURCES.get(P.op, ())
        ):
            return (
                False,
                "compiler compatibility may only modify existing private operator translation units",
            )

    trailers = git("show", "-s", "--format=%(trailers:only,unfold)", sha)
    failures = re.findall(r"^Compiler-Failure:\s*(.*)$", trailers, re.M)
    verifications = re.findall(r"^Compiler-Verification:\s*(.*)$", trailers, re.M)
    if len(failures) != 1 or len(verifications) != 1:
        return (
            False,
            "exactly one Compiler-Failure and Compiler-Verification trailer required",
        )
    failure = re.fullmatch(
        r"([0-9a-f]{40}) (https://\S+/-/jobs/[1-9][0-9]*)", failures[0]
    )
    if not failure or failure.group(1) != parents[0]:
        return (
            False,
            "Compiler-Failure must name the full immediate-parent SHA and a CI job URL",
        )
    verification = re.fullmatch(
        r"parent-fail/candidate-pass/codegen ([0-9a-f]{64}) (\S+)", verifications[0]
    )
    if not verification:
        return (
            False,
            "Compiler-Verification must name parent-fail/candidate-pass/codegen, report SHA256 and reference",
        )
    return True, "failed parent %s %s; verification SHA256 %s %s" % (
        parents[0],
        failure.group(2),
        verification.group(1),
        verification.group(2),
    )


def _regression_backed_fix(P, all_ops, sha, subject):
    """Accept a kernel fix only when its immediate parent is its tests-only repro."""
    fix = re.match(r"^fix\(([^()]+)\):\s+\S", subject)
    if not fix:
        return False, "not a scoped fix(<operator>): subject"
    scope = fix.group(1).casefold()
    if scope != P.op.casefold():
        return False, "fix scope does not match %s" % P.op

    parents = git("show", "-s", "--format=%P", sha).split()
    if len(parents) != 1:
        return False, "fix must have exactly one immediate parent"
    parent = parents[0]
    parent_subject = git("show", "-s", "--format=%s", parent).strip()
    test = re.match(r"^test\(([^()]+)\):\s+\S", parent_subject)
    if not test:
        return False, "immediate parent is not a scoped test(<operator>): commit"
    test_scope = test.group(1).casefold()
    if test_scope != scope:
        return False, "test and fix scopes do not match"

    test_files = [
        f.strip()
        for f in git("show", "--name-only", "--format=", parent).splitlines()
        if f.strip()
    ]
    if not test_files or any(
        not f.replace("\\", "/").startswith("tests/") for f in test_files
    ):
        return False, "immediate parent is not tests-only"
    if not any(_is_op_test(f, P, all_ops) for f in test_files):
        return False, "immediate parent has no operator-specific regression test"
    return True, parent


def _is_op_test(path, P, all_ops):
    """Return whether a tests/ filename belongs to the requested operator."""
    pl = path.replace("\\", "/")
    if not pl.startswith("tests/"):
        return False
    body = re.sub(r"[^a-z0-9]", "", Path(pl).stem.lower())
    for prefix in ("testop", "test"):
        if body.startswith(prefix):
            body = body.removeprefix(prefix)
            break
    candidates = [op for op in all_ops if body.startswith(op)] or (
        [P.op] if body.startswith(P.op) else []
    )
    return max(candidates, key=len, default=None) == P.op


# ----------------------------------------------------------------------------- small utils
def _load_json(p):
    t = read(p)
    if not t:
        return None
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        return None


def _clip_output(text, limit=1200):
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n... output truncated ..."


# ================================================================================ driver
def render_md(P, phase, findings):
    icon = {PASS: "✅", GAP: "❌", NA: "➖", MANUAL: "🔍"}
    lines = [f"# optimize-op: {P.Op} — phase={phase}", ""]
    counts = {}
    for f in findings:
        counts[f.status] = counts.get(f.status, 0) + 1
        lines.append(
            f"- {icon.get(f.status, '?')} **{f.status}** `{f.id}` — {f.summary}"
        )
        if f.evidence:
            lines.append(f"    - evidence: {f.evidence}")
        if f.status == GAP and f.fix:
            lines.append(f"    - fix: {f.fix}")
        if f.guideline:
            lines.append(f"    - ref: {f.guideline}")
    gaps, man = counts.get(GAP, 0), counts.get(MANUAL, 0)
    verdict = "PASS" if gaps == 0 and man == 0 else ("GAPS" if gaps else "NEEDS-REVIEW")
    lines += [
        "",
        f"**{phase} verdict: {verdict}** — "
        + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())),
        "",
        "Done = a re-run shows zero GAP and zero unresolved MANUAL.",
    ]
    return "\n".join(lines)


def render_json(P, phase, findings):
    counts = {}
    for f in findings:
        counts[f.status] = counts.get(f.status, 0) + 1
    return json.dumps(
        {
            "operator": P.Op,
            "op": P.op,
            "phase": phase,
            "findings": [vars(f) for f in findings],
            "counts": counts,
            "exit_gap": counts.get(GAP, 0),
            "exit_manual": counts.get(MANUAL, 0),
            "exit_blocking": counts.get(GAP, 0)
            + (counts.get(MANUAL, 0) if phase == "evidence" else 0),
        },
        indent=2,
        sort_keys=True,
    )


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Deterministic DoD checker for an optimization campaign (CV-CUDA)."
    )
    ap.add_argument("operator", help="Operator name (PascalCase, e.g. Resize)")
    ap.add_argument("--phase", required=True, choices=["preflight", "evidence"])
    ap.add_argument(
        "--base",
        default="main",
        help="git base ref for the changed-set diff (default: main)",
    )
    ap.add_argument(
        "--results",
        default=None,
        help="path to the MR description, or '-' for stdin",
    )
    ap.add_argument(
        "--mr-iid",
        default=None,
        help="GitLab MR IID used by the evidence-phase secondary-scope policy",
    )
    ap.add_argument("--format", default="md", choices=["md", "json"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    P = resolve_op(args.operator)
    if args.phase == "preflight":
        findings = check_preflight(P, args.base)
    else:
        findings = check_evidence(P, args.base, args.results, args.mr_iid)

    report = (
        render_md(P, args.phase, findings)
        if args.format == "md"
        else render_json(P, args.phase, findings)
    )
    print(report)
    if args.out:
        Path(args.out).write_text(report + "\n", encoding="utf-8")
    blocking = {GAP, MANUAL} if args.phase == "evidence" else {GAP}
    return 1 if any(f.status in blocking for f in findings) else 0


if __name__ == "__main__":
    sys.exit(main())
