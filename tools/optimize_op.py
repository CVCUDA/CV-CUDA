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
"""Deterministic definition-of-done checker for a single-operator optimization campaign.

Gates the *edges* of an optimization campaign per .agents/guidance/OPTIMIZATION_GUIDELINES.md — it does not
choose or implement optimizations (that is the agent/GPU-driven loop). Three phases:

  --phase preflight : readiness BEFORE optimizing (correctness + bench coverage + baseline
                      + profiling availability). Reuses tools/review_op.py where present.
  --phase evidence  : definition-of-done AFTER optimizing (regression test + pixelwise
                      evidence per changed config, baselines updated, results-summary format,
                      leads exhausted, memory-footprint and review/refactor gate evidence,
                      API/ABI unchanged, perf: commit hygiene).
  --phase summary   : initialize or refresh the bounded v1 MR optimization summary while
                      preserving its human-authored assessment, evidence, and learnings.

Read-only and deterministic (no network, no clocks, no randomness; the changed-set comes
from `git diff <base>..HEAD`). Emits PASS/GAP/N-A/MANUAL per item with evidence + a guideline
cite; the final evidence phase exits non-zero on any GAP or unresolved MANUAL. The agent
re-runs `--phase evidence` and uses the verdict to drive its optimize loop until the DoD holds.

Usage:
  python3 tools/optimize_op.py <Operator> --phase preflight|evidence|summary
          [--base <ref>] [--results <file>] [--mr-iid <iid>]
          [--benchmark-base <ref>] [--optimized-cases-file <file>] [--state provisional|final]
          [--bottleneck Memory-bound|Compute-bound] [--profile-evidence <text>]
          [--impact-metric cpp_time|python_overhead] [--secondary-operator <op>]...
          [--format md|json] [--out PATH]
"""

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GUIDE = ".agents/guidance/OPTIMIZATION_GUIDELINES.md"

AGENT_TOOLS = REPO / ".agents" / "tools"
if str(AGENT_TOOLS) not in sys.path:
    sys.path.insert(0, str(AGENT_TOOLS))
BENCH_TOOLS = REPO / "bench"
if str(BENCH_TOOLS) not in sys.path:
    sys.path.insert(0, str(BENCH_TOOLS))
CI_TOOLS = REPO / "ci"
if str(CI_TOOLS) not in sys.path:
    sys.path.insert(0, str(CI_TOOLS))

from optimization_summary import (  # noqa: E402
    SummaryError,
    SummaryMetadata,
    generate_summary,
    parse_summary,
    refresh_summary,
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


from binding_api import binding_api_snapshot as _binding_api_snapshot  # noqa: E402
from operator_source_map import SHARED_KERNEL_SOURCES  # noqa: E402
from _internal.quality import DEFAULT_BENCHMARK_QUALITY  # noqa: E402

PASS, GAP, NA, MANUAL = "PASS", "GAP", "N-A", "MANUAL"
# 10 MB in decimal bytes.
MEMORY_AUTO_ACCEPT_BYTES = 10_000_000


@dataclass
class Finding:
    id: str
    phase: str
    status: str
    summary: str
    evidence: str = ""
    guideline: str = ""
    fix: str = ""


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
    """Load a JSON file from a git revision without modifying the worktree."""
    text = git("show", f"{ref}:{path}")
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def resolve_commit(ref):
    return git("rev-parse", "--verify", f"{ref}^{{commit}}").strip()


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


def changed_paths(base):
    out = git("diff", "--name-only", f"{base}...HEAD")
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


def _has_pixelwise_assertion(evidence):
    """Recognize concrete C++/Python equality assertions, not descriptive prose."""
    evidence = evidence or ""
    if re.search(r"\bEXPECT_(?:EQ|NEAR)\s*\(", evidence):
        return True
    if re.search(
        r"\b(?:cupy|cp|numpy|np)\.testing\."
        r"assert_(?:array_equal|allclose|equal)\s*\(",
        evidence,
        re.I,
    ):
        return True
    return bool(
        re.search(
            r"\bassert\s+(?:"
            r"bool\s*\("
            r"|(?:cupy|cp|numpy|np|torch)\.(?:all|array_equal|allclose|equal)\s*\("
            r"|[A-Za-z_]\w*(?:\.[A-Za-z_]\w*|\[[^\]]+\])*\s*"
            r"(?:==|!=|<=|>=|<|>|\bis\b))",
            evidence,
            re.I,
        )
    )


# ============================================================================ PREFLIGHT
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

    # PRE-3: baseline captured
    cfg = read(P.bench_cfg)
    has_baseline = bool(cfg and re.search(r'"baselines"\s*:\s*\{[^}]*[A-Za-z]', cfg))
    out.append(
        Finding(
            "PRE-3",
            "preflight",
            PASS if has_baseline else GAP,
            (
                "Baseline captured for the surface"
                if has_baseline
                else "No baselines in the op config"
            ),
            rel(P.bench_cfg),
            g,
            (
                ""
                if has_baseline
                else "Capture/commit a baseline (CI on the bench-only SHA) before optimizing."
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

    # ODO-1: regression test + pixelwise-equality evidence per changed config
    test_f = review_op_findings(P, "test")
    tests_ok = test_f is not None and not [
        x for x in test_f if x["id"] in {"TST-1", "TST-2"} and x["status"] == "GAP"
    ]
    pixel_checked, pixel_evidence = _checklist_item(
        results, "Pixelwise equality to reference"
    )
    pixel_hard_evidence = bool(
        pixel_checked
        and _has_pixelwise_assertion(pixel_evidence)
        and re.search(r"\b(?:reference|oracle|gold|host|cpu)\b", pixel_evidence, re.I)
        and re.search(r"\b(?:pass|passed|green|\d+\s*/\s*\d+)\b", pixel_evidence, re.I)
    )
    if test_f is None:
        out.append(
            Finding(
                "ODO-1",
                "evidence",
                MANUAL,
                "Regression coverage: verify via review_op; confirm checked pixelwise evidence",
                "pixelwise checklist checked=%s" % pixel_checked,
                g,
            )
        )
    elif implementation_changed and (not tests_ok or not pixel_hard_evidence):
        out.append(
            Finding(
                "ODO-1",
                "evidence",
                GAP,
                "Changed implementation lacks regression test and/or hard pixelwise-equality evidence",
                "regression tests ok=%s; checklist checked=%s; changed=%s"
                % (
                    tests_ok,
                    pixel_checked,
                    [Path(c).name for c in implementation_changed],
                ),
                g,
                "Run the regression test against an independent reference, then check the "
                "pixelwise item with the test, assertion, branches, and passing result.",
            )
        )
    else:
        out.append(
            Finding(
                "ODO-1",
                "evidence",
                PASS,
                "Regression tests present + checked pixelwise-equality hard evidence",
                pixel_evidence,
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
    if not implementation_changed:
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
    elif baselines_touched:
        out.append(
            Finding(
                "ODO-2",
                "evidence",
                PASS,
                "Baselines updated for the operator",
                "%s baselines changed vs %s" % (rel(P.bench_cfg), base),
                g,
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
                "Regenerate via the CI fan-out + bench/_internal/update_baseline.py "
                "--from <artifacts-dir>/ --operator %s." % P.op,
            )
        )

    # ODO-4: leads exhausted
    out.append(_leads_exhausted(results, g))

    # ODO-5: API/ABI unchanged
    out.append(_api_abi(P, base, g))

    # ODO-6: perf: commit hygiene
    out.append(_perf_hygiene(P, base, all_ops, g))

    # ODO-7: regression-surface deltas and comparable committed-baseline guard.
    out.append(_baseline_regression_gate(P, base, g))

    # ODO-8: review/refactor lock-in gate
    out.append(_review_refactor_gate(P, results, implementation_changed, g))

    # ODO-9: memory-footprint growth gate
    out.append(_memory_footprint_gate(results, implementation_changed, g))

    touched_ops = _changed_perf_operators(changed, all_ops)
    out.append(_operator_scope_gate(P, touched_ops, results, mr_iid, g))

    # ODO-11: rerun the exact FakePlanar pairing contract at the final gate.
    # Benchmark coverage can drift after preflight as configs and baselines are
    # edited, so a preflight-only BEN-6 result is not durable evidence.
    out.append(_fake_planar_coverage_gate(P, g))

    # ODO-12: a binding-only campaign must prove the metric it changes. C++
    # timing can remain flat while Python overhead moves materially, so require
    # every claimed target/reference-SKU reduction to clear combined standard
    # error rather than accepting a mechanically flat C++ Impact table.
    out.append(_binding_impact_gate(P, results, implementation_changed, g))

    evidence_results = {
        "Pixelwise equality to reference": _finding_status(out, "ODO-1"),
        "Memory-footprint checks": _finding_status(out, "ODO-9"),
        "Baselines updated": _finding_status(out, "ODO-2"),
        "Baseline validation": _finding_status(out, "ODO-7"),
        "Lead exhaustion": _finding_status(out, "ODO-4"),
        "Review/refactor gate": _finding_status(out, "ODO-8"),
    }

    # Insert ODO-3 in numeric order after the underlying evidence has been
    # evaluated, so checked boxes can be correlated with hard gate results.
    odo3 = _results_format(P, results, results_path, g, evidence_results)
    insert_at = next((i for i, item in enumerate(out) if item.id == "ODO-4"), len(out))
    out.insert(insert_at, odo3)
    return out


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

    def metrics(value):
        fields = (
            "gpu_time_us_cpp",
            "gpu_time_us_python",
            "gpu_gap_stddev_us",
            "n_runs",
        )
        try:
            result = tuple(float(value[field]) for field in fields)
        except (KeyError, TypeError, ValueError):
            return None
        cpp, python, gap_stddev, runs = result
        if (
            not all(math.isfinite(item) for item in result)
            or cpp <= 0
            or python <= 0
            or gap_stddev < 0
            or runs < 2
            or not runs.is_integer()
        ):
            return None
        return result

    evidence = []
    failures = []
    for case_key in metadata.optimized_cases:
        short = case_key.split("[", 1)[0]
        for sku in references:
            before = metrics(payload(baseline, case_key, sku))
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
            standard_error = math.sqrt(
                b_gap_stddev**2 / b_runs + a_gap_stddev**2 / a_runs
            )
            reasons = []
            if reduction <= standard_error:
                reasons.append("reduction does not clear combined standard error")
            if DEFAULT_BENCHMARK_QUALITY.absolute_parity_exceeds_limit(after_gap):
                reasons.append(
                    "candidate gap exceeds "
                    f"{DEFAULT_BENCHMARK_QUALITY.max_perf_diff_us:.0f} us"
                )
            line = (
                f"{sku} {short}: gap {before_gap:.2f}->{after_gap:.2f} us; "
                f"reduction {reduction:.2f} us; paired-gap combined SE "
                f"{standard_error:.2f} us"
            )
            evidence.append(line)
            if reasons:
                failures.append(line + " (" + "; ".join(reasons) + ")")

    if failures:
        return Finding(
            gid,
            "evidence",
            GAP,
            "Binding-only Python-overhead improvement is not proven",
            _clip_output("; ".join(failures)),
            g,
            "Collect repeated paired C++/Python reference/candidate artifacts, reimport "
            "their baselines, or improve the binding until every target/reference-SKU "
            "reduction clears paired-gap combined standard error and the candidate gap "
            "stays within the parity limit.",
        )
    return Finding(
        gid,
        "evidence",
        PASS,
        "Binding-only Python-overhead improvement clears paired-gap combined standard error",
        _clip_output("; ".join(evidence)),
        g,
    )


def _results_format(P, results, results_path, g, evidence_results):
    gid = "ODO-3"
    if not results:
        return Finding(
            gid,
            "evidence",
            GAP,
            "Results summary not provided (pass --results <file> or the MR description)",
            "expected one bounded cvcuda-optimize-summary:v1 block",
            g,
            "Generate the v1 summary with --phase summary and put it in the MR description.",
        )
    try:
        summary = parse_summary(results)
    except SummaryError as exc:
        return Finding(
            gid,
            "evidence",
            GAP,
            "Results summary is missing or malformed",
            str(exc),
            g,
            "Regenerate the bounded v1 block with --phase summary.",
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
    if not git_success("merge-base", "--is-ancestor", metadata.baseline_commit, "HEAD"):
        return Finding(
            gid,
            "evidence",
            GAP,
            "Summary baseline is not an ancestor of the candidate",
            f"baseline={metadata.baseline_commit}; candidate={resolve_commit('HEAD')}",
            g,
            "Choose the committed pre-optimization benchmark revision and refresh the summary.",
        )
    baseline = git_json(metadata.baseline_commit, rel(P.bench_cfg))
    if baseline is None:
        return Finding(
            gid,
            "evidence",
            GAP,
            "Declared baseline benchmark config cannot be loaded",
            f"baseline={metadata.baseline_commit}:{rel(P.bench_cfg)}",
            g,
            "Choose a baseline commit containing valid benchmark config JSON "
            "for this operator, then refresh the summary.",
        )
    candidate = _load_json(P.bench_cfg)
    sku_map = _load_json(REPO / "bench/config/sku_map.json")
    result = validate_summary(
        summary,
        baseline_config=baseline,
        candidate_config=candidate,
        sku_map=sku_map,
        expected_operator=P.Op,
        expected_candidate_commit=resolve_commit("HEAD"),
        evidence_results=evidence_results,
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
            "Refresh the summary, supply the named hard evidence, and rerun the gate.",
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
        "optimized cases=%d; full cases=%d%s"
        % (
            len(metadata.optimized_cases),
            _case_count(candidate),
            "; warnings: " + "; ".join(warnings) if warnings else "",
        ),
        g,
    )


def _leads_exhausted(results, g):
    gid = "ODO-4"
    checked, evidence = _checklist_item(results, "Lead exhaustion")
    measured = bool(re.search(r"\d+(?:\.\d+)?\s*(?:%|x\b|u?s\b|ms\b)", evidence, re.I))
    at_ridge = bool(re.search(r"\b(?:at[- ]?ridge|ridge)\b", evidence, re.I))
    struck = bool(
        re.search(
            r"\b(?:3|three)\b[^\n]{0,120}\b(?:strike|struck|failed leads?)\b",
            evidence,
            re.I,
        )
    )
    if not checked or not measured or not (at_ridge or struck):
        return Finding(
            gid,
            "evidence",
            GAP,
            "Lead exhaustion lacks checked, measurement-backed hard evidence",
            evidence or "Lead exhaustion checklist item missing or unchecked",
            g,
            "Check the item only after citing measured at-ridge evidence or three "
            "measured post-win strikes.",
        )
    return Finding(
        gid,
        "evidence",
        PASS,
        "Lead exhaustion is checked with measurement-backed evidence",
        evidence,
        g,
    )


def _review_refactor_gate(P, results, implementation_changed, g):
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
    if not results:
        return Finding(
            gid,
            "evidence",
            GAP,
            "Review/refactor lock-in gate evidence missing",
            "provide --results with a Review/refactor gate section",
            g,
            "Run a review pass and `python3 tools/refactor_op.py %s --phase assess`, "
            "then record the disposition in the Results Summary." % P.Op,
        )

    checked, evidence = _checklist_item(results, "Review/refactor gate")
    if not checked:
        return Finding(
            gid,
            "evidence",
            GAP,
            "Review/refactor lock-in gate is missing or unchecked",
            evidence or "Review/refactor checklist item missing",
            g,
            "Run the review and refactor assessment, record hard evidence, then check it.",
        )

    missing = []
    if not re.search(r"(?:tools/)?refactor_op\.py", evidence):
        missing.append("refactor_op.py command")
    if not re.search(r"--phase\s+assess|\bphase\s*[=:]\s*assess\b", evidence, re.I):
        missing.append("--phase assess")
    if not re.search(
        r"\b(PASS|RECOMMENDATION|MANUAL|RED-\d+|recommendations?)\b",
        evidence,
        re.I,
    ):
        missing.append("recommendation disposition")

    applied = _refactor_applied(evidence)
    if applied:
        if not re.search(r"--phase\s+verify|\bphase\s*[=:]\s*verify\b", evidence, re.I):
            missing.append("--phase verify")
        if not re.search(
            r"\b(cvcuda_test|run_tests|pytest|frozen[- ]?test|tests?\s+passed)\b",
            evidence,
            re.I,
        ):
            missing.append("frozen operator tests")
        if not re.search(
            r"\b(run_bench|benchmark|benchmarks?\s+passed|no regression|performance preserved)\b",
            evidence,
            re.I,
        ):
            missing.append("post-refactor benchmark proof")

    if missing:
        return Finding(
            gid,
            "evidence",
            GAP,
            "Review/refactor lock-in gate evidence incomplete",
            "missing: " + ", ".join(missing),
            g,
            "Record the review pass, `refactor_op.py --phase assess` result, and if a "
            "refactor was applied, the verify/test/benchmark proof.",
        )

    return Finding(
        gid,
        "evidence",
        PASS,
        "Review/refactor lock-in gate evidence present",
        "refactor applied=%s; %s" % ("yes" if applied else "no", evidence),
        g,
    )


def _memory_footprint_sections(markdown):
    """Return visible, canonical Memory footprint sections from Markdown."""
    markdown = re.sub(r"<!--.*?(?:-->|$)", "", markdown or "", flags=re.S)
    visible = []
    fence = None
    for line in markdown.splitlines():
        marker = re.match(r"^\s*(`{3,}|~{3,})", line)
        if fence is None and marker:
            fence = (marker.group(1)[0], len(marker.group(1)))
            visible.append("")
        elif fence is not None:
            closing = re.fullmatch(r"\s*(`{3,}|~{3,})\s*", line)
            if (
                closing
                and closing.group(1)[0] == fence[0]
                and len(closing.group(1)) >= fence[1]
            ):
                fence = None
            visible.append("")
        else:
            visible.append(line)

    headings = [
        index
        for index, line in enumerate(visible)
        if re.fullmatch(r"## Memory footprint\s*", line)
    ]

    sections = []
    for start in headings:
        end = len(visible)
        for index in range(start + 1, len(visible)):
            if re.match(r"^#{1,2}\s+", visible[index]):
                end = index
                break
        sections.append("\n".join(visible[start:end]))
    return sections


def _memory_footprint_gate(results, _implementation_changed, g):
    gid = "ODO-9"
    sections = _memory_footprint_sections(results)
    if len(sections) != 1:
        return Finding(
            gid,
            "evidence",
            GAP,
            "Memory footprint section missing or duplicated",
            "expected exactly one visible `## Memory footprint` section; found %d"
            % len(sections),
            g,
            "Add one canonical section outside comments/code fences and remove duplicates.",
        )
    section = sections[0]

    fields = (
        ("Peak attributable increase", r"([0-9]{1,20}) B"),
        ("New runtime CUDA allocations/frees", r"(no|yes)"),
        ("Evidence", r"(.+?)"),
    )
    occurrences = {
        label: len(re.findall(r"^%s\s*:" % re.escape(label), section, re.M))
        for label, _ in fields
    }
    errors = [
        "%s: found %d occurrences" % (label, count)
        for label, count in occurrences.items()
        if count != 1
    ]
    values = {}
    if not errors:
        for label, value_pattern in fields:
            match = re.search(
                r"^%s:[ \t]*%s[ \t]*$" % (re.escape(label), value_pattern),
                section,
                re.M,
            )
            if not match:
                errors.append("%s: malformed declaration" % label)
                continue
            values[label] = match.group(1)

    if errors:
        return Finding(
            gid,
            "evidence",
            GAP,
            "Memory footprint declarations are missing, duplicated, or malformed",
            "; ".join(errors),
            g,
            "Provide exactly one canonical aggregate peak-live increase, allocation/free "
            "flag, and evidence line; do not net unrelated decreases.",
        )

    evidence = values["Evidence"].strip()
    if not evidence or re.fullmatch(
        r"(?:[-.]+|none|<[^>]+>|n/?a|tbd|todo|pending|placeholder|unknown|unmeasured|"
        r"not\s+(?:measured|available|provided)|"
        r"(?:measurement|evidence|review)(?:\s+is)?\s+"
        r"(?:pending|unknown|unavailable|not\s+(?:available|provided)))\.?",
        evidence,
        re.I,
    ):
        return Finding(
            gid,
            "evidence",
            GAP,
            "Memory footprint evidence is a placeholder",
            "Evidence: %s" % evidence,
            g,
            "Replace the placeholder with the measurement or source-inspection evidence.",
        )

    try:
        increase = int(values["Peak attributable increase"])
    except ValueError:
        return Finding(
            gid,
            "evidence",
            GAP,
            "Memory footprint declarations are missing, duplicated, or malformed",
            "Peak attributable increase: malformed declaration",
            g,
            "Use a non-negative decimal integer followed by `B`.",
        )
    new_allocations = values["New runtime CUDA allocations/frees"] == "yes"
    if increase > MEMORY_AUTO_ACCEPT_BYTES or new_allocations:
        reasons = []
        if increase > MEMORY_AUTO_ACCEPT_BYTES:
            reasons.append("%d B exceeds the 10 MB limit" % increase)
        if new_allocations:
            reasons.append("a new runtime CUDA allocation/free path is declared")
        return Finding(
            gid,
            "evidence",
            MANUAL,
            "Memory footprint change requires explicit human review",
            "%s; Evidence: %s" % ("; ".join(reasons), evidence),
            g,
            "Obtain explicit human approval outside this offline checker.",
        )

    return Finding(
        gid,
        "evidence",
        PASS,
        "Memory footprint change is within the 10 MB limit",
        "%d B increase; no new runtime CUDA allocations/frees; Evidence: %s"
        % (increase, evidence),
        g,
    )


def _checklist_item(results, label):
    if not results:
        return False, ""
    match = re.search(
        r"^- \[([ xX])\] \*\*%s\*\*\s+—\s+(.+?)\s*$" % re.escape(label),
        results,
        re.M,
    )
    if not match:
        return False, ""
    return match.group(1).lower() == "x", match.group(2).strip()


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
    """Attribute deterministic per-operator config, binding, and priv deltas."""
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
    try:
        metadata = parse_summary(results or "").metadata
        metadata_operator = metadata.operator
        declared = tuple(metadata.secondary_operators)
    except SummaryError:
        pass

    if not declared:
        foreign = sorted(actual - {primary})
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
                "Split the work into one perf MR and one v1 summary per operator."
                if foreign
                else ""
            ),
        )

    normalized_declared = {operator.casefold() for operator in declared}
    expected = {primary, *normalized_declared}
    reviewed = metadata_operator.casefold() == primary and is_reviewed_secondary_scope(
        mr_iid, metadata_operator, declared
    )
    exact_changes = actual == expected
    if reviewed and exact_changes:
        return Finding(
            "ODO-10",
            "evidence",
            PASS,
            "Optimization MR uses a code-reviewed secondary baseline scope",
            "MR !%s; attributed operators=%s" % (mr_iid, ", ".join(sorted(actual))),
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
    return Finding(
        "ODO-10",
        "evidence",
        GAP,
        "Optimization MR secondary operator scope is not authorized",
        "; ".join(reasons),
        guide,
        "Use one operator, or obtain a code-reviewed exact MR/primary/secondary "
        "policy entry and make the changed baseline configs match the declaration.",
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
    for sha, subj in commits:
        files = git("show", "--name-only", "--format=", sha).splitlines()
        touches_implementation = any(
            _is_op_implementation(f.strip(), P, all_ops) for f in files if f.strip()
        )
        if not touches_implementation or re.match(r"^perf(?:\([^)]+\))?:\s+\S", subj):
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
    return Finding(
        gid,
        "evidence",
        PASS,
        "Implementation-changing commits use perf: or a scoped regression-backed fix pair",
        detail,
        g,
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


def _refactor_applied(section):
    if re.search(
        r"no refactors? applied|refactors? applied\s*[:|=]\s*(?:no|false|n/a)\b",
        section,
        re.I,
    ):
        return False
    return bool(
        re.search(
            r"refactors? applied\s*[:|=]\s*(?:yes|true)\b|"
            r"applied refactor|refactor commit|RED-\d+\s+resolved",
            section,
            re.I,
        )
    )


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


def build_summary(P, args):
    """Initialize or refresh the canonical bounded MR summary."""
    candidate = _load_json(P.bench_cfg)
    sku_map = _load_json(REPO / "bench/config/sku_map.json")
    if candidate is None:
        raise SummaryError(
            f"cannot load candidate benchmark config: {rel(P.bench_cfg)}"
        )
    if sku_map is None:
        raise SummaryError("cannot load bench/config/sku_map.json")

    if args.results:
        description = read_results(args.results)
        if description is None:
            raise SummaryError(f"cannot read MR description: {args.results}")
        parsed = parse_summary(description)
        baseline = git_json(parsed.metadata.baseline_commit, rel(P.bench_cfg))
        if baseline is None:
            raise SummaryError(
                "cannot load baseline benchmark config at "
                f"{parsed.metadata.baseline_commit}:{rel(P.bench_cfg)}"
            )
        return refresh_summary(
            description,
            baseline_config=baseline,
            candidate_config=candidate,
            sku_map=sku_map,
            state=args.state,
            candidate_commit=resolve_commit("HEAD"),
            impact_metric=args.impact_metric,
            secondary_operators=args.secondary_operator,
        )

    missing = [
        flag
        for flag, value in (
            ("--benchmark-base", args.benchmark_base),
            ("--optimized-cases-file", args.optimized_cases_file),
            ("--state", args.state),
            ("--bottleneck", args.bottleneck),
            ("--profile-evidence", args.profile_evidence),
        )
        if not value
    ]
    if missing:
        raise SummaryError("initializing a summary requires " + ", ".join(missing))

    baseline_commit = resolve_commit(args.benchmark_base)
    candidate_commit = resolve_commit("HEAD")
    if not baseline_commit:
        raise SummaryError(f"cannot resolve benchmark base {args.benchmark_base!r}")
    if not candidate_commit:
        raise SummaryError("cannot resolve HEAD")
    if baseline_commit == candidate_commit:
        raise SummaryError("benchmark base must precede HEAD, not equal it")
    if not git_success(
        "merge-base", "--is-ancestor", baseline_commit, candidate_commit
    ):
        raise SummaryError("benchmark base must be an ancestor of HEAD")
    baseline = git_json(baseline_commit, rel(P.bench_cfg))
    if baseline is None:
        raise SummaryError(
            f"cannot load baseline benchmark config at {baseline_commit}:{rel(P.bench_cfg)}"
        )
    try:
        optimized_cases = tuple(
            line.strip()
            for line in Path(args.optimized_cases_file)
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        )
    except OSError as exc:
        raise SummaryError(
            f"cannot read optimized cases file {args.optimized_cases_file}: {exc}"
        ) from exc
    metadata = SummaryMetadata(
        operator=P.Op,
        state=args.state,
        baseline_commit=baseline_commit,
        candidate_commit=candidate_commit,
        optimized_cases=optimized_cases,
        impact_metric=args.impact_metric or "cpp_time",
        secondary_operators=tuple(args.secondary_operator or ()),
    )
    return generate_summary(
        metadata,
        baseline_config=baseline,
        candidate_config=candidate,
        sku_map=sku_map,
        bottleneck=args.bottleneck,
        profile_evidence=args.profile_evidence,
    )


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Deterministic DoD checker for an optimization campaign (CV-CUDA)."
    )
    ap.add_argument("operator", help="Operator name (PascalCase, e.g. Resize)")
    ap.add_argument(
        "--phase", required=True, choices=["preflight", "evidence", "summary"]
    )
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
        "--benchmark-base",
        default=None,
        help="pre-optimization benchmark revision used to initialize a summary",
    )
    ap.add_argument(
        "--optimized-cases-file",
        default=None,
        help="newline-delimited exact expanded case keys used to initialize a summary",
    )
    ap.add_argument("--state", choices=["provisional", "final"], default=None)
    ap.add_argument(
        "--impact-metric",
        choices=["cpp_time", "python_overhead"],
        default=None,
        help="Impact metric for summary initialization or migration",
    )
    ap.add_argument(
        "--secondary-operator",
        action="append",
        default=None,
        help=(
            "declare a reviewed secondary baseline operator (repeatable; "
            "does not grant authorization by itself)"
        ),
    )
    ap.add_argument(
        "--mr-iid",
        default=None,
        help="GitLab MR IID used by the evidence-phase secondary-scope policy",
    )
    ap.add_argument(
        "--bottleneck",
        choices=["Memory-bound", "Compute-bound"],
        default=None,
    )
    ap.add_argument("--profile-evidence", default=None)
    ap.add_argument("--format", default="md", choices=["md", "json"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    P = resolve_op(args.operator)
    if args.phase == "summary":
        try:
            report = build_summary(P, args)
        except (OSError, SummaryError, ValueError) as exc:
            print(f"optimize-op summary: {exc}", file=sys.stderr)
            return 2
        if args.out:
            Path(args.out).write_text(report.rstrip() + "\n", encoding="utf-8")
        else:
            print(report)
        return 0
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
