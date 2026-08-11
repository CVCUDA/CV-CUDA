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
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

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


def test_phase_is_required_and_validated():
    assert run("Resize").returncode == 2  # missing --phase
    assert run("Resize", "--phase", "bogus").returncode == 2  # invalid choice


def test_summary_initialization_requires_explicit_artifact_inputs():
    r = run("Resize", "--phase", "summary")
    assert r.returncode == 2
    assert "--benchmark-base" in r.stderr
    assert "--optimized-cases-file" in r.stderr


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
            metadata=SimpleNamespace(operator="OSD", secondary_operators=("BndBox",))
        ),
    )

    finding = m._operator_scope_gate(osd, {"osd", "bndbox"}, "summary", "123", "guide")

    assert finding.status == m.PASS
    assert "code-reviewed secondary" in finding.summary


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
                        operator=metadata_operator,
                        secondary_operators=secondary,
                    )
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
            metadata=SimpleNamespace(operator="OSD", secondary_operators=())
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


def test_kernel_attribution_explicit_shared_legacy_sources():
    m = _module()
    ops = m.all_operator_stems()
    histogram = m.resolve_op("Histogram")
    histogram_eq = m.resolve_op("HistogramEq")
    warp_affine = m.resolve_op("WarpAffine")
    warp_perspective = m.resolve_op("WarpPerspective")

    calc_hist = "src/cvcuda/priv/legacy/calc_hist.cu"
    assert m._is_op_kernel(calc_hist, histogram, ops) is True
    assert m._is_op_kernel(calc_hist, histogram_eq, ops) is False

    shared_warp = "src/cvcuda/priv/legacy/warp.cu"
    assert m._is_op_kernel(shared_warp, warp_affine, ops) is True
    assert m._is_op_kernel(shared_warp, warp_perspective, ops) is True


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


def test_pixelwise_assertion_requires_a_concrete_assertion_name():
    m = _module()

    for evidence in (
        "EXPECT_EQ(output, reference)",
        "EXPECT_NEAR(output, reference, 1e-6)",
        "assert bool(cupy.all(output == reference))",
        "assert output == reference",
        "cupy.testing.assert_array_equal(output, reference)",
    ):
        assert m._has_pixelwise_assertion(evidence), evidence

    for evidence in (
        "We assert every pixel matches the independent reference.",
        "EXPECT_EQ compares every output pixel",
        "cupy.testing.assert_array_equal verifies the output",
        "The Python test asserts exact output",
        "The equality assertion passed",
        "Pixel equality was checked",
    ):
        assert not m._has_pixelwise_assertion(evidence), evidence


def test_binding_only_change_triggers_implementation_evidence_gates(monkeypatch):
    m = _module()
    hq = m.resolve_op("HQResize")
    binding = "python/mod_cvcuda/operators/OpHQResize.cpp"
    reviewed = []

    monkeypatch.setattr(m, "all_operator_stems", lambda: {"hqresize"})
    monkeypatch.setattr(m, "changed_paths", lambda _base: [binding])
    monkeypatch.setattr(
        m,
        "read_results",
        lambda _path: (
            "- [x] **Pixelwise equality to reference** — `assert "
            "bool(cupy.all(output == expected))` checks every pixel against a "
            "constant-value oracle; 6 passed.\n"
        ),
    )
    monkeypatch.setattr(m, "review_op_findings", lambda *_args: [])
    monkeypatch.setattr(
        m,
        "git",
        lambda *args: '+                "gpu_gap_stddev_us": 4.25'
        if args[0] == "diff"
        else "",
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
    assert next(item for item in findings if item.id == "ODO-9").status == "GAP"
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
):
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
        module, "parse_summary", lambda _results: SimpleNamespace(metadata=metadata)
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
            )
        ),
    )

    finding = m._binding_impact_gate(
        hq,
        "summary",
        ["python/mod_cvcuda/operators/OpHQResize.cpp"],
        "guide",
    )

    assert finding.status == "PASS"
    assert "paired-gap combined standard error" in finding.summary


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


def test_results_declared_baseline_config_unavailable_is_precise_gap(monkeypatch):
    m = _module()
    resize = m.resolve_op("Resize")
    baseline_commit = "1" * 40
    candidate_commit = "2" * 40
    summary = SimpleNamespace(
        metadata=SimpleNamespace(
            baseline_commit=baseline_commit,
            candidate_commit=candidate_commit,
        )
    )
    monkeypatch.setattr(m, "parse_summary", lambda _results: summary)
    monkeypatch.setattr(m, "git_success", lambda *_args: True)
    monkeypatch.setattr(m, "git_json", lambda *_args: None)

    finding = m._results_format(
        resize,
        "v1 summary",
        Path("description.md"),
        "guideline",
        {},
    )

    assert finding.id == "ODO-3"
    assert finding.status == "GAP"
    assert finding.summary == "Declared baseline benchmark config cannot be loaded"
    assert baseline_commit in finding.evidence
    assert "bench/config/operators/resize.json" in finding.evidence


# ---- ODO-4 lead-exhaustion hard evidence ------------------------------------------------
def test_lead_exhaustion_requires_checked_measured_evidence():
    m = _module()
    unchecked = "- [ ] **Lead exhaustion** — ncu showed 93% Memory SOL at ridge.\n"
    unmeasured = (
        "- [x] **Lead exhaustion** — profiling says the implementation is at ridge.\n"
    )
    assert m._leads_exhausted(unchecked, "guide").status == "GAP"
    assert m._leads_exhausted(unmeasured, "guide").status == "GAP"


def test_lead_exhaustion_accepts_ridge_or_three_measured_strikes():
    m = _module()
    ridge = "- [x] **Lead exhaustion** — ncu showed 93% Memory SOL at ridge; PASS.\n"
    strikes = (
        "- [x] **Lead exhaustion** — three failed leads measured at -2.1%, "
        "-0.8%, and 0.1%; PASS.\n"
    )
    assert m._leads_exhausted(ridge, "guide").status == "PASS"
    assert m._leads_exhausted(strikes, "guide").status == "PASS"


# ---- ODO-8 review/refactor lock-in gate evidence ----------------------------------------
def test_review_refactor_gate_requires_assess_evidence():
    m = _module()
    resize = m.resolve_op("Resize")
    result = m._review_refactor_gate(
        resize,
        "- [x] **Review/refactor gate** — reviewed changed implementation; PASS.\n",
        ["src/cvcuda/priv/OpResize.cu"],
        "guide",
    )
    assert result.status == "GAP"
    assert "refactor_op.py" in result.fix


def test_review_refactor_gate_passes_with_clean_assess_result():
    m = _module()
    resize = m.resolve_op("Resize")
    results = (
        "- [x] **Review/refactor gate** — reviewed changed implementation; "
        "tools/refactor_op.py Resize --phase assess PASS with 0 recommendations; "
        "no refactor applied.\n"
    )
    result = m._review_refactor_gate(
        resize, results, ["src/cvcuda/priv/OpResize.cu"], "guide"
    )
    assert result.status == "PASS"


def test_review_refactor_gate_requires_verify_tests_and_bench_for_applied_refactor():
    m = _module()
    resize = m.resolve_op("Resize")
    missing = (
        "- [x] **Review/refactor gate** — reviewed changed implementation; "
        "tools/refactor_op.py Resize --phase assess RECOMMENDATION RED-1; "
        "Refactor applied: yes.\n"
    )
    result = m._review_refactor_gate(
        resize, missing, ["src/cvcuda/priv/OpResize.cu"], "guide"
    )
    assert result.status == "GAP"
    assert "--phase verify" in result.evidence
    assert "frozen operator tests" in result.evidence
    assert "post-refactor benchmark proof" in result.evidence

    complete = (
        "- [x] **Review/refactor gate** — reviewed changed implementation and fixed RED-1; "
        "tools/refactor_op.py Resize --phase assess RECOMMENDATION RED-1; "
        "Refactor applied: yes; tools/refactor_op.py Resize --phase verify PASS; "
        "cvcuda_test_system tests passed; run_bench.py benchmark passed with no regression.\n"
    )
    result = m._review_refactor_gate(
        resize, complete, ["src/cvcuda/priv/OpResize.cu"], "guide"
    )
    assert result.status == "PASS"


# ---- ODO-9 memory-footprint growth gate -----------------------------------------------
def _memory_results(increase=0, allocations="no", evidence=None):
    if evidence is None:
        evidence = (
            "Measured aggregate peak-live added bytes across changed memory paths."
        )
    return f"""## Memory footprint
Peak attributable increase: {increase} B
New runtime CUDA allocations/frees: {allocations}
Evidence: {evidence}
"""


def test_memory_footprint_gate_accepts_zero_and_ten_mb_boundaries():
    m = _module()
    for increase in (0, 9_999_999, 10_000_000):
        result = m._memory_footprint_gate(
            _memory_results(increase), ["src/cvcuda/priv/OpResize.cu"], "guide"
        )
        assert result.status == "PASS", increase


def test_memory_footprint_gate_requires_review_above_ten_mb():
    m = _module()
    result = m._memory_footprint_gate(_memory_results(10_000_001), [], "guide")
    assert result.status == "MANUAL"
    assert "10000001 B exceeds" in result.evidence


def test_memory_footprint_gate_requires_review_for_new_cuda_allocation_path():
    m = _module()
    for increase in (0, 10_000_000):
        result = m._memory_footprint_gate(
            _memory_results(increase, allocations="yes"), [], "guide"
        )
        assert result.status == "MANUAL", increase
        assert "allocation/free path" in result.evidence


def test_memory_footprint_gate_rejects_missing_fields():
    m = _module()
    lines = _memory_results().splitlines()
    for label in (
        "Peak attributable increase:",
        "New runtime CUDA allocations/frees:",
        "Evidence:",
    ):
        results = "\n".join(line for line in lines if not line.startswith(label))
        result = m._memory_footprint_gate(results, [], "guide")
        assert result.status == "GAP", label
        assert "missing, duplicated, or malformed" in result.summary


def test_memory_footprint_gate_rejects_duplicate_fields():
    m = _module()
    duplicates = (
        "Peak attributable increase: 0 B",
        "New runtime CUDA allocations/frees: no",
        "Evidence: Independently inspected the changed source.",
    )
    for duplicate in duplicates:
        result = m._memory_footprint_gate(
            _memory_results() + duplicate + "\n", [], "guide"
        )
        assert result.status == "GAP", duplicate
        assert "found 2 occurrences" in result.evidence


def test_memory_footprint_gate_rejects_malformed_increase():
    m = _module()
    for value in ("-1 B", "1.5 B", "1 KiB", "1", "9" * 5000 + " B"):
        results = _memory_results().replace(
            "Peak attributable increase: 0 B",
            "Peak attributable increase: " + value,
        )
        result = m._memory_footprint_gate(results, [], "guide")
        assert result.status == "GAP", value
        assert "malformed declaration" in result.evidence


def test_memory_footprint_gate_rejects_malformed_allocation_flag():
    m = _module()
    for value in ("No", "maybe", "false"):
        result = m._memory_footprint_gate(
            _memory_results(allocations=value), [], "guide"
        )
        assert result.status == "GAP", value


def test_memory_footprint_gate_rejects_placeholder_evidence_before_manual():
    m = _module()
    for evidence in (
        "",
        "TBD",
        "pending",
        "measurement pending",
        "n/a",
        "<measurement evidence>",
        "\nSupporting prose must not fill a blank declaration.",
    ):
        result = m._memory_footprint_gate(
            _memory_results(10_000_001, allocations="yes", evidence=evidence),
            [],
            "guide",
        )
        assert result.status == "GAP", evidence


def test_memory_footprint_gate_allows_substantive_evidence_with_placeholder_words():
    m = _module()
    result = m._memory_footprint_gate(
        _memory_results(
            evidence="Code inspection found no pending CUDA allocation/free paths."
        ),
        [],
        "guide",
    )
    assert result.status == "PASS"


def test_memory_footprint_gate_hidden_fields_do_not_satisfy_declarations():
    m = _module()
    fields = _memory_results().removeprefix("## Memory footprint\n")
    for hidden in (
        "<!--\n" + fields + "-->",
        "<!--\n" + fields,
        "```text\n" + fields + "```",
        "~~~~text\n" + fields + "~~~~",
    ):
        result = m._memory_footprint_gate("## Memory footprint\n" + hidden, [], "guide")
        assert result.status == "GAP"
        assert "found 0 occurrences" in result.evidence


def test_memory_footprint_gate_ignores_hidden_duplicate_fields():
    m = _module()
    fields = _memory_results().removeprefix("## Memory footprint\n")
    results = _memory_results() + "<!--\n" + fields + "-->\n"
    result = m._memory_footprint_gate(results, [], "guide")
    assert result.status == "PASS"


def test_memory_footprint_gate_ignores_hidden_or_fenced_headings():
    m = _module()
    for results in (
        "<!--\n" + _memory_results() + "-->",
        "<!--\n" + _memory_results(),
        "```markdown\n" + _memory_results() + "```",
        "````markdown\n```\n" + _memory_results() + "````",
    ):
        result = m._memory_footprint_gate(results, [], "guide")
        assert result.status == "GAP"
        assert "missing or duplicated" in result.summary


def test_memory_footprint_gate_requires_canonical_heading():
    m = _module()
    for heading in (
        "# Memory footprint",
        "### Memory footprint",
        "## 9. Memory footprint",
    ):
        result = m._memory_footprint_gate(
            _memory_results().replace("## Memory footprint", heading), [], "guide"
        )
        assert result.status == "GAP", heading


def test_memory_footprint_gate_rejects_duplicate_sections():
    m = _module()
    result = m._memory_footprint_gate(
        _memory_results() + "\n" + _memory_results(), [], "guide"
    )
    assert result.status == "GAP"
    assert "found 2" in result.evidence


def test_memory_footprint_gate_applies_without_implementation_changes():
    m = _module()
    result = m._memory_footprint_gate("", [], "guide")
    assert result.status == "GAP"
    assert "missing or duplicated" in result.summary


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
