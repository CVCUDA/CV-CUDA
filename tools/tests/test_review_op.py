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
"""Contract/smoke tests for the deterministic ``/review-op`` checker (tools/review_op.py).

These guard the harness's core promises — determinism, the GAP/exit-code contract, the
status vocabulary, and graceful degradation — without coupling to the live coverage state
of any specific operator (so legitimately fixing an operator's gaps never breaks the suite).

Stdlib-only. Run with ``python3 -m pytest tools/tests/``.
"""
import json
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
TOOL = REPO / "tools" / "review_op.py"
sys.path.insert(0, str(REPO / "tools"))
import review_op as ro  # noqa: E402

VALID_STATUS = {"PASS", "GAP", "N-A", "MANUAL", "RECOMMENDATION"}


def run(*args):
    return subprocess.run(
        [sys.executable, str(TOOL), *args],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_runs_and_emits_a_report():
    r = run("Resize", "--format", "json")
    assert r.stdout.strip(), "expected a non-empty report"
    data = json.loads(r.stdout)
    assert data["operator"] == "Resize"
    assert data["findings"], "expected at least one finding"


def test_report_is_deterministic():
    """The headline guarantee: same tree -> byte-identical report."""
    a = run("Resize", "--format", "json")
    b = run("Resize", "--format", "json")
    assert a.stdout == b.stdout


def test_exit_code_matches_gap_contract():
    """Exit is non-zero iff (and only iff) a GAP is present; statuses are from the vocabulary.

    Asserts the contract on whatever the live tree contains, not a hardcoded gap list.
    """
    r = run("Resize", "--format", "json")
    findings = json.loads(r.stdout)["findings"]
    assert all(f["status"] in VALID_STATUS for f in findings)
    has_gap = any(f["status"] == "GAP" for f in findings)
    assert r.returncode in (0, 1)
    assert (r.returncode == 1) == has_gap


def test_findings_carry_evidence_and_guideline():
    for f in json.loads(run("Resize", "--format", "json").stdout)["findings"]:
        assert f["id"] and f["domain"] and f["status"]
        # non-N-A findings should point somewhere (evidence or guideline ref)
        if f["status"] != "N-A":
            assert f.get("evidence") or f.get("guideline")


def test_python_test_path_prefers_op_derived_name(monkeypatch):
    # even when the pybind API name differs, the op-derived file wins when it exists
    monkeypatch.setattr(ro, "resolve_pyname", lambda pybind, op: "short_name")

    paths = ro.resolve_op("Resize")

    assert paths.test_py.name == "test_opresize.py"


def test_python_test_path_falls_back_to_api_name():
    # legacy tests named after the pybind API name stay resolvable
    assert ro.resolve_op("NonMaximumSuppression").test_py.name == "test_opnms.py"


def test_coordinate_tensor_python_image_layout_is_na():
    assert ro.python_image_layout_na({"bench_layout_na": True, "bench_rgb_na": True})
    assert not ro.python_image_layout_na(
        {"bench_layout_na": True, "bench_rgb_na": False}
    )


@pytest.mark.parametrize("operator", ["CenterCrop", "Histogram"])
def test_tensor_only_op_marks_varshape_na(operator):
    """Curated Tensor-only operators report SUP-2 as N-A rather than GAP."""
    r = run(operator, "--domain", "support", "--format", "json")
    sup2 = [f for f in json.loads(r.stdout)["findings"] if f["id"] == "SUP-2"]
    assert sup2 and sup2[0]["status"] == "N-A"


def test_generic_submit_is_classified_by_primary_input_container():
    """Tensor outputs/parameters do not turn a legacy ImageBatch-input API into Tensor support."""
    header = """
    CVCUDA_PUBLIC NVCVStatus cvcudaExampleSubmit(
        NVCVOperatorHandle handle, cudaStream_t stream, NVCVImageBatchHandle in,
        NVCVTensorHandle out, NVCVTensorHandle params);
    """

    support = ro.detect_c_api_containers("Example", header)

    assert not support.tensor
    assert [signature.primary_container for signature in support.varshape] == [
        "NVCVImageBatchHandle"
    ]
    assert support.generic_varshape == support.varshape


def test_tensor_batch_submit_forms_keep_their_established_container_names():
    header = """
    CVCUDA_PUBLIC NVCVStatus cvcudaExampleSubmit(
        NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorBatchHandle in,
        NVCVTensorHandle out);
    CVCUDA_PUBLIC NVCVStatus cvcudaExampleVarShapeSubmit(
        NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorBatchHandle in,
        NVCVTensorBatchHandle out);
    """

    support = ro.detect_c_api_containers("Example", header)

    assert [signature.name for signature in support.tensor] == ["cvcudaExampleSubmit"]
    assert [signature.name for signature in support.varshape] == [
        "cvcudaExampleVarShapeSubmit"
    ]
    assert not support.generic_varshape


@pytest.mark.parametrize("operator", ["CropFlipNormalizeReformat", "PadAndStack"])
def test_legacy_generic_varshape_submit_has_no_false_tensor_gap(operator):
    r = run(operator, "--domain", "support", "--format", "json")
    findings = {f["id"]: f for f in json.loads(r.stdout)["findings"]}

    assert findings["SUP-1"]["status"] == "N-A"
    assert "VarShape primary input" in findings["SUP-1"]["summary"]
    assert "NVCVImageBatchHandle" in findings["SUP-1"]["evidence"]
    assert findings["SUP-2"]["status"] == "PASS"
    assert findings["SUP-3"]["status"] == "PASS"


def test_legacy_generic_varshape_submit_does_not_require_tensor_benchmark():
    r = run("CropFlipNormalizeReformat", "--domain", "bench", "--format", "json")
    ben14 = next(f for f in json.loads(r.stdout)["findings"] if f["id"] == "BEN-14")

    assert "Tensor" not in ben14["summary"]
    assert "VarShape" in ben14["evidence"]


def test_generic_varshape_correctness_pattern_rejects_unrelated_tests():
    pattern = ro._varshape_positive_test_pattern("Example", generic_varshape=True)

    assert re.search(pattern, "TYPED_TEST(OpExample, correct_output)")
    assert re.search(pattern, "TEST_P(OpExample, correct_output)")
    assert not re.search(pattern, "TEST(OpExample, smoke)")
    assert not re.search(pattern, "TEST(OpExample_Negative, invalid_input)")
    assert not re.search(pattern, "TEST(OpExample, incorrect_input)")
    assert not re.search(pattern, "TEST(OpExample, VarShape_incorrect_output)")


@pytest.mark.parametrize("macro", ["TEST", "TEST_P", "TYPED_TEST"])
def test_specialized_tensor_correctness_parser_recognizes_test_macros(macro):
    source = "\n".join(
        (
            f"{macro}(OpExampleTensor, correct_output)",
            f"{macro}(OpExampleTensor2D, correct_output)",
            f"{macro}(OpExampleTensorBatch, matches_reference)",
        )
    )

    hits = ro._tensor_positive_test_hits("Example", source)

    assert [line for _, line in hits] == source.splitlines()


def test_tensor_correctness_parser_rejects_nonpositive_and_unrelated_tests():
    source = "\n".join(
        (
            "TEST(OpExample, smoke)",
            "TEST(OpExample, varshape_correct_output)",
            "TEST(OpExampleNegative, tensor_correct_output)",
            "TEST_P(OpExampleTensor, rejects_invalid_shape)",
            "TEST_P(OpExampleTensorNegative, correct_output)",
            "TYPED_TEST(OpExampleTensor2D_Negative, correct_output)",
            "TYPED_TEST(OpExampleTensorSmoke, basic_functionality)",
            "TYPED_TEST(OpExampleImageBatch, tensor_correct_output)",
            "TEST(OpExampleSiblingTensor, correct_output)",
            "TEST(OpUnrelatedTensor, correct_output)",
            "// TEST(OpExampleTensor, correct_output)",
            "/* TYPED_TEST(OpExampleTensor2D, matches_reference) */",
        )
    )

    assert not ro._tensor_positive_test_hits("Example", source)


def test_tensor_correctness_parser_retains_generic_positive_names():
    source = "\n".join(
        (
            "TEST_P(OpExample, CustomCrop_packed)",
            "TEST(OpExample, Histogram_mask)",
            "TYPED_TEST(OpExample, correct_output)",
        )
    )

    hits = ro._tensor_positive_test_hits("Example", source)

    assert [line for _, line in hits] == source.splitlines()


@pytest.mark.parametrize(
    "operator", ["AdvCvtColor", "CustomCrop", "HQResize", "Morphology", "Stack"]
)
def test_current_tensor_positive_suites_remain_recognized(operator):
    findings = json.loads(run(operator, "--domain", "test", "--format", "json").stdout)[
        "findings"
    ]
    tst2 = next(finding for finding in findings if finding["id"] == "TST-2")

    assert tst2["status"] == "PASS"


@pytest.mark.parametrize("operator", ["CropFlipNormalizeReformat", "PadAndStack"])
def test_legacy_generic_varshape_correctness_suite_is_recognized(operator):
    r = run(operator, "--domain", "test", "--format", "json")
    findings = {f["id"]: f for f in json.loads(r.stdout)["findings"]}

    assert findings["TST-2"]["status"] == "N-A"
    assert findings["TST-3"]["status"] == "PASS"


def test_hqresize_image_batch_submit_is_recognized_as_varshape():
    findings = json.loads(
        run("HQResize", "--domain", "support", "--format", "json").stdout
    )["findings"]
    sup2 = next(f for f in findings if f["id"] == "SUP-2")
    assert sup2["status"] == "PASS"


def test_layout_defining_benchmark_marks_layout_axis_na():
    """A dummy NHWC axis must not be required for an operator that defines layout."""
    r = run("Reformat", "--domain", "bench", "--format", "json")
    ben5 = [f for f in json.loads(r.stdout)["findings"] if f["id"] == "BEN-5"]
    assert ben5 and ben5[0]["status"] == "N-A"


def test_rgb_na_tensor_benchmark_does_not_require_fake_planar():
    findings = json.loads(run("SIFT", "--domain", "bench", "--format", "json").stdout)[
        "findings"
    ]
    ben6 = next(f for f in findings if f["id"] == "BEN-6")

    assert ben6["status"] == "PASS"
    assert "NCHW_FAKE N-A for scalar/RGB-N-A tensor benchmark" in ben6["summary"]


def _image_bench_entry(tier, dtype, layout, shape="8x16x32"):
    return {
        "tier": tier,
        "dtypes": [dtype],
        "string_axes": {
            "shape": [shape],
            "layout": [layout],
            "inputKind": ["Tensor"],
        },
        "baselines": {},
    }


def _ben6_for_synthetic_invert(monkeypatch, configs):
    return _ben6_for_synthetic_operator(monkeypatch, "Invert", configs)


def _ben6_for_synthetic_operator(monkeypatch, operator, configs):
    paths = ro.resolve_op(operator)
    _, support_info = ro.check_support(paths, ro.load_curated())
    config = {"benchmark": paths.op, "configs": configs}
    load_bench_cfg = ro.load_bench_cfg

    def load_synthetic(path):
        return config if path == paths.bench_cfg else load_bench_cfg(path)

    monkeypatch.setattr(ro, "load_bench_cfg", load_synthetic)
    return next(
        finding
        for finding in ro.check_bench(paths, support_info, do_run=False)
        if finding.id == "BEN-6"
    )


@pytest.mark.parametrize("operator", ["SIFT", "CropFlipNormalizeReformat"])
def test_ben6_validates_declared_fake_planar_even_when_inferred_na(
    monkeypatch, operator
):
    ben6 = _ben6_for_synthetic_operator(
        monkeypatch,
        operator,
        {
            "native_rgb_basic": _image_bench_entry("basic", "uchar3", "NCHW"),
            "fake_rgb_advanced": _image_bench_entry("advanced", "uchar3", "NCHW_FAKE"),
        },
    )

    assert ben6.status == "GAP"
    assert "same-tier" in ben6.summary
    assert "fake_rgb_advanced" in ben6.evidence


def test_ben6_rejects_fake_planar_whose_only_native_match_is_cross_tier(monkeypatch):
    ben6 = _ben6_for_synthetic_invert(
        monkeypatch,
        {
            "native_rgb_basic": _image_bench_entry("basic", "uchar3", "NCHW"),
            "fake_rgb_advanced": _image_bench_entry("advanced", "uchar3", "NCHW_FAKE"),
        },
    )

    assert ben6.status == "GAP"
    assert "same-tier" in ben6.summary
    assert "fake_rgb_advanced" in ben6.evidence


def test_ben6_rejects_one_orphan_in_an_otherwise_valid_fake_planar_surface(
    monkeypatch,
):
    ben6 = _ben6_for_synthetic_invert(
        monkeypatch,
        {
            "native_rgba_advanced": _image_bench_entry(
                "advanced", "uchar4", "NCHW", shape="4x16x32"
            ),
            "fake_rgba_advanced": _image_bench_entry(
                "advanced", "uchar4", "NCHW_FAKE", shape="4x16x32"
            ),
            "native_rgb_basic": _image_bench_entry("basic", "uchar3", "NCHW"),
            "fake_rgb_advanced": _image_bench_entry("advanced", "uchar3", "NCHW_FAKE"),
        },
    )

    assert ben6.status == "GAP"
    assert "1 unmatched" in ben6.summary
    assert "fake_rgb_advanced" in ben6.evidence
    assert "fake_rgba_advanced" not in ben6.evidence


def test_rgb_na_tensor_benchmark_still_requires_native_planar(monkeypatch):
    paths = ro.resolve_op("SIFT")
    _, support_info = ro.check_support(paths, ro.load_curated())
    config = json.loads(json.dumps(ro.load_bench_cfg(paths.bench_cfg)))
    for entry in config["configs"].values():
        layouts = entry["string_axes"]["layout"]
        entry["string_axes"]["layout"] = [
            layout for layout in layouts if layout != "NCHW"
        ]

    load_bench_cfg = ro.load_bench_cfg

    def load_without_native_planar(path):
        return config if path == paths.bench_cfg else load_bench_cfg(path)

    monkeypatch.setattr(ro, "load_bench_cfg", load_without_native_planar)
    ben6 = next(
        finding
        for finding in ro.check_bench(paths, support_info, do_run=False)
        if finding.id == "BEN-6"
    )

    assert ben6.status == "GAP"
    assert "Add native NCHW configs" in ben6.fix
    assert "and NCHW_FAKE" not in ben6.fix


def test_minmaxloc_benchmark_accepts_truthful_layout_axis():
    bench = json.loads(
        run("MinMaxLoc", "--domain", "bench", "--format", "json").stdout
    )["findings"]

    assert next(f for f in bench if f["id"] == "BEN-5")["status"] == "PASS"


def test_varshape_only_coverage_stats_do_not_invent_tensor_support():
    findings = json.loads(
        run("PadAndStack", "--domain", "bench", "--format", "json").stdout
    )["findings"]
    ben13 = next(f for f in findings if f["id"] == "BEN-13")
    assert "VarShape:basic" in ben13["evidence"]
    assert "Tensor:none" not in ben13["evidence"]


def test_layout_na_coverage_stats_do_not_recommend_dummy_layouts():
    findings = json.loads(
        run("Reformat", "--domain", "bench", "--format", "json").stdout
    )["findings"]
    ben13 = next(f for f in findings if f["id"] == "BEN-13")
    assert "layout: N-A" in ben13["evidence"]
    ben16 = [f for f in findings if f["id"] == "BEN-16"]
    assert not ben16 or "layout=" not in ben16[0]["evidence"]


def test_layout_na_benchmark_rejects_a_dummy_layout_axis(monkeypatch):
    paths = ro.resolve_op("Reformat")
    _, support_info = ro.check_support(paths, ro.load_curated())
    real_load = ro.load_bench_cfg

    def load_with_dummy_layout(path):
        if path == paths.bench_cfg:
            return {
                "configs": {
                    "reformat_basic": {
                        "tier": "basic",
                        "dtypes": ["uchar3"],
                        "string_axes": {
                            "shape": ["1x2x3"],
                            "layout": ["NHWC"],
                            "inputKind": ["Tensor"],
                        },
                        "baselines": {},
                    }
                }
            }
        return real_load(path)

    monkeypatch.setattr(ro, "load_bench_cfg", load_with_dummy_layout)
    ben5 = next(
        f for f in ro.check_bench(paths, support_info, False) if f.id == "BEN-5"
    )
    assert ben5.status == "GAP"


def test_hqresize_channel_axis_satisfies_rgb_floor_and_reports_tensorbatch():
    r = run("HQResize", "--domain", "bench", "--format", "json")
    findings = json.loads(r.stdout)["findings"]
    ben14 = [f for f in findings if f["id"] == "BEN-14"]
    ben13 = [f for f in findings if f["id"] == "BEN-13"]
    assert ben14 and ben14[0]["status"] == "PASS"
    assert ben13 and "TensorBatch" in ben13[0]["evidence"]


def test_curated_benchmark_applicability_lists_are_loaded():
    _, varshape_only, _, layout_na, rgb_na = ro.load_curated()
    assert "cropflipnormalizereformat" in varshape_only
    assert "findhomography" in layout_na
    assert "histogram" in rgb_na


def test_typed_suite_satisfies_parametrized_test_gate():
    r = run("CropFlipNormalizeReformat", "--domain", "test", "--format", "json")
    findings = {finding["id"]: finding for finding in json.loads(r.stdout)["findings"]}
    assert findings["TST-4"]["status"] == "PASS"


def test_unknown_operator_degrades_gracefully():
    """A bogus operator must not crash; it should report a GAP for the missing tensor entry point."""
    r = run("NotARealOperator", "--domain", "support", "--format", "json")
    assert "Traceback" not in r.stderr
    sup1 = [f for f in json.loads(r.stdout)["findings"] if f["id"] == "SUP-1"]
    assert sup1 and sup1[0]["status"] == "GAP"


def test_invalid_domain_is_a_usage_error():
    r = run("Resize", "--domain", "bogus")
    assert r.returncode == 2  # argparse usage error


def test_domain_filter_scopes_output():
    findings = json.loads(
        run("Resize", "--domain", "support", "--format", "json").stdout
    )["findings"]
    assert findings and all(f["domain"] == "support" for f in findings)


def test_gaussian_shared_sources_are_checked():
    assert ro.SHARED_KERNEL_SOURCES["gaussian"] == [
        "legacy/filter.cu",
        "legacy/filter_var_shape.cu",
    ]


def test_planar_not_applicable_declarations_are_colocated():
    declarations = {}
    for header in sorted((REPO / "src/cvcuda/include/cvcuda").glob("Op*.h")):
        policy = ro.parse_planar_policy(header.read_text())
        assert not policy["error"], f"{header}: {policy['error']}"
        if policy["not_applicable"]:
            declarations[header.stem[2:].lower()] = policy["reason"]

    assert set(declarations) == {
        "findhomography",
        "minarearect",
        "nonmaximumsuppression",
        "pairwisematcher",
    }
    assert all(declarations.values())


@pytest.mark.parametrize(
    ("header_text", "declared", "status"),
    [
        ("Limitations:\nData Layout: [NHWC]", False, "GAP"),
        ("Limitations:\nData Layout: [NCHW]", True, "PASS"),
        (
            "Planar image layouts: Not applicable\nReason: Inputs are not images.",
            False,
            "N-A",
        ),
        ("Planar image layouts: Not applicable", False, "GAP"),
        (
            "Planar image layouts: Unsupported\nReason: Inputs are not images.",
            False,
            "GAP",
        ),
        (
            "Planar image layouts: Not applicable\nReason: Inputs are not images.",
            True,
            "GAP",
        ),
    ],
)
def test_planar_policy_verdict(header_text, declared, status, tmp_path):
    operator = SimpleNamespace(header=tmp_path / "OpExample.h")
    finding = ro.planar_policy_verdict(operator, header_text, declared)
    assert finding.status == status


def test_plain_layout_names_are_parsed_as_planar():
    lim = ro.parse_limitations(
        """
        Limitations:
          Input:
            Data Layout: [HW, NHW, HWC, NHWC, CHW, NCHW]
            Channels: [1]
        """
    )

    assert {"CHW", "NCHW"} <= lim["layouts"]


@pytest.mark.parametrize(
    "operator",
    ["MinMaxLoc", "Reformat", "ResizeCropConvertReformat", "SIFT", "Stack"],
)
def test_supported_planar_ops_are_not_reported_na(operator):
    r = run(operator, "--domain", "support", "--format", "json")
    sup10 = [f for f in json.loads(r.stdout)["findings"] if f["id"] == "SUP-10"]
    assert sup10 and sup10[0]["status"] == "PASS"


@pytest.mark.parametrize(
    "operator",
    ["FindHomography", "MinAreaRect", "NonMaximumSuppression", "PairwiseMatcher"],
)
def test_non_image_ops_report_operator_local_planar_reason(operator):
    r = run(operator, "--domain", "support", "--format", "json")
    sup10 = [f for f in json.loads(r.stdout)["findings"] if f["id"] == "SUP-10"]
    assert sup10 and sup10[0]["status"] == "N-A"
    assert "not images" in sup10[0]["evidence"]


def test_ben7_reports_missing_declared_case_key():
    skus = [
        entry["stem"]
        for entry in ro.load_bench_cfg(REPO / "bench/config/sku_map.json")["entries"]
    ]
    configs = {
        "resize_basic": {
            "dtypes": ["uint8"],
            "string_axes": {
                "shape": ["1x2x3"],
                "inputKind": ["Tensor", "VarShape"],
            },
            "baselines": {
                "resize_basic[InOutDataType=uint8][shape=1x2x3][inputKind=Tensor]": {
                    sku: {
                        "n_runs": 1,
                        "gpu_time_us_cpp": 100.0,
                        "gpu_time_us_python": 100.0,
                        "gpu_noise_us_cpp": 1.0,
                        "gpu_noise_us_python": 1.0,
                        "gpu_bwutil_cpp": 0.1,
                        "gpu_bwutil_python": 0.1,
                    }
                    for sku in skus
                }
            },
        }
    }

    finding = ro.baseline_completeness(
        SimpleNamespace(op="resize"), configs, "guideline"
    )

    assert finding.status == "GAP"
    assert "VarShape" in finding.evidence


def test_ben7_reports_malformed_case_payload_as_gap():
    case_key = "resize_basic[InOutDataType=uint8][shape=1x2x3][inputKind=Tensor]"
    configs = {
        "resize_basic": {
            "dtypes": ["uint8"],
            "string_axes": {
                "shape": ["1x2x3"],
                "inputKind": ["Tensor"],
            },
            "baselines": {
                case_key: ["not", "a", "sku", "mapping"],
            },
        }
    }

    finding = ro.baseline_completeness(
        SimpleNamespace(op="resize"), configs, "guideline"
    )

    assert finding.status == "GAP"
    assert "baseline case payload must be an object keyed by SKU" in finding.evidence


def test_legacy_kernel_files_match_across_underscores():
    """Op names drop the underscores their legacy kernel files keep; the longest op name
    owns a file so a prefix op never steals it (regression for pillow_resize/convert_to)."""
    ops = ro.all_op_names()
    assert ro.legacy_belongs("pillow_resize", "pillowresize", ops)
    assert ro.legacy_belongs("convert_to", "convertto", ops)
    assert ro.legacy_belongs("resize_var_shape", "resize", ops)
    assert ro.legacy_belongs("gaussian_noise", "gaussiannoise", ops)
    assert not ro.legacy_belongs("gaussian_noise", "gaussian", ops)  # longer op wins
    assert not ro.legacy_belongs("pillow_resize", "resize", ops)  # not the owner


def test_histogram_resolves_non_derivable_legacy_kernel_source():
    histogram = ro.resolve_op("Histogram")
    relative_priv = {
        path.relative_to(ro.REPO / "src/cvcuda/priv").as_posix()
        for path in histogram.priv
    }
    assert "legacy/calc_hist.cu" in relative_priv


def test_load_curated_normalizes_valid_data(monkeypatch):
    monkeypatch.setattr(
        ro,
        "load_bench_cfg",
        lambda path: {
            "tensor_only": ["CenterCrop"],
            "varshape_only": ["Conv2D"],
            "bench_layout_na": ["Reformat"],
            "bench_rgb_na": ["Histogram"],
            "basic_expected": {"Resize": ["resize_basic"]},
        },
    )

    assert ro.load_curated() == (
        {"centercrop"},
        {"conv2d"},
        {"resize": ["resize_basic"]},
        {"reformat"},
        {"histogram"},
    )


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([], "`root` must be an object"),
        ({"tensor_only": None}, "`tensor_only` must be a list of strings"),
        ({"varshape_only": ["Resize", 7]}, "`varshape_only[1]` must be a string"),
        ({"bench_layout_na": None}, "`bench_layout_na` must be a list of strings"),
        ({"bench_rgb_na": [7]}, "`bench_rgb_na[0]` must be a string"),
        (
            {"basic_expected": []},
            "`basic_expected` must be an object mapping operator names to string lists",
        ),
        (
            {"basic_expected": {"Resize": ["ok", 4]}},
            "`basic_expected.Resize[1]` must be a string",
        ),
    ],
)
def test_load_curated_rejects_malformed_data(monkeypatch, payload, message):
    monkeypatch.setattr(ro, "load_bench_cfg", lambda path: payload)

    with pytest.raises(ValueError, match=re.escape(message)):
        ro.load_curated()
