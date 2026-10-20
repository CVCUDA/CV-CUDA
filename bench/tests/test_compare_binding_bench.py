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

import copy
from itertools import count
import json
import xml.etree.ElementTree as ET

import pytest

from binding_bench import CASES, SUITES, build_report
from compare_binding_bench import compare, main

MEASUREMENTS = count()


def report(cases=CASES):
    metadata = {
        "timestamp_utc": f"test measurement {next(MEASUREMENTS)}",
        "benchmark_script_sha256": "a" * 64,
        "python_version": "3.14.4",
        "python_build": "CPython test build",
        "gil_enabled": True,
        "runtime_image": "test-image@sha256:123",
        "worker_hostname": "worker",
        "cpu_model": "test CPU",
        "cpu_affinity": [2],
        "gpu_inventory": ["0, Test GPU, GPU-test, driver"],
        "cuda_major": "13",
        "cuda_visible_devices": None,
        "nvidia_visible_devices": "all",
        "cvcuda_version": "test-wheel",
        "cvcuda_extension_sha256": "b" * 64,
    }
    return build_report(
        [
            {"name": name, "group": "test", "iterations": 10, "window_ns": [10000] * 5}
            for name in cases
        ],
        {"selected_cases": list(cases), "samples": 5, "warmups": 5},
        metadata,
    )


def scale(data, index, factor):
    data["results"][index]["window_ns"] = [
        round(x * factor) for x in data["results"][index]["window_ns"]
    ]


@pytest.mark.parametrize(
    "factor,status", [(0.5, "PASS"), (1, "PASS"), (1.10, "PASS"), (1.101, "REGRESSION")]
)
def test_threshold_uses_raw_windows_and_one_case_cannot_be_masked(factor, status):
    before, after = report(), report()
    for i in range(1, len(CASES)):
        scale(after, i, 0.5)
    after["results"][1] = copy.deepcopy(before["results"][1])
    scale(after, 1, factor)
    # Deliberately leave the summary stale: decisions must use the raw windows.
    assert compare(before, after)[1][-1] == status


def test_noop_is_diagnostic_not_a_binding_slowdown():
    before, after = report(), report()
    scale(after, 0, 2)
    assert compare(before, after)[0][-1] == "CONTROL"


def test_noise_is_invalid_even_when_candidate_is_faster():
    before, after = report(), report()
    after["results"][1]["window_ns"] = [1000, 2000, 3000, 4000, 5000]
    assert compare(before, after)[1][-1] == "INVALID (noise)"


def test_process_medians_are_averaged_without_pooling_windows():
    before, after = [report() for _ in range(5)], [report() for _ in range(5)]
    for run, median in zip(after, [1000, 1000, 1200, 1400, 1600]):
        run["results"][1]["window_ns"] = [median * 10] * 3 + [100000] * 2
    # The five medians average 1240 ns; pooling or taking their median differs.
    result = compare(before, after)[1]
    assert result[2] == 1240
    assert result[3] == pytest.approx(24)
    assert result[-1] == "PASS"  # Broad spread does not support a regression.


def test_five_process_regression_and_single_slow_process():
    before, after = [report() for _ in range(5)], [report() for _ in range(5)]
    scale(after[0], 1, 1.2)
    assert compare(before, after)[1][-1] == "PASS"
    for run in after[1:]:
        scale(run, 1, 1.2)
    assert compare(before, after)[1][-1] == "REGRESSION"


def test_window_noise_remains_diagnostic_with_five_processes():
    before, after = [report() for _ in range(5)], [report() for _ in range(5)]
    for run in before + after:
        run["results"][1]["window_ns"] = [8000, 8000, 10000, 12000, 12000]
    assert compare(before, after)[1][-1] == "PASS"
    for run in after:
        scale(run, 1, 1.15)
    assert compare(before, after)[1][-1] == "REGRESSION"
    assert compare(before[0], after[0])[1][-1] == "INVALID (noise)"


def test_one_slow_process_does_not_produce_a_supported_regression():
    before, after = [report() for _ in range(5)], [report() for _ in range(5)]
    scale(after[-1], 1, 1.6)
    result = compare(before, after)[1]
    assert result[3] == pytest.approx(12)
    assert result[-1] == "PASS"


def test_two_band_majority_change_does_not_produce_a_regression():
    before, after = [report() for _ in range(5)], [report() for _ in range(5)]
    for run, factor in zip(before, [1, 1, 1, 1.2, 1.2]):
        scale(run, 1, factor)
    for run, factor in zip(after, [1, 1, 1.2, 1.2, 1.2]):
        scale(run, 1, factor)
    assert compare(before, after)[1][-1] == "PASS"


@pytest.mark.parametrize("shift,expected", [(30, "PASS"), (30.1, "REGRESSION")])
def test_significance_margin_is_strict_even_for_large_effect(shift, expected):
    before, after = [report() for _ in range(5)], [report() for _ in range(5)]
    for baseline, candidate, value in zip(before, after, [90, 95, 100, 105, 110]):
        baseline["results"][1]["window_ns"] = [value * 10] * 5
        candidate["results"][1]["window_ns"] = [round((value + shift) * 10)] * 5
    assert compare(before, after)[1][-1] == expected


def test_fixed_significance_guard_covers_current_binding_case_count():
    assert len(CASES) - 1 == 25  # Revisit the documented family bound if cases grow.
    assert all(len(cases) - 1 <= 25 for cases in SUITES.values())


def test_export_suite_preserves_regression_rule_and_rejects_mixed_or_partial_suites():
    cases = SUITES["export-metadata"]
    before, after = report(cases), report(cases)
    scale(after, -1, 1.2)
    assert compare(before, after)[-1][-1] == "REGRESSION"
    with pytest.raises(ValueError, match="parameters"):
        compare(report(), after)
    for incomplete in (cases[:-1], (*cases, CASES[1])):
        with pytest.raises(ValueError, match="complete binding suite"):
            compare(before, report(incomplete))


@pytest.mark.parametrize("counts", [(0, 0), (3, 3), (5, 4), (1, 5), (6, 6)])
def test_partial_or_extra_processes_are_rejected(counts):
    with pytest.raises(ValueError, match="one or five"):
        compare(
            [report() for _ in range(counts[0])], [report() for _ in range(counts[1])]
        )


@pytest.mark.parametrize(
    "key", ["runtime_image", "cvcuda_version", "cvcuda_extension_sha256"]
)
def test_last_process_provenance_is_checked(key):
    before, after = [report() for _ in range(5)], [report() for _ in range(5)]
    after[-1]["metadata"][key] = "different"
    with pytest.raises(ValueError):
        compare(before, after)


def test_cli_five_processes_and_duplicate_input(tmp_path):
    paths = []
    for run in range(5):
        path = tmp_path / f"run-{run}.json"
        path.write_text(json.dumps(report()))
        paths.append(str(path))
    assert main(["--reference", *paths, "--candidate", *paths]) == 0
    assert main(["--reference", *([paths[0]] * 5), "--candidate", *paths]) == 2


@pytest.mark.parametrize("role", ["reference", "candidate"])
def test_cli_copied_reports_are_invalid_even_under_distinct_paths(tmp_path, role):
    before, after = report(), report()
    for data in (before, after):
        data["results"][1]["window_ns"] = [8000, 8000, 10000, 12000, 12000]
    scale(after, 1, 1.2)
    assert compare(before, after)[1][-1] == "INVALID (noise)"
    arguments = []
    for label, data in (("reference", before), ("candidate", after)):
        arguments.append(f"--{label}")
        for index in range(5):
            run = copy.deepcopy(data)
            if label != role:
                run["metadata"]["timestamp_utc"] = f"measurement {index}"
            path = tmp_path / f"{label}-{index}.json"
            path.write_text(json.dumps(run))
            arguments.append(str(path))
    junit = tmp_path / "comparison.xml"
    assert main([*arguments, "--junit", str(junit)]) == 2
    assert "duplicate measurement" in ET.parse(junit).find(".//error").get("message")


@pytest.mark.parametrize(
    "key",
    [
        "python_build",
        "gil_enabled",
        "runtime_image",
        "worker_hostname",
        "cpu_affinity",
        "gpu_inventory",
        "cuda_major",
        "benchmark_script_sha256",
    ],
)
def test_environment_mismatch_is_invalid(key):
    before, after = report(), report()
    value = after["metadata"][key]
    after["metadata"][key] = {
        "gil_enabled": False,
        "cpu_affinity": [3],
        "gpu_inventory": ["other GPU"],
        "cuda_major": "12",
    }.get(key, str(value) + "-other")
    with pytest.raises(ValueError, match="incompatible"):
        compare(before, after)


def test_expected_identity_differences_are_allowed():
    before, after = report(), report()
    for key in (
        "timestamp_utc",
        "cvcuda_version",
        "cvcuda_extension_sha256",
        "python_executable",
        "cpu_frequency_before_khz",
    ):
        before["metadata"][key] = "reference"
        after["metadata"][key] = "candidate"
    assert all(row[-1] in ("PASS", "CONTROL") for row in compare(before, after))


@pytest.mark.parametrize("kind", ["missing", "duplicate", "extra", "empty"])
def test_case_coverage_must_be_complete(kind):
    before, after = report(), report()
    if kind == "missing":
        after["results"].pop()
    elif kind == "duplicate":
        after["results"][-1] = after["results"][0]
    elif kind == "extra":
        after["results"].append(after["results"][0])
    else:
        after["results"] = []
    with pytest.raises(ValueError, match="cases"):
        compare(before, after)


@pytest.mark.parametrize("number", [0, -1, float("nan"), float("inf"), True, "100"])
def test_invalid_duration(number):
    before, after = report(), report()
    after["results"][1]["window_ns"][0] = number
    with pytest.raises(ValueError):
        compare(before, after)


def test_null_runtime_image_is_invalid():
    data = report()
    data["metadata"]["runtime_image"] = "null"
    with pytest.raises(ValueError, match="runtime_image"):
        compare(data, data)


@pytest.mark.parametrize("number", [-1, float("nan"), float("inf")])
@pytest.mark.parametrize("argument", ["threshold", "max_noise"])
def test_invalid_threshold(number, argument):
    with pytest.raises(ValueError, match="thresholds"):
        compare(report(), report(), **{argument: number})


def test_parameters_and_iterations_must_match():
    before, after = report(), report()
    after["parameters"]["warmups"] = 4
    with pytest.raises(ValueError, match="parameters"):
        compare(before, after)
    after = report()
    after["results"][1]["iterations_per_window"] = 20
    with pytest.raises(ValueError, match="iteration counts"):
        compare(before, after)


@pytest.mark.parametrize(
    "kind,rc,tag",
    [
        ("pass", 0, None),
        ("slow", 1, "failure"),
        ("invalid", 2, "error"),
        ("noise", 2, "error"),
    ],
)
def test_cli_verdict_and_junit_survive_failure(tmp_path, kind, rc, tag):
    before, after = report(), report()
    if kind == "slow":
        scale(after, 1, 1.2)
    elif kind == "invalid":
        after = {}
    elif kind == "noise":
        after["results"][1]["window_ns"] = [1000, 2000, 3000, 4000, 5000]
    reference, candidate, junit = (
        tmp_path / name for name in ("reference.json", "candidate.json", "result.xml")
    )
    reference.write_text(json.dumps(before))
    candidate.write_text(json.dumps(after))
    assert (
        main(
            [
                "--reference",
                str(reference),
                "--candidate",
                str(candidate),
                "--junit",
                str(junit),
            ]
        )
        == rc
    )
    tree = ET.parse(junit)
    if tag:
        assert tree.find(f".//{tag}") is not None
    else:
        assert tree.find(".//failure") is None and tree.find(".//error") is None
