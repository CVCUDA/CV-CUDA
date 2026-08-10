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

"""Unit tests for JSON-backed compare_to_baseline.py."""

from __future__ import annotations

import json

import pytest

from _internal.baselines import (
    BaselineError,
    baseline_updates_from_payload,
    case_key_from_axis_values,
    load_config_index,
    load_sku_map,
    parse_case_key,
    sku_stems,
)
from compare_to_baseline import Thresholds, compare_updates, format_markdown


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=4) + "\n")


def _write_sku_map(path, *entries):
    _write_json(
        path,
        {
            "entries": [
                {
                    "gpu_name": name,
                    "power_cap_w": cap,
                    "locked_sm_clock_mhz": clock,
                    "stem": stem,
                }
                for name, cap, clock, stem in entries
            ]
        },
    )


def _case(config_key="resize_basic", dtype="uint8", shape="1x2x3", kernel=3):
    return case_key_from_axis_values(
        config_key,
        [
            ("InOutDataType", dtype),
            ("shape", shape),
            ("inputKind", "Tensor"),
            ("kernelSize", str(kernel)),
        ],
    )


def _write_config(
    root,
    *,
    baseline_time=100.0,
    include_advanced=True,
    include_other_operator=False,
    sku="TESTSKU",
):
    operators = root / "operators"
    basic_case = _case()
    configs = {
        "resize_basic": {
            "tier": "basic",
            "dtypes": ["uint8"],
            "string_axes": {
                "shape": ["1x2x3"],
                "inputKind": ["Tensor"],
            },
            "int64_axes": {
                "kernelSize": [3],
            },
            "baselines": {
                basic_case: {
                    sku: {
                        "n_runs": 2,
                        "gpu_time_us_cpp": baseline_time,
                        "gpu_time_us_python": baseline_time + 10.0,
                        "gpu_noise_us_cpp": 1.0,
                        "gpu_noise_us_python": 2.0,
                        "gpu_bwutil_cpp": 0.42,
                        "gpu_bwutil_python": 0.41,
                    }
                }
            },
        }
    }
    if include_advanced:
        advanced_case = _case("resize_advanced", shape="9x9x9")
        configs["resize_advanced"] = {
            "tier": "advanced",
            "dtypes": ["uint8"],
            "string_axes": {
                "shape": ["9x9x9"],
                "inputKind": ["Tensor"],
            },
            "int64_axes": {
                "kernelSize": [3],
            },
            "baselines": {
                advanced_case: {
                    sku: {
                        "n_runs": 2,
                        "gpu_time_us_cpp": 900.0,
                        "gpu_time_us_python": 910.0,
                        "gpu_noise_us_cpp": 1.0,
                        "gpu_noise_us_python": 1.0,
                        "gpu_bwutil_cpp": 0.12,
                        "gpu_bwutil_python": 0.11,
                    }
                }
            },
        }
    _write_json(operators / "resize.json", {"benchmark": "resize", "configs": configs})
    if include_other_operator:
        other_case = _case("gaussian_basic")
        _write_json(
            operators / "gaussian.json",
            {
                "benchmark": "gaussian",
                "configs": {
                    "gaussian_basic": {
                        "tier": "basic",
                        "dtypes": ["uint8"],
                        "string_axes": {
                            "shape": ["1x2x3"],
                            "inputKind": ["Tensor"],
                        },
                        "int64_axes": {"kernelSize": [3]},
                        "baselines": {
                            other_case: {
                                sku: {
                                    "n_runs": 2,
                                    "gpu_time_us_cpp": baseline_time,
                                    "gpu_time_us_python": baseline_time + 10.0,
                                    "gpu_noise_us_cpp": 1.0,
                                    "gpu_noise_us_python": 2.0,
                                    "gpu_bwutil_cpp": 0.42,
                                    "gpu_bwutil_python": 0.41,
                                }
                            }
                        },
                    }
                },
            },
        )
    return basic_case


def _current_payload(
    config_key="resize_basic",
    sku="TESTSKU",
    cpp_us=100.0,
    python_us=110.0,
    include_cpp=True,
    include_python=True,
    shape="1x2x3",
    kernel=3,
):
    metrics = {"n_runs": 1}
    if include_cpp:
        metrics.update(
            {
                "gpu_time_us_cpp": cpp_us,
                "gpu_noise_us_cpp": 1.0,
                "gpu_bwutil_cpp": 0.42,
            }
        )
    if include_python:
        metrics.update(
            {
                "gpu_time_us_python": python_us,
                "gpu_noise_us_python": 1.0,
                "gpu_bwutil_python": 0.41,
            }
        )
    return {
        _case(config_key, shape=shape, kernel=kernel): {
            sku: metrics,
        }
    }


def _updates(tmp_path, payload, *, sku="TESTSKU", include_other_operator=False):
    config_dir = tmp_path / "config"
    _write_config(
        config_dir,
        sku="TESTSKU",
        include_other_operator=include_other_operator,
    )
    sku_map = config_dir / "sku_map.json"
    _write_sku_map(
        sku_map,
        ("NVIDIA Test GPU", 350, 1095, "TESTSKU"),
        ("NVIDIA Test GPU 2", 250, 1095, sku),
    )
    index = load_config_index(config_dir / "operators")
    updates = baseline_updates_from_payload(
        payload,
        index=index,
        sku_stem_set=sku_stems(sku_map, strict=True),
        source=tmp_path / "bench_output.json",
    )
    return index, updates, tmp_path / "bench_output.json"


def test_case_key_parse_round_trip():
    case_key = _case()
    assert parse_case_key(case_key) == (
        "resize_basic",
        (
            ("InOutDataType", "uint8"),
            ("shape", "1x2x3"),
            ("inputKind", "Tensor"),
            ("kernelSize", "3"),
        ),
    )


def test_compare_passes_within_threshold(tmp_path):
    index, measurements, _ = _updates(
        tmp_path, _current_payload(cpp_us=105.0, python_us=115.0)
    )
    result = compare_updates(measurements, index=index, sku="TESTSKU")
    assert not result.any_fail
    assert result.matched == 2


def test_compare_flags_regression_and_improvement(tmp_path):
    index, measurements, _ = _updates(
        tmp_path, _current_payload(cpp_us=120.0, python_us=80.0)
    )
    result = compare_updates(measurements, index=index, sku="TESTSKU")
    assert len(result.regressions) == 1
    assert len(result.improvements) == 1


def test_compare_reports_missing_case_key(tmp_path):
    index, measurements, _ = _updates(tmp_path, _current_payload(include_python=False))
    result = compare_updates(measurements, index=index, sku="TESTSKU")
    assert len(result.missing_in_current) == 1
    assert result.missing_in_current[0].language == "python"


def test_compare_reports_missing_sku(tmp_path):
    index, measurements, _ = _updates(
        tmp_path,
        _current_payload(sku="NEWSKU"),
        sku="NEWSKU",
    )
    result = compare_updates(measurements, index=index, sku="NEWSKU")
    assert len(result.missing_sku) == 2


def test_tier_scoping_ignores_advanced_baselines_for_basic_run(tmp_path):
    index, measurements, _ = _updates(tmp_path, _current_payload())
    result = compare_updates(measurements, index=index, sku="TESTSKU")
    assert result.missing_in_current == []


def test_operator_scoping_ignores_unselected_operator_baselines(tmp_path):
    index, measurements, _ = _updates(
        tmp_path,
        _current_payload(),
        include_other_operator=True,
    )

    unscoped = compare_updates(measurements, index=index, sku="TESTSKU")
    scoped = compare_updates(
        measurements,
        index=index,
        sku="TESTSKU",
        operators={"resize"},
    )

    assert len(unscoped.missing_in_current) == 2
    assert not scoped.any_fail


def test_json_validation_rejects_unknown_config_key(tmp_path):
    with pytest.raises(BaselineError, match="does not exist"):
        _updates(tmp_path, _current_payload(config_key="unknown_basic"))


def test_json_validation_rejects_out_of_order_case_key(tmp_path):
    bad_case = (
        "resize_basic[kernelSize=3][InOutDataType=uint8]"
        "[shape=1x2x3][inputKind=Tensor]"
    )
    with pytest.raises(BaselineError, match="axes"):
        _updates(
            tmp_path,
            {
                bad_case: {
                    "TESTSKU": {
                        "n_runs": 1,
                        "gpu_time_us_cpp": 100.0,
                        "gpu_noise_us_cpp": 1.0,
                        "gpu_bwutil_cpp": 0.42,
                    }
                }
            },
        )


def test_json_validation_rejects_unknown_sku(tmp_path):
    with pytest.raises(BaselineError, match="unknown SKU"):
        _updates(tmp_path, _current_payload(sku="UNKNOWN"))


def test_load_sku_map_skips_invalid_entries_when_not_strict(tmp_path):
    path = tmp_path / "sku_map.json"
    path.write_text(
        '{"entries": ['
        '{"gpu_name": "bad", "power_cap_w": 1},'
        '{"gpu_name": "NVIDIA Test GPU", "power_cap_w": 350, '
        '"locked_sm_clock_mhz": 1095, "stem": "TESTSKU"}'
        "]}"
    )
    assert load_sku_map(path) == {("NVIDIA Test GPU", 350, 1095): "TESTSKU"}


def test_markdown_summary_reports_delta_stats(tmp_path):
    index, measurements, current = _updates(
        tmp_path, _current_payload(cpp_us=120.0, python_us=115.0)
    )
    result = compare_updates(measurements, index=index, sku="TESTSKU")
    report = format_markdown(result, "TESTSKU", "operators", current, Thresholds())
    assert "all-rows |Delta|:" in report
    assert "Regressions (1)" in report
