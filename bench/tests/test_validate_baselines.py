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

"""Unit tests for validate_baselines.py and shared JSON validation."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from _internal import validate_baselines
from _internal.baselines import (
    case_key_from_axis_values,
    load_config_index,
    split_operator_payload,
    sku_stems,
    validate_baselines_in_document,
)
from _internal.quality import DEFAULT_BENCHMARK_QUALITY


def test_internal_command_help_is_self_contained(tmp_path):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    result = subprocess.run(
        [sys.executable, validate_baselines.__file__, "--help"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "OPERATOR_JSON" in result.stdout
    assert "Examples:" in result.stdout
    assert "Exit status:" in result.stdout


def test_repo_root_after_internal_move():
    assert validate_baselines._repo_root() == Path(__file__).resolve().parents[2]


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=4) + "\n")


def _case(config_key="resize_basic", *, dtype="uint8", shape="1x2x3"):
    return case_key_from_axis_values(
        config_key,
        [
            ("InOutDataType", dtype),
            ("shape", shape),
            ("inputKind", "Tensor"),
        ],
    )


def _payload():
    case = _case()
    return {
        "benchmark": "resize",
        "configs": {
            "resize_basic": {
                "tier": "basic",
                "dtypes": ["uint8"],
                "string_axes": {
                    "shape": ["1x2x3"],
                    "inputKind": ["Tensor"],
                },
                "baselines": {
                    case: {
                        "TESTSKU": {
                            "n_runs": 2,
                            "gpu_time_us_cpp": 100.0,
                            "gpu_time_us_python": 110.0,
                            "gpu_noise_us_cpp": 1.0,
                            "gpu_noise_us_python": 2.0,
                            "gpu_bwutil_cpp": 0.42,
                            "gpu_bwutil_python": 0.41,
                        }
                    }
                },
            }
        },
    }


def _write_tree(tmp_path, payload):
    config_dir = tmp_path / "config"
    _write_json(config_dir / "operators" / "resize.json", payload)
    _write_json(
        config_dir / "sku_map.json",
        {
            "entries": [
                {
                    "gpu_name": "NVIDIA Test GPU",
                    "power_cap_w": 350,
                    "locked_sm_clock_mhz": 1095,
                    "stem": "TESTSKU",
                }
            ]
        },
    )
    index = load_config_index(config_dir / "operators")
    doc = index.docs_by_path[config_dir / "operators" / "resize.json"]
    stems = sku_stems(config_dir / "sku_map.json", strict=True)
    return validate_baselines_in_document(doc, index, stems)


def _write_config_dir(tmp_path, payload):
    config_dir = tmp_path / "config"
    _write_json(config_dir / "operators" / "resize.json", payload)
    _write_json(
        config_dir / "sku_map.json",
        {
            "entries": [
                {
                    "gpu_name": "NVIDIA Test GPU",
                    "power_cap_w": 350,
                    "locked_sm_clock_mhz": 1095,
                    "stem": "TESTSKU",
                }
            ]
        },
    )
    return config_dir


def _config_dir_with_ref_baseline(
    tmp_path, monkeypatch, *, cpp_time_us, python_time_us
):
    base_payload = _payload()
    current_payload = copy.deepcopy(base_payload)
    metrics = current_payload["configs"]["resize_basic"]["baselines"][_case()][
        "TESTSKU"
    ]
    metrics["gpu_time_us_cpp"] = cpp_time_us
    metrics["gpu_time_us_python"] = python_time_us
    config_dir = _write_config_dir(tmp_path, current_payload)
    base_doc = split_operator_payload(
        config_dir / "operators" / "resize.json", base_payload
    )

    monkeypatch.setattr(validate_baselines, "_verify_git_ref", lambda ref: None)
    monkeypatch.setattr(
        validate_baselines,
        "_operator_doc_at_ref",
        lambda ref, path: base_doc,
    )
    return config_dir


def test_valid_operator_json_passes(tmp_path):
    assert _write_tree(tmp_path, _payload()) == []


def test_valid_optional_paired_gap_stddev_passes(tmp_path):
    payload = _payload()
    payload["configs"]["resize_basic"]["baselines"][_case()]["TESTSKU"][
        "gpu_gap_stddev_us"
    ] = 3.0
    assert _write_tree(tmp_path, payload) == []


def test_wrong_config_key_in_case_key_fails(tmp_path):
    payload = _payload()
    case_payload = payload["configs"]["resize_basic"]["baselines"].pop(_case())
    payload["configs"]["resize_basic"]["baselines"][_case("other_basic")] = case_payload
    errors = _write_tree(tmp_path, payload)
    assert any("does not match nested config" in error for error in errors)


def test_out_of_order_axis_fails(tmp_path):
    payload = _payload()
    case_payload = payload["configs"]["resize_basic"]["baselines"].pop(_case())
    bad_case = "resize_basic[shape=1x2x3][InOutDataType=uint8][inputKind=Tensor]"
    payload["configs"]["resize_basic"]["baselines"][bad_case] = case_payload
    errors = _write_tree(tmp_path, payload)
    assert any("has axes" in error for error in errors)


def test_invalid_axis_value_fails(tmp_path):
    payload = _payload()
    case_payload = payload["configs"]["resize_basic"]["baselines"].pop(_case())
    payload["configs"]["resize_basic"]["baselines"][_case(shape="9x9x9")] = case_payload
    errors = _write_tree(tmp_path, payload)
    assert any("is not declared" in error for error in errors)


def test_missing_declared_case_key_fails(tmp_path):
    payload = _payload()
    payload["configs"]["resize_basic"]["string_axes"]["inputKind"] = [
        "Tensor",
        "VarShape",
    ]

    errors = _write_tree(tmp_path, payload)

    assert any(
        "missing baseline case" in error and "VarShape" in error for error in errors
    )


def test_unknown_sku_fails(tmp_path):
    payload = _payload()
    payload["configs"]["resize_basic"]["baselines"][_case()]["UNKNOWN"] = payload[
        "configs"
    ]["resize_basic"]["baselines"][_case()].pop("TESTSKU")
    errors = _write_tree(tmp_path, payload)
    assert any("unknown SKU" in error for error in errors)


def test_missing_metric_field_fails(tmp_path):
    payload = _payload()
    del payload["configs"]["resize_basic"]["baselines"][_case()]["TESTSKU"][
        "gpu_time_us_python"
    ]
    errors = _write_tree(tmp_path, payload)
    assert any("missing metric field" in error for error in errors)


def test_missing_bwutil_metric_field_fails(tmp_path):
    payload = _payload()
    del payload["configs"]["resize_basic"]["baselines"][_case()]["TESTSKU"][
        "gpu_bwutil_python"
    ]
    errors = _write_tree(tmp_path, payload)
    assert any("missing metric field" in error for error in errors)


def test_invalid_n_runs_fails(tmp_path):
    payload = _payload()
    payload["configs"]["resize_basic"]["baselines"][_case()]["TESTSKU"]["n_runs"] = 0
    errors = _write_tree(tmp_path, payload)
    assert any("n_runs" in error for error in errors)


def test_non_numeric_metric_fails(tmp_path):
    payload = copy.deepcopy(_payload())
    payload["configs"]["resize_basic"]["baselines"][_case()]["TESTSKU"][
        "gpu_noise_us_cpp"
    ] = "nope"
    errors = _write_tree(tmp_path, payload)
    assert any("must be numeric" in error for error in errors)


def test_non_numeric_optional_paired_gap_stddev_fails(tmp_path):
    payload = _payload()
    payload["configs"]["resize_basic"]["baselines"][_case()]["TESTSKU"][
        "gpu_gap_stddev_us"
    ] = "nope"
    errors = _write_tree(tmp_path, payload)
    assert any("gpu_gap_stddev_us must be numeric" in error for error in errors)


def test_optional_paired_gap_stddev_requires_repeated_runs(tmp_path):
    payload = _payload()
    metrics = payload["configs"]["resize_basic"]["baselines"][_case()]["TESTSKU"]
    metrics["n_runs"] = 1
    metrics["gpu_gap_stddev_us"] = 0.0
    errors = _write_tree(tmp_path, payload)
    assert any("gpu_gap_stddev_us requires n_runs >= 2" in error for error in errors)


def test_noise_above_quality_threshold_fails(tmp_path):
    payload = _payload()
    metrics = payload["configs"]["resize_basic"]["baselines"][_case()]["TESTSKU"]
    metrics["gpu_noise_us_cpp"] = (
        metrics["gpu_time_us_cpp"]
        * (DEFAULT_BENCHMARK_QUALITY.max_noise_pct + 1.0)
        / 100.0
    )

    errors = _write_tree(tmp_path, payload)

    assert any("cpp noise" in error and "exceeds" in error for error in errors)


def test_relative_parity_above_quality_threshold_fails(tmp_path):
    payload = _payload()
    metrics = payload["configs"]["resize_basic"]["baselines"][_case()]["TESTSKU"]
    metrics["gpu_time_us_python"] = metrics["gpu_time_us_cpp"] * (
        1.0 + (DEFAULT_BENCHMARK_QUALITY.max_perf_diff_pct + 1.0) / 100.0
    )

    errors = _write_tree(tmp_path, payload)

    assert any("C++/Python parity" in error and "%" in error for error in errors)


def test_absolute_parity_above_quality_threshold_fails(tmp_path):
    payload = _payload()
    metrics = payload["configs"]["resize_basic"]["baselines"][_case()]["TESTSKU"]
    metrics["gpu_time_us_cpp"] = DEFAULT_BENCHMARK_QUALITY.max_perf_diff_us * 100.0
    metrics["gpu_time_us_python"] = (
        metrics["gpu_time_us_cpp"] + DEFAULT_BENCHMARK_QUALITY.max_perf_diff_us + 1.0
    )

    errors = _write_tree(tmp_path, payload)

    assert any("C++/Python parity" in error and "us" in error for error in errors)


def test_main_rejects_same_key_baseline_regression_against_ref(
    tmp_path, monkeypatch, capsys
):
    config_dir = _config_dir_with_ref_baseline(
        tmp_path,
        monkeypatch,
        cpp_time_us=120.0,
        python_time_us=125.0,
    )

    rc = validate_baselines.main(
        [
            "--config-dir",
            str(config_dir),
            "--reject-regressions-from",
            "origin/main",
        ]
    )

    assert rc == 1
    captured = capsys.readouterr()
    assert "same-key baseline regressed +20.00%" in captured.err
    assert "origin/main" in captured.err


def test_main_allows_reviewed_same_key_baseline_regression_against_ref(
    tmp_path, monkeypatch, capsys
):
    config_dir = _config_dir_with_ref_baseline(
        tmp_path,
        monkeypatch,
        cpp_time_us=120.0,
        python_time_us=125.0,
    )

    assert (
        validate_baselines.main(
            [
                "--config-dir",
                str(config_dir),
                "--reject-regressions-from",
                "origin/main",
                "--allow-regressions",
            ]
        )
        == 0
    )
    assert (
        "reviewed same-key baseline regressions are allowed" in capsys.readouterr().err
    )


def test_allow_regressions_requires_comparison_ref(tmp_path, capsys):
    config_dir = _write_config_dir(tmp_path, _payload())

    assert (
        validate_baselines.main(
            ["--config-dir", str(config_dir), "--allow-regressions"]
        )
        == 2
    )
    assert (
        "--allow-regressions requires --reject-regressions-from"
        in capsys.readouterr().err
    )


def test_allow_regressions_rejects_invalid_comparison_ref(
    tmp_path, monkeypatch, capsys
):
    config_dir = _write_config_dir(tmp_path, _payload())

    def reject_ref(ref):
        raise validate_baselines.BaselineError(
            f"baseline regression ref does not exist: {ref}"
        )

    monkeypatch.setattr(validate_baselines, "_verify_git_ref", reject_ref)

    assert (
        validate_baselines.main(
            [
                "--config-dir",
                str(config_dir),
                "--reject-regressions-from",
                "missing-ref",
                "--allow-regressions",
            ]
        )
        == 2
    )
    assert (
        "baseline regression ref does not exist: missing-ref" in capsys.readouterr().err
    )


def test_allow_regressions_does_not_bypass_baseline_validation(tmp_path):
    payload = _payload()
    del payload["configs"]["resize_basic"]["baselines"][_case()]["TESTSKU"][
        "gpu_noise_us_cpp"
    ]
    config_dir = _write_config_dir(tmp_path, payload)

    assert (
        validate_baselines.main(
            [
                "--config-dir",
                str(config_dir),
                "--reject-regressions-from",
                "origin/main",
                "--allow-regressions",
            ]
        )
        == 1
    )


def test_main_accepts_same_key_baseline_change_within_ref_threshold(
    tmp_path, monkeypatch
):
    config_dir = _config_dir_with_ref_baseline(
        tmp_path,
        monkeypatch,
        cpp_time_us=105.0,
        python_time_us=115.0,
    )

    assert (
        validate_baselines.main(
            [
                "--config-dir",
                str(config_dir),
                "--reject-regressions-from",
                "origin/main",
            ]
        )
        == 0
    )


def test_git_ref_commands_trust_only_the_repository_root(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(_payload()),
            stderr="",
        )

    monkeypatch.setattr(validate_baselines.subprocess, "run", fake_run)

    validate_baselines._verify_git_ref("origin/main")
    validate_baselines._operator_doc_at_ref(
        "origin/main",
        validate_baselines._repo_root()
        / "bench"
        / "config"
        / "operators"
        / "resize.json",
    )

    safe_directory = f"safe.directory={validate_baselines._repo_root()}"
    assert len(calls) == 2
    assert all(command[:3] == ["git", "-c", safe_directory] for command, _ in calls)
    assert all(kwargs["cwd"] == validate_baselines._repo_root() for _, kwargs in calls)
    assert all(
        kwargs["timeout"] == validate_baselines.GIT_COMMAND_TIMEOUT_SECONDS
        for _, kwargs in calls
    )


@pytest.mark.parametrize("operation", ["verify", "show"])
def test_git_ref_commands_report_timeouts(monkeypatch, operation):
    def fake_run(command, **kwargs):
        raise subprocess.TimeoutExpired(
            cmd=command,
            timeout=kwargs["timeout"],
        )

    monkeypatch.setattr(validate_baselines.subprocess, "run", fake_run)

    with pytest.raises(validate_baselines.BaselineError, match="timed out"):
        if operation == "verify":
            validate_baselines._verify_git_ref("origin/main")
        else:
            validate_baselines._operator_doc_at_ref(
                "origin/main",
                validate_baselines._repo_root()
                / "bench"
                / "config"
                / "operators"
                / "resize.json",
            )
