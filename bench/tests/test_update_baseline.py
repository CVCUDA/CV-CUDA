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

"""Unit tests for JSON-backed update_baseline.py."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from _internal import update_baseline
from _internal.baselines import case_key_from_axis_values
from _internal.quality import DEFAULT_BENCHMARK_QUALITY


def test_internal_command_help_is_self_contained(tmp_path):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    result = subprocess.run(
        [sys.executable, update_baseline.__file__, "--help"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--from JSON_OR_DIR" in result.stdout
    assert "Examples:" in result.stdout
    assert "Exit status:" in result.stdout


def test_repo_root_after_internal_move():
    assert update_baseline._repo_root() == Path(__file__).resolve().parents[2]


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=4) + "\n")


def _case(config_key, dtype="uint8", shape="1x2x3"):
    return case_key_from_axis_values(
        config_key,
        [
            ("InOutDataType", dtype),
            ("shape", shape),
            ("inputKind", "Tensor"),
        ],
    )


def _config_entry(dtype="uint8", shape="1x2x3", baselines=None):
    entry = {
        "tier": "basic",
        "dtypes": [dtype],
        "string_axes": {
            "shape": [shape],
            "inputKind": ["Tensor"],
        },
    }
    if baselines is not None:
        entry["baselines"] = baselines
    return entry


def _write_config_tree(
    root, *, include_second_sku=False, include_resize_baseline=False
):
    entries = [
        {
            "gpu_name": "NVIDIA Test GPU",
            "power_cap_w": 350,
            "locked_sm_clock_mhz": 1095,
            "stem": "TESTSKU",
        }
    ]
    if include_second_sku:
        entries.append(
            {
                "gpu_name": "NVIDIA Test GPU 2",
                "power_cap_w": 250,
                "locked_sm_clock_mhz": 1095,
                "stem": "TESTSKU2",
            }
        )
    _write_json(
        root / "sku_map.json",
        {"entries": entries},
    )
    resize_baselines = None
    if include_resize_baseline:
        resize_case = _case("resize_basic")
        resize_baselines = {
            resize_case: {
                "TESTSKU": {
                    "n_runs": 1,
                    "gpu_time_us_cpp": 100.0,
                    "gpu_time_us_python": 110.0,
                    "gpu_noise_us_cpp": 1.0,
                    "gpu_noise_us_python": 1.0,
                    "gpu_bwutil_cpp": 0.42,
                    "gpu_bwutil_python": 0.41,
                }
            }
        }

    _write_json(
        root / "operators" / "resize.json",
        {
            "benchmark": "resize",
            "configs": {
                "resize_basic": _config_entry(baselines=resize_baselines),
            },
        },
    )
    gaussian_case = _case("gaussian_basic", shape="4x5x6")
    _write_json(
        root / "operators" / "gaussian.json",
        {
            "benchmark": "gaussian",
            "configs": {
                "gaussian_basic": _config_entry(
                    shape="4x5x6",
                    baselines={
                        gaussian_case: {
                            "TESTSKU": {
                                "n_runs": 1,
                                "gpu_time_us_cpp": 200.0,
                                "gpu_time_us_python": 210.0,
                                "gpu_noise_us_cpp": 2.0,
                                "gpu_noise_us_python": 3.0,
                                "gpu_bwutil_cpp": 0.22,
                                "gpu_bwutil_python": 0.21,
                            }
                        }
                    },
                ),
            },
        },
    )


def _baseline_payload(
    config_key="resize_basic",
    sku="TESTSKU",
    cpp_us=100.0,
    python_us=110.0,
    cpp_noise_us=1.0,
    python_noise_us=1.0,
    cpp_bwutil=0.42,
    python_bwutil=0.41,
    n_runs=1,
    gap_stddev_us=None,
    shape="1x2x3",
):
    case = _case(config_key, shape=shape)
    metrics = {
        "n_runs": n_runs,
        "gpu_time_us_cpp": cpp_us,
        "gpu_time_us_python": python_us,
        "gpu_noise_us_cpp": cpp_noise_us,
        "gpu_noise_us_python": python_noise_us,
        "gpu_bwutil_cpp": cpp_bwutil,
        "gpu_bwutil_python": python_bwutil,
    }
    if gap_stddev_us is not None:
        metrics["gpu_gap_stddev_us"] = gap_stddev_us
    return {
        case: {
            sku: metrics,
        }
    }


def _write_run_json(path, *payloads):
    merged = {}
    for payload in payloads:
        for case_key, case_payload in payload.items():
            merged.setdefault(case_key, {}).update(case_payload)
    _write_json(path, merged)


def _run(config_dir, *args):
    return update_baseline.main(["--config-dir", str(config_dir), *args])


def _load_operator(config_dir, name):
    return json.loads((config_dir / "operators" / f"{name}.json").read_text())


def test_normalize_operators_handles_repeats_and_commas():
    assert update_baseline._normalize_operators(None) is None
    assert update_baseline._normalize_operators(["resize,gaussian", "resize"]) == {
        "resize",
        "gaussian",
    }


def test_update_writes_nested_json_baseline(tmp_path):
    config_dir = tmp_path / "config"
    _write_config_tree(config_dir)
    run_json = tmp_path / "run.json"
    _write_run_json(run_json, _baseline_payload())

    assert _run(config_dir, "--from", str(run_json)) == 0

    raw = _load_operator(config_dir, "resize")
    case = _case("resize_basic")
    metrics = raw["configs"]["resize_basic"]["baselines"][case]["TESTSKU"]
    assert metrics == {
        "n_runs": 1,
        "gpu_time_us_cpp": 100.0,
        "gpu_time_us_python": 110.0,
        "gpu_noise_us_cpp": 1.0,
        "gpu_noise_us_python": 1.0,
        "gpu_bwutil_cpp": 0.42,
        "gpu_bwutil_python": 0.41,
    }


def test_multi_input_averages_repeated_runs(tmp_path):
    config_dir = tmp_path / "config"
    _write_config_tree(config_dir)
    runs = tmp_path / "runs"
    runs.mkdir()
    _write_run_json(
        runs / "r1.json",
        _baseline_payload(cpp_us=100.0, python_us=105.0),
    )
    _write_run_json(
        runs / "r2.json",
        _baseline_payload(cpp_us=110.0, python_us=115.0),
    )

    _run(config_dir, "--from", str(runs))

    case = _case("resize_basic")
    metrics = _load_operator(config_dir, "resize")["configs"]["resize_basic"][
        "baselines"
    ][case]["TESTSKU"]
    assert metrics["n_runs"] == 2
    assert metrics["gpu_time_us_cpp"] == pytest.approx(105.0)
    assert metrics["gpu_time_us_python"] == pytest.approx(110.0)
    assert metrics["gpu_bwutil_cpp"] == pytest.approx(0.42)
    assert metrics["gpu_bwutil_python"] == pytest.approx(0.41)
    assert metrics["gpu_gap_stddev_us"] == pytest.approx(0.0)


def test_multi_input_computes_paired_artifact_gap_stddev(tmp_path):
    config_dir = tmp_path / "config"
    _write_config_tree(config_dir)
    runs = tmp_path / "runs"
    runs.mkdir()
    _write_run_json(
        runs / "r1.json",
        _baseline_payload(cpp_us=100.0, python_us=105.0),
    )
    _write_run_json(
        runs / "r2.json",
        _baseline_payload(cpp_us=200.0, python_us=215.0),
    )

    _run(config_dir, "--from", str(runs))

    case = _case("resize_basic")
    metrics = _load_operator(config_dir, "resize")["configs"]["resize_basic"][
        "baselines"
    ][case]["TESTSKU"]
    assert metrics["gpu_time_us_cpp"] == pytest.approx(150.0)
    assert metrics["gpu_time_us_python"] == pytest.approx(160.0)
    assert metrics["gpu_gap_stddev_us"] == pytest.approx(math.sqrt(50.0))


def test_multi_input_pools_preaggregated_paired_gap_stddev(tmp_path):
    config_dir = tmp_path / "config"
    _write_config_tree(config_dir)
    run1 = tmp_path / "run1.json"
    run2 = tmp_path / "run2.json"
    _write_run_json(
        run1,
        _baseline_payload(
            cpp_us=100.0,
            python_us=105.0,
            n_runs=2,
            gap_stddev_us=math.sqrt(2.0),
        ),
    )
    _write_run_json(
        run2,
        _baseline_payload(
            cpp_us=200.0,
            python_us=210.0,
            n_runs=3,
            gap_stddev_us=2.0,
        ),
    )

    _run(config_dir, "--from", str(run1), "--from", str(run2))

    case = _case("resize_basic")
    metrics = _load_operator(config_dir, "resize")["configs"]["resize_basic"][
        "baselines"
    ][case]["TESTSKU"]
    assert metrics["n_runs"] == 5
    assert metrics["gpu_time_us_cpp"] == pytest.approx(160.0)
    assert metrics["gpu_time_us_python"] == pytest.approx(168.0)
    assert metrics["gpu_gap_stddev_us"] == pytest.approx(math.sqrt(10.0))


def test_preaggregated_input_without_gap_dispersion_does_not_invent_it(tmp_path):
    config_dir = tmp_path / "config"
    _write_config_tree(config_dir)
    run_json = tmp_path / "run.json"
    _write_run_json(run_json, _baseline_payload(n_runs=2))

    _run(config_dir, "--from", str(run_json))

    case = _case("resize_basic")
    metrics = _load_operator(config_dir, "resize")["configs"]["resize_basic"][
        "baselines"
    ][case]["TESTSKU"]
    assert "gpu_gap_stddev_us" not in metrics


def test_update_drops_stale_gap_dispersion_when_new_pairing_is_unavailable(tmp_path):
    config_dir = tmp_path / "config"
    _write_config_tree(config_dir, include_resize_baseline=True)
    case = _case("resize_basic")
    raw = _load_operator(config_dir, "resize")
    existing = raw["configs"]["resize_basic"]["baselines"][case]["TESTSKU"]
    existing["n_runs"] = 2
    existing["gpu_gap_stddev_us"] = 3.0
    _write_json(config_dir / "operators" / "resize.json", raw)
    run_json = tmp_path / "run.json"
    _write_run_json(run_json, _baseline_payload(n_runs=2))

    _run(config_dir, "--from", str(run_json))

    metrics = _load_operator(config_dir, "resize")["configs"]["resize_basic"][
        "baselines"
    ][case]["TESTSKU"]
    assert "gpu_gap_stddev_us" not in metrics


def test_operator_filter_preserves_unrelated_operator_json(tmp_path):
    config_dir = tmp_path / "config"
    _write_config_tree(config_dir)
    before = (config_dir / "operators" / "gaussian.json").read_text()
    run_json = tmp_path / "run.json"
    _write_run_json(
        run_json,
        _baseline_payload(),
        _baseline_payload(
            config_key="gaussian_basic",
            cpp_us=999.0,
            python_us=999.0,
            shape="4x5x6",
        ),
    )

    _run(config_dir, "--from", str(run_json), "--operator", "resize")

    assert (config_dir / "operators" / "gaussian.json").read_text() == before


def test_dry_run_writes_nothing(tmp_path, capsys):
    config_dir = tmp_path / "config"
    _write_config_tree(config_dir)
    before = (config_dir / "operators" / "resize.json").read_text()
    run_json = tmp_path / "run.json"
    _write_run_json(run_json, _baseline_payload())

    assert _run(config_dir, "--from", str(run_json), "--dry-run") == 0
    assert (config_dir / "operators" / "resize.json").read_text() == before
    assert "dry run" in capsys.readouterr().out


def test_update_rejects_same_key_regression_without_writing(tmp_path, capsys):
    config_dir = tmp_path / "config"
    _write_config_tree(config_dir, include_resize_baseline=True)
    before = (config_dir / "operators" / "resize.json").read_text()
    run_json = tmp_path / "run.json"
    _write_run_json(run_json, _baseline_payload(cpp_us=120.0, python_us=125.0))

    with pytest.raises(SystemExit) as exc:
        _run(config_dir, "--from", str(run_json))

    assert exc.value.code == 1
    assert (config_dir / "operators" / "resize.json").read_text() == before
    captured = capsys.readouterr()
    assert "same-key baseline regressions" in captured.err
    assert "resize_basic" in captured.err
    assert "--allow-regressions" in captured.err


def test_dry_run_rejects_same_key_regression_without_writing(tmp_path):
    config_dir = tmp_path / "config"
    _write_config_tree(config_dir, include_resize_baseline=True)
    before = (config_dir / "operators" / "resize.json").read_text()
    run_json = tmp_path / "run.json"
    _write_run_json(run_json, _baseline_payload(cpp_us=120.0, python_us=125.0))

    with pytest.raises(SystemExit):
        _run(config_dir, "--from", str(run_json), "--dry-run")

    assert (config_dir / "operators" / "resize.json").read_text() == before


def test_update_accepts_same_key_change_within_regression_threshold(tmp_path):
    config_dir = tmp_path / "config"
    _write_config_tree(config_dir, include_resize_baseline=True)
    run_json = tmp_path / "run.json"
    _write_run_json(run_json, _baseline_payload(cpp_us=105.0, python_us=115.0))

    assert _run(config_dir, "--from", str(run_json)) == 0

    case = _case("resize_basic")
    metrics = _load_operator(config_dir, "resize")["configs"]["resize_basic"][
        "baselines"
    ][case]["TESTSKU"]
    assert metrics["gpu_time_us_cpp"] == 105.0
    assert metrics["gpu_time_us_python"] == 115.0


def test_update_allows_same_key_regression_with_explicit_override(tmp_path):
    config_dir = tmp_path / "config"
    _write_config_tree(config_dir, include_resize_baseline=True)
    run_json = tmp_path / "run.json"
    _write_run_json(run_json, _baseline_payload(cpp_us=120.0, python_us=125.0))

    assert _run(config_dir, "--from", str(run_json), "--allow-regressions") == 0

    case = _case("resize_basic")
    metrics = _load_operator(config_dir, "resize")["configs"]["resize_basic"][
        "baselines"
    ][case]["TESTSKU"]
    assert metrics["gpu_time_us_cpp"] == 120.0
    assert metrics["gpu_time_us_python"] == 125.0


def test_non_json_input_is_rejected_without_writing(tmp_path):
    config_dir = tmp_path / "config"
    _write_config_tree(config_dir)
    before = (config_dir / "operators" / "resize.json").read_text()
    text = tmp_path / "bad.txt"
    text.write_text("not,supported\n")

    with pytest.raises(SystemExit):
        _run(config_dir, "--from", str(text))
    assert (config_dir / "operators" / "resize.json").read_text() == before


def test_routes_multiple_skus_in_one_invocation(tmp_path):
    config_dir = tmp_path / "config"
    _write_config_tree(config_dir, include_second_sku=True)
    runs = tmp_path / "runs"
    runs.mkdir()
    _write_run_json(
        runs / "sku1.json",
        _baseline_payload(),
    )
    _write_run_json(
        runs / "sku2.json",
        _baseline_payload(sku="TESTSKU2", cpp_us=200.0, python_us=220.0),
    )

    _run(config_dir, "--from", str(runs))

    case = _case("resize_basic")
    case_payload = _load_operator(config_dir, "resize")["configs"]["resize_basic"][
        "baselines"
    ][case]
    assert set(case_payload) == {"TESTSKU", "TESTSKU2"}


def test_update_imports_raw_baseline_json(tmp_path):
    config_dir = tmp_path / "config"
    _write_config_tree(config_dir)
    case = _case("resize_basic")
    baseline_json = tmp_path / "run.json"
    _write_run_json(
        baseline_json,
        _baseline_payload(python_noise_us=2.0),
    )

    _run(config_dir, "--from", str(baseline_json))

    metrics = _load_operator(config_dir, "resize")["configs"]["resize_basic"][
        "baselines"
    ][case]["TESTSKU"]
    assert metrics == {
        "n_runs": 1,
        "gpu_time_us_cpp": 100.0,
        "gpu_time_us_python": 110.0,
        "gpu_noise_us_cpp": 1.0,
        "gpu_noise_us_python": 2.0,
        "gpu_bwutil_cpp": 0.42,
        "gpu_bwutil_python": 0.41,
    }


def test_update_averages_json_repeated_runs(tmp_path):
    config_dir = tmp_path / "config"
    _write_config_tree(config_dir)
    case = _case("resize_basic")
    run1 = tmp_path / "run1.json"
    run2 = tmp_path / "run2.json"
    _write_run_json(run1, _baseline_payload())
    _write_run_json(
        run2,
        _baseline_payload(
            cpp_us=120.0,
            python_us=130.0,
            cpp_noise_us=3.0,
            python_noise_us=5.0,
            cpp_bwutil=0.44,
            python_bwutil=0.43,
        ),
    )

    _run(config_dir, "--from", str(run1), "--from", str(run2))

    metrics = _load_operator(config_dir, "resize")["configs"]["resize_basic"][
        "baselines"
    ][case]["TESTSKU"]
    assert metrics["n_runs"] == 2
    assert metrics["gpu_time_us_cpp"] == pytest.approx(110.0)
    assert metrics["gpu_time_us_python"] == pytest.approx(120.0)
    assert metrics["gpu_noise_us_cpp"] == pytest.approx(2.0)
    assert metrics["gpu_noise_us_python"] == pytest.approx(3.0)
    assert metrics["gpu_bwutil_cpp"] == pytest.approx(0.43)
    assert metrics["gpu_bwutil_python"] == pytest.approx(0.42)
    assert metrics["gpu_gap_stddev_us"] == pytest.approx(0.0)


def test_invalid_json_aborts_without_writing(tmp_path):
    config_dir = tmp_path / "config"
    _write_config_tree(config_dir)
    before = (config_dir / "operators" / "resize.json").read_text()
    case = _case("resize_basic")
    baseline_json = tmp_path / "bad.json"
    _write_json(
        baseline_json,
        {
            case: {
                "UNKNOWN": {
                    "n_runs": 1,
                    "gpu_time_us_cpp": 100.0,
                    "gpu_noise_us_cpp": 1.0,
                    "gpu_bwutil_cpp": 0.42,
                }
            }
        },
    )

    with pytest.raises(SystemExit):
        _run(config_dir, "--from", str(baseline_json))
    assert (config_dir / "operators" / "resize.json").read_text() == before


def test_quality_violating_json_aborts_without_writing(tmp_path):
    config_dir = tmp_path / "config"
    _write_config_tree(config_dir)
    before = (config_dir / "operators" / "resize.json").read_text()
    run_json = tmp_path / "run.json"
    _write_run_json(
        run_json,
        _baseline_payload(
            python_us=100.0
            * (1.0 + (DEFAULT_BENCHMARK_QUALITY.max_perf_diff_pct + 1.0) / 100.0)
        ),
    )

    with pytest.raises(SystemExit):
        _run(config_dir, "--from", str(run_json))

    assert (config_dir / "operators" / "resize.json").read_text() == before


def test_dry_run_validates_quality_without_writing(tmp_path):
    config_dir = tmp_path / "config"
    _write_config_tree(config_dir)
    before = (config_dir / "operators" / "resize.json").read_text()
    run_json = tmp_path / "run.json"
    _write_run_json(
        run_json,
        _baseline_payload(cpp_noise_us=DEFAULT_BENCHMARK_QUALITY.max_noise_pct + 1.0),
    )

    with pytest.raises(SystemExit):
        _run(config_dir, "--from", str(run_json), "--dry-run")

    assert (config_dir / "operators" / "resize.json").read_text() == before


def test_single_language_json_aborts_without_writing(tmp_path):
    config_dir = tmp_path / "config"
    _write_config_tree(config_dir)
    before = (config_dir / "operators" / "resize.json").read_text()
    case = _case("resize_basic")
    run_json = tmp_path / "run.json"
    _write_json(
        run_json,
        {
            case: {
                "TESTSKU": {
                    "n_runs": 1,
                    "gpu_time_us_cpp": 100.0,
                    "gpu_noise_us_cpp": 1.0,
                    "gpu_bwutil_cpp": 0.42,
                }
            }
        },
    )

    with pytest.raises(SystemExit):
        _run(config_dir, "--from", str(run_json))

    assert (config_dir / "operators" / "resize.json").read_text() == before


def test_from_diff_filename_mapping(monkeypatch):
    class _Stub:
        returncode = 0
        stdout = (
            "src/cvcuda/priv/OpGaussian.cpp\n"
            "src/cvcuda/priv/OpGaussianNoise.cu\n"
            "docs/index.rst\n"
        )
        stderr = ""

    monkeypatch.setattr(update_baseline.subprocess, "run", lambda *a, **k: _Stub())
    assert update_baseline._ops_from_git_diff(
        "origin/main", {"gaussian", "gaussiannoise", "resize"}
    ) == {"gaussian", "gaussiannoise"}
