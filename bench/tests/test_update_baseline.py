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
from types import SimpleNamespace

import pytest

from _internal import update_baseline
from _internal.baselines import (
    build_baseline_document,
    case_key_from_axis_values,
    normalize_operator_names,
)
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
    root,
    *,
    include_second_sku=False,
    include_resize_baseline=False,
    include_second_resize=False,
):
    entries = [
        {
            "gpu_name": "NVIDIA Test GPU",
            "power_cap_w": 350,
            "locked_sm_clock_mhz": 1095,
            "cuda_major": 13,
            "stem": "TESTSKU",
        }
    ]
    if include_second_sku:
        entries.append(
            {
                "gpu_name": "NVIDIA Test GPU 2",
                "power_cap_w": 250,
                "locked_sm_clock_mhz": 1095,
                "cuda_major": 13,
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

    resize_configs = {
        "resize_basic": _config_entry(baselines=resize_baselines),
    }
    if include_second_resize:
        secondary_case = _case("resize_secondary", shape="2x3x4")
        resize_configs["resize_secondary"] = _config_entry(
            shape="2x3x4",
            baselines={
                secondary_case: {
                    "TESTSKU": {
                        "n_runs": 5,
                        "gpu_time_us_cpp": 200.0,
                        "gpu_time_us_python": 210.0,
                        "gpu_noise_us_cpp": 1.0,
                        "gpu_noise_us_python": 1.0,
                        "gpu_bwutil_cpp": 0.42,
                        "gpu_bwutil_python": 0.41,
                    }
                }
            },
        )
    _write_json(
        root / "operators" / "resize.json",
        {"benchmark": "resize", "configs": resize_configs},
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


def _write_stability_wave(
    root,
    values,
    *,
    python_values=None,
    include_second_resize=False,
    include_gaussian=False,
):
    paths = []
    for index, cpp_us in enumerate(values, 1):
        path = root / f"run-{index}-aaaaaaaa.json"
        python_us = python_values[index - 1] if python_values else cpp_us + 10
        baselines = _baseline_payload(cpp_us=cpp_us, python_us=python_us)
        if include_second_resize:
            baselines.update(
                _baseline_payload(
                    config_key="resize_secondary",
                    cpp_us=190 + index / 10,
                    python_us=200 + index / 10,
                    shape="2x3x4",
                )
            )
        if include_gaussian:
            baselines.update(
                _baseline_payload(
                    config_key="gaussian_basic",
                    cpp_us=190 + index / 10,
                    python_us=200 + index / 10,
                    shape="4x5x6",
                )
            )
        _write_json(
            path,
            build_baseline_document(
                baselines,
                run_metadata={
                    "git_sha": "a" * 40,
                    "git_ref": "test/baselines",
                    "run_context": "mr",
                    "cuda_major": 13,
                    "gpu_name": "NVIDIA Test GPU",
                    "power_cap_w": 350,
                    "locked_sm_clock_mhz": 1095,
                    "sm_clock_device_verified": True,
                    "vbios_version": "1.0",
                    "driver_version": "600.0",
                    "k8s_node_name": f"node-{index}",
                },
            ),
        )
        paths.append(path)
    return paths


def _run(config_dir, *args):
    argv = list(args)
    if "--stable-only" in argv and "--expect-revision" not in argv:
        argv.extend(["--expect-revision", "A" * 40])
    return update_baseline.main(["--config-dir", str(config_dir), *argv])


def _load_operator(config_dir, name):
    return json.loads((config_dir / "operators" / f"{name}.json").read_text())


def test_normalize_operators_handles_repeats_and_commas():
    assert normalize_operator_names(None) is None
    assert normalize_operator_names(["resize,gaussian", "resize"]) == {
        "resize",
        "gaussian",
    }


def test_stable_only_requires_both_language_rows():
    key = ("resize_basic", _case("resize_basic"), "TESTSKU")
    update = SimpleNamespace()
    rows = [
        SimpleNamespace(
            config_key=key[0],
            case_key=key[1],
            sku=key[2],
            language=language,
            stable=stable,
        )
        for language, stable in (("cpp", True), ("python", False))
    ]

    assert (
        update_baseline._stable_updates(
            {key: update},
            SimpleNamespace(
                issues=[], rows=rows, minimum_cluster_runs=5, expected_runs=5
            ),
        )
        == {}
    )

    rows[1].stable = True
    assert update_baseline._stable_updates(
        {key: update},
        SimpleNamespace(issues=[], rows=rows, minimum_cluster_runs=5, expected_runs=5),
    ) == {key: update}

    assert (
        update_baseline._stable_updates(
            {key: update},
            SimpleNamespace(
                issues=[], rows=rows[:1], minimum_cluster_runs=5, expected_runs=5
            ),
        )
        == {}
    )


def test_stable_only_keys_rows_by_case_and_sku():
    case = _case("resize_basic")
    other_case = _case("resize_basic", shape="2x3x4")
    keep = ("resize_basic", case, "TESTSKU")
    unstable_sku = ("resize_basic", case, "TESTSKU2")
    unstable_case = ("resize_basic", other_case, "TESTSKU")
    updates = {key: SimpleNamespace() for key in (keep, unstable_sku, unstable_case)}
    rows = [
        SimpleNamespace(
            config_key=key[0],
            case_key=key[1],
            sku=key[2],
            language=language,
            stable=stable,
        )
        for key, stability in (
            (keep, (True, True)),
            (unstable_sku, (True, False)),
            (unstable_case, (False, True)),
        )
        for language, stable in zip(("cpp", "python"), stability)
    ]

    assert update_baseline._stable_updates(
        updates,
        SimpleNamespace(issues=[], rows=rows, minimum_cluster_runs=5, expected_runs=5),
    ) == {keep: updates[keep]}


def test_regression_findings_select_only_changes_over_the_threshold():
    keep = ("resize_basic", "keep", "TESTSKU")
    withhold = ("resize_basic", "withhold", "TESTSKU")
    new = ("resize_basic", "new", "TESTSKU")
    diffs = [
        {
            "config_key": key[0],
            "case_key": key[1],
            "sku": key[2],
            "pct": pct,
            "language": "cpp",
            "axes": (),
            "prev_us": 100.0,
            "new_us": 101.01,
        }
        for key, pct in ((keep, 1.0 + 1e-12), (withhold, 1.01), (new, None))
    ]

    withheld, errors = update_baseline._regression_findings(diffs, 1.0)

    assert withheld == {withhold}
    assert len(errors) == 1
    assert "+1.01%" in errors[0]


def test_stable_only_cli_accepts_scoped_wave_and_custom_sku_map(
    tmp_path, capsys, monkeypatch
):
    config_dir = tmp_path / "config"
    _write_config_tree(config_dir, include_resize_baseline=True)
    custom_sku_map = tmp_path / "custom-sku-map.json"
    (config_dir / "sku_map.json").replace(custom_sku_map)
    wave = tmp_path / "wave"
    _write_stability_wave(wave, [100, 100.1, 100.2, 100.3, 100.4])
    load_calls = 0
    real_load_artifacts = update_baseline.load_artifacts

    def counted_load_artifacts(paths):
        nonlocal load_calls
        load_calls += 1
        return real_load_artifacts(paths)

    monkeypatch.setattr(update_baseline, "load_artifacts", counted_load_artifacts)
    monkeypatch.setattr(
        update_baseline.baseline_stability, "load_artifacts", counted_load_artifacts
    )

    assert (
        _run(
            config_dir,
            "--from",
            str(wave),
            "--sku-map",
            str(custom_sku_map),
            "--operator",
            "resize",
            "--stable-only",
            "--dry-run",
        )
        == 0
    )
    output = capsys.readouterr().out
    assert load_calls == 1
    assert "--stable-only: withholding 0 of 1 entries" in output
    assert "dry run" in output


def test_stable_only_cli_withholds_unstable_entry(tmp_path, capsys):
    config_dir = tmp_path / "config"
    _write_config_tree(config_dir, include_resize_baseline=True)
    before = (config_dir / "operators" / "resize.json").read_text()
    wave = tmp_path / "wave"
    _write_stability_wave(wave, [100, 100.1, 100.2, 100.3, 110])

    assert (
        _run(
            config_dir,
            "--from",
            str(wave),
            "--operator",
            "resize",
            "--stable-only",
            "--dry-run",
        )
        == 1
    )
    assert (config_dir / "operators" / "resize.json").read_text() == before
    output = capsys.readouterr().out
    assert "--stable-only: withholding 1 of 1 entries" in output
    assert "withheld entries" in output
    assert "unstable or incomplete wave" in output
    assert "resize_basic" in output
    assert "temporary investigation snapshot" not in output
    assert (
        f"{config_dir / 'operators' / 'resize.json'}: partial-refresh marker only"
        in output
    )


def test_stable_only_rejects_unstable_new_entry_without_committed_value(
    tmp_path, capsys
):
    config_dir = tmp_path / "config"
    _write_config_tree(config_dir)
    wave = tmp_path / "wave"
    _write_stability_wave(wave, [100, 100.1, 100.2, 100.3, 110])

    with pytest.raises(SystemExit) as exc:
        _run(
            config_dir,
            "--from",
            str(wave),
            "--operator",
            "resize",
            "--stable-only",
            "--dry-run",
        )

    assert exc.value.code == 1
    error = capsys.readouterr().err
    assert "without a committed baseline" in error
    assert "must be remeasured until stable" in error
    assert "TESTSKU resize_basic" in error


def test_stable_only_imports_stable_new_entry(tmp_path):
    config_dir = tmp_path / "config"
    _write_config_tree(config_dir)
    wave = tmp_path / "wave"
    _write_stability_wave(wave, [100, 100.1, 100.2, 100.3, 100.4])

    assert (
        _run(
            config_dir,
            "--from",
            str(wave),
            "--operator",
            "resize",
            "--stable-only",
            "--write-summary",
            str(tmp_path / "summary.md"),
        )
        == 0
    )

    payload = _load_operator(config_dir, "resize")
    metrics = payload["configs"]["resize_basic"]["baselines"][_case("resize_basic")][
        "TESTSKU"
    ]
    assert metrics["gpu_time_us_cpp"] == pytest.approx(100.2)
    assert metrics["n_runs"] == 5
    assert "baseline_refresh_incomplete" not in payload


def test_stable_only_imports_only_the_selected_cluster(tmp_path):
    config_dir = tmp_path / "config"
    _write_config_tree(config_dir, include_resize_baseline=True)
    wave = tmp_path / "wave"
    _write_stability_wave(
        wave,
        [100, 100.1, 100.2, 100.3, 100.4, 120, 121, 122, 123, 124],
    )

    assert (
        _run(
            config_dir,
            "--from",
            str(wave),
            "--operator",
            "resize",
            "--stable-only",
            "--clustered",
            "--allow-regressions",
            "--write-summary",
            str(tmp_path / "summary.md"),
        )
        == 0
    )

    payload = _load_operator(config_dir, "resize")
    metrics = payload["configs"]["resize_basic"]["baselines"][_case("resize_basic")][
        "TESTSKU"
    ]
    assert metrics["gpu_time_us_cpp"] == pytest.approx(100.2)
    assert metrics["gpu_time_us_python"] == pytest.approx(110.2)
    assert metrics["n_runs"] == 5


def test_stable_only_cli_rejects_incomplete_wave(tmp_path, capsys):
    config_dir = tmp_path / "config"
    _write_config_tree(config_dir, include_resize_baseline=True)
    config_path = config_dir / "operators" / "resize.json"
    before = config_path.read_bytes()
    summary = tmp_path / "summary.md"
    wave = tmp_path / "wave"
    _write_stability_wave(wave, [100, 100.1, 100.2, 100.3])

    with pytest.raises(SystemExit) as exc:
        _run(
            config_dir,
            "--from",
            str(wave),
            "--operator",
            "resize",
            "--stable-only",
            "--write-summary",
            str(summary),
        )

    assert exc.value.code == 1
    error = capsys.readouterr().err
    assert "refusing stable-only import" in error
    assert "expected 5 artifacts exactly, found 4" in error
    assert config_path.read_bytes() == before
    assert not summary.exists()


def test_stable_only_withholds_one_incomplete_row_and_imports_complete_row(
    tmp_path, capsys
):
    config_dir = tmp_path / "config"
    _write_config_tree(
        config_dir,
        include_resize_baseline=True,
        include_second_resize=True,
    )
    wave = tmp_path / "wave"
    paths = _write_stability_wave(
        wave,
        [100, 100.1, 100.2, 100.3, 100.4],
        include_second_resize=True,
    )
    first_payload = json.loads(paths[0].read_text())
    first_payload["baselines"].pop(_case("resize_basic"))
    _write_json(paths[0], first_payload)
    summary = tmp_path / "summary.md"

    assert (
        _run(
            config_dir,
            "--from",
            str(wave),
            "--operator",
            "resize",
            "--stable-only",
            "--write-summary",
            str(summary),
        )
        == 0
    )

    configs = _load_operator(config_dir, "resize")["configs"]
    first = configs["resize_basic"]["baselines"][_case("resize_basic")]["TESTSKU"]
    second = configs["resize_secondary"]["baselines"][
        _case("resize_secondary", shape="2x3x4")
    ]["TESTSKU"]
    assert first["gpu_time_us_cpp"] == 100.0
    assert first["n_runs"] == 1
    assert second["gpu_time_us_cpp"] == pytest.approx(190.3)
    assert second["n_runs"] == 5
    assert _load_operator(config_dir, "resize")["baseline_refresh_incomplete"] is True
    output = capsys.readouterr().out
    assert "--stable-only: withholding 1 of 2 entries" in output
    assert "case/SKU entries without 5 raw runs: 1" in output
    summary_text = summary.read_text()
    assert "## Coverage issues" in summary_text
    assert "case/SKU entries without 5 raw runs: 1" in summary_text
    assert "## Withheld entries" in summary_text
    assert "unstable or incomplete wave" in summary_text
    assert "resize_basic" in summary_text
    assert "temporary investigation snapshot" not in summary_text


def test_stable_only_reports_declared_rows_absent_from_wave(tmp_path, capsys):
    config_dir = tmp_path / "config"
    _write_config_tree(
        config_dir,
        include_resize_baseline=True,
        include_second_resize=True,
    )
    wave = tmp_path / "wave"
    _write_stability_wave(wave, [100, 100.1, 100.2, 100.3, 100.4])
    summary = tmp_path / "summary.md"

    assert (
        _run(
            config_dir,
            "--from",
            str(wave),
            "--operator",
            "resize",
            "--stable-only",
            "--write-summary",
            str(summary),
        )
        == 0
    )

    issue = "missing declared case/SKU entries: 1"
    assert f"--stable-only coverage: {issue}" in capsys.readouterr().out
    assert issue in summary.read_text()
    assert _load_operator(config_dir, "resize")["baseline_refresh_incomplete"] is True


def test_stable_only_scoped_wave_requires_operator(tmp_path, capsys):
    config_dir = tmp_path / "config"
    _write_config_tree(config_dir, include_resize_baseline=True)
    gaussian_path = config_dir / "operators" / "gaussian.json"
    gaussian = _load_operator(config_dir, "gaussian")
    gaussian["configs"]["gaussian_basic"].pop("baselines")
    _write_json(gaussian_path, gaussian)
    gaussian_before = _load_operator(config_dir, "gaussian")["configs"]
    wave = tmp_path / "wave"
    _write_stability_wave(wave, [100, 100.1, 100.2, 100.3, 100.4])

    with pytest.raises(SystemExit) as exc:
        _run(
            config_dir,
            "--from",
            str(wave),
            "--stable-only",
            "--dry-run",
        )

    assert exc.value.code == 1
    error = capsys.readouterr().err
    assert "do not cover every configured operator" in error
    assert "Pass --operator for a scoped wave" in error
    gaussian = _load_operator(config_dir, "gaussian")
    assert "baseline_refresh_incomplete" not in gaussian
    assert gaussian["configs"] == gaussian_before


def test_stable_only_unscoped_wave_skips_preexisting_gap_in_partial_operator(tmp_path):
    config_dir = tmp_path / "config"
    _write_config_tree(
        config_dir,
        include_resize_baseline=True,
        include_second_resize=True,
    )
    resize_path = config_dir / "operators" / "resize.json"
    resize = _load_operator(config_dir, "resize")
    resize["configs"]["resize_secondary"].pop("baselines")
    _write_json(resize_path, resize)
    gaussian_path = config_dir / "operators" / "gaussian.json"
    gaussian = _load_operator(config_dir, "gaussian")
    gaussian["baseline_refresh_incomplete"] = True
    _write_json(gaussian_path, gaussian)
    wave = tmp_path / "wave"
    _write_stability_wave(
        wave,
        [100, 100.1, 100.2, 100.3, 110],
        include_gaussian=True,
    )

    assert (
        _run(
            config_dir,
            "--from",
            str(wave),
            "--stable-only",
            "--write-summary",
            str(tmp_path / "summary.md"),
        )
        == 0
    )

    resize = _load_operator(config_dir, "resize")
    gaussian = _load_operator(config_dir, "gaussian")
    assert resize["baseline_refresh_incomplete"] is True
    assert "baselines" not in resize["configs"]["resize_secondary"]
    assert "baseline_refresh_incomplete" not in gaussian


def test_stable_only_withholds_regression_and_records_it_in_summary(tmp_path, capsys):
    config_dir = tmp_path / "config"
    _write_config_tree(
        config_dir,
        include_resize_baseline=True,
        include_second_resize=True,
    )
    wave = tmp_path / "wave"
    _write_stability_wave(
        wave,
        [105, 105.1, 105.2, 105.3, 105.4],
        include_second_resize=True,
    )
    summary = tmp_path / "summary.md"

    assert (
        _run(
            config_dir,
            "--from",
            str(wave),
            "--operator",
            "resize",
            "--stable-only",
            "--withhold-regressions",
            "--dry-run",
        )
        == 0
    )
    dry_run_output = capsys.readouterr().out
    assert "candidate regression > 1.00%" in dry_run_output
    assert "temporary investigation snapshot" in dry_run_output
    assert "resolve every slowdown" in dry_run_output

    assert (
        _run(
            config_dir,
            "--from",
            str(wave),
            "--operator",
            "resize",
            "--stable-only",
            "--withhold-regressions",
            "--write-summary",
            str(summary),
        )
        == 0
    )

    configs = _load_operator(config_dir, "resize")["configs"]
    first = configs["resize_basic"]["baselines"][_case("resize_basic")]["TESTSKU"]
    second = configs["resize_secondary"]["baselines"][
        _case("resize_secondary", shape="2x3x4")
    ]["TESTSKU"]
    assert first["gpu_time_us_cpp"] == 100.0
    assert second["gpu_time_us_cpp"] == pytest.approx(190.3)
    assert _load_operator(config_dir, "resize")["baseline_refresh_incomplete"] is True
    text = summary.read_text()
    assert f"- measured revision: `{'a' * 40}`" in text
    assert "## Withheld entries" in text
    assert "candidate regression > 1.00%" in text
    assert "resize_basic" in text
    assert "temporary investigation snapshot" in text
    assert "resolve every slowdown" in text
    output = capsys.readouterr().out
    assert "temporary investigation snapshot" in output
    assert "resolve every slowdown" in output


def test_stable_only_rejects_regression_without_withholding(tmp_path, capsys):
    config_dir = tmp_path / "config"
    _write_config_tree(config_dir, include_resize_baseline=True)
    config_path = config_dir / "operators" / "resize.json"
    before = config_path.read_bytes()
    wave = tmp_path / "wave"
    _write_stability_wave(wave, [105, 105.1, 105.2, 105.3, 105.4])

    with pytest.raises(SystemExit) as exc:
        _run(
            config_dir,
            "--from",
            str(wave),
            "--operator",
            "resize",
            "--stable-only",
            "--write-summary",
            str(tmp_path / "summary.md"),
        )

    assert exc.value.code == 1
    assert "same-key baseline regressions > 1.00%" in capsys.readouterr().err
    assert config_path.read_bytes() == before


def test_stable_only_allow_regressions_imports_reviewed_reset(tmp_path):
    config_dir = tmp_path / "config"
    _write_config_tree(config_dir, include_resize_baseline=True)
    wave = tmp_path / "wave"
    _write_stability_wave(wave, [105, 105.1, 105.2, 105.3, 105.4])
    summary = tmp_path / "summary.md"

    assert (
        _run(
            config_dir,
            "--from",
            str(wave),
            "--operator",
            "resize",
            "--stable-only",
            "--allow-regressions",
            "--write-summary",
            str(summary),
        )
        == 0
    )

    metrics = _load_operator(config_dir, "resize")["configs"]["resize_basic"][
        "baselines"
    ][_case("resize_basic")]["TESTSKU"]
    assert metrics["gpu_time_us_cpp"] == pytest.approx(105.2)
    text = summary.read_text()
    assert f"- measured revision: `{'a' * 40}`" in text
    assert "+5.20%" in text
    assert "`resize_basic`" in text


def test_complete_stable_refresh_clears_partial_marker(tmp_path):
    config_dir = tmp_path / "config"
    _write_config_tree(config_dir, include_resize_baseline=True)
    config_path = config_dir / "operators" / "resize.json"
    payload = json.loads(config_path.read_text())
    payload["baseline_refresh_incomplete"] = True
    _write_json(config_path, payload)
    wave = tmp_path / "wave"
    _write_stability_wave(wave, [100, 100.1, 100.2, 100.3, 100.4])

    assert (
        _run(
            config_dir,
            "--from",
            str(wave),
            "--operator",
            "resize",
            "--stable-only",
            "--write-summary",
            str(tmp_path / "summary.md"),
        )
        == 0
    )

    assert "baseline_refresh_incomplete" not in _load_operator(config_dir, "resize")


def test_plain_import_preserves_partial_marker(tmp_path):
    config_dir = tmp_path / "config"
    _write_config_tree(config_dir, include_resize_baseline=True)
    config_path = config_dir / "operators" / "resize.json"
    payload = json.loads(config_path.read_text())
    payload["baseline_refresh_incomplete"] = True
    _write_json(config_path, payload)
    artifact = tmp_path / "run.json"
    _write_run_json(artifact, _baseline_payload(cpp_us=95, python_us=100))

    assert _run(config_dir, "--from", str(artifact), "--allow-regressions") == 0

    refreshed = _load_operator(config_dir, "resize")
    assert refreshed["baseline_refresh_incomplete"] is True
    metrics = refreshed["configs"]["resize_basic"]["baselines"][_case("resize_basic")][
        "TESTSKU"
    ]
    assert metrics["gpu_time_us_cpp"] == 95


def test_stable_only_requires_expected_revision(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        update_baseline.main(
            [
                "--config-dir",
                str(tmp_path / "config"),
                "--from",
                str(tmp_path / "missing.json"),
                "--stable-only",
                "--dry-run",
            ]
        )

    assert exc.value.code == 2
    assert "--stable-only requires --expect-revision" in capsys.readouterr().err


def test_stable_only_rejects_unexpected_revision(tmp_path, capsys):
    config_dir = tmp_path / "config"
    _write_config_tree(config_dir, include_resize_baseline=True)
    wave = tmp_path / "wave"
    _write_stability_wave(wave, [100, 100.1, 100.2, 100.3, 100.4])

    with pytest.raises(SystemExit) as exc:
        _run(
            config_dir,
            "--from",
            str(wave),
            "--operator",
            "resize",
            "--stable-only",
            "--expect-revision",
            "b" * 40,
            "--dry-run",
        )

    assert exc.value.code == 1
    assert f"does not match --expect-revision {'b' * 40}" in capsys.readouterr().err


def test_stable_only_rejects_full_revision_with_only_matching_prefix(tmp_path, capsys):
    config_dir = tmp_path / "config"
    _write_config_tree(config_dir, include_resize_baseline=True)
    wave = tmp_path / "wave"
    paths = _write_stability_wave(wave, [100, 100.1, 100.2, 100.3, 100.4])
    for path in paths:
        payload = json.loads(path.read_text())
        payload["run_metadata"]["git_sha"] = "a" * 39 + "b"
        _write_json(path, payload)

    with pytest.raises(SystemExit) as exc:
        _run(
            config_dir,
            "--from",
            str(wave),
            "--operator",
            "resize",
            "--stable-only",
            "--dry-run",
        )

    assert exc.value.code == 1
    assert f"does not match --expect-revision {'a' * 40}" in capsys.readouterr().err


def test_stable_only_reports_input_issues_before_revision_mismatch(tmp_path, capsys):
    config_dir = tmp_path / "config"
    _write_config_tree(config_dir, include_resize_baseline=True)
    wave = tmp_path / "wave"
    paths = _write_stability_wave(wave, [100, 100.1, 100.2, 100.3, 100.4])
    for path in paths:
        payload = json.loads(path.read_text())
        payload["run_metadata"]["git_sha"] = "not-a-revision"
        payload["run_metadata"].pop("power_cap_w")
        _write_json(path, payload)

    with pytest.raises(SystemExit) as exc:
        _run(
            config_dir,
            "--from",
            str(wave),
            "--operator",
            "resize",
            "--stable-only",
            "--dry-run",
        )

    assert exc.value.code == 1
    error = capsys.readouterr().err
    assert "refusing stable-only import due to invalid stability inputs" in error
    assert "missing metadata ['power_cap_w']" in error
    assert "metadata git_sha is not an 8-40 digit hex revision" in error


def test_withhold_regressions_requires_stable_only(tmp_path, capsys):
    config_dir = tmp_path / "config"
    _write_config_tree(config_dir)
    run_json = tmp_path / "run.json"
    _write_run_json(run_json, _baseline_payload())

    with pytest.raises(SystemExit) as exc:
        _run(config_dir, "--from", str(run_json), "--withhold-regressions")

    assert exc.value.code == 2
    assert "--withhold-regressions requires --stable-only" in capsys.readouterr().err


def test_negative_regression_threshold_is_rejected(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        _run(
            tmp_path / "config",
            "--from",
            str(tmp_path / "missing.json"),
            "--max-regression-pct",
            "-1",
        )

    assert exc.value.code == 1
    assert "--max-regression-pct must be >= 0" in capsys.readouterr().err


@pytest.mark.parametrize("revision", ["a" * 8, "a" * 39, "not-a-sha"])
def test_invalid_expected_revision_is_rejected(capsys, revision):
    with pytest.raises(SystemExit) as exc:
        update_baseline.main(["--from", "missing.json", "--expect-revision", revision])

    assert exc.value.code == 2
    assert "must be a 40 digit hex revision" in capsys.readouterr().err


def test_expected_revision_requires_stable_only(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        _run(
            tmp_path / "config",
            "--from",
            str(tmp_path / "missing.json"),
            "--expect-revision",
            "a" * 40,
        )

    assert exc.value.code == 2
    assert "--expect-revision requires --stable-only" in capsys.readouterr().err


def test_markdown_summary_caps_updated_rows_but_includes_all_withheld_rows(monkeypatch):
    row = {
        "benchmark": "resize",
        "config_key": "resize_basic",
        "case_key": "resize_basic[shape=1x2x3]",
        "language": "cpp",
        "axes": (("shape", "1x2x3"),),
        "sku": "TESTSKU",
        "prev_us": 100.0,
        "new_us": 101.0,
        "pct": 1.0,
        "n_runs": 5,
        "withheld_reason": "unstable",
        "withheld_kind": "unstable",
    }
    count = update_baseline._MAX_CONSOLE_ROWS + 1

    text = update_baseline._format_markdown_summary([row] * count, None, [row] * count)

    assert "_1 more updated row(s) omitted._" in text
    assert text.count("| unstable |") == count
    assert "more withheld row(s) omitted" not in text

    row["pct"] = None
    text = update_baseline._format_markdown_summary([row], None)
    assert "| TESTSKU | N/A | 100.00 |" in text

    text = update_baseline._format_markdown_summary([], None, [row])
    assert "| unstable | TESTSKU | N/A |" in text

    first = {**row, "config_key": "aaa", "case_key": "aaa[shape=1x2x3]"}
    monkeypatch.setattr(update_baseline, "_MAX_CONSOLE_ROWS", 1)
    text = update_baseline._format_markdown_summary([row, first], None)
    assert "`aaa`" in text
    assert "`resize_basic`" not in text


def test_console_summary_renders_nonpositive_prior_value(capsys):
    row = {
        "config_key": "resize_basic",
        "case_key": "resize_basic",
        "language": "cpp",
        "axes": (),
        "sku": "TESTSKU",
        "benchmark": "resize",
        "prev_us": 0.0,
        "new_us": 101.0,
        "pct": None,
    }

    update_baseline._print_summary([row], None)

    assert "N/A" in capsys.readouterr().out


def test_summary_write_failure_does_not_modify_baselines(tmp_path, capsys):
    config_dir = tmp_path / "config"
    _write_config_tree(config_dir, include_resize_baseline=True)
    config_path = config_dir / "operators" / "resize.json"
    before = config_path.read_bytes()
    wave = tmp_path / "wave"
    _write_stability_wave(wave, [100, 100.1, 100.2, 100.3, 100.4])

    with pytest.raises(SystemExit) as exc:
        _run(
            config_dir,
            "--from",
            str(wave),
            "--operator",
            "resize",
            "--stable-only",
            "--write-summary",
            str(tmp_path / "missing" / "summary.md"),
        )

    assert exc.value.code == 1
    assert "could not write markdown summary" in capsys.readouterr().err
    assert config_path.read_bytes() == before


def test_summary_directory_does_not_modify_baselines(tmp_path, capsys):
    config_dir = tmp_path / "config"
    _write_config_tree(config_dir, include_resize_baseline=True)
    config_path = config_dir / "operators" / "resize.json"
    before = config_path.read_bytes()
    wave = tmp_path / "wave"
    _write_stability_wave(wave, [100, 100.1, 100.2, 100.3, 100.4])
    summary = tmp_path / "summary"
    summary.mkdir()

    with pytest.raises(SystemExit) as exc:
        _run(
            config_dir,
            "--from",
            str(wave),
            "--operator",
            "resize",
            "--stable-only",
            "--write-summary",
            str(summary),
        )

    assert exc.value.code == 1
    assert "markdown summary path is not a file" in capsys.readouterr().err
    assert config_path.read_bytes() == before


def test_summary_is_published_only_after_baselines(tmp_path, monkeypatch):
    config_dir = tmp_path / "config"
    _write_config_tree(config_dir, include_resize_baseline=True)
    wave = tmp_path / "wave"
    _write_stability_wave(wave, [100, 100.1, 100.2, 100.3, 100.4])
    summary = tmp_path / "summary.md"

    def fail_write(*_args):
        raise OSError("baseline write failed")

    monkeypatch.setattr(update_baseline, "write_json", fail_write)
    with pytest.raises(OSError, match="baseline write failed"):
        _run(
            config_dir,
            "--from",
            str(wave),
            "--operator",
            "resize",
            "--stable-only",
            "--write-summary",
            str(summary),
        )

    assert not summary.exists()
    assert not list(tmp_path.glob(".summary.md.*.tmp"))


def test_withhold_and_allow_regressions_are_mutually_exclusive(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        _run(
            tmp_path / "config",
            "--from",
            str(tmp_path / "missing.json"),
            "--stable-only",
            "--withhold-regressions",
            "--allow-regressions",
        )

    assert exc.value.code == 2
    assert "not allowed with argument --withhold-regressions" in capsys.readouterr().err


def test_nondry_withholding_requires_summary(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        _run(
            tmp_path / "config",
            "--from",
            str(tmp_path / "missing.json"),
            "--stable-only",
            "--withhold-regressions",
        )

    assert exc.value.code == 1
    assert "requires --write-summary" in capsys.readouterr().err


def test_nondry_stable_only_requires_summary(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        _run(
            tmp_path / "config",
            "--from",
            str(tmp_path / "missing.json"),
            "--stable-only",
        )

    assert exc.value.code == 1
    assert "baseline withholding requires --write-summary" in capsys.readouterr().err


def test_regression_withholding_keeps_the_whole_language_pair(tmp_path):
    config_dir = tmp_path / "config"
    _write_config_tree(config_dir, include_resize_baseline=True)
    wave = tmp_path / "wave"
    _write_stability_wave(
        wave,
        [105, 105.1, 105.2, 105.3, 105.4],
        python_values=[100, 100.1, 100.2, 100.3, 100.4],
    )
    summary = tmp_path / "summary.md"

    assert (
        _run(
            config_dir,
            "--from",
            str(wave),
            "--operator",
            "resize",
            "--stable-only",
            "--withhold-regressions",
            "--max-regression-pct",
            "1",
            "--write-summary",
            str(summary),
        )
        == 1
    )

    metrics = _load_operator(config_dir, "resize")["configs"]["resize_basic"][
        "baselines"
    ][_case("resize_basic")]["TESTSKU"]
    assert metrics["gpu_time_us_cpp"] == 100.0
    assert metrics["gpu_time_us_python"] == 110.0
    assert summary.read_text().count("`resize_basic`") == 2


def test_all_withheld_regressions_emit_warning_and_summary(tmp_path, capsys):
    config_dir = tmp_path / "config"
    _write_config_tree(config_dir, include_resize_baseline=True)
    wave = tmp_path / "wave"
    _write_stability_wave(wave, [105, 105.1, 105.2, 105.3, 105.4])
    summary = tmp_path / "summary.md"

    assert (
        _run(
            config_dir,
            "--from",
            str(wave),
            "--operator",
            "resize",
            "--stable-only",
            "--withhold-regressions",
            "--max-regression-pct",
            "1",
            "--write-summary",
            str(summary),
        )
        == 1
    )

    output = capsys.readouterr().out
    assert "all matching rows were withheld" in output
    assert "temporary investigation snapshot" in output
    assert "candidate regression > 1.00%" in summary.read_text()
    resize = _load_operator(config_dir, "resize")
    assert resize["baseline_refresh_incomplete"] is True
    metrics = resize["configs"]["resize_basic"]["baselines"][_case("resize_basic")][
        "TESTSKU"
    ]
    assert metrics["gpu_time_us_cpp"] == 100.0
    assert metrics["gpu_time_us_python"] == 110.0


def test_empty_artifact_is_not_a_successful_update(tmp_path, capsys):
    config_dir = tmp_path / "config"
    _write_config_tree(config_dir)
    artifact = tmp_path / "empty.json"
    _write_json(artifact, {})

    assert _run(config_dir, "--from", str(artifact)) == 1
    assert "no input row matched the selected operators" in capsys.readouterr().out


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
