# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for the two-wheel Python benchmark tool."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import zipfile
from argparse import Namespace
from pathlib import Path

import pandas as pd
import pytest

import compare_wheels
from _internal.baselines import load_config_index
from _internal.quality import DEFAULT_BENCHMARK_QUALITY


def _write_wheel(path: Path, *, name="cvcuda-cu12", version="1.2.3") -> Path:
    dist_info = name.replace("-", "_") + f"-{version}.dist-info"
    with zipfile.ZipFile(path, "w") as wheel:
        wheel.writestr(
            f"{dist_info}/METADATA",
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
        )
    return path


def _operator_document(*, dtypes=None):
    return {
        "benchmark": "resize",
        "configs": {
            "resize_basic": {
                "tier": "basic",
                "dtypes": dtypes or ["uint8"],
                "string_axes": {"shape": ["1x2x3"], "inputKind": ["Tensor"]},
            },
            "resize_advanced": {
                "tier": "advanced",
                "dtypes": ["float32"],
                "string_axes": {"shape": ["1x4x5"], "inputKind": ["Tensor"]},
            },
        },
    }


def _write_config(root: Path, *, dtypes=None):
    operators = root / "config" / "operators"
    operators.mkdir(parents=True)
    (operators / "resize.json").write_text(
        json.dumps(_operator_document(dtypes=dtypes))
    )
    return load_config_index(operators)


def _row(
    *,
    config_key="resize_basic",
    dtype="uint8",
    shape="1x2x3",
    time=100.0,
    noise=1.0,
    status="PASS",
    device="NVIDIA Test GPU",
    power=350,
    clock=1095,
):
    return {
        "Benchmark": "resize",
        "config_key": config_key,
        "Language": "python",
        "InOutDataType": dtype,
        "shape": shape,
        "inputKind": "Tensor",
        "GPU Time (µs)": time,
        "GPU Noise (%)": noise,
        "Status": status,
        "Device Name": device,
        "Power Cap (W)": power,
        "Locked SM Clock (MHz)": clock,
        "vBIOS Version": "1.0",
    }


def _write_run(path: Path, rows) -> Path:
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _args(**overrides):
    values = {
        "operator": None,
        "config_key": None,
        "tier": "basic",
        "max_noise_pct": DEFAULT_BENCHMARK_QUALITY.max_noise_pct,
        "warmup_cap": None,
        "bench_min_time": None,
        "bench_max_noise": None,
    }
    values.update(overrides)
    return Namespace(**values)


def _expected(index, *config_keys):
    return compare_wheels.expected_case_operators(index, set(config_keys))


def _stage_source(root: Path, *, flat: bool) -> Path:
    root.mkdir()
    for filename in compare_wheels.RUNNER_FILES:
        (root / filename).write_text(f"# {filename}\n")
    internal = root / "_internal"
    internal.mkdir()
    for filename in compare_wheels.INTERNAL_FILES:
        (internal / filename).write_text(f"# {filename}\n")
    config = root / "config"
    (config / "operators").mkdir(parents=True)
    (config / "bench_params.json").write_text(
        json.dumps(
            {
                "operators": {
                    "resize": {
                        "config": "operators/resize.json",
                        "python": "bench_resize.py",
                    }
                }
            }
        )
    )
    (config / "operators" / "resize.json").write_text(json.dumps(_operator_document()))
    for filename in ("sku_map.json", "load_config.py", "axis_order.py"):
        (config / filename).write_text(
            "{}\n" if filename.endswith(".json") else "# helper\n"
        )
    python_root = root if flat else root / "python"
    ops_root = root if flat else root / "python" / "ops"
    ops_root.mkdir(parents=True, exist_ok=True)
    (python_root / "python_bench_utils.py").write_text("# utils\n")
    (ops_root / "bench_resize.py").write_text("# resize\n")
    return root


def test_read_wheel_metadata_and_reject_non_cvcuda(tmp_path):
    metadata = compare_wheels.read_wheel_metadata(_write_wheel(tmp_path / "cvcuda.whl"))
    assert (metadata.name, metadata.version, len(metadata.sha256)) == (
        "cvcuda-cu12",
        "1.2.3",
        64,
    )

    with pytest.raises(compare_wheels.WheelComparisonError, match="not a CV-CUDA"):
        compare_wheels.read_wheel_metadata(
            _write_wheel(tmp_path / "other.whl", name="other")
        )


@pytest.mark.parametrize("flat", [False, True])
def test_stage_benchmark_harness_supports_source_and_installed_layout(tmp_path, flat):
    source = _stage_source(tmp_path / "source", flat=flat)
    destination = tmp_path / "work" / "harness-bin"

    compare_wheels.stage_benchmark_harness(source, destination)

    assert (destination / "run_bench.py").is_file()
    for filename in compare_wheels.INTERNAL_FILES:
        assert (destination / "_internal" / filename).is_file()
    assert (destination / "python_bench_utils.py").is_file()
    assert (destination / "bench_resize.py").is_file()
    assert (destination / "config" / "operators" / "resize.json").is_file()


def test_flattened_build_layout_can_start_clis(tmp_path):
    destination = tmp_path / "bin"
    compare_wheels.stage_benchmark_harness(compare_wheels.BENCH_DIR, destination)
    shutil.copy2(compare_wheels.__file__, destination / "compare_wheels.py")
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    for script, expected in (
        ("run_bench.py", "--lang {cpp,python,both}"),
        ("compare_wheels.py", "REFERENCE_WHEEL"),
    ):
        result = subprocess.run(
            [sys.executable, str(destination / script), "--help"],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        assert expected in result.stdout


def test_create_wheel_environment_uses_exact_wheel_in_system_site_venv(
    tmp_path, monkeypatch
):
    wheel = compare_wheels.read_wheel_metadata(_write_wheel(tmp_path / "cvcuda.whl"))
    commands = []

    def fake_run(command, _log_path, **_kwargs):
        commands.append(command)
        return 0

    monkeypatch.setattr(compare_wheels, "_run_logged", fake_run)
    python = compare_wheels.create_wheel_environment(
        Path("/usr/bin/python3"), tmp_path / "env", wheel, tmp_path / "install.log"
    )

    assert python == tmp_path / "env" / "bin" / "python"
    assert commands[0][2:] == ["venv", "--system-site-packages", str(tmp_path / "env")]
    assert {"--force-reinstall", "--no-deps"} <= set(commands[1])
    assert commands[1][-1] == str(wheel.path)


def test_build_runner_command_reuses_runner_defaults(tmp_path):
    args = _args(config_key="resize_basic", warmup_cap=3, bench_min_time=0.01)
    command = compare_wheels.build_runner_command(
        args, Path("/env/python"), tmp_path / "harness", tmp_path / "out.csv"
    )

    assert command[:2] == ["/env/python", str(tmp_path / "harness" / "run_bench.py")]
    assert command[-1] == str(tmp_path / "harness")
    assert command[command.index("--config-key") + 1] == "resize_basic"
    assert "--max-retries" not in command
    assert "--skip-validation" not in command


def test_compare_flags_regression_but_not_improvement(tmp_path):
    index = _write_config(tmp_path)
    expected = _expected(index, "resize_basic")
    baseline = compare_wheels.load_measurements(
        _write_run(tmp_path / "baseline.csv", [_row(time=100)]), index
    )

    regression = compare_wheels.compare_measurements(
        baseline,
        compare_wheels.load_measurements(
            _write_run(tmp_path / "slow.csv", [_row(time=120)]), index
        ),
        expected_case_operators=expected,
        threshold=0.10,
    )
    assert regression.regressions.iloc[0]["candidate_delta_pct"] == pytest.approx(20)
    assert regression.failed

    improvement = compare_wheels.compare_measurements(
        baseline,
        compare_wheels.load_measurements(
            _write_run(tmp_path / "fast.csv", [_row(time=80)]), index
        ),
        expected_case_operators=expected,
        threshold=0.10,
    )
    assert len(improvement.improvements) == 1
    assert improvement.mean_operator_speedup == pytest.approx(1.25)
    assert not improvement.failed


def test_speedup_is_mean_over_operators_of_per_operator_geomeans():
    table = pd.DataFrame(
        [
            {
                "Benchmark": "many_cases",
                "candidate_speedup": speedup,
                "comparison_status": "neutral",
            }
            for speedup in (16.0, 16.0, 16.0, 16.0)
        ]
        + [
            {
                "Benchmark": "one_case",
                "candidate_speedup": 1.0,
                "comparison_status": "neutral",
            }
        ]
    )
    comparison = compare_wheels.Comparison(
        table=table,
        expected_case_operators={},
        missing_baseline_cases=(),
        missing_candidate_cases=(),
        baseline_fingerprint=None,
        candidate_fingerprint=None,
        baseline_clocks=(),
        candidate_clocks=(),
        baseline_row_count=5,
        candidate_row_count=5,
        threshold=0.10,
    )

    assert comparison.operator_geomean_speedups.to_dict() == {
        "many_cases": pytest.approx(16.0),
        "one_case": pytest.approx(1.0),
    }
    assert comparison.mean_operator_speedup == pytest.approx(8.5)


def test_compatibility_distinguishes_full_partial_and_zero_overlap():
    table = pd.DataFrame(
        [
            ("full", "full-1", "neutral", 1.0),
            ("full", "full-2", "improvement", 2.0),
            ("partial", "partial-1", "neutral", 1.0),
            ("partial", "partial-2", "only_in_candidate", None),
            ("absent", "absent-1", "only_in_candidate", None),
            ("failed", "failed-1", "excluded", 1.0),
        ],
        columns=(
            "Benchmark",
            "case_key",
            "comparison_status",
            "candidate_speedup",
        ),
    )
    comparison = compare_wheels.Comparison(
        table=table,
        expected_case_operators={
            row.case_key: row.Benchmark for row in table.itertuples(index=False)
        },
        missing_baseline_cases=("absent-1", "partial-2"),
        missing_candidate_cases=(),
        baseline_fingerprint=None,
        candidate_fingerprint=None,
        baseline_clocks=(),
        candidate_clocks=(),
        baseline_row_count=4,
        candidate_row_count=6,
        threshold=0.10,
    )

    assert comparison.operator_compatibility == {
        "absent": "incompatible_or_not_present",
        "failed": "fully_compatible",
        "full": "fully_compatible",
        "partial": "partially_compatible",
    }
    items = dict(compare_wheels._compatibility_items(comparison))
    assert items["operators with any incompatible or missing configurations"] == "2/4"
    assert items["operators present in both"] == "3/4"
    assert items["operators with no shared configurations"] == "1/4"
    assert items["operators partially compatible"] == "1/4"
    assert items["operators fully compatible"] == "2/4"
    assert (
        items["expanded benchmark configurations incompatible or not present"] == "2/6"
    )
    assert items["expanded benchmark configurations present in both"] == "4/6"
    assert items["present in both but non-PASS"] == "1"


def test_clock_drift_is_diagnostic_but_gpu_mismatch_is_rejected(tmp_path):
    index = _write_config(tmp_path)
    expected = _expected(index, "resize_basic")
    baseline = compare_wheels.load_measurements(
        _write_run(tmp_path / "baseline.csv", [_row(clock=1095)]), index
    )
    drifted = compare_wheels.load_measurements(
        _write_run(tmp_path / "drifted.csv", [_row(clock=1110)]), index
    )
    comparison = compare_wheels.compare_measurements(
        baseline, drifted, expected_case_operators=expected, threshold=0.10
    )
    assert comparison.baseline_clocks != comparison.candidate_clocks
    assert not comparison.failed

    different_gpu = compare_wheels.load_measurements(
        _write_run(tmp_path / "other.csv", [_row(device="GPU B")]), index
    )
    with pytest.raises(
        compare_wheels.WheelComparisonError, match="fingerprints differ"
    ):
        compare_wheels.compare_measurements(
            baseline,
            different_gpu,
            expected_case_operators=expected,
            threshold=0.10,
        )


def test_comparison_reports_nonpass_asymmetric_and_both_missing_cases(tmp_path):
    index = _write_config(tmp_path, dtypes=["uint8", "int16"])
    expected = _expected(index, "resize_basic")
    baseline = compare_wheels.load_measurements(
        _write_run(tmp_path / "baseline.csv", [_row(status="FAIL (noise)")]), index
    )
    candidate = compare_wheels.load_measurements(
        _write_run(tmp_path / "candidate.csv", [_row(status="FAIL (noise)")]), index
    )

    comparison = compare_wheels.compare_measurements(
        baseline, candidate, expected_case_operators=expected, threshold=0.10
    )

    assert len(comparison.rows("excluded")) == 1
    assert len(comparison.rows("missing_from_both")) == 1
    assert len(comparison.missing_baseline_cases) == 1
    assert len(comparison.missing_candidate_cases) == 1
    assert comparison.failed


def test_load_measurements_rejects_duplicate_cases(tmp_path):
    index = _write_config(tmp_path)
    path = _write_run(tmp_path / "run.csv", [_row(), _row(time=101)])
    with pytest.raises(compare_wheels.WheelComparisonError, match="duplicate"):
        compare_wheels.load_measurements(path, index)


def test_selectors_reuse_shared_parsers(tmp_path):
    index = _write_config(tmp_path)
    assert compare_wheels.selected_config_keys(index, _args()) == {"resize_basic"}
    assert compare_wheels.selected_config_keys(
        index, _args(tier="basic,advanced", operator="resize")
    ) == {"resize_basic", "resize_advanced"}
    with pytest.raises(ValueError, match="duplicate"):
        compare_wheels.selected_config_keys(
            index, _args(config_key="resize_basic,resize_basic")
        )


def test_report_and_comparison_csv_include_all_failure_kinds(tmp_path):
    index = _write_config(tmp_path, dtypes=["uint8", "int16"])
    baseline = compare_wheels.load_measurements(
        _write_run(tmp_path / "baseline.csv", [_row(time=100)]), index
    )
    candidate = compare_wheels.load_measurements(
        _write_run(tmp_path / "candidate.csv", [_row(time=120)]), index
    )
    comparison = compare_wheels.compare_measurements(
        baseline,
        candidate,
        expected_case_operators=_expected(index, "resize_basic"),
        threshold=0.10,
    )
    baseline_wheel = compare_wheels.read_wheel_metadata(
        _write_wheel(tmp_path / "baseline.whl", version="1.0")
    )
    candidate_wheel = compare_wheels.read_wheel_metadata(
        _write_wheel(tmp_path / "candidate.whl", version="2.0")
    )

    output_csv = tmp_path / "comparison.csv"
    comparison.table.to_csv(output_csv, index=False)
    report = compare_wheels.format_report(
        baseline_wheel,
        candidate_wheel,
        comparison,
        baseline_rc=0,
        candidate_rc=0,
        output_dir=tmp_path,
    )

    written = pd.read_csv(output_csv)
    assert set(written["comparison_status"]) == {"regression", "missing_from_both"}
    assert (
        "overall candidate speedup vs baseline: **0.8333x** "
        "(mean of per-operator geomeans across matched benchmarks)" in report
    )
    assert "valid compatible configurations: 1/2" in report
    assert "valid compatible operators: 1/1" in report
    assert report.index("valid compatible operators") < report.index(
        "valid compatible configurations"
    )
    assert "## Compatibility" in report
    assert "operators partially compatible: 1/1" in report
    assert (
        "expanded benchmark configurations incompatible or not present: 1/2" in report
    )
    assert "## Top 5 operator improvements\n\n_none_" in report
    assert "| resize | 0.8333x | +20.00% | 1 |" in report
    assert "reference `run_bench.py`: **PASS** (exit code 0)" in report
    assert "candidate `run_bench.py`: **PASS** (exit code 0)" in report
    artifact_lines = (
        f"- artifacts: `{tmp_path}`\n"
        "  - reference benchmark CSV: `baseline/bench_output.csv`\n"
        "  - candidate benchmark CSV: `candidate/bench_output.csv`\n"
        "  - comparison CSV: `comparison.csv`"
    )
    assert artifact_lines in report
    summary = report.split("## Summary\n\n", 1)[1]
    assert summary.startswith("- overall candidate speedup vs baseline:")
    assert report.index("## Compatibility") < report.index(
        "## Top 5 operator improvements"
    )
    assert "## Result" not in report
    assert "comparison status: **FAIL**" in report
    assert (
        "1 configuration regression over threshold; "
        "1 configuration missing from both" in report
    )


def _mock_run_dependencies(
    monkeypatch, *, baseline_row=True, baseline_rc=0, candidate_rc=0
):
    calls = []

    def fake_stage(_source, destination):
        _write_config(destination)

    def fake_run(_args, _python, _harness, output_csv, _log):
        side = output_csv.parent.name
        calls.append(side)
        if side == "candidate" or baseline_row:
            _write_run(output_csv, [_row(time=105 if side == "candidate" else 100)])
        return baseline_rc if side == "baseline" else candidate_rc

    monkeypatch.setattr(compare_wheels, "stage_benchmark_harness", fake_stage)
    monkeypatch.setattr(
        compare_wheels,
        "create_wheel_environment",
        lambda _base, env, _wheel, _log: env / "bin" / "python",
    )
    monkeypatch.setattr(compare_wheels, "probe_wheel_environment", lambda *_args: None)
    monkeypatch.setattr(compare_wheels, "run_benchmark", fake_run)
    return calls


def _run_args(tmp_path, output_dir):
    baseline = _write_wheel(tmp_path / "baseline.whl", version="1.0")
    candidate = _write_wheel(tmp_path / "candidate.whl", version="2.0")
    return compare_wheels.parse_args(
        [
            str(baseline),
            str(candidate),
            "--config-key",
            "resize_basic",
            "--output-dir",
            str(output_dir),
            "--python",
            sys.executable,
        ]
    )


def test_run_passes_writes_artifacts_and_cleans_workdir(tmp_path, monkeypatch):
    calls = _mock_run_dependencies(monkeypatch)
    output_dir = tmp_path / "result"

    assert compare_wheels.run(_run_args(tmp_path, output_dir)) == 0
    assert calls == ["baseline", "candidate"]
    assert (output_dir / "summary.md").is_file()
    assert (output_dir / "comparison.csv").is_file()
    assert (output_dir / "baseline" / "bench_output.csv").is_file()
    assert (output_dir / "candidate" / "bench_output.csv").is_file()
    assert not list(output_dir.glob("work-*"))


def test_run_reports_missing_csv_after_running_both_sides(tmp_path, monkeypatch):
    calls = _mock_run_dependencies(monkeypatch, baseline_row=False, baseline_rc=1)
    output_dir = tmp_path / "result"

    assert compare_wheels.run(_run_args(tmp_path, output_dir)) == 1
    assert calls == ["baseline", "candidate"]
    summary = (output_dir / "summary.md").read_text()
    assert "reference `run_bench.py`: **FAIL** (exit code 1)" in summary
    assert "candidate `run_bench.py`: **PASS** (exit code 0)" in summary
    assert "comparison status: **FAIL**" in summary
    comparison = pd.read_csv(output_dir / "comparison.csv")
    assert set(comparison["comparison_status"]) == {"only_in_candidate"}


def test_main_returns_two_for_output_path_file(tmp_path, capsys):
    output_path = tmp_path / "not-a-directory"
    output_path.write_text("occupied")
    baseline = _write_wheel(tmp_path / "baseline.whl")
    candidate = _write_wheel(tmp_path / "candidate.whl")

    assert (
        compare_wheels.main(
            [str(baseline), str(candidate), "--output-dir", str(output_path)]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert "output path is not a directory" in captured.err
    assert "Traceback" not in captured.err


def test_parse_args_uses_shared_noise_default(tmp_path):
    args = compare_wheels.parse_args(
        [str(tmp_path / "baseline.whl"), str(tmp_path / "candidate.whl")]
    )
    assert args.max_noise_pct == DEFAULT_BENCHMARK_QUALITY.max_noise_pct
