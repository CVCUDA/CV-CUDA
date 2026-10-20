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

"""Unit tests for benchmark tier configs and exact --config-key discovery."""

from __future__ import annotations

import json
import os
import sys
from itertools import product
from types import SimpleNamespace

import pandas as pd
import pytest

import run_bench
from _internal.baselines import (
    BaselineError,
    MissingClockConfigurationError,
    MissingClockProfileError,
    baseline_payload_from_dataframe,
    baseline_updates_from_payload,
    expected_case_keys_for_entry,
    fake_planar_pairing_issues,
    load_config_index,
    parse_case_key,
)
from _internal.quality import DEFAULT_BENCHMARK_QUALITY
from config.load_config import (
    generate_axis_args,
    get_configs_for_benchmark,
    load_bench_config,
    load_bench_manifest,
    parse_config_key_arg,
    parse_operator_arg,
    parse_tier_arg,
)


class DummyRunner(run_bench.BenchmarkRunner):
    def __init__(
        self,
        bench_folder,
        config,
        config_keys,
        language,
        operators=None,
        tiers=None,
    ):
        super().__init__(
            "bench_",
            str(bench_folder),
            operators=operators,
            language=language,
            config_keys=config_keys,
            tiers=tiers,
        )
        self.config = config
        self.operator_manifest = {
            entry["benchmark"]: {
                "config": f"operators/{entry['benchmark']}.json",
                "cpp": f"bench_{entry['benchmark']}",
                "python": f"bench_{entry['benchmark']}.py",
            }
            for entry in config.values()
            if isinstance(entry, dict) and "benchmark" in entry
        }

        from config.load_config import (  # noqa: PLC0415
            get_operator_from_benchmark_name,
            get_configs_for_benchmark,
        )

        self.get_operator_from_benchmark_name = get_operator_from_benchmark_name
        self.get_configs_for_benchmark = get_configs_for_benchmark

    def build_command(self, benchmark_path, extra_args, output_file, config_key=None):
        return []


def _clock_test_runner(tmp_path):
    runner = DummyRunner(tmp_path, {}, None, "cpp")
    runner.build_command = lambda *args, **kwargs: ["bench_test"]
    runner._check_noise_from_scratch = lambda _path: True
    return runner


def test_clock_protocol_markers_must_start_the_line():
    assert (
        run_bench._first_line_with(
            "note: ignoring SM clock mismatch: transient sample",
            run_bench._SM_CLOCK_MISMATCH_PREFIX,
        )
        is None
    )
    assert (
        run_bench._first_line_with(
            "  SM clock mismatch: expected 1005 MHz, measured 1020 MHz",
            run_bench._SM_CLOCK_MISMATCH_PREFIX,
        )
        == "SM clock mismatch: expected 1005 MHz, measured 1020 MHz"
    )


def test_clock_failure_relocks_and_retries_fresh_process(monkeypatch, tmp_path):
    runner = _clock_test_runner(tmp_path)
    output_path = tmp_path / "out_test_config_cpp.nvbench"
    results = iter(
        [
            SimpleNamespace(
                returncode=1,
                stdout="",
                stderr="SM clock mismatch: expected 1005 MHz, measured 1020 MHz",
            ),
            SimpleNamespace(
                returncode=0,
                stdout="SM clock verified: 1005 MHz",
                stderr="",
            ),
        ]
    )
    run_calls = []
    relock_calls = []
    monkeypatch.setattr(run_bench, "_clock_lock_required_mhz", 1005)
    monkeypatch.setattr(run_bench, "_clock_lock_device_verified", False)

    def fake_run(*args, **kwargs):
        if not run_calls:
            output_path.write_text("partial clock-mismatched output")
        else:
            assert not output_path.exists()
        run_calls.append(args)
        return next(results)

    monkeypatch.setattr(run_bench.subprocess, "run", fake_run)
    monkeypatch.setattr(
        run_bench,
        "_reapply_required_sm_clock",
        lambda: relock_calls.append(True) or run_bench._CLOCK_LOCK_OK,
    )

    output, passed = runner.run_benchmark("bench_test", [], "test_config")

    assert output == "out_test_config_cpp.nvbench"
    assert passed is True
    assert len(run_calls) == 2
    assert len(relock_calls) == 1
    assert run_bench._clock_lock_device_verified is True


def test_third_clock_failure_is_fatal(monkeypatch, tmp_path):
    runner = _clock_test_runner(tmp_path)
    run_calls = []
    relock_calls = []
    errors = []
    monkeypatch.setattr(run_bench, "_clock_lock_required_mhz", 1005)
    monkeypatch.setenv("K8S_NODE_NAME", "benchmark-node-7")
    monkeypatch.setattr(run_bench, "log_error", errors.append)
    monkeypatch.setattr(
        run_bench.subprocess,
        "run",
        lambda *args, **kwargs: run_calls.append(args)
        or SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="SM clock mismatch: expected 1005 MHz, measured 1020 MHz",
        ),
    )
    monkeypatch.setattr(
        run_bench,
        "_reapply_required_sm_clock",
        lambda: relock_calls.append(True) or run_bench._CLOCK_LOCK_OK,
    )

    with pytest.raises(SystemExit) as exc:
        runner.run_benchmark("bench_test", [], "test_config")

    assert exc.value.code == run_bench._CLOCK_INELIGIBLE_EXIT_CODE
    assert len(run_calls) == 3
    assert len(relock_calls) == 2
    assert errors[-1].endswith("1005 MHz on node benchmark-node-7")


@pytest.mark.parametrize("returncode", [-11, 3])
def test_preexisting_cuda_error_is_an_ordinary_benchmark_failure(
    monkeypatch, tmp_path, returncode
):
    runner = _clock_test_runner(tmp_path)
    run_calls = []
    relock_calls = []
    monkeypatch.setattr(run_bench, "_clock_lock_required_mhz", 1005)
    monkeypatch.setattr(
        run_bench.subprocess,
        "run",
        lambda *args, **kwargs: run_calls.append(args)
        or SimpleNamespace(
            returncode=returncode,
            stdout="",
            stderr="Pre-existing CUDA error: out of memory",
        ),
    )
    monkeypatch.setattr(
        run_bench,
        "_reapply_required_sm_clock",
        lambda: relock_calls.append(True) or run_bench._CLOCK_LOCK_OK,
    )

    assert runner.run_benchmark("bench_test", [], "test_config") == (None, False)
    assert len(run_calls) == 1
    assert not relock_calls


def test_success_without_clock_verification_ack_is_fatal(monkeypatch, tmp_path):
    runner = _clock_test_runner(tmp_path)
    output_path = tmp_path / "out_test_config_cpp.nvbench"
    output_path.write_text("Benchmark,Skipped\ntimed_state,No\n")
    errors = []
    monkeypatch.setattr(run_bench, "_clock_lock_required_mhz", 1005)
    monkeypatch.setattr(run_bench, "_clock_lock_device_verified", False)
    monkeypatch.setattr(run_bench, "log_error", errors.append)
    monkeypatch.setattr(
        run_bench.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    with pytest.raises(SystemExit) as exc:
        runner.run_benchmark("bench_test", [], "test_config")

    assert exc.value.code == 1
    assert not output_path.exists()
    assert run_bench._clock_lock_device_verified is False
    assert errors[-1].endswith("did not report device clock verification")


def test_strict_clock_protocol_exception_fails_only_the_benchmark(
    monkeypatch, tmp_path
):
    runner = _clock_test_runner(tmp_path)
    output_path = tmp_path / "out_test_config_cpp.nvbench"
    output_path.write_text("Benchmark,Skipped\ntimed_state,No\n")
    monkeypatch.setattr(run_bench, "_clock_lock_required_mhz", 1005)
    monkeypatch.setattr(
        run_bench.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    monkeypatch.setattr(
        run_bench.os,
        "remove",
        lambda _path: (_ for _ in ()).throw(PermissionError("read-only scratch")),
    )

    assert runner.run_benchmark("bench_test", [], "test_config") == (None, False)


def test_strict_clock_timeout_fails_only_the_benchmark(monkeypatch, tmp_path):
    runner = _clock_test_runner(tmp_path)
    monkeypatch.setattr(run_bench, "_clock_lock_required_mhz", 1005)
    monkeypatch.setattr(
        run_bench.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            run_bench.subprocess.TimeoutExpired(args[0], kwargs["timeout"])
        ),
    )

    assert runner.run_benchmark("bench_test", [], "test_config") == (None, False)


def test_all_skipped_states_do_not_turn_missing_ack_into_process_failure(
    monkeypatch, tmp_path
):
    runner = _clock_test_runner(tmp_path)
    output_path = tmp_path / "out_test_config_cpp.nvbench"
    output_path.write_text("Benchmark,Skipped\nskipped_state,Yes\n")
    runner._check_noise_from_scratch = lambda _path: False
    monkeypatch.setattr(run_bench, "_clock_lock_required_mhz", 1005)
    monkeypatch.setattr(run_bench, "_clock_lock_device_verified", False)
    monkeypatch.setattr(
        run_bench.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    assert runner.run_benchmark("bench_test", [], "test_config") == (
        "out_test_config_cpp.nvbench",
        False,
    )
    assert output_path.exists()
    assert run_bench._clock_lock_device_verified is False


def test_mixed_skipped_and_timed_states_fail_closed_without_ack(monkeypatch, tmp_path):
    runner = _clock_test_runner(tmp_path)
    output_path = tmp_path / "out_test_config_cpp.nvbench"
    output_path.write_text("Benchmark,Skipped\nskipped_state,Yes\ntimed_state,No\n")
    monkeypatch.setattr(run_bench, "_clock_lock_required_mhz", 1005)
    monkeypatch.setattr(
        run_bench.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    with pytest.raises(SystemExit) as exc:
        runner.run_benchmark("bench_test", [], "test_config")

    assert exc.value.code == 1
    assert not output_path.exists()


def test_all_skipped_scratch_is_read_once(monkeypatch, tmp_path):
    scratch_path = tmp_path / "scratch.nvbench"
    scratch_path.touch()
    reads = []

    def read_scratch(path):
        reads.append(path)
        if len(reads) > 1:
            raise pd.errors.ParserError("scratch changed between reads")
        return pd.DataFrame({"Skipped": ["Yes"]})

    monkeypatch.setattr(run_bench.pd, "read_csv", read_scratch)

    assert run_bench.BenchmarkRunner._scratch_reports_all_skipped(str(scratch_path))
    assert reads == [str(scratch_path)]


def test_non_numeric_noise_fails_only_the_benchmark(tmp_path):
    scratch_path = tmp_path / "scratch.nvbench"
    pd.DataFrame({"Skipped": ["No"], "Noise.1": ["invalid"]}).to_csv(
        scratch_path, index=False
    )
    runner = DummyRunner(tmp_path, {}, None, "cpp")

    assert runner._check_noise_from_scratch(str(scratch_path)) is False


def test_noise_check_fails_closed_without_skipped_column(tmp_path):
    scratch_path = tmp_path / "scratch.nvbench"
    pd.DataFrame({"Noise.1": [0.01]}).to_csv(scratch_path, index=False)
    runner = DummyRunner(tmp_path, {}, None, "cpp")

    assert runner._check_noise_from_scratch(str(scratch_path)) is False


def test_header_only_scratch_fails_closed_when_ack_is_missing(monkeypatch, tmp_path):
    runner = _clock_test_runner(tmp_path)
    output_path = tmp_path / "out_test_config_cpp.nvbench"
    output_path.write_text("Benchmark,Skipped\n")
    monkeypatch.setattr(run_bench, "_clock_lock_required_mhz", 1005)
    monkeypatch.setattr(
        run_bench.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    with pytest.raises(SystemExit) as exc:
        runner.run_benchmark("bench_test", [], "test_config")

    assert exc.value.code == 1
    assert not output_path.exists()


def test_unparseable_scratch_fails_closed_when_ack_is_missing(monkeypatch, tmp_path):
    runner = _clock_test_runner(tmp_path)
    output_path = tmp_path / "out_test_config_cpp.nvbench"
    output_path.write_text('Benchmark,Skipped\n"truncated')
    monkeypatch.setattr(run_bench, "_clock_lock_required_mhz", 1005)
    monkeypatch.setattr(
        run_bench.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    with pytest.raises(SystemExit) as exc:
        runner.run_benchmark("bench_test", [], "test_config")

    assert exc.value.code == 1
    assert not output_path.exists()


def test_missing_scratch_fails_closed_when_ack_is_missing(monkeypatch, tmp_path):
    runner = _clock_test_runner(tmp_path)
    monkeypatch.setattr(run_bench, "_clock_lock_required_mhz", 1005)
    monkeypatch.setattr(
        run_bench.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    with pytest.raises(SystemExit) as exc:
        runner.run_benchmark("bench_test", [], "test_config")

    assert exc.value.code == 1


def test_best_effort_success_does_not_require_clock_ack(monkeypatch, tmp_path):
    runner = _clock_test_runner(tmp_path)
    monkeypatch.setattr(run_bench, "_clock_lock_required_mhz", None)
    monkeypatch.setattr(
        run_bench.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    assert runner.run_benchmark("bench_test", [], "test_config") == (
        "out_test_config_cpp.nvbench",
        True,
    )


def test_best_effort_failure_does_not_retry_or_exclude_node(monkeypatch, tmp_path):
    runner = _clock_test_runner(tmp_path)
    run_calls = []
    relock_calls = []
    monkeypatch.setattr(run_bench, "_clock_lock_required_mhz", None)
    monkeypatch.setattr(
        run_bench.subprocess,
        "run",
        lambda *args, **kwargs: run_calls.append(args)
        or SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="SM clock mismatch: expected 1005 MHz, measured 1020 MHz",
        ),
    )
    monkeypatch.setattr(
        run_bench,
        "_reapply_required_sm_clock",
        lambda: relock_calls.append(True) or run_bench._CLOCK_LOCK_OK,
    )

    assert runner.run_benchmark("bench_test", [], "test_config") == (None, False)
    assert len(run_calls) == 1
    assert not relock_calls


@pytest.mark.parametrize("device_name", ["NVIDIA Other GPU", None, "missing"])
def test_collect_results_rejects_missing_or_mismatched_device(
    monkeypatch, tmp_path, device_name
):
    output = tmp_path / "out_test_config_cpp.nvbench"
    row = {
        "Benchmark": "test",
        "Skipped": "No",
        "BWUtil": 0.5,
        "GPU Time (sec)": 0.001,
    }
    if device_name != "missing":
        row["Device Name"] = device_name
    pd.DataFrame([row]).to_csv(output, index=False)
    runner = DummyRunner(
        tmp_path,
        {"test_config": {"benchmark": "test", "tier": "basic"}},
        None,
        "cpp",
    )
    runner._transform_scratch_columns = lambda frame: frame
    errors = []
    monkeypatch.setattr(run_bench, "_clock_policy_gpu_name", "NVIDIA H100 PCIe")
    monkeypatch.setattr(run_bench, "_clock_lock_required_mhz", 1005)
    monkeypatch.setattr(run_bench, "log_error", errors.append)

    assert runner.collect_results(output.name, "test_config") is False
    assert runner.results == []
    if device_name == "missing":
        assert errors == ["test_config is missing required columns: ['Device Name']"]
    elif device_name is None:
        assert errors[-1].endswith("did not report a device name")
    else:
        assert "NVIDIA Other GPU" in errors[-1]
        assert "NVIDIA H100 PCIe" in errors[-1]


def test_collect_results_does_not_gate_device_identity_for_best_effort_run(
    monkeypatch, tmp_path
):
    output = tmp_path / "out_test_config_cpp.nvbench"
    pd.DataFrame(
        [
            {
                "Benchmark": "test",
                "Skipped": "No",
                "BWUtil": 0.5,
                "GPU Time (sec)": 0.001,
                "Device Name": "NVIDIA Other GPU",
            }
        ]
    ).to_csv(output, index=False)
    runner = DummyRunner(
        tmp_path,
        {"test_config": {"benchmark": "test", "tier": "basic"}},
        None,
        "cpp",
    )
    runner._transform_scratch_columns = lambda frame: frame
    monkeypatch.setattr(run_bench, "_clock_policy_gpu_name", "NVIDIA H100 PCIe")
    monkeypatch.setattr(run_bench, "_clock_lock_required_mhz", None)

    assert runner.collect_results(output.name, "test_config") is True
    assert len(runner.results) == 1


@pytest.mark.parametrize(
    "axis_group,values,expected",
    [
        ("string_axes", ["5x5", "7x7"], ["5x5", "7x7"]),
        ("int64_axes", ["3", "5.0"], [3, 5]),
    ],
)
def test_results_preserve_declared_axis_types(tmp_path, axis_group, values, expected):
    output = tmp_path / "axes.nvbench"
    pd.DataFrame(
        {
            "Benchmark": "test",
            "Skipped": "No",
            "BWUtil": 0.5,
            "Device Name": "test GPU",
            "GPU Time (sec)": 0.001,
            "GPU Time (µs)": 1000,
            "GPU Noise (%)": 0.1,
            "kernelSize": values,
            "batch": [2.0, None],
        }
    ).to_csv(output, index=False)
    config = {
        "test_config": {
            "benchmark": "test",
            "tier": "basic",
            axis_group: {"kernelSize": expected},
        }
    }
    runner = DummyRunner(tmp_path, config, None, "cpp")
    runner._transform_scratch_columns = lambda frame: frame
    assert runner.collect_results(output.name, "test_config")

    processor = run_bench.ResultsProcessor(10.0, 10.0, 100.0)
    result = processor.process(runner.results, compute_parity=False)

    assert result["kernelSize"].tolist() == expected
    assert str(result["batch"].dtype) == "Int64"
    assert result["batch"].iloc[0] == 2
    assert pd.isna(result["batch"].iloc[1])
    if axis_group == "int64_axes":
        assert str(result["kernelSize"].dtype) == "Int64"


def test_best_effort_probe_error_is_an_ordinary_benchmark_failure(
    monkeypatch, tmp_path
):
    runner = _clock_test_runner(tmp_path)
    monkeypatch.setattr(run_bench, "_clock_lock_required_mhz", None)
    monkeypatch.setattr(
        run_bench.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="SM clock verification error: stale child setting",
        ),
    )

    assert runner.run_benchmark("bench_test", [], "test_config") == (None, False)


def test_invalid_device_probe_excludes_node(monkeypatch, tmp_path):
    runner = _clock_test_runner(tmp_path)
    monkeypatch.setattr(run_bench, "_clock_lock_required_mhz", 1005)
    monkeypatch.setattr(
        run_bench.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="SM clock device error: invalid multiprocessor count: 0",
        ),
    )

    with pytest.raises(SystemExit) as exc:
        runner.run_benchmark("bench_test", [], "test_config")

    assert exc.value.code == run_bench._CLOCK_INELIGIBLE_EXIT_CODE


def test_benchmark_env_scopes_required_clock_to_children(monkeypatch, tmp_path):
    runner = _clock_test_runner(tmp_path)
    monkeypatch.setenv("CVCUDA_BENCH_EXPECT_SM_CLOCK_MHZ", "stale")

    monkeypatch.setattr(run_bench, "_clock_lock_required_mhz", 1005)
    assert runner.benchmark_env()["CVCUDA_BENCH_EXPECT_SM_CLOCK_MHZ"] == "1005"

    monkeypatch.setattr(run_bench, "_clock_lock_required_mhz", None)
    assert "CVCUDA_BENCH_EXPECT_SM_CLOCK_MHZ" not in runner.benchmark_env()
    assert os.environ["CVCUDA_BENCH_EXPECT_SM_CLOCK_MHZ"] == "stale"


def test_clock_failure_stops_when_reapply_fails(monkeypatch, tmp_path):
    runner = _clock_test_runner(tmp_path)
    run_calls = []
    monkeypatch.setattr(run_bench, "_clock_lock_required_mhz", 1005)
    monkeypatch.setattr(
        run_bench.subprocess,
        "run",
        lambda *args, **kwargs: run_calls.append(args)
        or SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="SM clock mismatch: expected 1005 MHz, measured 1020 MHz",
        ),
    )
    monkeypatch.setattr(
        run_bench,
        "_reapply_required_sm_clock",
        lambda: run_bench._CLOCK_LOCK_RESET_FAILED,
    )

    with pytest.raises(SystemExit) as exc:
        runner.run_benchmark("bench_test", [], "test_config")

    assert exc.value.code == run_bench._CLOCK_INELIGIBLE_EXIT_CODE
    assert len(run_calls) == 1


@pytest.mark.parametrize(
    "reapply_status,expected_exit",
    [
        (
            run_bench._CLOCK_LOCK_QUERY_FAILED,
            1,
        ),
        (
            run_bench._CLOCK_LOCK_REQUEST_FAILED,
            1,
        ),
        (run_bench._CLOCK_LOCK_NO_TARGET, run_bench._CLOCK_INELIGIBLE_EXIT_CODE),
    ],
)
def test_clock_reapply_status_controls_retry_classification(
    monkeypatch, tmp_path, reapply_status, expected_exit
):
    runner = _clock_test_runner(tmp_path)
    monkeypatch.setattr(run_bench, "_clock_lock_required_mhz", 1005)
    monkeypatch.setattr(
        run_bench.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="SM clock mismatch: expected 1005 MHz, measured 1020 MHz",
        ),
    )
    monkeypatch.setattr(run_bench, "_reapply_required_sm_clock", lambda: reapply_status)

    with pytest.raises(SystemExit) as exc:
        runner.run_benchmark("bench_test", [], "test_config")

    assert exc.value.code == expected_exit


def test_strict_generic_failure_preserves_the_wave_and_removes_stale_output(
    monkeypatch, tmp_path
):
    runner = _clock_test_runner(tmp_path)
    output_path = tmp_path / "out_test_config_cpp.nvbench"
    output_path.write_text("stale")
    run_calls = []
    relock_calls = []
    monkeypatch.setattr(run_bench, "_clock_lock_required_mhz", 1005)
    monkeypatch.setattr(
        run_bench.subprocess,
        "run",
        lambda *args, **kwargs: run_calls.append(args)
        or SimpleNamespace(returncode=2, stdout="", stderr="benchmark failed"),
    )
    monkeypatch.setattr(
        run_bench,
        "_reapply_required_sm_clock",
        lambda: relock_calls.append(True) or run_bench._CLOCK_LOCK_OK,
    )

    assert runner.run_benchmark("bench_test", [], "test_config") == (None, False)
    assert len(run_calls) == 1
    assert not relock_calls
    assert not output_path.exists()


def _touch(path):
    path.write_text("")


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=4) + "\n")


def _write_baseline_config_tree(root):
    _write_json(
        root / "sku_map.json",
        {
            "entries": [
                {
                    "gpu_name": "NVIDIA Test GPU",
                    "power_cap_w": 250,
                    "locked_sm_clock_mhz": 1005,
                    "cuda_major": 13,
                    "stem": "A100_PCIE_40GB_250W_1005MHz",
                },
                {
                    "gpu_name": "NVIDIA Test GPU",
                    "power_cap_w": 250,
                    "locked_sm_clock_mhz": 1005,
                    "cuda_major": 12,
                    "stem": "A100_PCIE_40GB_250W_1005MHz_CUDA12",
                },
            ]
        },
    )
    _write_json(
        root / "operators" / "resize.json",
        {
            "benchmark": "resize",
            "configs": {
                "resize_contract_area_tensor_uchar3_basic": {
                    "tier": "basic",
                    "dtypes": ["uchar3"],
                    "string_axes": {
                        "shape": ["32x1080x1920"],
                        "kernelSize": ["5x5"],
                        "resizeType": ["CONTRACT"],
                        "interpolation": ["AREA"],
                        "layout": ["NHWC"],
                        "inputKind": ["Tensor"],
                    },
                }
            },
        },
    )


def _baseline_row(language="cpp", gpu_us=525.7525, noise_us=1.2, bwutil=0.42):
    return {
        "Benchmark": "resize",
        "tier": "basic",
        "config_key": "resize_contract_area_tensor_uchar3_basic",
        "Language": language,
        "GPU Time (µs)": gpu_us,
        "GPU Noise (µs)": noise_us,
        "BWUtil": bwutil,
        "Device Name": "NVIDIA Test GPU",
        "Power Cap (W)": 250,
        "Locked SM Clock (MHz)": 1005,
        "InOutDataType": "uchar3",
        "shape": "32x1080x1920",
        "resizeType": "CONTRACT",
        "interpolation": "AREA",
        "layout": "NHWC",
        "inputKind": "Tensor",
    }


def test_advanced_tier_is_not_discovered_by_basic_tier():
    config = {
        "resize_basic": {
            "benchmark": "resize",
            "tier": "basic",
        },
        "resize_advanced": {
            "benchmark": "resize",
            "tier": "advanced",
        },
    }

    assert get_configs_for_benchmark("resize", config, tiers={"basic"}) == [
        "resize_basic"
    ]
    assert get_configs_for_benchmark("resize", config, tiers={"advanced"}) == [
        "resize_advanced"
    ]
    assert get_configs_for_benchmark("resize", config, tiers={"basic", "advanced"}) == [
        "resize_basic",
        "resize_advanced",
    ]


def test_parse_tier_arg_accepts_basic_and_advanced():
    assert parse_tier_arg("basic") == {"basic"}
    assert parse_tier_arg("advanced") == {"advanced"}
    assert parse_tier_arg("basic, advanced") == {"basic", "advanced"}

    with pytest.raises(ValueError, match="unknown tier"):
        parse_tier_arg("ci")


def test_parse_config_key_arg_single_and_comma_separated():
    assert parse_config_key_arg("resize_advanced") == ["resize_advanced"]
    assert parse_config_key_arg("resize_advanced, gaussian_advanced ") == [
        "resize_advanced",
        "gaussian_advanced",
    ]


def test_parse_config_key_arg_rejects_empty_and_duplicate_values():
    with pytest.raises(ValueError, match="empty"):
        parse_config_key_arg(" , ")

    with pytest.raises(ValueError, match="duplicate"):
        parse_config_key_arg("resize_advanced,resize_advanced")


def test_parse_operator_arg_accepts_ordered_exact_names():
    assert parse_operator_arg("resize") == ["resize"]
    assert parse_operator_arg("resize, gaussian flip") == [
        "resize",
        "gaussian",
        "flip",
    ]


def test_parse_operator_arg_rejects_empty_and_duplicate_values():
    with pytest.raises(ValueError, match="empty"):
        parse_operator_arg(" , ")

    with pytest.raises(ValueError, match="duplicate"):
        parse_operator_arg("resize,resize")


def test_load_bench_config_expands_split_manifest(tmp_path):
    operators = tmp_path / "operators"
    operators.mkdir()
    _write_json(
        operators / "resize.json",
        {
            "resize_basic": {
                "benchmark": "resize",
                "tier": "basic",
            },
        },
    )
    _write_json(
        operators / "gaussian.json",
        {
            "gaussian_advanced": {
                "benchmark": "gaussian",
                "tier": "advanced",
            },
        },
    )
    _write_json(
        tmp_path / "bench_params.json",
        {"include": ["operators/resize.json", "operators/gaussian.json"]},
    )

    assert list(load_bench_config(str(tmp_path / "bench_params.json"))) == [
        "resize_basic",
        "gaussian_advanced",
    ]
    assert list(load_bench_config(str(tmp_path))) == [
        "resize_basic",
        "gaussian_advanced",
    ]


def test_load_bench_config_expands_operator_manifest(tmp_path):
    operators = tmp_path / "operators"
    operators.mkdir()
    _write_json(
        operators / "resize.json",
        {
            "resize_basic": {
                "benchmark": "resize",
                "tier": "basic",
            },
        },
    )
    _write_json(
        operators / "gaussian.json",
        {
            "gaussian_advanced": {
                "benchmark": "gaussian",
                "tier": "advanced",
            },
        },
    )
    _write_json(
        tmp_path / "bench_params.json",
        {
            "operators": {
                "resize": {
                    "config": "operators/resize.json",
                    "cpp": "bench_resize",
                    "python": "bench_resize.py",
                },
                "gaussian": {
                    "config": "operators/gaussian.json",
                    "cpp": "bench_gaussian",
                    "python": "bench_gaussian.py",
                },
            }
        },
    )

    assert list(load_bench_config(str(tmp_path / "bench_params.json"))) == [
        "resize_basic",
        "gaussian_advanced",
    ]
    assert list(load_bench_manifest(str(tmp_path / "bench_params.json"))) == [
        "resize",
        "gaussian",
    ]


def test_load_bench_config_rejects_duplicate_split_keys(tmp_path):
    operators = tmp_path / "operators"
    operators.mkdir()
    entry = {
        "resize_basic": {
            "benchmark": "resize",
            "tier": "basic",
        },
    }
    _write_json(operators / "resize.json", entry)
    _write_json(operators / "duplicate.json", entry)
    _write_json(
        tmp_path / "bench_params.json",
        {"include": ["operators/resize.json", "operators/duplicate.json"]},
    )

    with pytest.raises(ValueError, match="Duplicate config key"):
        load_bench_config(str(tmp_path / "bench_params.json"))


def _entry_string_axis_values(entry, axis):
    return entry.get("string_axes", {}).get(axis, [])


def _entry_row_count(entry):
    count = len(entry.get("dtypes", [])) or 1
    for axis_group in ("string_axes", "int64_axes", "float64_axes"):
        for values in entry.get(axis_group, {}).values():
            count *= len(values)
    return count


@pytest.mark.parametrize(
    ("key", "dtype", "shape"),
    (
        ("cvtcolor_lab_u8_advanced", "uint8", "128x1080x1920"),
        ("cvtcolor_lab_f16_advanced", "float16", "64x1080x1920"),
        ("cvtcolor_lab_f32_advanced", "float32", "32x1080x1920"),
    ),
)
def test_cvtcolor_lab_configs_cover_dtypes_containers_and_native_layouts(
    key, dtype, shape
):
    entry = load_bench_config()[key]

    assert entry["tier"] == "advanced"
    assert entry["dtypes"] == [dtype]
    assert entry["string_axes"] == {
        "shape": [shape],
        "code": [
            "BGR2Lab",
            "RGB2Lab",
            "Lab2BGR",
            "Lab2RGB",
            "LBGR2Lab",
            "LRGB2Lab",
            "Lab2LBGR",
            "Lab2LRGB",
        ],
        "inputKind": ["Tensor", "VarShape"],
        "layout": ["NHWC", "NCHW"],
    }
    assert _entry_row_count(entry) == 32


_RESIZE_ANISOTROPIC_TARGETS = {
    "CONTRACT": "TARGET_480x864",
    "EXPAND": "TARGET_2160x3840",
}
_RESIZE_ANISOTROPIC_INPUT_SPATIAL_SHAPES = {
    "TARGET_480x864": ["1080", "1920"],
    "TARGET_2160x3840": ["480", "864"],
}


def _resize_coverage_signature(operator, axes):
    fields = ["InOutDataType", "interpolation", "layout", "inputKind"]
    if operator == "hqresize":
        fields.extend(("antialias", "numChannels"))
    return tuple(axes[field] for field in fields)


def test_generate_axis_args_preserves_axis_order_and_grouping():
    config = {
        "synthetic_planar_basic": {
            "benchmark": "synthetic",
            "tier": "basic",
            "dtypes": ["uchar3"],
            "string_axes": {
                "shape": ["1x2x3"],
                "layout": ["NCHW"],
                "inputKind": ["Tensor", "VarShape"],
            },
        }
    }

    assert generate_axis_args("synthetic_planar_basic", config) == [
        "--axis",
        "InOutDataType=uchar3",
        "--axis",
        "shape=1x2x3",
        "--axis",
        "layout=NCHW",
        "--axis",
        "inputKind=[Tensor,VarShape]",
    ]


def test_repository_config_separates_basic_and_advanced_tiers():
    config = load_bench_config()
    manifest = load_bench_manifest()

    assert {
        entry.get("tier") for entry in config.values() if isinstance(entry, dict)
    } == {"basic", "advanced"}
    assert manifest["resize"] == {
        "config": "operators/resize.json",
        "cpp": "bench_resize",
        "python": "bench_resize.py",
    }
    removed_field = "manual" + "_only"
    assert [
        key
        for key, entry in config.items()
        if isinstance(entry, dict) and removed_field in entry
    ] == []
    assert [
        key
        for key, entry in config.items()
        if isinstance(entry, dict)
        and entry.get("tier") == "basic"
        and "advanced" in key
    ] == []
    assert [
        key
        for key, entry in config.items()
        if isinstance(entry, dict)
        and entry.get("tier") == "advanced"
        and not key.endswith("_advanced")
    ] == []

    planar_layouts = {"NCHW", "CHW", "NCHW_FAKE", "CHW_FAKE"}
    fake_planar_layouts = {"NCHW_FAKE", "CHW_FAKE"}
    planar_keys = []
    fake_planar_keys = []

    for key, entry in config.items():
        if not isinstance(entry, dict):
            continue

        string_axes = entry.get("string_axes", {})
        layouts = _entry_string_axis_values(entry, "layout")

        if any(token in key.lower() for token in ("planar", "nchw", "chw")):
            assert (
                "layout" in string_axes
            ), f"{key} names a planar layout but has no layout axis"

        if "fake" in key.lower():
            assert any(
                layout in fake_planar_layouts for layout in layouts
            ), f"{key} names a fake-planar layout but has no fake-planar axis"

        if any(layout in planar_layouts for layout in layouts):
            planar_keys.append(key)

        if any(layout in fake_planar_layouts for layout in layouts):
            fake_planar_keys.append(key)
            assert entry["tier"] == "advanced"
            input_kinds = _entry_string_axis_values(entry, "inputKind")
            assert input_kinds == [
                "Tensor"
            ], f"{key} fake-planar row is not Tensor-only"

    assert planar_keys
    assert fake_planar_keys

    # CvtColor keeps one basic case for every reviewed conversion branch.
    assert config["cvtcolor_basic"]["string_axes"]["code"] == [
        "RGB2GRAY",
        "RGB2HSV",
        "YUV2RGB",
        "YUV2RGB_NV12",
    ]
    assert config["cvtcolor_planar_nchw_1080p_basic"]["string_axes"]["code"] == [
        "RGB2GRAY",
        "RGB2HSV",
        "YUV2RGB",
    ]
    assert config["cvtcolor_rgb2rgba_uchar3_basic"]["string_axes"]["inputKind"] == [
        "Tensor",
        "VarShape",
    ]

    # CLAHE keeps the representative clip limit in basic; the alternate limit
    # and fake-planar comparison live in advanced.
    assert config["clahe_basic"]["int64_axes"]["clip10"] == [400]
    assert config["clahe_clip20_advanced"]["int64_axes"]["clip10"] == [20]
    assert config["clahe_clip20_advanced"]["tier"] == "advanced"
    assert config["clahe_fake_planar_advanced"]["tier"] == "advanced"

    # Normalize's RGB8 basic matrix also measures the tensor-free scalar-parameter path.
    assert config["normalize_rgb_u8_1080p_basic"]["string_axes"]["inputKind"] == [
        "Tensor",
        "TensorScalar",
        "VarShape",
    ]
    assert config["normalize_planar_rgb_u8_nchw_1080p_basic"]["string_axes"][
        "inputKind"
    ] == ["Tensor", "TensorScalar", "VarShape"]
    assert config["normalize_rgb_f32_1080p_basic"]["string_axes"]["inputKind"] == [
        "Tensor"
    ]
    assert config["normalize_planar_rgb_f32_nchw_1080p_basic"]["string_axes"][
        "inputKind"
    ] == ["Tensor"]
    assert config["normalize_rgb_f32_1080p_advanced"]["string_axes"]["inputKind"] == [
        "VarShape"
    ]

    # GammaContrast float3 is symmetric across native layouts for Tensor;
    # VarShape float3 remains advanced-only.
    assert config["gammacontrast_float3_lcg_basic"]["string_axes"] == {
        "shape": ["32x1080x1920"],
        "layout": ["NHWC", "NCHW"],
        "inputKind": ["Tensor"],
    }
    assert config["gammacontrast_planar_nchw_float3_varshape_lcg_advanced"][
        "string_axes"
    ]["inputKind"] == ["VarShape"]

    # The real container coverage added for non-standard APIs stays explicit.
    assert config["findhomography_basic"]["string_axes"]["inputKind"] == ["Tensor"]
    assert config["findhomography_varshape_basic"]["string_axes"]["inputKind"] == [
        "VarShape"
    ]
    assert config["stack_uchar3_basic"]["string_axes"] == {
        "shape": ["32x2160x3840"],
        "layout": ["NHWC", "NCHW"],
        "inputKind": ["Tensor", "VarShape"],
    }
    assert config["stack_uchar3_layout_compare_4k_advanced"]["string_axes"] == {
        "shape": ["4x2160x3840"],
        "layout": ["NHWC", "NCHW", "NCHW_FAKE"],
        "inputKind": ["Tensor"],
    }

    # These image-layout-bearing benchmarks now expose their truthful layout;
    # semantic N-A operators intentionally do not receive a dummy axis.
    layout_bearing = {
        "bndbox",
        "channelreorder",
        "composite",
        "cropflipnormalizereformat",
        "histogram",
        "label",
        "minmaxloc",
        "osd",
        "padandstack",
        "resizecropconvertreformat",
        "sift",
        "stack",
    }
    assert all(
        "layout" in entry.get("string_axes", {})
        for entry in config.values()
        if entry.get("benchmark") in layout_bearing
    )
    layout_na = {
        "findhomography",
        "minarearect",
        "nonmaximumsuppression",
        "pairwisematcher",
        "reformat",
    }
    assert all(
        "layout" not in entry.get("string_axes", {})
        for entry in config.values()
        if entry.get("benchmark") in layout_na
    )
    assert config["label_basic"]["string_axes"]["inputKind"] == ["Tensor"]
    assert config["pairwisematcher_uint8_basic"]["string_axes"]["inputKind"] == [
        "Tensor"
    ]

    # AdaptiveThreshold's ImageBatch path rejects planar formats; its NCHW
    # benchmark is therefore truthfully Tensor-only rather than a skipped row.
    assert config["adaptivethreshold_planar_nchw_basic"]["string_axes"][
        "inputKind"
    ] == ["Tensor"]

    assert "distancemap" not in manifest


@pytest.mark.parametrize(
    ("operator", "dtype"),
    [
        ("resize", "uchar3"),
        ("pillowresize", "uchar3"),
        ("hqresize", "uint8"),
    ],
)
def test_resize_family_has_basic_anisotropic_profiles(operator, dtype):
    config = load_bench_config()
    profiles = {
        "contract_480p_tensor": ("TARGET_480x864", "Tensor", ["1080", "1920"]),
        "contract_480p_varshape": (
            "TARGET_480x864",
            "VarShape",
            ["1080", "1920"],
        ),
        "expand_4k_tensor": ("TARGET_2160x3840", "Tensor", ["480", "864"]),
        "expand_4k_varshape": (
            "TARGET_2160x3840",
            "VarShape",
            ["480", "864"],
        ),
    }

    for suffix, (resize_type, input_kind, spatial_shape) in profiles.items():
        entry = config[f"{operator}_anisotropic_{suffix}_rgb8_basic"]
        axes = entry["string_axes"]
        assert entry["tier"] == "basic"
        assert entry["dtypes"] == [dtype]
        assert axes["layout"] == ["NHWC"]
        assert axes["resizeType"] == [resize_type]
        assert axes["inputKind"] == [input_kind]
        assert axes["shape"][0].split("x")[1:] == spatial_shape


_RESIZE_EXISTING_BASIC_ANISOTROPIC_SIGNATURES = {
    "resize": {
        ("TARGET_480x864", ("uchar3", "AREA", "NHWC", "Tensor")),
        ("TARGET_480x864", ("uchar3", "AREA", "NHWC", "VarShape")),
        ("TARGET_2160x3840", ("uchar3", "LINEAR", "NHWC", "Tensor")),
        ("TARGET_2160x3840", ("uchar3", "LINEAR", "NHWC", "VarShape")),
    },
    "pillowresize": {
        ("TARGET_480x864", ("uchar3", "CUBIC", "NHWC", "Tensor")),
        ("TARGET_480x864", ("uchar3", "CUBIC", "NHWC", "VarShape")),
        ("TARGET_2160x3840", ("uchar3", "CUBIC", "NHWC", "Tensor")),
        ("TARGET_2160x3840", ("uchar3", "CUBIC", "NHWC", "VarShape")),
    },
    "hqresize": {
        (
            "TARGET_480x864",
            ("uint8", "CUBIC", "NHWC", "Tensor", "1", "3"),
        ),
        (
            "TARGET_480x864",
            ("uint8", "CUBIC", "NHWC", "VarShape", "1", "3"),
        ),
        (
            "TARGET_2160x3840",
            ("uint8", "CUBIC", "NHWC", "Tensor", "0", "3"),
        ),
        (
            "TARGET_2160x3840",
            ("uint8", "CUBIC", "NHWC", "VarShape", "0", "3"),
        ),
    },
}


@pytest.mark.parametrize("operator", ["split", "merge"])
def test_channel_copy_configuration_counts(operator):
    entries = {
        key: entry
        for key, entry in load_bench_config().items()
        if entry.get("benchmark") == operator
    }
    counts = {
        tier: sum(
            len(expected_case_keys_for_entry(key, entry))
            for key, entry in entries.items()
            if entry["tier"] == tier
        )
        for tier in ("basic", "advanced")
    }
    assert counts == {"basic": 12, "advanced": 42}
    assert sum(counts.values()) == {"split": 54, "merge": 54}[operator]
    assert len(entries) == 15
    assert fake_planar_pairing_issues(entries) == ()
    baseline_index = load_config_index()
    tensor_counts = {"basic": 0, "advanced": 0}
    varshape_cases = set()
    for key, entry in entries.items():
        assert isinstance(baseline_index.require(key).entry["baselines"], dict)
        assert set(entry["string_axes"]) == {"shape", "layout", "inputKind"}
        assert set(entry["int64_axes"]) == {"channels"}
        for case in expected_case_keys_for_entry(key, entry):
            axes = dict(parse_case_key(case)[1])
            if axes["inputKind"] == "Tensor":
                tensor_counts[entry["tier"]] += 1
                assert "@" not in axes["shape"]
                continue
            assert axes["layout"] in ("NHWC", "NCHW")
            shape, separator, variation = axes["shape"].partition("@")
            dtype = {"uchar3": "uint8", "float3": "float32"}.get(
                axes["InOutDataType"], axes["InOutDataType"]
            )
            variant = (
                dtype,
                int(axes["channels"]),
                axes["layout"],
                int(variation) if separator else 0,
            )
            assert variant not in varshape_cases
            varshape_cases.add(variant)
            if entry["tier"] == "basic":
                tensor = entries[key.replace("_varshape_", "_")]
                assert shape in tensor["string_axes"]["shape"]
                assert axes["channels"] == "3"
                assert dtype in ("uint8", "float32")
    assert tensor_counts == {"basic": 4, "advanced": 22}
    assert varshape_cases == {
        (dtype, 3, layout, variation)
        for dtype in ("uint8", "float32")
        for layout in ("NHWC", "NCHW")
        for variation in (0, 17)
    } | {
        ("uint8", channels, layout, 17)
        for channels in (1, 2, 4)
        for layout in ("NHWC", "NCHW")
    } | {
        (dtype, 3, layout, 17)
        for dtype in (
            "int8",
            "uint16",
            "int16",
            "uint32",
            "int32",
            "float16",
            "float64",
        )
        for layout in ("NHWC", "NCHW")
    }


@pytest.mark.parametrize(
    "operator,basic_rows,advanced_rows",
    [
        ("resize", 22, 233),
        ("pillowresize", 22, 247),
        ("hqresize", 26, 132),
    ],
)
def test_resize_family_anisotropic_coverage_mirrors_isotropic_matrix(
    operator, basic_rows, advanced_rows
):
    config = {
        key: entry
        for key, entry in load_bench_config().items()
        if entry.get("benchmark") == operator
    }
    tier_rows = {
        tier: sum(
            _entry_row_count(entry)
            for entry in config.values()
            if entry["tier"] == tier
        )
        for tier in ("basic", "advanced")
    }
    assert tier_rows == {"basic": basic_rows, "advanced": advanced_rows}

    isotropic_signatures = set()
    anisotropic_cases = {}
    for key, entry in config.items():
        for case_key in expected_case_keys_for_entry(key, entry):
            axes = dict(parse_case_key(case_key)[1])
            resize_type = axes["resizeType"]
            signature = _resize_coverage_signature(operator, axes)

            if resize_type in _RESIZE_ANISOTROPIC_TARGETS:
                isotropic_signatures.add(
                    (_RESIZE_ANISOTROPIC_TARGETS[resize_type], signature)
                )
            elif resize_type in _RESIZE_ANISOTROPIC_TARGETS.values():
                assert axes["shape"].split("x")[1:] == (
                    _RESIZE_ANISOTROPIC_INPUT_SPATIAL_SHAPES[resize_type]
                )
                semantic_case = (resize_type, signature)
                assert semantic_case not in anisotropic_cases, (
                    f"duplicate {operator} anisotropic semantic case: "
                    f"{semantic_case} in {anisotropic_cases[semantic_case][0]} and {key}"
                )
                anisotropic_cases[semantic_case] = (key, entry["tier"])

    existing_basic = _RESIZE_EXISTING_BASIC_ANISOTROPIC_SIGNATURES[operator]
    assert {
        semantic_case
        for semantic_case, (_, tier) in anisotropic_cases.items()
        if tier == "basic"
    } == existing_basic

    # Some pre-existing basic profiles (for example anisotropic EXPAND
    # VarShape) intentionally have no isotropic source. They remain in the
    # matrix, while every projected case that was added by this extension is
    # advanced-only.
    assert set(anisotropic_cases) == isotropic_signatures | existing_basic
    for semantic_case in isotropic_signatures - existing_basic:
        assert anisotropic_cases[semantic_case][1] == "advanced"

    target_configs = {
        key: entry
        for key, entry in config.items()
        if any(
            resize_type in _RESIZE_ANISOTROPIC_TARGETS.values()
            for resize_type in entry["string_axes"]["resizeType"]
        )
    }
    assert fake_planar_pairing_issues(target_configs) == ()

    layout_batches = {}
    for key, entry in target_configs.items():
        for case_key in expected_case_keys_for_entry(key, entry):
            _, case_axes = parse_case_key(case_key)
            axes = dict(case_axes)
            layout = axes["layout"]
            if layout not in {"NHWC", "NCHW", "NCHW_FAKE"}:
                continue
            shape = axes["shape"].split("x")
            identity = tuple(
                sorted(
                    (name, value)
                    for name, value in case_axes
                    if name not in {"layout", "shape"}
                )
            ) + (("spatialShape", "x".join(shape[1:])),)
            layout_batches.setdefault(identity, {})[layout] = int(shape[0])

    for identity, batches in layout_batches.items():
        if "NHWC" in batches and {"NCHW", "NCHW_FAKE"} & batches.keys():
            assert (
                len(set(batches.values())) == 1
            ), f"{operator} layout batch mismatch for {dict(identity)}: {batches}"


@pytest.mark.parametrize(
    "operator",
    [
        "autocontrast",
        "bilateralfilter",
        "bndbox",
        "boxblur",
        "brightnesscontrast",
        "centercrop",
        "clahe",
        "convertto",
        "copymakeborder",
        "cvtcolor",
        "erase",
        "flip",
        "gaussiannoise",
        "inpaint",
        "invert",
        "morphology",
    ],
)
def test_priority_operator_fake_planar_rows_have_native_twins(operator):
    config = {
        key: entry
        for key, entry in load_bench_config().items()
        if entry.get("benchmark") == operator
    }

    assert any(
        layout in {"NCHW_FAKE", "CHW_FAKE"}
        for entry in config.values()
        for layout in _entry_string_axis_values(entry, "layout")
    )
    assert fake_planar_pairing_issues(config) == ()


def test_reformat_single_call_calibration_case_keys_are_stable():
    config = load_bench_config()

    expected = {
        "reformat_uint8_pitched_fallback_advanced": (
            "reformat_uint8_pitched_fallback_advanced"
            "[InOutDataType=uint8][shape=3072x480x640]"
            "[inputKind=Tensor][rowAlignment=256]"
        ),
        "reformat_uint8_pitched_large_image_advanced": (
            "reformat_uint8_pitched_large_image_advanced"
            "[InOutDataType=uint8][shape=160x1080x1920]"
            "[inputKind=Tensor][rowAlignment=256]"
        ),
        "reformat_uint8_pitched_large_image_b448_advanced": (
            "reformat_uint8_pitched_large_image_b448_advanced"
            "[InOutDataType=uint8][shape=448x1080x1920]"
            "[inputKind=Tensor][rowAlignment=256]"
        ),
    }

    for config_key, case_key in expected.items():
        entry = config[config_key]
        assert expected_case_keys_for_entry(config_key, entry) == [case_key]
        assert "timed_repetitions" not in entry.get("metadata", {})

    assert "reformat_uint8_pitched_fullhd_advanced" not in config


def test_brightnesscontrast_benchmark_signatures_are_canonical_and_unique():
    config = {
        key: entry
        for key, entry in load_bench_config().items()
        if entry.get("benchmark") == "brightnesscontrast"
    }
    assert {
        "brightnesscontrast_rgb_f32_1080p_advanced",
        "brightnesscontrast_planar_rgb_f32_nchw_1080p_advanced",
    } <= config.keys()

    signatures = {}
    for key, entry in config.items():
        for case in expected_case_keys_for_entry(key, entry):
            _, axes = parse_case_key(case)
            signature = (entry["tier"], tuple(sorted(axes)))
            assert signature not in signatures, (
                f"{case} duplicates canonical benchmark signature from "
                f"{signatures.get(signature)}"
            )
            signatures[signature] = case


def test_parse_args_rejects_unknown_config_key(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_bench.py", "--config-key", "definitely_not_a_config_key"],
    )

    with pytest.raises(SystemExit) as exc:
        run_bench.parse_args()

    assert exc.value.code == 2


def test_parse_args_accepts_exact_operator_names(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_bench.py", "--operator", "resize,gaussian"],
    )

    args = run_bench.parse_args()

    assert args.operators == ["resize", "gaussian"]


def test_parse_args_forwards_arguments_after_explicit_bench_folder(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_bench.py", ".", "--", "--axis", "shape=1x2x3"],
    )

    args = run_bench.parse_args()

    assert args.bench_folder == "."
    assert args.bench_args[-2:] == ["--axis", "shape=1x2x3"]


def test_parse_args_defaults_to_csv_output(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_bench.py"])

    args = run_bench.parse_args()

    assert args.output.endswith("bench_output.csv")


def test_parse_args_uses_shared_quality_defaults(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_bench.py"])

    args = run_bench.parse_args()

    assert args.max_noise_pct == DEFAULT_BENCHMARK_QUALITY.max_noise_pct
    assert args.max_perf_diff_pct == DEFAULT_BENCHMARK_QUALITY.max_perf_diff_pct
    assert args.max_perf_diff_us == DEFAULT_BENCHMARK_QUALITY.max_perf_diff_us


def test_parse_args_accepts_warmup_cap(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_bench.py", "--warmup-cap", "50"])

    args = run_bench.parse_args()

    assert args.warmup_cap == 50


@pytest.mark.parametrize("value", ["-1", "abc", "1.5"])
def test_parse_args_rejects_invalid_warmup_cap(monkeypatch, value):
    monkeypatch.setattr(sys, "argv", ["run_bench.py", "--warmup-cap", value])

    with pytest.raises(SystemExit) as exc:
        run_bench.parse_args()

    assert exc.value.code == 2


def test_parse_args_uses_csv_output_for_custom_config_file(monkeypatch, tmp_path):
    config_dir = tmp_path / "config"
    _write_json(
        config_dir / "bench_params_smoke.json",
        {
            "operators": {
                "resize": {
                    "config": "operators/resize.json",
                    "cpp": "bench_resize",
                    "python": "bench_resize.py",
                }
            }
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_bench.py", "--config-file", str(config_dir / "bench_params_smoke.json")],
    )

    args = run_bench.parse_args()

    assert args.output.endswith("bench_output_smoke.csv")


def test_parse_args_accepts_keep_outputs_and_json_output(monkeypatch, tmp_path):
    out = tmp_path / "bench_output.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_bench.py",
            "--operator",
            "resize",
            "--keep-outputs",
            "--output",
            str(out),
            "--cuda-major",
            "13",
        ],
    )

    args = run_bench.parse_args()

    assert args.output == str(out)
    assert args.keep_outputs is True


def test_parse_args_rejects_unknown_operator(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_bench.py", "--operator", "res"],
    )

    with pytest.raises(SystemExit) as exc:
        run_bench.parse_args()

    assert exc.value.code == 2


def test_deprecated_benchmarks_arg_uses_exact_operator_names(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_bench.py", "--benchmarks", "resize"],
    )

    args = run_bench.parse_args()

    assert args.operators == ["resize"]


def test_baseline_payload_from_dataframe_emits_raw_baselines_json(tmp_path):
    config_dir = tmp_path / "config"
    _write_baseline_config_tree(config_dir)
    index = load_config_index(config_dir / "operators")
    df = pd.DataFrame(
        [
            _baseline_row(language="cpp"),
            _baseline_row(
                language="python", gpu_us=535.7935, noise_us=3.88, bwutil=0.41
            ),
        ]
    )

    payload = baseline_payload_from_dataframe(
        df,
        index=index,
        sku_map_path=config_dir / "sku_map.json",
        cuda_major=12,
        source=tmp_path / "bench_output.json",
    )

    case = (
        "resize_contract_area_tensor_uchar3_basic"
        "[InOutDataType=uchar3]"
        "[shape=32x1080x1920]"
        "[kernelSize=5x5]"
        "[resizeType=CONTRACT]"
        "[interpolation=AREA]"
        "[layout=NHWC]"
        "[inputKind=Tensor]"
    )
    assert payload == {
        case: {
            "A100_PCIE_40GB_250W_1005MHz_CUDA12": {
                "n_runs": 1,
                "gpu_time_us_cpp": 525.7525,
                "gpu_time_us_python": 535.7935,
                "gpu_noise_us_cpp": 1.2,
                "gpu_noise_us_python": 3.88,
                "gpu_bwutil_cpp": 0.42,
                "gpu_bwutil_python": 0.41,
            }
        }
    }


def test_informative_payload_records_an_unmapped_sku(tmp_path):
    config_dir = tmp_path / "config"
    _write_baseline_config_tree(config_dir)
    index = load_config_index(config_dir / "operators")
    row = _baseline_row(language="cpp")
    row["Device Name"] = "NVIDIA A10"
    row["Power Cap (W)"] = 150
    row["Locked SM Clock (MHz)"] = 345
    df = pd.DataFrame([row])
    source = tmp_path / "bench_output.json"

    with pytest.raises(BaselineError, match=r"is not in sku_map\.json"):
        baseline_payload_from_dataframe(
            df,
            index=index,
            sku_map_path=config_dir / "sku_map.json",
            cuda_major=12,
            source=source,
        )

    payload = baseline_payload_from_dataframe(
        df,
        index=index,
        sku_map_path=config_dir / "sku_map.json",
        cuda_major=12,
        source=source,
        allow_unmapped_sku=True,
    )
    case_key, by_sku = next(iter(payload.items()))
    assert list(by_sku) == ["unmapped_NVIDIA_A10_150W_345MHz_CUDA12"]

    updates = baseline_updates_from_payload(
        payload,
        index=index,
        sku_stem_set=set(),
        source=source,
    )
    assert [update.sku for update in updates] == [
        "unmapped_NVIDIA_A10_150W_345MHz_CUDA12"
    ]

    renamed = {case_key: {"not_a_sku": next(iter(by_sku.values()))}}
    with pytest.raises(BaselineError, match="unknown SKU"):
        baseline_updates_from_payload(
            renamed,
            index=index,
            sku_stem_set=set(),
            source=source,
        )


def test_parse_args_accepts_informative(monkeypatch, tmp_path):
    out = tmp_path / "bench_output.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_bench.py",
            "--informative",
            "--output",
            str(out),
            "--cuda-major",
            "12",
        ],
    )

    args = run_bench.parse_args()

    assert args.informative is True
    assert args.output == str(out)


def test_baseline_payload_from_dataframe_allows_single_language(tmp_path):
    config_dir = tmp_path / "config"
    _write_baseline_config_tree(config_dir)
    index = load_config_index(config_dir / "operators")

    payload = baseline_payload_from_dataframe(
        pd.DataFrame([_baseline_row(language="cpp")]),
        index=index,
        sku_map_path=config_dir / "sku_map.json",
        cuda_major=13,
        source=tmp_path / "bench_output.json",
    )

    metrics = next(iter(next(iter(payload.values())).values()))
    assert metrics == {
        "n_runs": 1,
        "gpu_time_us_cpp": 525.7525,
        "gpu_noise_us_cpp": 1.2,
        "gpu_bwutil_cpp": 0.42,
    }


def test_baseline_payload_from_dataframe_accepts_explicit_sku(tmp_path):
    config_dir = tmp_path / "config"
    _write_baseline_config_tree(config_dir)
    sku_map_path = config_dir / "sku_map.json"
    sku_map = json.loads(sku_map_path.read_text())
    sku_map["entries"].append(
        {
            "routing": "explicit",
            "cuda_major": 12,
            "stem": "Jetson_Orin_R36_2_MAXN_CUDA12",
            "baseline": {"languages": ["python"], "n_runs": 5},
        }
    )
    _write_json(sku_map_path, sku_map)
    row = _baseline_row(language="python")
    for column in ("Device Name", "Power Cap (W)", "Locked SM Clock (MHz)"):
        row.pop(column)

    payload = baseline_payload_from_dataframe(
        pd.DataFrame([row]),
        index=load_config_index(config_dir / "operators"),
        sku_map_path=sku_map_path,
        cuda_major=12,
        source=tmp_path / "bench_output.json",
        baseline_sku="Jetson_Orin_R36_2_MAXN_CUDA12",
    )

    metrics_by_sku = next(iter(payload.values()))
    assert list(metrics_by_sku) == ["Jetson_Orin_R36_2_MAXN_CUDA12"]
    assert set(metrics_by_sku["Jetson_Orin_R36_2_MAXN_CUDA12"]) == {
        "n_runs",
        "gpu_time_us_python",
        "gpu_noise_us_python",
        "gpu_bwutil_python",
    }


def test_baseline_payload_from_dataframe_rejects_missing_ambiguous_axis(tmp_path):
    config_dir = tmp_path / "config"
    _write_baseline_config_tree(config_dir)
    config_path = config_dir / "operators" / "resize.json"
    config = json.loads(config_path.read_text())
    config["configs"]["resize_contract_area_tensor_uchar3_basic"]["string_axes"][
        "kernelSize"
    ] = ["3x3", "5x5"]
    _write_json(config_path, config)

    index = load_config_index(config_dir / "operators")

    with pytest.raises(BaselineError, match="missing axis column 'kernelSize'"):
        baseline_payload_from_dataframe(
            pd.DataFrame([_baseline_row(language="cpp")]),
            index=index,
            sku_map_path=config_dir / "sku_map.json",
            cuda_major=13,
            source=tmp_path / "bench_output.json",
        )


def test_parse_args_requires_cuda_major_for_json(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys, "argv", ["run_bench.py", "--output", str(tmp_path / "out.json")]
    )

    with pytest.raises(SystemExit) as exc:
        run_bench.parse_args()

    assert exc.value.code == 2


def test_parse_args_accepts_cuda_major_for_json(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_bench.py",
            "--output",
            str(tmp_path / "out.json"),
            "--cuda-major",
            "12",
        ],
    )

    assert run_bench.parse_args().cuda_major == 12


@pytest.mark.parametrize("run_context", ["mr", "nightly-dev", "nightly", "main"])
def test_parse_args_requires_cuda_major_for_non_local_runs(
    monkeypatch, capsys, run_context
):
    monkeypatch.setattr(sys, "argv", ["run_bench.py", "--run-context", run_context])

    with pytest.raises(SystemExit) as exc:
        run_bench.parse_args()

    assert exc.value.code == 2
    assert "--cuda-major is required for non-local runs" in capsys.readouterr().err


def test_baseline_config_paths_uses_run_manifest(tmp_path):
    config_dir = tmp_path / "config"
    _write_baseline_config_tree(config_dir)
    _write_json(
        config_dir / "bench_params.json",
        {
            "operators": {
                "resize": {
                    "config": "operators/resize.json",
                    "cpp": "bench_resize",
                    "python": "bench_resize.py",
                }
            }
        },
    )

    paths, sku_map_path = run_bench._baseline_config_paths(
        str(config_dir / "bench_params.json")
    )

    assert paths == [(config_dir / "operators" / "resize.json").resolve()]
    assert sku_map_path == config_dir / "sku_map.json"


def test_mapped_lock_target_uses_sku_map(tmp_path):
    sku_map = tmp_path / "sku_map.json"
    entries = [
        {
            "gpu_name": "NVIDIA H100 PCIe",
            "power_cap_w": 350,
            "locked_sm_clock_mhz": 1005,
            "cuda_major": 13,
            "stem": "target",
        },
        *[
            {
                "gpu_name": name,
                "power_cap_w": power,
                "locked_sm_clock_mhz": clock,
                "cuda_major": cuda,
                "stem": f"decoy_{index}",
            }
            for index, (name, power, clock, cuda) in enumerate(
                [
                    ("NVIDIA A100-PCIE-40GB", 350, 1095, 13),
                    ("NVIDIA H100 PCIe", 310, 900, 13),
                    ("NVIDIA H100 PCIe", 350, 1095, 12),
                ]
            )
        ],
    ]
    _write_json(
        sku_map,
        {"entries": entries},
    )

    assert (
        run_bench._resolve_required_sm_clock(
            sku_map,
            gpu_name="NVIDIA H100 PCIe",
            power_cap_w=350,
            cuda_major=13,
        )
        == 1005
    )


def test_unknown_gpu_has_no_required_sm_clock(tmp_path):
    sku_map = tmp_path / "sku_map.json"
    _write_json(sku_map, {"entries": []})

    assert (
        run_bench._resolve_required_sm_clock(
            sku_map,
            gpu_name="Unknown GPU",
            power_cap_w=None,
            cuda_major=13,
        )
        is None
    )


def test_mapped_gpu_must_match_a_baseline_profile(tmp_path):
    sku_map = tmp_path / "sku_map.json"
    _write_json(
        sku_map,
        {
            "entries": [
                {
                    "gpu_name": "NVIDIA H100 PCIe",
                    "power_cap_w": 350,
                    "locked_sm_clock_mhz": 1005,
                    "cuda_major": 13,
                    "stem": "H100_PCIe_350W_1005MHz",
                }
            ]
        },
    )

    with pytest.raises(BaselineError, match="no baseline-backed SM clock target"):
        run_bench._resolve_required_sm_clock(
            sku_map,
            gpu_name="NVIDIA H100 PCIe",
            power_cap_w=310,
            cuda_major=13,
        )

    with pytest.raises(
        run_bench._ClockIneligibleError, match="could not determine power cap"
    ):
        run_bench._resolve_required_sm_clock(
            sku_map,
            gpu_name="NVIDIA H100 PCIe",
            power_cap_w=None,
            cuda_major=13,
        )


def test_mapped_power_profile_missing_cuda_major_is_configuration_error(tmp_path):
    sku_map = tmp_path / "sku_map.json"
    _write_json(
        sku_map,
        {
            "entries": [
                {
                    "gpu_name": "NVIDIA H100 PCIe",
                    "power_cap_w": 350,
                    "locked_sm_clock_mhz": 1095,
                    "cuda_major": 12,
                    "stem": "H100_CUDA12",
                }
            ]
        },
    )

    with pytest.raises(BaselineError, match=r"configured CUDA majors: \[12\]") as exc:
        run_bench._resolve_required_sm_clock(
            sku_map,
            gpu_name="NVIDIA H100 PCIe",
            power_cap_w=350,
            cuda_major=13,
        )

    assert not isinstance(exc.value, MissingClockProfileError)


def test_clock_control_uses_required_clock_despite_idle_readback(monkeypatch, tmp_path):
    requests = []
    monkeypatch.setattr(run_bench, "_clock_lock_request_accepted", False)
    monkeypatch.setattr(run_bench, "_clock_lock_applied_mhz", None)
    monkeypatch.setattr(run_bench, "_clock_lock_required_mhz", None)
    monkeypatch.setattr(run_bench, "_clock_sampler_thread", None)
    monkeypatch.setattr(
        run_bench,
        "_try_lock_sm_clock",
        lambda preferred, **_kwargs: requests.append(preferred)
        or (run_bench._CLOCK_LOCK_OK, 375),
    )
    monkeypatch.setattr(run_bench, "_write_gpu_fingerprint", lambda _out_dir: None)
    monkeypatch.setattr(run_bench.threading.Thread, "start", lambda _self: None)

    run_bench.init_clock_control(out_dir=str(tmp_path), required_mhz=1005)

    assert requests == [[1005]]
    assert run_bench._clock_lock_required_mhz == 1005
    assert run_bench._clock_lock_applied_mhz == 1005
    assert run_bench._clock_lock_request_accepted is True


def test_clock_cleanup_is_registered_before_post_lock_setup(monkeypatch, tmp_path):
    events = []
    monkeypatch.setattr(run_bench, "_clock_lock_request_accepted", False)
    monkeypatch.setattr(run_bench, "_clock_lock_applied_mhz", None)
    monkeypatch.setattr(run_bench, "_clock_lock_required_mhz", None)
    monkeypatch.setattr(run_bench, "_clock_sampler_thread", None)
    monkeypatch.setattr(
        run_bench,
        "_try_lock_sm_clock",
        lambda *_args, **_kwargs: events.append("lock")
        or (run_bench._CLOCK_LOCK_OK, 1005),
    )
    monkeypatch.setattr(
        run_bench.atexit,
        "register",
        lambda callback: events.append(("register", callback)),
    )

    def fail_fingerprint(_out_dir):
        events.append("fingerprint")
        raise RuntimeError("unwritable diagnostics")

    monkeypatch.setattr(run_bench, "_write_gpu_fingerprint", fail_fingerprint)

    with pytest.raises(RuntimeError, match="unwritable diagnostics"):
        run_bench.init_clock_control(str(tmp_path), required_mhz=1005)

    assert events == [
        "lock",
        ("register", run_bench.shutdown_clock_control),
        "fingerprint",
    ]


def test_gpu_info_classifies_missing_binary_as_unavailable(monkeypatch):
    monkeypatch.setattr(
        run_bench.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(FileNotFoundError()),
    )

    assert run_bench._query_gpu_name() is None
    with pytest.raises(run_bench._ClockIneligibleError, match="determine GPU"):
        run_bench._require_gpu_name()


def test_supported_clock_query_uses_preflight_timeout(monkeypatch):
    calls = []
    monkeypatch.setattr(
        run_bench,
        "_nvidia_smi",
        lambda *args, **kwargs: calls.append((args, kwargs)) or None,
    )

    assert run_bench._query_supported_sm_clocks_mhz() is None
    assert calls == [(("-q", "-d", "SUPPORTED_CLOCKS", "-i", "0"), {"timeout": 10.0})]


@pytest.mark.parametrize(
    "supported,lock_accepted,expected_status,expect_lock_command",
    [
        (None, False, run_bench._CLOCK_LOCK_QUERY_FAILED, False),
        ([900], False, run_bench._CLOCK_LOCK_NO_TARGET, False),
        ([1005], False, run_bench._CLOCK_LOCK_REQUEST_FAILED, True),
        ([1005], True, run_bench._CLOCK_LOCK_OK, True),
    ],
)
def test_strict_clock_lock_reports_direct_outcome(
    monkeypatch,
    supported,
    lock_accepted,
    expected_status,
    expect_lock_command,
):
    commands = []

    def run_command(command, **_kwargs):
        commands.append(command)
        accepted = lock_accepted and "-lgc" in command
        return SimpleNamespace(returncode=0 if accepted else 1)

    monkeypatch.setattr(run_bench.subprocess, "run", run_command)
    monkeypatch.setattr(run_bench, "_query_supported_sm_clocks_mhz", lambda: supported)
    monkeypatch.setattr(run_bench, "_query_current_sm_clock_mhz", lambda: 375)

    status, sampled = run_bench._try_lock_sm_clock(
        [1005], require_supported_clocks=True
    )

    assert status == expected_status
    assert sampled == 375
    lock_command = [
        "sudo",
        "-n",
        "nvidia-smi",
        "-lgc",
        "1005,1005",
        "-i",
        "0",
    ]
    assert (lock_command in commands) is expect_lock_command


def test_gpu_name_query_targets_clock_control_device(monkeypatch):
    calls = []
    monkeypatch.setattr(
        run_bench,
        "_nvidia_smi",
        lambda *args: calls.append(args) or "NVIDIA H100 PCIe\n",
    )

    assert run_bench._query_gpu_name() == "NVIDIA H100 PCIe"
    assert calls == [("--query-gpu=name", "--format=csv,noheader", "-i", "0")]


def test_gpu_info_classifies_nonzero_exit_as_unavailable(monkeypatch):
    monkeypatch.setattr(
        run_bench.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=1, stdout="", stderr="NVML failed"
        ),
    )

    assert run_bench._query_gpu_name() is None


def test_clock_control_shutdown_resets_state_and_allows_sampler_restart(
    monkeypatch, tmp_path
):
    stop_event = run_bench.threading.Event()
    stop_event.set()
    starts = []
    joins = []

    class FakeThread:
        def __init__(self, **_kwargs):
            pass

        def start(self):
            starts.append(stop_event.is_set())

        def join(self, timeout):
            joins.append(timeout)

    monkeypatch.setattr(run_bench, "_clock_lock_request_accepted", False)
    monkeypatch.setattr(run_bench, "_clock_lock_applied_mhz", None)
    monkeypatch.setattr(run_bench, "_clock_lock_required_mhz", None)
    monkeypatch.setattr(run_bench, "_clock_lock_device_verified", False)
    monkeypatch.setattr(run_bench, "_clock_sampler_stop", stop_event)
    monkeypatch.setattr(run_bench, "_clock_sampler_thread", None)
    monkeypatch.setattr(
        run_bench,
        "_try_lock_sm_clock",
        lambda _preferred, **_kwargs: (run_bench._CLOCK_LOCK_OK, 375),
    )
    monkeypatch.setattr(run_bench, "_try_reset_sm_clock", lambda: True)
    monkeypatch.setattr(run_bench, "_write_gpu_fingerprint", lambda _out_dir: None)
    monkeypatch.setattr(run_bench.threading, "Thread", FakeThread)
    monkeypatch.setattr(run_bench.atexit, "register", lambda _callback: None)

    run_bench.init_clock_control(out_dir=str(tmp_path), required_mhz=1005)
    monkeypatch.setattr(run_bench, "_clock_lock_device_verified", True)
    run_bench.shutdown_clock_control()

    assert stop_event.is_set()
    assert run_bench._clock_sampler_thread is None
    assert run_bench._clock_lock_request_accepted is False
    assert run_bench._clock_lock_applied_mhz is None
    assert run_bench._clock_lock_required_mhz is None
    assert run_bench._clock_lock_device_verified is True

    run_bench.init_clock_control(out_dir=str(tmp_path), required_mhz=1005)

    assert starts == [False, False]
    assert joins == [3.0]
    assert run_bench._clock_lock_device_verified is False


def test_clock_control_shutdown_warns_when_reset_fails(monkeypatch):
    warnings = []
    monkeypatch.setattr(run_bench, "_clock_sampler_thread", None)
    monkeypatch.setattr(run_bench, "_clock_lock_request_accepted", True)
    monkeypatch.setattr(run_bench, "_try_reset_sm_clock", lambda: False)
    monkeypatch.setattr(run_bench, "_query_current_sm_clock_mhz", lambda: 1005)
    monkeypatch.setattr(run_bench, "log_warning", warnings.append)
    monkeypatch.setenv("K8S_NODE_NAME", "gpu-node-7")

    run_bench.shutdown_clock_control()

    assert warnings == [
        "Could not reset the SM clock during benchmark cleanup; "
        "node=gpu-node-7 residual_sm_clock_mhz=1005"
    ]
    assert run_bench._clock_lock_request_accepted is False


def test_clock_control_requires_lock_request(monkeypatch, tmp_path):
    fingerprints = []
    monkeypatch.setattr(run_bench, "_clock_lock_request_accepted", False)
    monkeypatch.setattr(run_bench, "_clock_lock_applied_mhz", None)
    monkeypatch.setattr(run_bench, "_clock_sampler_thread", None)
    monkeypatch.setattr(
        run_bench,
        "_try_lock_sm_clock",
        lambda preferred, **_kwargs: (run_bench._CLOCK_LOCK_REQUEST_FAILED, 375),
    )
    monkeypatch.setattr(
        run_bench,
        "_query_supported_sm_clocks_mhz",
        lambda: pytest.fail("init must use the lock attempt's reason"),
    )
    monkeypatch.setattr(run_bench, "_write_gpu_fingerprint", fingerprints.append)

    with pytest.raises(RuntimeError, match="required 1005 MHz"):
        run_bench.init_clock_control(out_dir=str(tmp_path), required_mhz=1005)

    assert run_bench._clock_lock_applied_mhz == 375
    assert fingerprints == [str(tmp_path)]


def test_clock_control_classifies_unsupported_required_clock_as_ineligible(
    monkeypatch, tmp_path
):
    fingerprints = []
    monkeypatch.setattr(run_bench, "_clock_lock_request_accepted", False)
    monkeypatch.setattr(run_bench, "_clock_lock_applied_mhz", None)
    monkeypatch.setattr(run_bench, "_clock_sampler_thread", None)
    monkeypatch.setattr(
        run_bench,
        "_try_lock_sm_clock",
        lambda preferred, **_kwargs: (run_bench._CLOCK_LOCK_NO_TARGET, 375),
    )
    monkeypatch.setattr(
        run_bench,
        "_query_supported_sm_clocks_mhz",
        lambda: pytest.fail("init must use the lock attempt's reason"),
    )
    monkeypatch.setattr(run_bench, "_write_gpu_fingerprint", fingerprints.append)

    with pytest.raises(run_bench._ClockIneligibleError, match="not supported"):
        run_bench.init_clock_control(out_dir=str(tmp_path), required_mhz=1005)

    assert run_bench._clock_lock_applied_mhz == 375
    assert fingerprints == [str(tmp_path)]


def test_best_effort_clock_control_tolerates_lock_failure(monkeypatch, tmp_path):
    requests = []
    monkeypatch.setattr(run_bench, "_clock_lock_request_accepted", False)
    monkeypatch.setattr(run_bench, "_clock_lock_applied_mhz", None)
    monkeypatch.setattr(run_bench, "_clock_lock_required_mhz", None)
    monkeypatch.setattr(run_bench, "_clock_sampler_thread", None)
    monkeypatch.setattr(
        run_bench,
        "_try_lock_sm_clock",
        lambda preferred, **_kwargs: requests.append(preferred)
        or (run_bench._CLOCK_LOCK_REQUEST_FAILED, 375),
    )
    monkeypatch.setattr(run_bench, "_write_gpu_fingerprint", lambda _out_dir: None)
    monkeypatch.setattr(run_bench.threading.Thread, "start", lambda _self: None)
    monkeypatch.setattr(run_bench.atexit, "register", lambda _callback: None)

    run_bench.init_clock_control(out_dir=str(tmp_path), required_mhz=None)

    assert requests == [run_bench._CLOCK_LOCK_PREFERRED_MHZ]
    assert run_bench._clock_lock_required_mhz is None
    assert run_bench._clock_lock_applied_mhz == 375
    assert run_bench._clock_lock_request_accepted is False


def test_strict_clock_control_rejects_unqueryable_supported_clocks(
    monkeypatch, tmp_path
):
    fingerprints = []
    monkeypatch.setattr(run_bench, "_clock_lock_request_accepted", False)
    monkeypatch.setattr(run_bench, "_clock_lock_applied_mhz", None)
    monkeypatch.setattr(run_bench, "_clock_sampler_thread", None)
    monkeypatch.setattr(
        run_bench,
        "_try_lock_sm_clock",
        lambda preferred, **_kwargs: (run_bench._CLOCK_LOCK_QUERY_FAILED, 375),
    )
    monkeypatch.setattr(run_bench, "_write_gpu_fingerprint", fingerprints.append)

    with pytest.raises(run_bench._ClockRequestError, match="required 1005 MHz"):
        run_bench.init_clock_control(out_dir=str(tmp_path), required_mhz=1005)

    assert run_bench._clock_lock_applied_mhz == 375
    assert fingerprints == [str(tmp_path)]


def test_lock_target_requires_supported_clock_query_only_for_strict_runs(monkeypatch):
    monkeypatch.setattr(run_bench, "_query_supported_sm_clocks_mhz", lambda: None)

    assert run_bench._pick_lock_target([1005]) == (1005, False)
    assert run_bench._pick_lock_target([1005], require_supported_clocks=True) == (
        None,
        False,
    )


def test_best_effort_lock_prefers_baseline_clock(monkeypatch):
    monkeypatch.setattr(
        run_bench, "_query_supported_sm_clocks_mhz", lambda: [1005, 1095]
    )

    assert run_bench._pick_lock_target(run_bench._CLOCK_LOCK_PREFERRED_MHZ) == (
        1005,
        True,
    )


def test_clock_reset_retries_and_targets_gpu_zero(monkeypatch):
    commands = []
    monkeypatch.setattr(
        run_bench.subprocess,
        "run",
        lambda command, **_kwargs: commands.append(command)
        or SimpleNamespace(returncode=0 if len(commands) == 3 else 1),
    )

    assert run_bench._try_reset_sm_clock() is True

    assert commands == [
        ["sudo", "-n", "nvidia-smi", "-rgc", "-i", "0"],
        ["nvidia-smi", "-rgc", "-i", "0"],
        ["sudo", "-n", "nvidia-smi", "-rgc", "-i", "0"],
    ]


def test_reapply_required_clock_resets_before_locking(monkeypatch):
    calls = []
    monkeypatch.setattr(run_bench, "_clock_lock_required_mhz", 1005)
    monkeypatch.setattr(run_bench, "_clock_lock_request_accepted", True)
    monkeypatch.setattr(run_bench, "_clock_lock_applied_mhz", 1005)
    monkeypatch.setattr(
        run_bench, "_try_reset_sm_clock", lambda: calls.append("reset") or True
    )
    monkeypatch.setattr(
        run_bench,
        "_try_lock_sm_clock",
        lambda preferred, **_kwargs: calls.append(("lock", preferred))
        or (run_bench._CLOCK_LOCK_OK, 375),
    )

    assert run_bench._reapply_required_sm_clock() == run_bench._CLOCK_LOCK_OK
    assert calls == ["reset", ("lock", [1005])]
    assert run_bench._clock_lock_applied_mhz == 1005
    assert run_bench._clock_lock_request_accepted is True


def test_reapply_required_clock_stops_when_reset_fails(monkeypatch):
    monkeypatch.setattr(run_bench, "_clock_lock_required_mhz", 1005)
    monkeypatch.setattr(run_bench, "_try_reset_sm_clock", lambda: False)
    monkeypatch.setattr(
        run_bench,
        "_try_lock_sm_clock",
        lambda _preferred, **_kwargs: pytest.fail(
            "lock must not run after reset failure"
        ),
    )

    assert run_bench._reapply_required_sm_clock() == run_bench._CLOCK_LOCK_RESET_FAILED


@pytest.mark.parametrize(
    "run_context,resolved_clock,expected_clock",
    [
        ("local", 1005, None),
        ("mr", 1005, 1005),
        ("nightly-dev", 1005, 1005),
        ("nightly", 1005, 1005),
        ("mr", None, None),
    ],
)
def test_main_applies_clock_policy(
    monkeypatch, tmp_path, run_context, resolved_clock, expected_clock
):
    class StopAfterClockSetup(Exception):
        pass

    args = SimpleNamespace(
        config_file=None,
        cuda_major=13,
        output=str(tmp_path / "nested" / "output.json"),
        run_context=run_context,
    )
    resolve_calls = []
    init_calls = []
    monkeypatch.setattr(run_bench, "parse_args", lambda: args)
    monkeypatch.setattr(
        run_bench,
        "_baseline_config_paths",
        lambda _path: ([], tmp_path / "sku_map.json"),
    )
    monkeypatch.setattr(run_bench, "_query_gpu_name", lambda: "NVIDIA H100 PCIe")
    monkeypatch.setattr(run_bench, "query_power_cap_w", lambda: 350)
    monkeypatch.setattr(
        run_bench,
        "_resolve_required_sm_clock",
        lambda *_args, **_kwargs: resolve_calls.append(None) or resolved_clock,
    )

    def record_clock_control(**kwargs):
        assert os.path.isdir(kwargs["out_dir"])
        init_calls.append(kwargs)

    monkeypatch.setattr(run_bench, "init_clock_control", record_clock_control)
    monkeypatch.setattr(
        run_bench,
        "_run_paired_benchmarks",
        lambda _args: (_ for _ in ()).throw(StopAfterClockSetup()),
    )
    monkeypatch.setenv("CVCUDA_BENCH_EXPECT_SM_CLOCK_MHZ", "stale")

    with pytest.raises(StopAfterClockSetup):
        run_bench.main()

    assert len(resolve_calls) == (run_context != "local")
    if run_context == "local":
        assert run_bench._clock_policy_gpu_name is None
    else:
        assert run_bench._clock_policy_gpu_name == "NVIDIA H100 PCIe"
    assert init_calls == [
        {"out_dir": str(tmp_path / "nested"), "required_mhz": expected_clock}
    ]
    assert os.environ["CVCUDA_BENCH_EXPECT_SM_CLOCK_MHZ"] == "stale"


def test_main_reports_output_directory_error_as_execution_failure(
    monkeypatch, tmp_path
):
    output_parent = tmp_path / "not-a-directory"
    output_parent.write_text("occupied")
    args = SimpleNamespace(
        config_file=None,
        output=str(output_parent / "bench_output.csv"),
        run_context="local",
    )
    monkeypatch.setattr(run_bench, "parse_args", lambda: args)
    monkeypatch.setattr(
        run_bench,
        "init_clock_control",
        lambda **_kwargs: pytest.fail("clock setup must not run"),
    )

    with pytest.raises(SystemExit) as exc:
        run_bench.main()

    assert exc.value.code == 1


def test_main_does_not_hide_unexpected_clock_setup_error(monkeypatch, tmp_path):
    args = SimpleNamespace(
        config_file=None,
        output=str(tmp_path / "bench_output.csv"),
        run_context="local",
    )
    monkeypatch.setattr(run_bench, "parse_args", lambda: args)
    monkeypatch.setattr(
        run_bench,
        "init_clock_control",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("unexpected defect")),
    )

    with pytest.raises(RuntimeError, match="unexpected defect"):
        run_bench.main()


def test_main_classifies_missing_mapped_profile_as_node_ineligible(
    monkeypatch, tmp_path
):
    args = SimpleNamespace(
        config_file=None,
        cuda_major=13,
        output=str(tmp_path / "output.json"),
        run_context="mr",
    )
    monkeypatch.setattr(run_bench, "parse_args", lambda: args)
    monkeypatch.setattr(
        run_bench,
        "_baseline_config_paths",
        lambda _path: ([], tmp_path / "sku_map.json"),
    )
    monkeypatch.setattr(run_bench, "_query_gpu_name", lambda: "NVIDIA H100 PCIe")
    monkeypatch.setattr(run_bench, "query_power_cap_w", lambda: None)
    monkeypatch.setattr(
        run_bench,
        "_resolve_required_sm_clock",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            MissingClockProfileError("no mapped profile")
        ),
    )

    with pytest.raises(SystemExit) as exc:
        run_bench.main()

    assert exc.value.code == run_bench._CLOCK_INELIGIBLE_EXIT_CODE


def test_main_classifies_missing_cuda_clock_as_configuration_error(
    monkeypatch, tmp_path
):
    args = SimpleNamespace(
        config_file=None,
        cuda_major=13,
        output=str(tmp_path / "output.json"),
        run_context="mr",
    )
    monkeypatch.setattr(run_bench, "parse_args", lambda: args)
    monkeypatch.setattr(
        run_bench,
        "_baseline_config_paths",
        lambda _path: ([], tmp_path / "sku_map.json"),
    )
    monkeypatch.setattr(run_bench, "_query_gpu_name", lambda: "NVIDIA H100 PCIe")
    monkeypatch.setattr(run_bench, "query_power_cap_w", lambda: 350)
    monkeypatch.setattr(
        run_bench,
        "_resolve_required_sm_clock",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            MissingClockConfigurationError("CUDA 13 profile missing")
        ),
    )

    with pytest.raises(SystemExit) as exc:
        run_bench.main()

    assert exc.value.code == 1


def test_main_excludes_unresolvable_strict_gpu_identity_before_benchmark(
    monkeypatch, tmp_path
):
    args = SimpleNamespace(
        config_file=None,
        cuda_major=13,
        output=str(tmp_path / "output.json"),
        run_context="mr",
    )
    monkeypatch.setattr(run_bench, "parse_args", lambda: args)
    monkeypatch.setattr(
        run_bench,
        "_baseline_config_paths",
        lambda _path: ([], tmp_path / "sku_map.json"),
    )
    monkeypatch.setattr(run_bench, "_query_gpu_name", lambda: None)
    monkeypatch.setattr(
        run_bench,
        "_run_paired_benchmarks",
        lambda _args: pytest.fail("benchmarks must not run without a GPU identity"),
    )

    with pytest.raises(SystemExit) as exc:
        run_bench.main()

    assert exc.value.code == run_bench._CLOCK_INELIGIBLE_EXIT_CODE


def test_clock_ineligible_exit_propagates_through_paired_runner(monkeypatch, tmp_path):
    class ExitRunner:
        def __init__(self, *_args, **_kwargs):
            self.summary = SimpleNamespace(total=0, successful=0, failed=0)
            self.results = []

        def run_benchmark(self, *_args, **_kwargs):
            raise SystemExit(run_bench._CLOCK_INELIGIBLE_EXIT_CODE)

    args = SimpleNamespace(
        bench_folder=str(tmp_path),
        operators=[],
        keep_outputs=False,
        max_noise_pct=10.0,
        config_file=None,
        tiers=None,
        config_keys=None,
        warmup_cap=None,
        lang="cpp",
        max_perf_diff_pct=10.0,
        max_perf_diff_us=100.0,
        max_retries=0,
        bench_args=[],
    )
    monkeypatch.setattr(run_bench, "CppBenchmarkRunner", ExitRunner)
    monkeypatch.setattr(
        run_bench,
        "discover_operator_pairs",
        lambda *_args: [("test_config", "bench_test", None)],
    )
    monkeypatch.setattr(run_bench, "_query_gpu_name", lambda: "NVIDIA H100 PCIe")

    with pytest.raises(SystemExit) as exc:
        run_bench._run_paired_benchmarks(args)

    assert exc.value.code == run_bench._CLOCK_INELIGIBLE_EXIT_CODE


def test_write_output_json_writes_enveloped_payload(tmp_path):
    config_dir = tmp_path / "config"
    _write_baseline_config_tree(config_dir)
    _write_json(
        config_dir / "bench_params.json",
        {
            "operators": {
                "resize": {
                    "config": "operators/resize.json",
                    "cpp": "bench_resize",
                    "python": "bench_resize.py",
                }
            }
        },
    )
    out = tmp_path / "bench_baselines.json"
    args = SimpleNamespace(
        config_file=str(config_dir / "bench_params.json"),
        output=str(out),
        cuda_major=13,
        run_context="local",
        tier="basic",
    )
    df = pd.DataFrame(
        [
            _baseline_row(language="cpp"),
            _baseline_row(
                language="python", gpu_us=535.7935, noise_us=3.88, bwutil=0.41
            ),
        ]
    )

    run_bench._write_output(args, df)

    document = json.loads(out.read_text())
    assert document["schema_version"] == 1
    # A run that does not declare itself nightly must not be consumable by the
    # drift analysis.
    assert document["run_metadata"]["is_nightly"] is False

    case = next(iter(document["baselines"]))
    assert case == (
        "resize_contract_area_tensor_uchar3_basic"
        "[InOutDataType=uchar3]"
        "[shape=32x1080x1920]"
        "[kernelSize=5x5]"
        "[resizeType=CONTRACT]"
        "[interpolation=AREA]"
        "[layout=NHWC]"
        "[inputKind=Tensor]"
    )


def test_run_metadata_records_device_verification_not_lock_request(monkeypatch):
    args = SimpleNamespace(
        cuda_major=13,
        git_ref="test-ref",
        run_context="mr",
        tier="basic",
    )
    monkeypatch.setattr(run_bench, "_git_revision", lambda: "deadbeef")
    monkeypatch.setattr(run_bench, "_nvidia_smi", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(run_bench, "_clock_lock_request_accepted", True)
    monkeypatch.setattr(run_bench, "_clock_lock_device_verified", False)

    metadata = run_bench._run_metadata(args, pd.DataFrame([_baseline_row()]))

    assert metadata["sm_clock_device_verified"] is False
    monkeypatch.setattr(run_bench, "_clock_lock_device_verified", True)
    assert (
        run_bench._run_metadata(args, pd.DataFrame([_baseline_row()]))[
            "sm_clock_device_verified"
        ]
        is True
    )


@pytest.mark.parametrize(
    "run_context,is_nightly", [("nightly-dev", False), ("nightly", True)]
)
def test_run_metadata_distinguishes_nightly_rehearsals(
    monkeypatch, run_context, is_nightly
):
    args = SimpleNamespace(
        cuda_major=13,
        git_ref="test-ref",
        run_context=run_context,
        tier="basic",
    )
    monkeypatch.setattr(run_bench, "_git_revision", lambda: "deadbeef")
    monkeypatch.setattr(run_bench, "_nvidia_smi", lambda *_args, **_kwargs: None)

    metadata = run_bench._run_metadata(args, pd.DataFrame([_baseline_row()]))

    assert metadata["is_nightly"] is is_nightly


def test_write_output_csv_writes_legacy_combined_csv(tmp_path):
    out = tmp_path / "bench_output.csv"
    args = SimpleNamespace(
        output=str(out),
        max_noise_pct=DEFAULT_BENCHMARK_QUALITY.max_noise_pct,
        max_perf_diff_pct=DEFAULT_BENCHMARK_QUALITY.max_perf_diff_pct,
        max_perf_diff_us=DEFAULT_BENCHMARK_QUALITY.max_perf_diff_us,
    )
    df = pd.DataFrame(
        [
            {
                **_baseline_row(gpu_us=525.75251, noise_us=1.234),
                "Device": 0,
                "GPU Noise (%)": 0.23456,
                "CPU Time (µs)": 42.4242,
                "CPU Noise (%)": 0.1234,
                "CPU Noise (µs)": 0.5678,
                "Py overhead (%)": 1.2345,
                "Py overhead (µs)": 2.3456,
            }
        ]
    )

    run_bench._write_output(args, df)

    written = pd.read_csv(out)
    assert "Device" not in written.columns
    assert written.loc[0, "config_key"] == "resize_contract_area_tensor_uchar3_basic"
    assert written.loc[0, "GPU Time (µs)"] == pytest.approx(525.75)
    assert written.loc[0, "BWUtil"] == pytest.approx(0.42)


def test_discover_operator_pairs_uses_manifest_operator_names(tmp_path):
    for name in ("resize", "gaussian"):
        _touch(tmp_path / f"bench_{name}")
        _touch(tmp_path / f"bench_{name}.py")

    config = {
        "resize_basic": {
            "benchmark": "resize",
            "tier": "basic",
        },
        "gaussian_basic": {
            "benchmark": "gaussian",
            "tier": "basic",
        },
    }

    cpp_runner = DummyRunner(
        tmp_path,
        config,
        config_keys=None,
        language="cpp",
        operators=["gaussian", "resize"],
        tiers={"basic"},
    )
    python_runner = DummyRunner(
        tmp_path,
        config,
        config_keys=None,
        language="python",
        operators=["gaussian", "resize"],
        tiers={"basic"},
    )

    pairs = run_bench.discover_operator_pairs(cpp_runner, python_runner)

    assert [
        (key, os.path.basename(cpp), os.path.basename(py)) for key, cpp, py in pairs
    ] == [
        ("gaussian_basic", "bench_gaussian", "bench_gaussian.py"),
        ("resize_basic", "bench_resize", "bench_resize.py"),
    ]


def test_discover_operator_pairs_with_config_keys_supports_mixed_operators(tmp_path):
    for name in ("resize", "gaussian"):
        _touch(tmp_path / f"bench_{name}")
        _touch(tmp_path / f"bench_{name}.py")

    config = {
        "resize_advanced": {
            "benchmark": "resize",
            "tier": "advanced",
        },
        "gaussian_advanced": {
            "benchmark": "gaussian",
            "tier": "advanced",
        },
    }
    config_keys = ["resize_advanced", "gaussian_advanced"]

    cpp_runner = DummyRunner(
        tmp_path,
        config,
        config_keys,
        language="cpp",
    )
    python_runner = DummyRunner(
        tmp_path,
        config,
        config_keys,
        language="python",
    )

    pairs = run_bench.discover_operator_pairs(
        cpp_runner, python_runner, config_keys=config_keys
    )

    assert [
        (key, os.path.basename(cpp), os.path.basename(py)) for key, cpp, py in pairs
    ] == [
        ("resize_advanced", "bench_resize", "bench_resize.py"),
        ("gaussian_advanced", "bench_gaussian", "bench_gaussian.py"),
    ]


def test_tvresize_calibrated_anchors_preserve_diagnostics_and_layout_pairs():
    configs = {
        key: entry
        for key, entry in load_bench_config().items()
        if entry.get("benchmark") == "tvresize"
    }
    anchors = {key: entry for key, entry in configs.items() if "_calibrated_" in key}
    assert len(anchors) == 8
    assert {entry["dtypes"][0] for entry in anchors.values()} == {
        "uchar3",
        "half3",
        "float3",
        "uint16",
        "int16",
        "uchar4",
        "half4",
        "float4",
    }
    diagnostics = {key: entry for key, entry in configs.items() if key not in anchors}
    for tier, expected in (("basic", 54), ("advanced", 121)):
        assert (
            sum(
                len(expected_case_keys_for_entry(key, entry))
                for key, entry in diagnostics.items()
                if entry["tier"] == tier
            )
            == expected
        )
    assert (
        sum(
            len(expected_case_keys_for_entry(key, entry))
            for key, entry in configs.items()
        )
        == 199
    )
    assert fake_planar_pairing_issues(configs) == ()
    heterogeneous = {
        key: entry
        for key, entry in configs.items()
        if key.startswith("tvresize_heterogeneous_varshape_")
    }
    assert {dtype for entry in heterogeneous.values() for dtype in entry["dtypes"]} == {
        "uchar3",
        "float3",
    }
    for entry in heterogeneous.values():
        axes = entry["string_axes"]
        count, height, width = map(int, axes["shape"][0].split("x"))
        assert count > 0 and count % 3 == 0
        assert (height, width) == (1080, 1920)
        assert axes["batchCase"] == ["ordinary", "mixed-singleton"]
    geometry_outputs = set()
    for key, entry in diagnostics.items():
        axes = entry["string_axes"]
        if key in heterogeneous:
            continue
        count, height, width = map(int, axes["shape"][0].split("x"))
        assert count >= 1
        assert (height, width) == (1080, 1920)
        if key.startswith("tvresize_geometry_"):
            assert len(axes["outputShape"]) == 1
            geometry_outputs.update(axes["outputShape"])
        else:
            assert axes["outputShape"] == ["720x1280"]
    assert geometry_outputs == {
        "1080x1920",
        "1653x2950",
        "507x2950",
        "1653x1405",
        "1x1",
    }
    for key, entry in anchors.items():
        assert entry["tier"] == "advanced"
        assert len(entry["dtypes"]) == 1
        assert len(expected_case_keys_for_entry(key, entry)) == 3
        axes = entry["string_axes"]
        assert axes["layout"] == ["NHWC", "NCHW", "NCHW_FAKE"]
        assert len(axes["shape"]) == 1
        count, height, width = map(int, axes["shape"][0].split("x"))
        assert count >= 1
        assert (height, width) == (1080, 1920)
        assert axes["outputShape"] == ["720x1280"]
        assert axes["inputKind"] == ["Tensor"]
        assert axes["batchCase"] == ["uniform"]
        assert axes["interpolation"] == ["LINEAR"]
        assert entry["int64_axes"]["antialias"] == [1]
        dtype = entry["dtypes"][0]
        channels = int(dtype[-1]) if dtype[-1] in "34" else 1
        assert entry["int64_axes"]["numChannels"] == [channels]


def test_tvresize_recalibration_preserves_dtype_and_layout_workloads():
    configs = {
        key: entry
        for key, entry in load_bench_config().items()
        if entry.get("benchmark") == "tvresize"
    }
    batches_by_workload = {}
    lane_bytes = {
        "uchar": 1,
        "uint8": 1,
        "half": 2,
        "float": 4,
        "uint16": 2,
        "int16": 2,
    }
    for key, entry in configs.items():
        assert len(entry["dtypes"]) == 1
        dtype = entry["dtypes"][0]
        base_dtype = dtype[:-1] if dtype[-1] in "34" else dtype
        axes = entry["string_axes"]
        assert len(axes["shape"]) == 1
        count, height, width = map(int, axes["shape"][0].split("x"))
        for output, kind, batch_case, interpolation, channels, antialias in product(
            axes["outputShape"],
            axes["inputKind"],
            axes["batchCase"],
            axes["interpolation"],
            entry["int64_axes"]["numChannels"],
            entry["int64_axes"]["antialias"],
        ):
            workload = (
                "_calibrated_" in key,
                dtype,
                height,
                width,
                output,
                kind,
                batch_case,
                interpolation,
                channels,
                antialias,
            )
            assert batches_by_workload.setdefault(workload, count) == count
            if kind == "Tensor":
                output_height, output_width = map(int, output.split("x"))
                pixels = max(height * width, output_height * output_width)
                assert count * pixels * channels * lane_bytes[base_dtype] < 2**31
        assert entry["warmup_iterations"] == 10


def test_invert_default_axis_does_not_expand_old_configs():
    old = {"invert_uchar3_basic": {"benchmark": "invert", "dtypes": ["uchar3"]}}
    args = generate_axis_args("invert_uchar3_basic", old)
    assert "inputRange=nonnegative" in args
    explicit = {"invert": {"string_axes": {"inputRange": ["nonnegative", "mixed"]}}}
    args = generate_axis_args("invert", explicit)
    assert "inputRange=[nonnegative,mixed]" in args
    assert "inputRange=nonnegative" not in args


@pytest.mark.parametrize(
    "explicit",
    [
        {},
        {"scaleMode": ["scale", "stddev"]},
        {"storage": ["Offset2Pitch2"]},
        {"paramKind": ["RowBaseColumnScale"]},
    ],
)
def test_normalize_optional_axes_preserve_implicit_defaults(explicit):
    entry = {"benchmark": "normalize", "string_axes": explicit}
    args = generate_axis_args("normalize_case", {"normalize_case": entry})
    for axis, default in (
        ("paramKind", "PerSample"),
        ("scaleMode", "stddev"),
        ("storage", "Native"),
    ):
        actual = [arg for arg in args if arg.startswith(f"{axis}=")]
        values = explicit.get(axis, [default])
        expected = values[0] if len(values) == 1 else f"[{','.join(values)}]"
        assert actual == [f"{axis}={expected}"]


@pytest.mark.parametrize(
    "explicit",
    [
        {},
        {"storage": ["Offset2Pitch2"]},
        {"paramKind": ["RowBaseColumnScale"]},
        {"scaleMode": ["scale"]},
    ],
)
def test_normalize_results_keep_only_declared_optional_axes(tmp_path, explicit):
    output = tmp_path / "normalize.nvbench"
    row = {
        "Benchmark": "normalize",
        "Skipped": "No",
        "BWUtil": 0.5,
        "GPU Time (sec)": 0.001,
        "Device Name": "test GPU",
        "paramKind": "PerSample",
        "scaleMode": "stddev",
        "storage": "Native",
    }
    row.update({axis: values[0] for axis, values in explicit.items()})
    pd.DataFrame([row]).to_csv(output, index=False)
    runner = DummyRunner(
        tmp_path,
        {
            "normalize_case": {
                "benchmark": "normalize",
                "tier": "advanced",
                "string_axes": explicit,
            }
        },
        None,
        "cpp",
    )
    assert runner.collect_results(output.name, "normalize_case")
    result = runner.results[0]
    for axis in ("paramKind", "scaleMode", "storage"):
        assert (axis in result.columns) == (axis in explicit)
