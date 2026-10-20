# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import run_bench
from _internal.raw_capture import CaptureError, captured_run, record_clock


def runner(tmp_path, monkeypatch, raw=True, language="cpp"):
    cls = (
        run_bench.CppBenchmarkRunner
        if language == "cpp"
        else run_bench.PythonBenchmarkRunner
    )
    obj = cls(str(tmp_path), [])
    obj.raw_output_dir = str(tmp_path / "raw") if raw else None
    obj.build_command = lambda *a, **k: ["fake", "--csv", str(tmp_path / a[2])]
    obj._check_noise_from_scratch = lambda _: False
    monkeypatch.chdir(tmp_path)
    return obj


def native_fixture(cmd):
    folder = Path(cmd[cmd.index("--jsonbin") + 1]).parent
    binary = folder / "nvbench.json-bin/0.bin"
    binary.parent.mkdir()
    binary.write_bytes(b"\x00" * 8)
    summaries = [
        {"tag": "nv/cold/sample_size"},
        {
            "hint": "file/sample_times",
            "data": [
                {"name": "filename", "value": str(binary)},
                {"name": "size", "value": "2"},
            ],
        },
    ]
    (folder / "nvbench.json").write_text(
        json.dumps({"benchmarks": [{"states": [{"summaries": summaries}]}]})
    )
    (folder / "nvbench.csv").write_text("native csv")
    return folder


@pytest.mark.parametrize("language", ["cpp", "python"])
@pytest.mark.parametrize("failure", ["exit", "timeout", "launch", "clock", "interrupt"])
def test_failed_attempt_retained(tmp_path, monkeypatch, failure, language):
    obj = runner(tmp_path, monkeypatch, language=language)

    def fake(cmd, **kwargs):
        native_fixture(cmd)
        if failure == "timeout":
            raise subprocess.TimeoutExpired(
                cmd, 1, output=b"partial out", stderr=b"partial err"
            )
        if failure == "launch":
            raise OSError("launch failed")
        if failure == "interrupt":
            raise KeyboardInterrupt()
        return SimpleNamespace(returncode=1, stdout="out", stderr="err")

    monkeypatch.setattr(run_bench.subprocess, "run", fake)
    monkeypatch.setattr(
        run_bench,
        "_classify_clock_outcome",
        lambda *a, **k: (
            run_bench._CLOCK_OUTCOME_MISSING_ACK if failure == "clock" else "failed",
            "bad",
        ),
    )
    if failure == "interrupt":
        with pytest.raises(KeyboardInterrupt):
            obj.run_benchmark("bench_invert", [], "test")
    elif failure == "clock":
        with pytest.raises(SystemExit):
            obj.run_benchmark("bench_invert", [], "test")
    else:
        obj.run_benchmark("bench_invert", [], "test")
    folders = list((tmp_path / "raw").glob("attempt-*"))
    assert len(folders) == 1
    assert (folders[0] / "nvbench.csv").read_text() == "native csv"
    assert (folders[0] / "nvbench.json-bin/0.bin").stat().st_size == 8
    meta = json.loads((folders[0] / "metadata.json").read_text())
    assert meta["returncode"] == (
        None if failure in ("launch", "timeout", "interrupt") else 1
    )
    assert (
        meta["exception"]
        if failure in ("launch", "timeout", "interrupt")
        else meta["clock_outcome"]
    )
    assert meta["language"] == language
    assert (folders[0] / "stdout.txt").exists()
    assert (folders[0] / "stderr.txt").exists()
    if failure == "timeout":
        assert (folders[0] / "stdout.txt").read_bytes() == b"partial out"
        assert (folders[0] / "stderr.txt").read_bytes() == b"partial err"


@pytest.mark.parametrize("cwd_kind", ["default", "none", "str", "path"])
def test_capture_subprocess_cwd(tmp_path, monkeypatch, cwd_kind):
    monkeypatch.chdir(tmp_path)
    child = tmp_path / "child"
    child.mkdir()
    kwargs = {
        "default": {},
        "none": {"cwd": None},
        "str": {"cwd": str(child)},
        "path": {"cwd": Path("child")},
    }[cwd_kind]

    def fake(cmd, **actual_kwargs):
        assert actual_kwargs == kwargs
        native_fixture(cmd)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(run_bench.subprocess, "run", fake)
    _, folder = captured_run(
        ["fake", "--csv", str(tmp_path / "scratch")],
        tmp_path / "scratch",
        tmp_path / "raw",
        {},
        **kwargs,
    )
    assert json.loads((folder / "metadata.json").read_text())["cwd"] == str(
        child if cwd_kind in ("str", "path") else tmp_path
    )


def test_capture_default_off_exact_passthrough(tmp_path, monkeypatch):
    command = ["fake", "--csv", "scratch"]
    kwargs = {
        "shell": False,
        "capture_output": True,
        "timeout": 300,
        "env": {"BENCH_X": "x"},
    }
    expected = SimpleNamespace(returncode=1, stdout="out", stderr="err")
    calls = []

    def fake_run(cmd, **kw):
        calls.append((cmd, kw))
        return expected

    monkeypatch.setattr(run_bench.subprocess, "run", fake_run)
    result, folder = captured_run(command, "scratch", None, {}, **kwargs)
    assert result is expected
    assert folder is None
    ((actual_command, actual_kwargs),) = calls
    assert actual_command is command
    assert actual_kwargs == kwargs
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("language", ["cpp", "python"])
def test_unique_clock_attempts_and_cleanup(tmp_path, monkeypatch, language):
    obj = runner(tmp_path, monkeypatch, language=language)
    monkeypatch.setattr(run_bench, "_clock_lock_required_mhz", 1005)
    monkeypatch.setattr(
        run_bench, "_reapply_required_sm_clock", lambda: run_bench._CLOCK_LOCK_OK
    )
    monkeypatch.setattr(
        run_bench,
        "_classify_clock_outcome",
        lambda *a, **k: (run_bench._CLOCK_OUTCOME_MISMATCH, "mismatch"),
    )

    def fake(cmd, **kwargs):
        native_fixture(cmd)
        return SimpleNamespace(returncode=1, stdout="out", stderr="err")

    monkeypatch.setattr(run_bench.subprocess, "run", fake)
    with pytest.raises(SystemExit) as caught:
        obj.run_benchmark("bench_invert", [], "test")
    assert caught.value.code == run_bench._CLOCK_INELIGIBLE_EXIT_CODE
    folders = list((tmp_path / "raw").glob("attempt-*"))
    assert len(folders) == run_bench._CLOCK_VERIFY_MAX_ATTEMPTS
    assert not (tmp_path / run_bench.get_output_filename("test", language)).exists()
    assert {
        json.loads((p / "metadata.json").read_text())["clock_attempt"] for p in folders
    } == set(range(1, run_bench._CLOCK_VERIFY_MAX_ATTEMPTS + 1))
    assert all((p / "nvbench.json-bin/0.bin").exists() for p in folders)


@pytest.mark.parametrize("missing", ["json", "binary", "truncated", None])
def test_native_output_required_on_success(tmp_path, monkeypatch, missing):
    obj = runner(tmp_path, monkeypatch)
    monkeypatch.setattr(
        run_bench,
        "_classify_clock_outcome",
        lambda *a, **k: (run_bench._CLOCK_OUTCOME_SUCCESS, None),
    )

    def fake(cmd, **kwargs):
        folder = native_fixture(cmd)
        if missing == "json":
            (folder / "nvbench.json").unlink()
        elif missing == "binary":
            (folder / "nvbench.json-bin/0.bin").unlink()
        elif missing == "truncated":
            (folder / "nvbench.json-bin/0.bin").write_bytes(b"")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(run_bench.subprocess, "run", fake)
    if missing:
        with pytest.raises(CaptureError):
            obj.run_benchmark("bench_invert", [], "test")
    else:
        assert obj.run_benchmark("bench_invert", [], "test")[0]
    meta = json.loads(
        next((tmp_path / "raw").glob("attempt-*/metadata.json")).read_text()
    )
    assert meta["returncode"] == 0
    assert meta["clock_outcome"] == run_bench._CLOCK_OUTCOME_SUCCESS


def test_capture_attempt_readable_by_archive_reader(tmp_path, monkeypatch):
    root = tmp_path / "raw"
    root.mkdir(mode=0o750)
    root_mode = root.stat().st_mode

    def fake(cmd, **kwargs):
        folder = native_fixture(cmd)
        assert folder.stat().st_mode & 0o777 == 0o755
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(run_bench.subprocess, "run", fake)
    _, folder = captured_run(
        ["fake", "--csv", "scratch"], tmp_path / "scratch", root, {}
    )
    assert folder.stat().st_mode & 0o777 == 0o755
    assert root.stat().st_mode == root_mode


def test_capture_attempt_permission_failure(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        run_bench.subprocess,
        "run",
        lambda *a, **k: calls.append(a)
        or SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    def fail_chmod(*args, **kwargs):
        raise OSError("permission denied")

    monkeypatch.setattr(Path, "chmod", fail_chmod)
    with pytest.raises(CaptureError, match="permission denied"):
        captured_run(
            ["fake", "--csv", "scratch"],
            tmp_path / "scratch",
            tmp_path / "raw",
            {},
        )
    assert calls == []


def test_capture_directory_failure(tmp_path):
    root = tmp_path / "raw"
    root.write_text("not a directory")
    with pytest.raises(CaptureError):
        captured_run(["fake", "--csv", "scratch"], "scratch", root, {})


def test_clock_metadata_failure_propagates(tmp_path, monkeypatch):
    from _internal import raw_capture

    obj = runner(tmp_path, monkeypatch)
    monkeypatch.setattr(
        run_bench.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=1, stdout="", stderr=""),
    )

    def fail_record_clock(*args):
        raise CaptureError("full")

    monkeypatch.setattr(raw_capture, "record_clock", fail_record_clock)
    with pytest.raises(CaptureError):
        obj.run_benchmark("bench_invert", [], "test")


def test_record_clock_write_failure(tmp_path, monkeypatch):
    (tmp_path / "metadata.json").write_text("{}")

    def fail_write(*args, **kwargs):
        raise OSError("full")

    monkeypatch.setattr(Path, "write_text", fail_write)
    with pytest.raises(CaptureError):
        record_clock(tmp_path, "failed", None)


@pytest.mark.parametrize("language", ["cpp", "python"])
@pytest.mark.parametrize("raw", [False, True])
def test_validation_retries_keep_policy_and_all_attempts(
    tmp_path, monkeypatch, language, raw
):
    import pandas as pd

    obj = runner(tmp_path, monkeypatch, raw=raw, language=language)
    calls = []

    def fake(cmd, **kwargs):
        calls.append(cmd)
        if raw:
            native_fixture(cmd)
        else:
            Path(cmd[cmd.index("--csv") + 1]).write_text("native csv")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    def collect(*args):
        obj.results.append(pd.DataFrame([{"Status": "FAIL (noise)"}]))
        return False

    monkeypatch.setattr(run_bench.subprocess, "run", fake)
    monkeypatch.setattr(
        run_bench,
        "_classify_clock_outcome",
        lambda *a, **k: (run_bench._CLOCK_OUTCOME_SUCCESS, None),
    )
    monkeypatch.setattr(obj, "collect_results", collect)
    cls = "CppBenchmarkRunner" if language == "cpp" else "PythonBenchmarkRunner"
    monkeypatch.setattr(run_bench, cls, lambda *a, **k: obj)
    pair = (
        "test",
        "bench_invert" if language == "cpp" else None,
        "bench_invert.py" if language == "python" else None,
    )
    monkeypatch.setattr(run_bench, "discover_operator_pairs", lambda *a: [pair])
    monkeypatch.setattr(run_bench, "_query_gpu_name", lambda: "test GPU")
    monkeypatch.setattr(
        run_bench.ResultsProcessor, "process", lambda self, dfs, **k: dfs[0]
    )
    monkeypatch.setattr(run_bench, "print_parity_summary_table", lambda *a: None)
    args = SimpleNamespace(
        bench_folder=str(tmp_path),
        operators=[],
        keep_outputs=False,
        max_noise_pct=10,
        config_file=None,
        tiers=None,
        config_keys=None,
        warmup_cap=None,
        lang=language,
        max_perf_diff_pct=10,
        max_perf_diff_us=100,
        max_retries=2,
        bench_args=[],
        raw_output_dir=obj.raw_output_dir,
    )
    run_bench._run_paired_benchmarks(args)
    assert len(calls) == 3
    assert len(obj.results) == 1
    assert obj.results[0].iloc[0]["Status"] == "FAIL (noise)"
    folders = list((tmp_path / "raw").glob("attempt-*"))
    assert len(folders) == (3 if raw else 0)
    assert all((p / "nvbench.json-bin/0.bin").exists() for p in folders)


def test_capture_redirects_managed_last_csv_argument(tmp_path, monkeypatch):
    forwarded = tmp_path / "user.csv"
    managed = tmp_path / "runner.csv"
    command = ["fake", "--csv", str(forwarded), "--csv", str(managed)]

    def fake(actual, **kwargs):
        csv_positions = [
            index for index, token in enumerate(actual) if token == "--csv"
        ]
        destination = Path(actual[csv_positions[-1] + 1])
        assert destination.parent.name.startswith("attempt-")
        assert actual[csv_positions[0] + 1] == str(forwarded)
        native_fixture(actual)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(run_bench.subprocess, "run", fake)
    _, folder = captured_run(command, managed, tmp_path / "raw", {})
    assert managed.read_text() == (folder / "nvbench.csv").read_text()
    assert command[-1] == str(managed)
