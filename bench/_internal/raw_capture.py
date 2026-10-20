# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in native NVBench output retention for each actual subprocess."""

import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path


class CaptureError(RuntimeError):
    """Opted-in evidence could not be retained."""


METADATA_FILE = "metadata.json"
NATIVE_JSON_FILE = "nvbench.json"
NATIVE_CSV_FILE = "nvbench.csv"


def write_metadata(folder, record):
    try:
        (folder / METADATA_FILE).write_text(json.dumps(record, indent=2) + "\n")
    except OSError as exc:
        raise CaptureError(f"Cannot write raw metadata in {folder}: {exc}") from exc


def _validate_state_output(folder, state):
    summaries = state.get("summaries", [])
    files = [item for item in summaries if item.get("hint") == "file/sample_times"]
    samples = [item for item in summaries if item.get("tag") == "nv/cold/sample_size"]
    if samples and not files:
        raise ValueError("timed state has no native sample-times file")
    for item in files:
        values = {entry["name"]: entry["value"] for entry in item["data"]}
        binary = Path(values["filename"])
        if not binary.is_absolute():
            binary = folder / binary
        if not binary.resolve().is_relative_to(folder.resolve()):
            raise ValueError("sample file is outside the captured attempt")
        if binary.stat().st_size != int(values["size"]) * 4:
            raise ValueError(f"missing or incomplete native sample file: {binary}")


def validate_native_output(folder):
    try:
        document = json.loads((folder / NATIVE_JSON_FILE).read_text())
        if not (folder / NATIVE_CSV_FILE).is_file():
            raise ValueError("missing native CSV")
        for benchmark in document["benchmarks"]:
            for state in benchmark["states"]:
                _validate_state_output(folder, state)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        record = json.loads((folder / METADATA_FILE).read_text())
        record["capture_error"] = str(exc)
        write_metadata(folder, record)
        raise CaptureError(
            f"Incomplete native JSONBIN output in {folder}: {exc}"
        ) from exc


def captured_run(command, output_path, raw_output_dir, metadata, **kwargs):
    if not raw_output_dir:
        return subprocess.run(command, **kwargs), None
    try:
        root = Path(raw_output_dir).resolve()
        root.mkdir(parents=True, exist_ok=True)
        folder = Path(tempfile.mkdtemp(prefix="attempt-", dir=root))
        # The archive reader may run outside the container with a different UID.
        folder.chmod(0o755)
    except OSError as exc:
        raise CaptureError(f"Cannot create raw attempt directory: {exc}") from exc
    command = list(command)
    csv_index = len(command) - 1 - command[::-1].index("--csv")
    command[csv_index + 1] = str(folder / NATIVE_CSV_FILE)
    command.extend(["--jsonbin", str(folder / NATIVE_JSON_FILE)])
    record = dict(
        metadata,
        command=command,
        cwd=(
            str(Path(kwargs["cwd"]).resolve())
            if kwargs.get("cwd") is not None
            else os.getcwd()
        ),
        started_at=datetime.now(timezone.utc).isoformat(),
        returncode=None,
        exception=None,
        clock_outcome=None,
    )
    record["environment"] = {
        key: os.environ.get(key)
        for key in (
            "GIT_SHA",
            "COMMIT_SHA_FULL",
            "CI_COMMIT_SHA",
            "BUILD_NUMBER",
            "BUILD_URL",
            "NODE_NAME",
            "K8S_NODE_NAME",
            "BASELINE_SKU",
            "BURNIN_DEVICE",
            "BURNIN_IDX",
            "CUDA_VERSION",
        )
    }
    record["benchmark_environment"] = {
        key: value
        for key, value in (kwargs.get("env") or {}).items()
        if key.startswith(("CVCUDA_BENCH_", "NVBENCH_", "BENCH_"))
    }
    write_metadata(folder, record)
    stdout = stderr = ""
    try:
        result = subprocess.run(command, **kwargs)
        record["returncode"] = result.returncode
        stdout, stderr = result.stdout or "", result.stderr or ""
    except BaseException as exc:
        record["exception"] = {"type": type(exc).__name__, "message": str(exc)}
        stdout = getattr(exc, "stdout", None) or ""
        stderr = getattr(exc, "stderr", None) or ""
        raise
    finally:
        try:
            for name, text in (("stdout.txt", stdout), ("stderr.txt", stderr)):
                (folder / name).write_bytes(
                    text if isinstance(text, bytes) else text.encode()
                )
            record["finished_at"] = datetime.now(timezone.utc).isoformat()
            write_metadata(folder, record)
            if (folder / NATIVE_CSV_FILE).exists():
                shutil.copyfile(folder / NATIVE_CSV_FILE, output_path)
        except OSError as exc:
            raise CaptureError(f"Cannot retain raw outputs in {folder}: {exc}") from exc
    return result, folder


def record_clock(folder, outcome, detail):
    if folder is not None:
        try:
            record = json.loads((folder / METADATA_FILE).read_text())
            record.update(clock_outcome=outcome, clock_detail=detail)
            write_metadata(folder, record)
        except (OSError, ValueError) as exc:
            raise CaptureError(
                f"Cannot retain clock outcome in {folder}: {exc}"
            ) from exc
