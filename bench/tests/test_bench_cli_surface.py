# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for the supported benchmark command-line tools."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

BENCH_DIR = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("script", ("update_baseline.py", "validate_baselines.py"))
def test_baseline_maintenance_commands_are_internal(script):
    assert not (BENCH_DIR / script).exists()
    assert (BENCH_DIR / "_internal" / script).is_file()


@pytest.mark.parametrize(
    ("script", "signatures"),
    [
        ("run_bench.py", ("BENCH_FOLDER",)),
        ("compare_wheels.py", ("REFERENCE_WHEEL", "CANDIDATE_WHEEL")),
        ("compare_to_baseline.py", ("--current", "JSON")),
    ],
)
def test_supported_command_help_is_self_contained(tmp_path, script, signatures):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    result = subprocess.run(
        [sys.executable, str(BENCH_DIR / script), "--help"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert all(signature in result.stdout for signature in signatures)
    assert "Examples:" in result.stdout
    assert "Exit status:" in result.stdout
