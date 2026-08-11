# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the shared C++/Python benchmark warmup policy."""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

import pytest

import run_bench
from _internal.warmup import WARMUP_CAP_ENV, resolve_warmup_iterations


BENCH_DIR = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def warmup_policy_probe(tmp_path_factory):
    source = tmp_path_factory.mktemp("warmup-policy") / "probe.cpp"
    binary = source.with_suffix("")
    source.write_text(
        """
#include "cpp/WarmupPolicy.hpp"

#include <exception>
#include <iostream>

int main(int argc, char **argv)
{
    try
    {
        std::cout << benchutils::resolve_warmup_iterations(std::stoi(argv[1])) << '\\n';
        return 0;
    }
    catch (const std::exception &err)
    {
        std::cerr << err.what() << '\\n';
        return 2;
    }
}
"""
    )
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    subprocess.run(
        [*compiler, "-std=c++17", "-I", str(BENCH_DIR), str(source), "-o", str(binary)],
        check=True,
        capture_output=True,
        text=True,
    )
    return binary


def _run_probe(binary: Path, configured: int, cap: str | None):
    env = os.environ.copy()
    env.pop(WARMUP_CAP_ENV, None)
    if cap is not None:
        env[WARMUP_CAP_ENV] = cap
    return subprocess.run(
        [str(binary), str(configured)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    ("cap", "configured", "expected"),
    [
        (None, 200, 200),
        ("0", 200, 0),
        ("50", 0, 0),
        ("50", 20, 20),
        ("50", 50, 50),
        ("50", 51, 50),
        ("50", 200, 50),
        ("500", 200, 200),
    ],
)
def test_cpp_and_python_resolve_warmups_identically(
    warmup_policy_probe, cap, configured, expected
):
    env = {} if cap is None else {WARMUP_CAP_ENV: cap}
    assert resolve_warmup_iterations(configured, env) == expected

    result = _run_probe(warmup_policy_probe, configured, cap)
    assert result.returncode == 0, result.stderr
    assert int(result.stdout) == expected


@pytest.mark.parametrize("cap", ["", "-1", "abc", "1.5", "50junk", "2147483648"])
def test_cpp_and_python_reject_the_same_invalid_caps(warmup_policy_probe, cap):
    with pytest.raises(ValueError, match=WARMUP_CAP_ENV):
        resolve_warmup_iterations(200, {WARMUP_CAP_ENV: cap})

    result = _run_probe(warmup_policy_probe, 200, cap)
    assert result.returncode == 2
    assert result.stdout == ""
    assert WARMUP_CAP_ENV in result.stderr


class _Runner(run_bench.BenchmarkRunner):
    def build_command(self, benchmark_path, extra_args, output_file, config_key=None):
        return []


def test_runner_propagates_cap_and_clears_stale_environment(monkeypatch, tmp_path):
    monkeypatch.setenv(WARMUP_CAP_ENV, "999")
    capped = _Runner("bench_", str(tmp_path), [], "cpp", warmup_cap=50)
    uncapped = _Runner("bench_", str(tmp_path), [], "cpp")

    assert capped.benchmark_env()[WARMUP_CAP_ENV] == "50"
    assert WARMUP_CAP_ENV not in uncapped.benchmark_env()


def test_python_runner_preserves_cap_while_extending_environment(tmp_path):
    runner = object.__new__(run_bench.PythonBenchmarkRunner)
    runner.bench_folder = str(tmp_path)
    runner.warmup_cap = 50

    assert runner.benchmark_env()[WARMUP_CAP_ENV] == "50"
