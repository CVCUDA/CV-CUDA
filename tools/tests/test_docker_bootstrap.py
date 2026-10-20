# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import re
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def test_pip_bootstrap_reuses_download_for_all_interpreters(tmp_path):
    dockerfile = (ROOT / "docker/Dockerfile.devel.deps").read_text()
    instructions = dockerfile.replace("\\\n", " ").splitlines()
    commands = [
        line.removeprefix("RUN ")
        for line in instructions
        if line.startswith("RUN ") and "get-pip.py" in line
    ]
    assert len(commands) == 2
    download = next(line for line in instructions if line.startswith("ADD "))
    destination = str(Path(download.split()[-1]) / Path(download.split()[-2]).name)
    bootstrap = tmp_path / Path(destination).name
    bootstrap.write_text("verified bootstrap\n")
    interpreter_log = tmp_path / "interpreters"
    executable_dir = tmp_path / "bin"
    executable_dir.mkdir()
    for interpreter in ("python3", "python3.10", "python3.14t"):
        executable = executable_dir / interpreter
        executable.write_text(
            '#!/bin/sh\ntest -f "$1" || exit 1\n'
            'printf "%s\\n" "${0##*/}" >> "$INTERPRETER_LOG"\n'
        )
        executable.chmod(0o755)
    environment = {
        **os.environ,
        "PATH": f"{executable_dir}:{os.environ['PATH']}",
        "INTERPRETER_LOG": str(interpreter_log),
        "PYTHON_VERSIONS": "3.10 3.14t",
    }
    for command in commands:
        command = command.replace(destination, str(bootstrap))
        result = subprocess.run(
            ["sh", "-c", command], env=environment, capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr
    assert interpreter_log.read_text().splitlines() == [
        "python3",
        "python3.10",
        "python3.14t",
    ]
    assert not bootstrap.exists()


def test_pip_bootstrap_source_is_immutable_and_digest_is_pinned():
    dockerfile = (ROOT / "docker/Dockerfile.devel.deps").read_text()
    assert re.search(r"^ARG GET_PIP_COMMIT=[0-9a-f]{40}$", dockerfile, re.MULTILINE)
    assert re.search(r"^ARG GET_PIP_SHA256=[0-9a-f]{64}$", dockerfile, re.MULTILINE)
    assert (
        "https://raw.githubusercontent.com/pypa/get-pip/${GET_PIP_COMMIT}/public/get-pip.py"
        in dockerfile
    )
    assert "ADD --checksum=sha256:${GET_PIP_SHA256}" in dockerfile


@pytest.mark.parametrize("ubuntu", ["22.04", "26.04"])
def test_development_packages_have_no_empty_arguments(tmp_path, ubuntu):
    dockerfile = (ROOT / "docker/Dockerfile.devel.deps").read_text()
    dockerfile = "\n".join(
        line for line in dockerfile.splitlines() if not line.lstrip().startswith("#")
    )
    instructions = dockerfile.replace("\\\n", " ").splitlines()
    command = next(
        line.removeprefix("RUN ")
        for line in instructions
        if line.startswith("RUN ") and "CLANG_VERSION=14" in line
    )
    arguments = tmp_path / "arguments"
    executable_dir = tmp_path / "bin"
    executable_dir.mkdir()
    scripts = {
        "lsb_release": 'printf "%s\\n" "$TEST_UBUNTU"',
        "apt-fast": 'printf "%s\\n" "$@" >> "$APT_ARGUMENTS"',
        "rm": "exit 0",
    }
    for name, script in scripts.items():
        executable = executable_dir / name
        executable.write_text(f"#!/bin/sh\n{script}\n")
        executable.chmod(0o755)
    result = subprocess.run(
        ["sh", "-c", command],
        env={
            **os.environ,
            "PATH": f"{executable_dir}:{os.environ['PATH']}",
            "TEST_UBUNTU": ubuntu,
            "APT_ARGUMENTS": str(arguments),
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    packages = arguments.read_text().splitlines()
    assert "" not in packages
    assert ("libjitterentropy3-dev" in packages) == (ubuntu == "26.04")
    assert ("clang-14" if ubuntu == "22.04" else "clang-18") in packages
