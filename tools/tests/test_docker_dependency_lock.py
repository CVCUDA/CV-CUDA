# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT / "docker/requirements.sys_python.lock"


@pytest.mark.parametrize(
    "dockerfile", ["Dockerfile.builder.deps", "Dockerfile.devel.deps"]
)
def test_system_python_install_requires_locked_hashes(dockerfile):
    instructions = (ROOT / "docker" / dockerfile).read_text().replace("\\\n", " ")
    installs = [
        line
        for line in instructions.splitlines()
        if line.startswith("RUN ") and "pip install" in line
    ]
    locked = [line for line in installs if "requirements.sys_python.lock" in line]
    assert len(locked) == 1
    assert "--require-hashes" in locked[0]
    if dockerfile == "Dockerfile.devel.deps":
        assert installs[-1] == locked[0]
    assert "requirements.build.sys_python.txt" not in instructions
    assert "requirements.docs.txt" not in instructions


def test_lock_pins_and_hashes_every_dependency_and_preserves_direct_pins():
    records = [
        line.strip()
        for line in LOCK.read_text().replace("\\\n", " ").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    requirements = []
    for record in records:
        requirement = Requirement(record.split("--hash=", 1)[0].strip())
        assert requirement.url is None
        assert len(list(requirement.specifier)) == 1
        assert next(iter(requirement.specifier)).operator == "=="
        assert re.search(r"--hash=sha256:[0-9a-f]{64}(?:\s|$)", record)
        requirements.append(requirement)
    locked_versions = {
        canonicalize_name(requirement.name): str(requirement.specifier)
        for requirement in requirements
    }
    for source in (
        ROOT / "docker/requirements.build.sys_python.txt",
        ROOT / "docs/requirements.docs.txt",
    ):
        for line in source.read_text().splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            requirement = Requirement(line)
            assert locked_versions[canonicalize_name(requirement.name)] == str(
                requirement.specifier
            )
