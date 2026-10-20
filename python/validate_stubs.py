# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reject generated Python stubs that are not valid Python syntax."""

import ast
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} STUB_DIRECTORY")

    stub_dir = Path(sys.argv[1])
    stubs = sorted(stub_dir.rglob("*.pyi"))
    if not stubs:
        raise SystemExit(f"no Python stubs found under {stub_dir}")

    for stub in stubs:
        ast.parse(stub.read_text(encoding="utf-8"), filename=str(stub))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
