# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from channel_copy_bench import channel_copy  # noqa: E402
from python_bench_utils import run_benchmark  # noqa: E402


def split(state):
    return channel_copy(state, split=True)


if __name__ == "__main__":
    run_benchmark("split", split)
