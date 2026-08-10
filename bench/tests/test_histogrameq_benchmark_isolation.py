# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for HistogramEq benchmark process isolation."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path


BENCH_DIR = Path(__file__).resolve().parent.parent


def test_histogrameq_container_states_use_separate_config_keys():
    config_path = BENCH_DIR / "config/operators/histogrameq.json"
    configs = json.loads(config_path.read_text())["configs"]

    combined = {
        key: entry["string_axes"]["inputKind"]
        for key, entry in configs.items()
        if len(entry["string_axes"]["inputKind"]) != 1
    }

    assert combined == {}

    entries = list(configs.values())

    def signature(entry):
        axes = tuple(
            (name, tuple(values))
            for name, values in entry["string_axes"].items()
            if name != "inputKind"
        )
        return (
            entry["tier"],
            tuple(entry["dtypes"]),
            axes,
            entry.get("warmup_iterations"),
        )

    tensor_signatures = Counter(
        signature(entry)
        for entry in entries
        if entry["string_axes"]["inputKind"] == ["Tensor"]
    )
    varshape_signatures = Counter(
        signature(entry)
        for entry in entries
        if entry["string_axes"]["inputKind"] == ["VarShape"]
    )

    assert tensor_signatures
    assert varshape_signatures
    assert varshape_signatures - tensor_signatures == {}
