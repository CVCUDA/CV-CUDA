# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep the recalibrated control independent of the unchanged scalar fixtures."""

import json
from pathlib import Path

from _internal.baselines import expected_case_keys_for_entry


def test_normalize_u8_control_replaces_only_the_short_tensor_shape():
    path = Path(__file__).resolve().parents[1] / "config/operators/normalize.json"
    configs = json.loads(path.read_text())["configs"]
    control_key = "normalize_scalar_u8_nhwc_b512_1080p_advanced"
    control = configs[control_key]
    assert control["tier"] == "advanced"
    assert control["warmup_iterations"] == 200
    assert control["dtypes"] == ["uint8"]
    assert control["string_axes"] == {
        "shape": ["512x1080x1920"],
        "layout": ["NHWC"],
        "inputKind": ["Tensor"],
    }
    assert len(expected_case_keys_for_entry(control_key, control)) == 1
    all_cases = [
        case
        for key, entry in configs.items()
        for case in expected_case_keys_for_entry(key, entry)
    ]
    assert len(all_cases) == 69
    assert not any(
        all(
            token in case
            for token in (
                "[InOutDataType=uint8]",
                "[shape=128x1080x1920]",
                "[layout=NHWC]",
                "[inputKind=Tensor]",
            )
        )
        for case in all_cases
    )
    preserved = {
        "normalize_scalar_advanced": 4,
        "normalize_scalar_u8_varshape_advanced": 2,
        "normalize_scalar_u8_tensor_4k_advanced": 1,
    }
    for key, count in preserved.items():
        assert len(expected_case_keys_for_entry(key, configs[key])) == count
        assert configs[key]["warmup_iterations"] == 200
