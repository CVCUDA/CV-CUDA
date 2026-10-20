# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from _internal.baselines import fake_planar_pairing_issues
from config.load_config import load_bench_config


@pytest.mark.parametrize(
    "dtype,shape", [("float3", "8x720x1280"), ("uchar3", "16x720x1280")]
)
def test_jointbilateralfilter_advanced_fake_planar_has_native_tensor_pair(dtype, shape):
    configs = {
        key: entry
        for key, entry in load_bench_config().items()
        if entry.get("benchmark") == "jointbilateralfilter"
    }
    native = configs[f"jointbilateralfilter_{dtype}_planar_nchw_advanced"]
    fake = configs[f"jointbilateralfilter_{dtype}_fakeplanar_nchw_advanced"]

    assert native["tier"] == fake["tier"] == "advanced"
    assert native["dtypes"] == fake["dtypes"] == [dtype]
    assert native["string_axes"] == {
        **fake["string_axes"],
        "layout": ["NCHW"],
    }
    assert native["string_axes"]["shape"] == [shape]
    assert native["string_axes"]["inputKind"] == ["Tensor"]
    for field in ("int64_axes", "float64_axes", "warmup_iterations"):
        assert native[field] == fake[field]
    assert fake_planar_pairing_issues(configs) == ()
