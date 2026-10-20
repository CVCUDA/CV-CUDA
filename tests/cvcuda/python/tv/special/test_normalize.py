# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cvcuda.tv.normalize in-place aliasing and argument errors."""

import pytest

import cvcuda
from cvcuda import tv as cvcuda_tv

from conftest import as_torch, layout_source


@pytest.mark.parametrize("layout", ["CHW", "HWC", "NCHW", "NHWC"])
def test_normalize_inplace_alias(torch, layout):
    _, source = layout_source(torch, layout)
    expected = as_torch(
        torch, cvcuda_tv.normalize(source, [0.1, 0.2, 0.3], [0.5, 0.6, 0.7])
    ).clone()
    result = cvcuda_tv.normalize(source, [0.1, 0.2, 0.3], [0.5, 0.6, 0.7], inplace=True)
    assert result is source
    torch.testing.assert_close(as_torch(torch, result), expected)


def test_normalize_errors(torch):
    uint8_owner = torch.zeros((3, 7, 9), dtype=torch.uint8, device="cuda")
    uint8_source = cvcuda.as_tensor(uint8_owner, layout="CHW")
    with pytest.raises(TypeError, match="float"):
        cvcuda_tv.normalize(uint8_source, [0.1] * 3, [0.5] * 3)
    float_owner = uint8_owner.float()
    float_source = cvcuda.as_tensor(float_owner, layout="CHW")
    with pytest.raises(ValueError, match="division by zero"):
        cvcuda_tv.normalize(float_source, [0.1] * 3, [0.5, 0, 0.5])
