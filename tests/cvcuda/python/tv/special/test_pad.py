# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cvcuda.tv.pad argument errors."""

import pytest

from cvcuda import tv as cvcuda_tv


@pytest.mark.parametrize("padding", [2.5, [], [1, 2, 3], [1, 2, 3, 4, 5]])
def test_pad_errors(padding):
    with pytest.raises((TypeError, ValueError)):
        cvcuda_tv.pad(None, padding)


@pytest.mark.parametrize("padding_mode", ["bogus", "CONSTANT", "", None, []])
def test_pad_unknown_mode(padding_mode):
    with pytest.raises(
        ValueError, match="padding_mode.*constant.*edge.*reflect.*symmetric"
    ):
        cvcuda_tv.pad(None, 1, padding_mode=padding_mode)
