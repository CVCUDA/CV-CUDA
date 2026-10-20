# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cvcuda.tv.gaussian_blur argument errors."""

import pytest

from cvcuda import tv as cvcuda_tv


@pytest.mark.parametrize(
    "kernel_size", [None, [], [3], [3, 3, 3], [3.0, 3], [0, 3], [2, 3]]
)
def test_gaussian_blur_kernel_errors(kernel_size):
    with pytest.raises(ValueError, match="kernel_size"):
        cvcuda_tv.gaussian_blur(None, kernel_size)


@pytest.mark.parametrize("sigma", [[], [1, 2, 3], 0, -1, [1, 0]])
def test_gaussian_blur_sigma_errors(sigma):
    with pytest.raises(ValueError, match="sigma"):
        cvcuda_tv.gaussian_blur(None, 3, sigma)
