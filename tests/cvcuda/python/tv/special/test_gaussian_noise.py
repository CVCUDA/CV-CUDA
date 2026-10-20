# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cvcuda.tv.gaussian_noise: random, so checked by distribution and seed replay."""

import pytest

import cvcuda
from cvcuda import tv as cvcuda_tv

from conftest import as_torch


def test_gaussian_noise_distribution_and_seed(torch):
    torch.manual_seed(0)
    owner = torch.rand((8, 3, 128, 128), device="cuda")
    source = cvcuda.as_tensor(owner, layout="NCHW")
    kwargs = dict(mean=0.05, sigma=0.1, clip=False)
    first = as_torch(
        torch, cvcuda_tv.gaussian_noise(source, seed=98765, **kwargs)
    ).clone()
    replay = as_torch(
        torch, cvcuda_tv.gaussian_noise(source, seed=98765, **kwargs)
    ).clone()
    torch.testing.assert_close(first, replay, atol=0, rtol=0)
    delta = first - owner
    assert abs(float(delta.mean()) - 0.05) < 0.003
    assert abs(float(delta.std()) - 0.1) < 0.003
    advanced = as_torch(torch, cvcuda_tv.gaussian_noise(source, **kwargs)).clone()
    assert not torch.equal(first, advanced)


def test_gaussian_noise_errors(torch):
    owner = torch.zeros((3, 7, 9), device="cuda")
    source = cvcuda.as_tensor(owner, layout="CHW")
    with pytest.raises(ValueError, match="negative"):
        cvcuda_tv.gaussian_noise(source, sigma=-0.1)
    with pytest.raises(TypeError, match="seed"):
        cvcuda_tv.gaussian_noise(source, seed=1.5)
    with pytest.raises(ValueError, match="range"):
        cvcuda_tv.gaussian_noise(source, seed=1 << 64)
