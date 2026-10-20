# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cvcuda.tv.jpeg: lossy, so compared statistically rather than bit-exactly."""

import math

import pytest

import cvcuda
from cvcuda import tv as cvcuda_tv

from conftest import _from_planar, layout_pair, to_torch


def _smooth_image(torch, batch, channels, height=96, width=128):
    y, x = torch.meshgrid(
        torch.arange(height, device="cuda", dtype=torch.float64),
        torch.arange(width, device="cuda", dtype=torch.float64),
        indexing="ij",
    )
    planes = []
    for channel in range(channels):
        wave = torch.sin(2 * math.pi * x / (32 + 8 * channel)) * torch.sin(
            2 * math.pi * y / (48 + 8 * channel)
        )
        planes.append(127.5 + 80 * wave + 60 * ((x + y) / (width + height) - 0.5))
    image = torch.stack(planes).clamp(0, 255).to(torch.uint8)
    return image if batch is None else image.expand(batch, -1, -1, -1).contiguous()


@pytest.mark.parametrize("quality", [5, 75])
@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("batch", [None, 1, 3])
def test_jpeg_parity(torch, tvf, quality, channels, batch):
    image = _smooth_image(torch, batch, channels)
    expected = tvf.jpeg(image.cpu(), quality).cuda()

    for layout in layout_pair(image.ndim):
        owner = _from_planar(image, layout)
        got = cvcuda_tv.jpeg(cvcuda.as_tensor(owner, layout=layout), quality)
        actual = to_torch(torch, got, layout)
        diff = (actual.to(torch.float32) - expected).abs()
        assert int((diff > 25).sum()) <= 0.015 * diff.numel()
        bound = 0.5 if channels == 1 else (6.0 if quality <= 10 else 4.5)
        assert float(diff.mean()) <= bound


@pytest.mark.parametrize("quality", [0, 101])
def test_jpeg_quality_errors(quality):
    source = cvcuda.Tensor((1, 16, 24, 3), cvcuda.Type.U8, "NHWC")
    with pytest.raises(RuntimeError, match="quality"):
        cvcuda_tv.jpeg(source, quality)
