# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cvcuda.tv.equalize against the native Pillow-table histogram equalization."""

import pytest

import cvcuda
from cvcuda import tv as cvcuda_tv

from conftest import _from_planar, as_torch, image_params, layout_pair


@pytest.mark.parametrize("image", image_params(("uint8",), (1, 3, 4)), indirect=True)
def test_equalize_matches_histogrameq(torch, image):
    for layout in layout_pair(image.ndim):
        owner = _from_planar(image, layout)
        source = cvcuda.as_tensor(owner, layout=layout)
        got = cvcuda_tv.equalize(source)
        interleaved = "NHWC" if image.ndim == 4 else "HWC"
        expected_source = (
            source if layout == interleaved else cvcuda.reformat(source, interleaved)
        )
        # torchvision follows Pillow's table, so cvcuda_tv.equalize asks for it by name.
        expected = cvcuda.histogrameq(
            expected_source, expected_source.dtype, method=cvcuda.HistogramEq.PILLOW
        )
        if layout != interleaved:
            expected = cvcuda.reformat(expected, layout)
        torch.testing.assert_close(
            as_torch(torch, got),
            as_torch(torch, expected),
            atol=0,
            rtol=0,
        )
