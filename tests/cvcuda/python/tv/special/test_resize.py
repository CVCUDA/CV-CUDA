# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cvcuda.tv.resize contracts outside the one-call parity oracle."""

import pytest
from types import SimpleNamespace

import cvcuda
from cvcuda import tv as cvcuda_tv


@pytest.mark.parametrize("mode", ["nearest", "nearest-exact", "bilinear", "bicubic"])
def test_resize_identity_preserves_input(monkeypatch, mode):
    source = SimpleNamespace(dtype=cvcuda.Type.F16, layout="NCHW", shape=(2, 2, 7, 9))

    def unexpected_resize(*args, **kwargs):
        pytest.fail("An identity resize must not invoke the native operator")

    monkeypatch.setattr(cvcuda, "tvresize", unexpected_resize)
    assert cvcuda_tv.resize(source, [7, 9], interpolation=mode) is source


@pytest.mark.parametrize("operator", ["resize", "resized_crop"])
@pytest.mark.parametrize("mode", ["box", "hamming", "lanczos"])
def test_resize_rejects_pil_only_interpolation(operator, mode):
    source = cvcuda.Tensor((1, 13, 17, 3), cvcuda.Type.U8, "NHWC")
    arguments = {"size": [7, 9], "interpolation": mode}
    if operator == "resized_crop":
        arguments.update(top=1, left=1, height=9, width=11)
    with pytest.raises(NotImplementedError, match="interpolation is not supported"):
        getattr(cvcuda_tv, operator)(source, **arguments)
