# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cvcuda.tv.erase in-place aliasing."""

import pytest

import cvcuda
from cvcuda import tv as cvcuda_tv

from conftest import as_torch, layout_source


@pytest.mark.parametrize("layout", ["CHW", "HWC", "NCHW", "NHWC"])
def test_erase_inplace_alias(torch, layout):
    owner, source = layout_source(torch, layout)
    value_owner = torch.tensor([0.5], device="cuda")
    value = cvcuda.as_tensor(value_owner)
    result = cvcuda_tv.erase(source, 2, 3, 2, 3, value, inplace=True)
    assert result is source
    assert result.cuda().__cuda_array_interface__["data"][0] == owner.data_ptr()


def _erase_value(torch, shape, dtype):
    if shape == "chw":
        return torch.arange(3 * 2 * 3, device="cuda").reshape(3, 2, 3).to(dtype) + 0.7
    return torch.full(shape, 7.7, device="cuda").to(dtype)


@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("by_keyword", [False, True])
@pytest.mark.parametrize("shape", [(), (1,), (3, 1, 1), "chw"])
@pytest.mark.parametrize("image_shape", [(3, 7, 9), (2, 3, 7, 9)])
@pytest.mark.parametrize(
    "image_dtype,value_dtype",
    [("uint8", "float32"), ("uint8", "uint8"), ("float32", "float32")],
)
def test_erase_torch_value_matches_torchvision(
    torch, tvf, image_dtype, image_shape, shape, value_dtype, by_keyword, inplace
):
    dtype = getattr(torch, image_dtype)
    image = torch.randint(0, 200, image_shape, device="cuda").to(dtype)
    value = _erase_value(torch, shape, getattr(torch, value_dtype))
    expected = tvf.erase(image.clone(), 2, 3, 2, 3, value)
    if by_keyword:
        actual = cvcuda_tv.erase(image, 2, 3, 2, 3, v=value, inplace=inplace)
    else:
        actual = cvcuda_tv.erase(image, 2, 3, 2, 3, value, inplace)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    if inplace:
        assert actual.data_ptr() == image.data_ptr()
        torch.testing.assert_close(image, expected, atol=0, rtol=0)


def test_erase_cpu_torch_value_is_moved_like_torchvision(torch, tvf):
    image = torch.randint(0, 200, (3, 7, 9), device="cuda", dtype=torch.uint8)
    value = torch.tensor([3.0, 4.0, 5.0]).reshape(3, 1, 1)
    expected = tvf.erase(image.clone(), 2, 3, 2, 3, value)
    actual = cvcuda_tv.erase(image, 2, 3, 2, 3, value)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_erase_torch_value_with_native_image(torch, tvf):
    image = torch.randint(0, 200, (3, 7, 9), device="cuda", dtype=torch.uint8)
    value = torch.tensor([3.0, 4.0, 5.0], device="cuda").reshape(3, 1, 1)
    expected = tvf.erase(image.clone(), 2, 3, 2, 3, value)
    actual = cvcuda_tv.erase(cvcuda.as_tensor(image, "CHW"), 2, 3, 2, 3, value)
    assert type(actual) is cvcuda.Tensor
    torch.testing.assert_close(as_torch(torch, actual), expected, atol=0, rtol=0)
