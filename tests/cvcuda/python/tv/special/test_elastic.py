# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cvcuda.tv.elastic: the displacement field is an extra tensor argument."""

import pytest

import cvcuda
from cvcuda import tv as cvcuda_tv

from conftest import _from_planar, image_params, layout_pair, to_torch

InterpolationMode = pytest.importorskip("torchvision.transforms").InterpolationMode

# The Remap-based composition rounds a nearest sample that lands exactly halfway between
# two pixels the other way from grid_sample, so a tie picks the neighbouring pixel.
_NEAREST_TIE_GAP = pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="Remap-based elastic rounds nearest half-pixel ties away from grid_sample "
    "(fixed by the native Elastic operator MR)",
)

# torchvision's float16 grid_sample accumulates in half and lands up to 3.7e-3 from its own
# float32 result; CV-CUDA accumulates in float and stays within 2.4e-4 of that float32 result.
# Multi-channel F16 remap exposes more such pixels, so allow the measured gap plus margin.
_ELASTIC_F16_ATOL = 4.5e-3
# The fill blend adds up to 5.9e-3 (four half ULPs at 1.0) on top of that.
_ELASTIC_F16_FILL_ATOL = 6e-3
# Measured max |CV-CUDA - torchvision| is 1.8e-6 (float32 bilinear, 17x11); float32
# nearest and uint8 are bit-exact.
_ELASTIC_F32_ATOL = 1e-5

_INTERPOLATIONS = [InterpolationMode.NEAREST, InterpolationMode.BILINEAR]


def _displacement(torch, image, seed, amplitude):
    height, width = image.shape[-2:]
    generator = torch.Generator(device=image.device).manual_seed(seed)
    shape = (1, height, width, 2)
    noise = torch.rand(shape, generator=generator, device=image.device)
    return noise * (2 * amplitude) - amplitude


def _assert_elastic_parity(torch, tvf, image, displacement, **kwargs):
    expected = tvf.elastic(image, displacement, **kwargs)
    atol = 0
    if image.dtype == torch.float32:
        atol = _ELASTIC_F32_ATOL
    elif image.dtype == torch.float16:
        atol = _ELASTIC_F16_FILL_ATOL if "fill" in kwargs else _ELASTIC_F16_ATOL
    actual = cvcuda_tv.elastic(image, displacement, **kwargs)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=0)
    native_displacement = cvcuda.as_tensor(displacement, "NHWC")
    for layout in layout_pair(image.ndim):
        owner = _from_planar(image, layout)
        got = cvcuda_tv.elastic(
            cvcuda.as_tensor(owner, layout=layout), native_displacement, **kwargs
        )
        torch.testing.assert_close(
            to_torch(torch, got, layout), expected, atol=atol, rtol=0
        )


@pytest.mark.parametrize(
    "image",
    image_params(("uint8", "float16", "float32"), (1, 3, 4)),
    indirect=True,
)
@pytest.mark.parametrize("interpolation", _INTERPOLATIONS)
def test_elastic_parity(torch, tvf, image, interpolation):
    displacement = _displacement(torch, image, seed=1, amplitude=0.05)
    _assert_elastic_parity(torch, tvf, image, displacement, interpolation=interpolation)


# float16 nearest picks a different source pixel for ~1% of samples once the field is
# large: torchvision computes the sample coordinate in half, CV-CUDA in float.
_F16_NEAREST_GAP = pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="float16 nearest sample coordinate is rounded in half by torchvision "
    "(revisit with the native Elastic operator MR)",
)


@pytest.mark.parametrize(
    "image, interpolation",
    [
        pytest.param(
            image.values[0], interpolation, id=f"{image.id}-{interpolation.value}"
        )
        for image in image_params(("uint8", "float32"), (3,))
        for interpolation in _INTERPOLATIONS
    ]
    + [
        pytest.param(
            image.values[0],
            InterpolationMode.NEAREST,
            id=f"{image.id}-nearest",
            marks=_F16_NEAREST_GAP,
        )
        for image in image_params(("float16",), (3,))
    ]
    + [
        pytest.param(
            image.values[0], InterpolationMode.BILINEAR, id=f"{image.id}-bilinear"
        )
        for image in image_params(("float16",), (3,))
    ],
    indirect=["image"],
)
def test_elastic_per_channel_fill_parity(torch, tvf, image, interpolation):
    # Large enough to push samples off the image so the fill is exercised.
    displacement = _displacement(torch, image, seed=2, amplitude=0.3)
    fill = [0.25, 0.5, 0.75] if image.dtype.is_floating_point else [40.0, 120.0, 200.0]
    _assert_elastic_parity(
        torch, tvf, image, displacement, interpolation=interpolation, fill=fill
    )


@pytest.mark.parametrize(
    "image",
    image_params(("uint8", "float16", "float32"), (1, 3, 4), marks=_NEAREST_TIE_GAP),
    indirect=True,
)
def test_elastic_nearest_half_pixel_ties(torch, tvf, image):
    """Shift every sample by exactly half a pixel, so each nearest lookup is a tie."""
    height, width = image.shape[-2:]
    displacement = torch.empty((1, height, width, 2), device=image.device)
    displacement[..., 0] = 1 / width
    displacement[..., 1] = 1 / height
    _assert_elastic_parity(
        torch, tvf, image, displacement, interpolation=InterpolationMode.NEAREST
    )


@pytest.mark.parametrize("mode", [999, "bogus", []])
def test_elastic_invalid_interpolation(mode):
    with pytest.raises(NotImplementedError, match="interpolation is not supported"):
        cvcuda_tv.elastic(None, None, interpolation=mode)


def _relative_displacement(torch, height, width, device="cuda", scale=0.1):
    generator = torch.Generator(device=device).manual_seed(3)
    return (
        torch.rand((1, height, width, 2), generator=generator, device=device) - 0.5
    ) * scale


@pytest.mark.parametrize("by_keyword", [False, True])
@pytest.mark.parametrize("image_shape", [(3, 16, 20), (2, 3, 16, 20)])
@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_elastic_torch_displacement_matches_torchvision(
    torch, tvf, image_shape, device, by_keyword
):
    image = torch.randint(0, 256, image_shape, device="cuda", dtype=torch.uint8)
    displacement = _relative_displacement(torch, 16, 20, device)
    expected = tvf.elastic(image, displacement)
    if by_keyword:
        actual = cvcuda_tv.elastic(image, displacement=displacement)
    else:
        actual = cvcuda_tv.elastic(image, displacement)
    torch.testing.assert_close(actual, expected, atol=1, rtol=0)


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("shape", [(16, 20, 2), (2, 16, 20, 2), (1, 20, 16, 2)])
def test_elastic_displacement_shape_error(torch, native, shape):
    image = torch.zeros((3, 16, 20), device="cuda", dtype=torch.uint8)
    displacement = torch.zeros(shape, device="cuda")
    if native:
        image = cvcuda.as_tensor(image, "CHW")
        displacement = cvcuda.as_tensor(
            displacement, "NHWC" if len(shape) == 4 else "HWC"
        )
    with pytest.raises(
        ValueError, match=r"Argument displacement shape should be \(1, 16, 20, 2\)"
    ):
        cvcuda_tv.elastic(image, displacement)
