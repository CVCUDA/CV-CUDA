# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One declarative torchvision parity matrix per cvcuda.tv operator.

The shared assertion exercises Torch inputs and both native layouts on a caller
stream. Pattern, dispatch-gap and allocation regressions live in special/.
"""

import pytest

from cvcuda import tv as cvcuda_tv
from conftest import images

InterpolationMode = pytest.importorskip("torchvision.transforms").InterpolationMode


@images(dtypes=("uint8", "uint16", "float16", "float32", "float64"), channels=(1, 3))
@pytest.mark.parametrize("factor", [0.0, 0.5, 1.0, 5.0])
def test_adjust_brightness(assert_tv_parity, tvf, image, factor):
    assert_tv_parity(
        cvcuda_tv.adjust_brightness,
        tvf.adjust_brightness,
        image,
        factor,
        atol=(
            1e-3
            if image.is_floating_point() and image.element_size() == 2
            else 1e-6 if image.is_floating_point() else 1
        ),
    )


@pytest.mark.parametrize("factor", [0.1, 0.5, 1.0])
@images(dtypes=("uint8", "float32", "float16"), channels=(1, 3))
def test_adjust_contrast(assert_tv_parity, tvf, image, factor):
    # Torchvision computes in half while CV-CUDA widens to float before the blend.
    assert_tv_parity(
        cvcuda_tv.adjust_contrast,
        tvf.adjust_contrast,
        image,
        factor,
        atol=(
            5e-3
            if (image.is_floating_point() and image.element_size() == 2)
            else (1e-6 if image.is_floating_point() else 1)
        ),
    )


@pytest.mark.parametrize("gamma", [0.1, 0.5, 1.0])
@pytest.mark.parametrize("gain", [0.1, 1.0, 2.0])
@images(dtypes=("uint8", "float16", "float32"), channels=(1, 3, 4))
def test_adjust_gamma(assert_tv_parity, tvf, image, gamma, gain):
    assert_tv_parity(
        cvcuda_tv.adjust_gamma,
        tvf.adjust_gamma,
        image,
        gamma,
        gain=gain,
        atol=(
            4e-3
            if (image.is_floating_point() and image.element_size() == 2)
            else 5e-7 if image.is_floating_point() else 0
        ),
    )


@pytest.mark.parametrize("factor", [-0.5, -0.3, 0.0, 0.2, 0.5])
@images(dtypes=("uint8", "float32", "float16"), channels=(1, 3))
def test_adjust_hue(assert_tv_parity, tvf, image, factor):
    # The half-precision torchvision HSV round-trip accumulates error in both conversions.
    assert_tv_parity(
        cvcuda_tv.adjust_hue,
        tvf.adjust_hue,
        image,
        factor,
        atol=(
            1e-2
            if (image.is_floating_point() and image.element_size() == 2)
            else (2e-6 if image.is_floating_point() else 1)
        ),
    )


@pytest.mark.parametrize("factor", [0.1, 0.5, 1.0])
@images(dtypes=("uint8", "float32", "float16"), channels=(1, 3))
def test_adjust_saturation(assert_tv_parity, tvf, image, factor):
    # Torchvision computes the F16 luma blend in half; CV-CUDA computes it in float.
    assert_tv_parity(
        cvcuda_tv.adjust_saturation,
        tvf.adjust_saturation,
        image,
        factor,
        # uint8 atol=1 until the native blend-order fix lands (then 0).
        atol=(
            5e-3
            if (image.is_floating_point() and image.element_size() == 2)
            else (1e-6 if image.is_floating_point() else 1)
        ),
    )


@pytest.mark.parametrize("factor", [0.1, 0.5, 1.0])
@images(
    dtypes=("uint8", "uint16", "float16", "float32"),
    sizes=((17, 11), (2, 11), (11, 2)),
    channels=(1, 3),
)
def test_adjust_sharpness(assert_tv_parity, tvf, image, factor):
    # Torchvision's F16 convolution accumulates in half; CV-CUDA accumulates in float.
    assert_tv_parity(
        cvcuda_tv.adjust_sharpness,
        tvf.adjust_sharpness,
        image,
        factor,
        atol=(
            5e-3
            if (image.is_floating_point() and image.element_size() == 2)
            else (1e-6 if image.is_floating_point() else 0)
        ),
    )


@images(dtypes=("uint8", "uint16", "float16", "float32"), channels=(1, 3, 4))
@pytest.mark.parametrize(
    "angle,translate,scale,shear",
    [
        pytest.param(0.0, [0, 0], 1.0, 0.0, id="identity"),
        pytest.param(-10.9, [1, -3], 0.5, 35.6, id="scalar-shear"),
        pytest.param(18.0, [2.5, -1.5], 1.0, [5.4, 21.8], id="fractional-translation"),
        pytest.param(20.0, [2, -3], 1.2, [5.0, 8.0], id="scale-and-shear"),
        pytest.param(-10.9, [6.3, -0.6], 0.5, [-37.7], id="list-shear"),
        pytest.param(18.0, [1, -3], 1.0, (5.3,), id="tuple-shear"),
    ],
)
@pytest.mark.parametrize(
    "interpolation", [InterpolationMode.NEAREST, InterpolationMode.BILINEAR]
)
@pytest.mark.parametrize("fill", [None, 7, [7]])
def test_affine(
    assert_tv_parity, tvf, image, angle, translate, scale, shear, interpolation, fill
):
    half = image.is_floating_point() and image.element_size() == 2
    magnitude = (
        0
        if fill is None
        else abs(float(fill[0] if isinstance(fill, list) else fill)) + 1
    )
    assert_tv_parity(
        cvcuda_tv.affine,
        tvf.affine,
        image,
        angle,
        translate,
        scale,
        shear,
        interpolation=interpolation,
        fill=fill,
        atol=(3e-3 if half else 1e-5 if image.is_floating_point() else 1)
        + magnitude * (5e-2 if half else 2e-5 if image.is_floating_point() else 0),
        max_bad_frac=0.1 if half else 0,
    )


@images(
    dtypes=(
        "uint8",
        "uint16",
        "int16",
        "float16",
        "float32",
        pytest.param(
            "int32",
            marks=pytest.mark.xfail(
                strict=True,
                raises=RuntimeError,
                reason="native dispatch has no signed int32 kernel",
            ),
        ),
        pytest.param(
            "int64",
            marks=pytest.mark.xfail(
                strict=True,
                raises=RuntimeError,
                reason="native dispatch has no signed int64 kernel",
            ),
        ),
        pytest.param(
            "float64",
            marks=pytest.mark.xfail(
                strict=True,
                raises=RuntimeError,
                reason="native dispatch has no float64 kernel",
            ),
        ),
    ),
    channels=(1, 3),
)
def test_autocontrast(assert_tv_parity, tvf, image):
    assert_tv_parity(
        cvcuda_tv.autocontrast,
        tvf.autocontrast,
        image,
        atol=(
            2e-3
            if image.is_floating_point() and image.element_size() == 2
            else 0 if image.is_floating_point() else 1
        ),
    )


@images(
    dtypes=("uint8", "uint16", "float16", "float32"),
    channels=(1, 3, 4),
    sizes=((17, 11),),
)
@pytest.mark.parametrize(
    "size",
    [(3, 5), (5, 3), (4, 4), (21, 9), (13, 15), (19, 14), 3, (4,), [5], (17, 11)],
)
def test_center_crop(assert_tv_parity, tvf, image, size):
    assert_tv_parity(cvcuda_tv.center_crop, tvf.center_crop, image, size)


@images(
    dtypes=("uint8", "uint16", "float16", "float32", "float64"),
    channels=(1, 3, 4),
    sizes=((21, 11),),
)
@pytest.mark.parametrize(
    "top,left,height,width",
    [(5, 3, 10, 5), (0, 0, 21, 11), (0, 0, 1, 1), (20, 10, 1, 1)],
)
def test_crop(assert_tv_parity, tvf, image, top, left, height, width):
    assert_tv_parity(cvcuda_tv.crop, tvf.crop, image, top, left, height, width)


@images(
    dtypes=("uint8", "float16", "float32"),
    channels=(1, 3),
    sizes=((9, 13),),
    batches=(None, 2),
)
@pytest.mark.parametrize(
    "interpolation", [InterpolationMode.NEAREST, InterpolationMode.BILINEAR]
)
def test_elastic(assert_tv_parity, torch, tvf, image, interpolation):
    height, width = image.shape[-2:]
    displacement = (
        torch.rand((1, height, width, 2), device=image.device).mul_(0.1).sub_(0.05)
    )
    half = image.dtype == torch.float16
    assert_tv_parity(
        cvcuda_tv.elastic,
        tvf.elastic,
        image,
        displacement,
        interpolation=interpolation,
        atol=4.5e-3 if half else (1e-5 if image.is_floating_point() else 0),
    )


@images(
    dtypes=(
        "uint8",
        "float32",
        pytest.param(
            "float16",
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="torchvision half binning rounds the multiplier to 256 and wraps 1.0 to bin 0",
            ),
        ),
    ),
    channels=(1, 3),
)
def test_equalize(assert_tv_parity, tvf, image):
    assert_tv_parity(cvcuda_tv.equalize, tvf.equalize, image)


@pytest.mark.parametrize("inplace", [False, True])
@images(
    dtypes=(
        "uint8",
        "uint16",
        "float16",
        "float32",
        "float64",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint32",
    ),
    channels=(1, 2, 3, 4),
    sizes=((7, 9),),
)
def test_erase(assert_tv_parity, tvf, torch, image, inplace):
    value = torch.tensor(0.5, device=image.device)
    assert_tv_parity(
        cvcuda_tv.erase,
        tvf.erase,
        image,
        2,
        3,
        2,
        3,
        value,
        inplace=inplace,
    )


@pytest.mark.parametrize("size", [3, (3,), [3], (3, 5), [3, 5], (17, 11)])
@images(dtypes=("uint8", "uint16", "float16", "float32", "float64"), channels=(1, 3, 4))
def test_five_crop(assert_tv_parity, tvf, image, size):
    assert_tv_parity(cvcuda_tv.five_crop, tvf.five_crop, image, size)


@pytest.mark.parametrize("kernel_size", [1, 3, (3, 1), [3, 5]])
@pytest.mark.parametrize("sigma", [None, 1.0, (0.5,), [0.3], (0.3, 0.7), [0.9, 0.2]])
@images(dtypes=("uint8", "uint16", "float16", "float32"), channels=(1, 3, 4))
def test_gaussian_blur(assert_tv_parity, tvf, image, kernel_size, sigma):
    # Torchvision's F16 separable convolution accumulates in half; CV-CUDA uses float.
    assert_tv_parity(
        cvcuda_tv.gaussian_blur,
        tvf.gaussian_blur,
        image,
        kernel_size,
        sigma,
        atol=(
            5e-3
            if (image.is_floating_point() and image.element_size() == 2)
            else (1e-6 if image.is_floating_point() else 1)
        ),
        rtol=(
            0
            if (image.is_floating_point() and image.element_size() == 2)
            else (1e-6 if image.is_floating_point() else 0)
        ),
    )


@pytest.mark.parametrize("clip", [True, False])
@images(dtypes=("uint8", "float32", "float16"), channels=(1, 3))
def test_gaussian_noise(assert_tv_parity, tvf, image, clip):
    # F16 performs one add and clip, rounded once per implementation.
    assert_tv_parity(
        cvcuda_tv.gaussian_noise,
        tvf.gaussian_noise,
        image,
        mean=0.125,
        sigma=0.0,
        clip=clip,
        atol=2e-3 if (image.is_floating_point() and image.element_size() == 2) else 0,
    )


@images(dtypes=("uint8", "float16", "float32"), channels=(1, 3))
def test_grayscale_to_rgb(assert_tv_parity, tvf, image):
    assert_tv_parity(cvcuda_tv.grayscale_to_rgb, tvf.grayscale_to_rgb, image)


@images(dtypes=("uint8", "uint16", "float16", "float32"), channels=(1, 3, 4))
def test_horizontal_flip(assert_tv_parity, tvf, image):
    assert_tv_parity(cvcuda_tv.horizontal_flip, tvf.horizontal_flip, image)


@images(
    dtypes=("uint8", "int16", "int32", "int64", "float16", "float32", "float64"),
    channels=(1, 3),
)
def test_invert(assert_tv_parity, tvf, image):
    assert_tv_parity(
        cvcuda_tv.invert,
        tvf.invert,
        image,
        atol=(
            5e-4
            if (image.is_floating_point() and image.element_size() == 2)
            else 1e-6 if image.is_floating_point() else 0
        ),
    )


@images(dtypes=("uint8",), channels=(1, 3), sizes=((16, 20),), batches=(None, 2))
@pytest.mark.parametrize("quality", [80])
def test_jpeg(assert_tv_parity, tvf, image, quality):
    image.fill_(64)
    assert_tv_parity(
        cvcuda_tv.jpeg,
        lambda x, quality: tvf.jpeg(x.cpu(), quality).to(x.device),
        image,
        quality,
        fresh=True,
    )


@images(dtypes=("float16", "float32", "float64"), channels=(1, 3, 4))
def test_normalize(assert_tv_parity, tvf, image):
    channels = image.shape[-3]
    mean = [0.1 + index * 0.1 for index in range(channels)]
    std = [0.5 + index * 0.1 for index in range(channels)]
    assert_tv_parity(
        cvcuda_tv.normalize,
        tvf.normalize,
        image,
        mean,
        std,
        atol=(
            1e-3 if (image.is_floating_point() and image.element_size() == 2) else 1e-6
        ),
        rtol=1e-6,
    )


@images(dtypes=("uint8", "float16", "float32"), channels=(1, 3, 4), sizes=((17, 29),))
@pytest.mark.parametrize(
    "padding,fill,padding_mode",
    [
        (0, None, "constant"),
        (1, None, "constant"),
        ((1,), None, "constant"),
        ([1], None, "constant"),
        ((1, 2), None, "constant"),
        ([1, 2], None, "constant"),
        ((1, 2, 3, 4), None, "constant"),
        ([1, 2, 3, 4], None, "constant"),
        ([1, 2], 7, "constant"),
        ([1, 2], [7], "constant"),
        ([1, 2], None, "edge"),
        ([1, 2], None, "reflect"),
        ([1, 2], None, "symmetric"),
    ],
)
def test_pad(assert_tv_parity, tvf, image, padding, padding_mode, fill):
    assert_tv_parity(
        cvcuda_tv.pad, tvf.pad, image, padding, fill=fill, padding_mode=padding_mode
    )


@images(dtypes=("uint8", "float32", "float16"), channels=(3,))
@pytest.mark.parametrize("permutation", [[0, 1, 2], [2, 0, 1], [2, -3, 2]])
def test_permute_channels(assert_tv_parity, tvf, image, permutation):
    assert_tv_parity(
        cvcuda_tv.permute_channels, tvf.permute_channels, image, permutation
    )


@images(dtypes=("uint8", "uint16", "float16", "float32"), channels=(1, 3, 4))
@pytest.mark.parametrize(
    "startpoints,endpoints",
    [
        pytest.param(
            [[0, 0], [10, 0], [10, 16], [0, 16]],
            [[1, 1], [9, 2], [10, 14], [0, 15]],
            id="inward",
        ),
        pytest.param(
            [[1, 1], [9, 2], [10, 14], [0, 15]],
            [[0, 0], [10, 0], [10, 16], [0, 16]],
            id="outward",
        ),
    ],
)
@pytest.mark.parametrize(
    "interpolation", [InterpolationMode.NEAREST, InterpolationMode.BILINEAR]
)
@pytest.mark.parametrize("fill", [None, 7, [7]])
def test_perspective(
    assert_tv_parity, tvf, image, startpoints, endpoints, interpolation, fill
):
    half = image.is_floating_point() and image.element_size() == 2
    magnitude = (
        0
        if fill is None
        else abs(float(fill[0] if isinstance(fill, list) else fill)) + 1
    )
    assert_tv_parity(
        cvcuda_tv.perspective,
        tvf.perspective,
        image,
        startpoints,
        endpoints,
        interpolation=interpolation,
        fill=fill,
        atol=(3e-3 if half else 1e-5 if image.is_floating_point() else 1)
        + magnitude * (5e-2 if half else 2e-5 if image.is_floating_point() else 0),
        max_bad_frac=0.1 if half else 0,
    )


@pytest.mark.parametrize("bits", [0, 1, 4, 8])
@images(dtypes=("uint8", "float16", "float32"), channels=(1, 3, 4))
def test_posterize(assert_tv_parity, tvf, image, bits):
    assert_tv_parity(cvcuda_tv.posterize, tvf.posterize, image, bits)


@images(
    dtypes=("uint8", "int16", "float16", "float32"),
    channels=(1, 2, 3, 4),
    sizes=((17, 11), (11, 17), (7, 5)),
)
@pytest.mark.parametrize("size", [[12, 13], [22, 34], [11, 13], 9])
@pytest.mark.parametrize(
    "interpolation",
    [
        InterpolationMode.NEAREST,
        InterpolationMode.NEAREST_EXACT,
        InterpolationMode.BILINEAR,
        InterpolationMode.BICUBIC,
    ],
)
@pytest.mark.parametrize("antialias", [True, False, None])
def test_resize(assert_tv_parity, torch, tvf, image, size, interpolation, antialias):
    if image.dtype.is_floating_point:
        atol = 3e-3 if image.element_size() == 2 else 3e-5
    else:
        atol = 1
    if image.dtype == torch.int16 and interpolation is InterpolationMode.BICUBIC:
        image = image.remainder(256)
    if interpolation in (InterpolationMode.NEAREST, InterpolationMode.NEAREST_EXACT):
        atol = 0
    assert_tv_parity(
        cvcuda_tv.resize,
        tvf.resize,
        image,
        size,
        interpolation=interpolation,
        antialias=antialias,
        atol=atol,
    )


@images(
    dtypes=("uint8", "float32", "int16", "float16"),
    channels=(1, 3, 4),
    sizes=((17, 29),),
)
@pytest.mark.parametrize(
    "top,left,height,width,size",
    [
        (2, 3, 12, 18, [8, 12]),
        (2, 3, 12, 18, [24, 36]),
        (2, 3, 12, 18, [12, 18]),
        (2, 3, 12, 18, [18, 12]),
        (2, 2, 5, 7, (19, 32)),
        (0, 0, 17, 29, 224),
        (0, 0, 8, 12, [10, 14]),
        (9, 17, 8, 12, [10, 14]),
    ],
)
@pytest.mark.parametrize(
    "interpolation",
    [
        InterpolationMode.NEAREST_EXACT,
        InterpolationMode.BILINEAR,
        InterpolationMode.BICUBIC,
    ],
)
@pytest.mark.parametrize("antialias", [True, False])
def test_resized_crop(
    assert_tv_parity,
    torch,
    tvf,
    image,
    top,
    left,
    height,
    width,
    size,
    interpolation,
    antialias,
):
    half = image.dtype == torch.float16
    if image.dtype == torch.int16 and interpolation is InterpolationMode.BICUBIC:
        image = image.remainder(256)
    atol = (
        0
        if interpolation in (InterpolationMode.NEAREST, InterpolationMode.NEAREST_EXACT)
        else 3e-3 if half else 3e-5 if image.is_floating_point() else 1
    )
    assert_tv_parity(
        cvcuda_tv.resized_crop,
        tvf.resized_crop,
        image,
        top,
        left,
        height,
        width,
        size,
        interpolation=interpolation,
        antialias=antialias,
        atol=atol,
    )


@pytest.mark.parametrize("num_output_channels", [1, 3])
@images(dtypes=("uint8", "float16", "float32"), channels=(1, 3))
def test_rgb_to_grayscale(assert_tv_parity, tvf, image, num_output_channels):
    # Tolerance only matters for the 3-channel path (luma conversion); the
    # 1-channel short-circuit is a pass-through or GRAY2RGB replication with
    # zero numeric error.
    assert_tv_parity(
        cvcuda_tv.rgb_to_grayscale,
        tvf.rgb_to_grayscale,
        image,
        num_output_channels=num_output_channels,
        atol=(
            2e-3
            if (image.is_floating_point() and image.element_size() == 2)
            else (1.1e-4 if image.is_floating_point() else 1)
        ),
    )


@images(dtypes=("uint8", "uint16", "float16", "float32"), channels=(1, 3, 4))
@pytest.mark.parametrize("angle", [-10.9, 30.0, 60.0])
@pytest.mark.parametrize("center", [None, [4.5, 7.25], [3, 2]])
@pytest.mark.parametrize(
    "interpolation", [InterpolationMode.NEAREST, InterpolationMode.BILINEAR]
)
@pytest.mark.parametrize("fill", [None, 7, [7]])
@pytest.mark.parametrize("expand", [False, True])
def test_rotate(
    assert_tv_parity, tvf, image, angle, center, interpolation, fill, expand
):
    half = image.is_floating_point() and image.element_size() == 2
    magnitude = (
        0
        if fill is None
        else abs(float(fill[0] if isinstance(fill, list) else fill)) + 1
    )
    atol = (3e-3 if half else 1e-5 if image.is_floating_point() else 1) + magnitude * (
        5e-2 if half else 2e-5 if image.is_floating_point() else 0
    )
    assert_tv_parity(
        cvcuda_tv.rotate,
        tvf.rotate,
        image,
        angle,
        interpolation=interpolation,
        center=center,
        fill=fill,
        expand=expand,
        atol=atol if interpolation is InterpolationMode.BILINEAR else 0,
        max_bad_frac=(
            0.1 if half else 0.05 if interpolation is InterpolationMode.NEAREST else 0
        ),
    )


@pytest.mark.parametrize("threshold_factor", [0.0, 0.1, 0.5, 0.9, 1.0])
@images(dtypes=("uint8", "float16", "float32", "float64"), channels=(1, 3, 4))
def test_solarize(assert_tv_parity, tvf, image, threshold_factor):
    threshold = (
        image.new_tensor(threshold_factor).item()
        if image.is_floating_point()
        else threshold_factor * 255
    )
    assert_tv_parity(
        cvcuda_tv.solarize,
        tvf.solarize,
        image,
        threshold,
        atol=5e-4 if (image.is_floating_point() and image.element_size() == 2) else 0,
    )


@pytest.mark.parametrize("vertical_flip", [False, True])
@pytest.mark.parametrize("size", [3, (3,), [3], (3, 5), [3, 5], (17, 11)])
@images(dtypes=("uint8", "uint16", "float16", "float32"), channels=(1, 3, 4))
def test_ten_crop(assert_tv_parity, tvf, image, size, vertical_flip):
    assert_tv_parity(
        cvcuda_tv.ten_crop,
        tvf.ten_crop,
        image,
        size,
        vertical_flip=vertical_flip,
    )


@pytest.mark.parametrize("target", ["float16", "float32", "float64", "uint8"])
@pytest.mark.parametrize("scale", [True, False])
@images(dtypes=("uint8", "float16", "float32", "float64"), channels=(1, 3))
def test_to_dtype(assert_tv_parity, tvf, torch, image, target, scale):
    assert_tv_parity(
        cvcuda_tv.to_dtype,
        tvf.to_dtype,
        image,
        dtype=getattr(torch, target),
        scale=scale,
        atol=1e-3 if target == "float16" else 1e-6 if target.startswith("float") else 1,
    )


@images(dtypes=("uint8", "uint16", "float16", "float32"), channels=(1, 3, 4))
def test_vertical_flip(assert_tv_parity, tvf, image):
    assert_tv_parity(cvcuda_tv.vertical_flip, tvf.vertical_flip, image)
