# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pattern and dispatch regressions supplementing the declarative parity matrices."""

import itertools

import numpy as np
import pytest

from cvcuda import tv as cvcuda_tv
from cvcuda.tv._geometry import _resize_output_hw
from conftest import image_params

InterpolationMode = pytest.importorskip("torchvision.transforms").InterpolationMode
torch_module = pytest.importorskip("torch")

_TORCHVISION_DTYPES = ("uint8", "uint16", "float16", "float32", "float64")


_MINIMAL_DTYPES = ("uint8", "float32")


_UINTS = ("uint8", "uint16")


_CHANNELS = (1, 2, 3, 4)


_BATCHES = (None, 1, 3)


_RESIZE_INTERPOLATIONS = (
    InterpolationMode.NEAREST_EXACT,
    InterpolationMode.BILINEAR,
    InterpolationMode.BICUBIC,
)


_WARP_INTERPOLATIONS = (InterpolationMode.NEAREST, InterpolationMode.BILINEAR)


_C2_GAP = pytest.mark.xfail(
    strict=True,
    raises=(KeyError, RuntimeError),
    reason="this CV-CUDA path does not support two channels",
)


_INT64_BRIGHTNESS_GAP = pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "torchvision.adjust_brightness converts int64 to float32 for scaling; "
        "CV-CUDA uses double, which diverges for large values when factor > 1"
    ),
)


_F16_INTERP_ATOL = 3e-3


def _fill_blend_atol(dtype, fill):
    """Extra tolerance the coverage blend needs for ``fill``, zero when unfilled."""
    if fill is None:
        return 0.0
    values = [fill] if isinstance(fill, (int, float)) else list(fill)
    magnitude = max(abs(float(value)) for value in values)
    # Images are generated in [0, 1], so |image - fill| <= |fill| + 1 bounds the blend's span.
    # Coverage-weight rounding is amplified by fill; measured across 120 configurations.
    relative = {"torch.float16": 5e-2, "torch.float32": 2e-5}.get(str(dtype), 0.0)
    return relative * (magnitude + 1.0)


_WARP_DTYPE_GAP = pytest.mark.xfail(
    strict=True,
    raises=RuntimeError,
    reason="warp_affine/warp_perspective dispatch lacks S32, S64 and F64",
)


_RESIZE_DTYPE_GAP = pytest.mark.xfail(
    strict=True,
    raises=RuntimeError,
    reason="cvcuda.tvresize dispatch lacks S32, S64 and F64",
)


def _images(dtypes, channels=(1, 3), **kwargs):
    return image_params(dtypes, channels, **kwargs)


def _is_float16(image):
    return image.is_floating_point() and image.element_size() == 2


def _gap_params(
    channels,
    *values,
    case_id,
    gap=False,
    gap_mark=None,
    c2_gap_mark=None,
    c2_gap_layouts=("planar", "interleaved"),
):
    gap_marks = [gap_mark] if gap else []
    if channels != 2:
        return [pytest.param(None, *values, id=case_id, marks=gap_marks)]

    params = []
    for layout in ("planar", "interleaved"):
        marks = gap_marks
        if layout in c2_gap_layouts:
            mark = _C2_GAP
            if gap:
                mark = c2_gap_mark or gap_mark
            marks = [mark]
        params.append(
            pytest.param(layout, *values, id=f"{case_id}-{layout}", marks=marks)
        )
    return params


def _gap_images(dtypes, *, sizes=((17, 11),)):
    for dtype, channels, (height, width), batch in itertools.product(
        dtypes, _CHANNELS, sizes, _BATCHES
    ):
        yield from _gap_params(
            channels,
            (batch, channels, height, width, dtype),
            case_id=(
                f"{'chw' if batch is None else f'n{batch}chw'}-c{channels}-"
                f"{dtype}-{height}x{width}"
            ),
            c2_gap_layouts=("planar",),
        )


_VALUE_KINDS = ("full", "10-240", "50-200", "100-140", "constant", "two-value")


_VALUE_SHAPES = ((3, 64, 96), (2, 3, 64, 96), (1, 17, 23), (2, 1, 17, 23))


_FACTOR_SWEEP = tuple(float(f) for f in np.random.default_rng(0).uniform(0.0, 4.0, 24))


def _value_images(torch, dtype, kind):
    """Images whose levels follow ``kind`` on the uint8 scale, scaled to ``dtype``."""
    top = torch.iinfo(dtype).max
    for seed, shape in enumerate(_VALUE_SHAPES):
        generator = torch.Generator(device="cuda").manual_seed(seed)
        x = torch.randint(0, top + 1, shape, generator=generator, device="cuda")
        if kind == "constant":
            x = torch.full_like(x, 123 * top // 255)
        elif kind == "two-value":
            x = (x % 2) * (37 * top // 255) + 90 * top // 255
        elif kind != "full":
            low, high = (int(level) * top // 255 for level in kind.split("-"))
            x = x % (high - low) + low
        yield x.to(dtype)


_NATIVE_BLEND_ORDER = pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="native AdjustSaturation blend rounding order (native fix pending)",
)


_BLEND_ORDER_GAPS = {
    ("adjust_saturation", "uint8", 0.3): ("full", "10-240", "50-200"),
    ("adjust_saturation", "uint8", 1.0838091373443604): _VALUE_KINDS[:4],
    ("adjust_saturation", "uint8", 1.266): _VALUE_KINDS[:4],
    ("adjust_saturation", "uint8", 1.3): _VALUE_KINDS[:4],
    ("adjust_saturation", "uint8", 1.7): ("full", "10-240", "50-200"),
    ("adjust_saturation", "uint8", 7.3): _VALUE_KINDS[:4],
    ("adjust_saturation", "uint8", "sweep"): _VALUE_KINDS[:4],
}


_BLEND_FACTORS = (0.0, 0.3, 0.5, 0.6, 1.0, 1.0838091373443604, 1.266, 1.3, 1.7, 7.3)


_VALUE_RANGE_OPS = [
    ("adjust_brightness", "uint8", (0.3, 1.266, 1.7)),
    ("adjust_contrast", "uint8", (0.3, 1.266, 1.7)),
    ("adjust_gamma", "uint8", (0.5, 1.266, 2.2)),
    ("adjust_saturation", "uint8", (*_BLEND_FACTORS, "sweep")),
    ("adjust_sharpness", "uint8", (*_BLEND_FACTORS, "sweep")),
    ("adjust_sharpness", "uint16", (*_BLEND_FACTORS, "sweep")),
    ("equalize", "uint8", (None,)),
    ("invert", "uint8", (None,)),
    ("posterize", "uint8", (1, 4, 7)),
    ("rgb_to_grayscale", "uint8", (1, 3)),
    ("solarize", "uint8", (64, 128, 200)),
]


def _value_range_params():
    for name, dtype, factors in _VALUE_RANGE_OPS:
        for factor, kind in itertools.product(factors, _VALUE_KINDS):
            gaps = _BLEND_ORDER_GAPS.get((name, dtype, factor), ())
            yield pytest.param(
                name,
                dtype,
                factor,
                kind,
                id=f"{name}-{dtype}-{factor}-{kind}",
                marks=[_NATIVE_BLEND_ORDER] if kind in gaps else [],
            )


_RESIZE_SCENARIOS = [
    ("down-landscape", (240, 320), [120, 160], None),
    ("down-portrait", (320, 240), [160, 120], None),
    ("up", (120, 160), [240, 320], None),
    ("mixed", (240, 320), [320, 160], None),
    ("one-axis-down", (240, 320), [240, 160], None),
    ("odd-noop", (17, 29), [17, 29], None),
    ("scalar", (240, 320), 224, None),
    ("list-one", (320, 240), [224], None),
    ("tuple-one-cap", (240, 640), (224,), 320),
    ("cap-nonbinding", (240, 320), 224, 640),
    ("imagenet", (256, 256), [224, 224], None),
    ("inception", (320, 320), (299, 299), None),
    ("detector", (720, 1280), [640, 640], None),
    ("up-2x", (112, 112), [224, 224], None),
    ("square-noop", (224, 224), [224, 224], None),
]


def _assert_resize_parity(
    assert_tv_parity, tvf, image, size, max_size, interpolation, antialias
):
    if image.dtype == torch_module.int16 and interpolation is InterpolationMode.BICUBIC:
        image = image.remainder(256)
    atol = 1
    if image.is_floating_point():
        atol = _F16_INTERP_ATOL if _is_float16(image) else 3e-5
    if interpolation in (InterpolationMode.NEAREST, InterpolationMode.NEAREST_EXACT):
        atol = 0
    assert_tv_parity(
        cvcuda_tv.resize,
        tvf.resize,
        image,
        size,
        interpolation=interpolation,
        max_size=max_size,
        antialias=antialias,
        atol=atol,
    )


_NEAREST_LEGACY_SIZES = [
    ("down", 17, 11, [12, 13]),
    ("up", 11, 17, [22, 34]),
    ("nonint", 7, 5, [11, 13]),
]


_RESIZED_CROP_CASES = [
    (2, 3, 12, 18, [8, 12]),
    (2, 3, 12, 18, [24, 36]),
    (2, 3, 12, 18, [12, 18]),
    (2, 3, 12, 18, [18, 12]),
    (2, 2, 5, 7, (19, 32)),
    (0, 0, 17, 29, 224),
    (0, 0, 8, 12, [10, 14]),
    (9, 17, 8, 12, [10, 14]),
]


def _resized_crop_params(dtypes):
    for (
        (case_index, (top, left, height, width, size)),
        dtype,
        channels,
        batch,
        interpolation,
        antialias,
    ) in itertools.product(
        enumerate(_RESIZED_CROP_CASES),
        dtypes,
        _CHANNELS,
        _BATCHES,
        _RESIZE_INTERPOLATIONS,
        (True, False),
    ):
        atol = (
            0
            if interpolation is InterpolationMode.NEAREST_EXACT
            else (
                _F16_INTERP_ATOL
                if dtype == "float16"
                else 3e-5 if dtype == "float32" else 1
            )
        )
        yield from _gap_params(
            channels,
            (batch, channels, 17, 29, dtype),
            top,
            left,
            height,
            width,
            size,
            interpolation,
            antialias,
            atol,
            case_id=(
                f"case{case_index}-{dtype}-c{channels}-n{batch}-"
                f"{interpolation.value}-aa{antialias}"
            ),
            c2_gap_layouts=("planar",),
        )


_OUT_OF_BOUNDS_CROPS = [
    pytest.param(-5, -5, 30, 20, id="all-sides"),
    pytest.param(-5, 3, 10, 5, id="top"),
    pytest.param(16, 3, 10, 5, id="bottom"),
    pytest.param(5, -5, 10, 5, id="left"),
    pytest.param(5, 9, 10, 5, id="right"),
    pytest.param(-5, -5, 20, 10, id="corner-tl"),
    pytest.param(-5, 5, 20, 10, id="corner-tr"),
    pytest.param(5, -5, 20, 10, id="corner-bl"),
    pytest.param(5, 5, 20, 10, id="corner-br"),
    pytest.param(-100, -100, 5, 5, id="disjoint-before"),
    pytest.param(100, 100, 5, 5, id="disjoint-after"),
]


_CROP_PAD_GAP = pytest.mark.xfail(
    strict=True,
    raises=RuntimeError,
    reason=(
        "an out-of-bounds crop pads through copymakeborder, which takes two "
        "channels only for 8-bit unsigned"
    ),
)


_CENTER_CROP_SIZES = [
    (3, 5),
    (5, 3),
    (4, 4),
    (21, 9),
    (13, 15),
    (19, 14),
    3,
    (4,),
    [5],
    (17, 11),
]


def _center_crop_params(dtypes):
    for dtype, channels, batch, (size_index, size) in itertools.product(
        dtypes,
        _CHANNELS,
        _BATCHES,
        enumerate(_CENTER_CROP_SIZES),
    ):
        crop_height, crop_width = (
            (size, size)
            if isinstance(size, int)
            else (size[0], size[0]) if len(size) == 1 else size
        )
        interleaved_gap = dtype != "uint8" and (crop_height > 17 or crop_width > 11)
        yield from _gap_params(
            channels,
            (batch, channels, 17, 11, dtype),
            size,
            case_id=(
                f"{'chw' if batch is None else f'n{batch}chw'}-c{channels}-"
                f"{dtype}-17x11-size{size_index}"
            ),
            c2_gap_layouts=(
                ("planar", "interleaved") if interleaved_gap else ("planar",)
            ),
        )


_PAD_CASES = [
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
]


_PAD_U16_SYMMETRIC_GAP = pytest.mark.xfail(
    strict=True,
    raises=RuntimeError,
    reason="CV-CUDA symmetric padding does not support uint16",
)


def _pad_params(dtypes):
    for dtype, (padding, fill, padding_mode), channels, batch in itertools.product(
        dtypes, _PAD_CASES, _CHANNELS, _BATCHES
    ):
        u16_symmetric_gap = dtype == "uint16" and padding_mode == "symmetric"
        yield from _gap_params(
            channels,
            (batch, channels, 17, 29, dtype),
            padding,
            fill,
            padding_mode,
            case_id=f"{dtype}-c{channels}-n{batch}-{padding}-{padding_mode}",
            gap=u16_symmetric_gap,
            gap_mark=_PAD_U16_SYMMETRIC_GAP,
            c2_gap_layouts=(
                ("planar", "interleaved") if dtype != "uint8" else ("planar",)
            ),
        )


_AFFINE_CASES = [
    dict(angle=0.0, translate=[0, 0], scale=1.0, shear=0.0),
    dict(angle=-10.9, translate=[1, -3], scale=0.5, shear=35.6),
    dict(angle=18.0, translate=[2.5, -1.5], scale=1.0, shear=[5.4, 21.8]),
    dict(angle=20.0, translate=[2, -3], scale=1.2, shear=[5.0, 8.0]),
    # A one-element shear sequence shears *both* axes in torchvision, unlike a
    # bare number which shears x only. Cover both spellings of a single value.
    dict(angle=-10.9, translate=[6.3, -0.6], scale=0.5, shear=[-37.7]),
    dict(angle=18.0, translate=[1, -3], scale=1.0, shear=(5.3,)),
]


def _affine_params(dtypes):
    for (
        (case_index, kw),
        interpolation,
        fill,
        dtype,
        channels,
        batch,
    ) in itertools.product(
        enumerate(_AFFINE_CASES),
        _WARP_INTERPOLATIONS,
        (None, 7, [7]),
        dtypes,
        _CHANNELS,
        _BATCHES,
    ):
        yield from _gap_params(
            channels,
            (batch, channels, 17, 11, dtype),
            kw,
            interpolation,
            fill,
            case_id=f"case{case_index}-{dtype}-c{channels}-n{batch}-{interpolation.value}-fill{fill}",
        )


def _rotate_params(dtypes):
    for (
        angle,
        center,
        interpolation,
        fill,
        expand,
        dtype,
        channels,
        batch,
    ) in itertools.product(
        (-10.9, 30.0, 60.0),
        (None, [4.5, 7.25], [3, 2]),
        _WARP_INTERPOLATIONS,
        (None, 7, [7]),
        (False, True),
        dtypes,
        _CHANNELS,
        _BATCHES,
    ):
        yield from _gap_params(
            channels,
            (batch, channels, 17, 11, dtype),
            angle,
            center,
            interpolation,
            fill,
            expand,
            case_id=(
                f"{dtype}-c{channels}-n{batch}-a{angle}-center{center}-"
                f"{interpolation.value}-fill{fill}-expand{expand}"
            ),
        )


_PERSPECTIVE_CASES = [
    (
        [[0, 0], [10, 0], [10, 16], [0, 16]],
        [[1, 1], [9, 2], [10, 14], [0, 15]],
    ),
    (
        [[1, 1], [9, 2], [10, 14], [0, 15]],
        [[0, 0], [10, 0], [10, 16], [0, 16]],
    ),
]


def _perspective_params(dtypes):
    for (
        (case_index, (startpoints, endpoints)),
        interpolation,
        fill,
        dtype,
        channels,
        batch,
    ) in itertools.product(
        enumerate(_PERSPECTIVE_CASES),
        _WARP_INTERPOLATIONS,
        (None, 7, [7]),
        dtypes,
        _CHANNELS,
        _BATCHES,
    ):
        yield from _gap_params(
            channels,
            (batch, channels, 17, 11, dtype),
            startpoints,
            endpoints,
            interpolation,
            fill,
            case_id=f"case{case_index}-{dtype}-c{channels}-n{batch}-{interpolation.value}-fill{fill}",
        )


@pytest.mark.parametrize("name, dtype, factor, kind", list(_value_range_params()))
def test_integer_value_range_bit_exact(
    assert_tv_parity, torch, tvf, name, dtype, factor, kind
):
    factors = _FACTOR_SWEEP if factor == "sweep" else (factor,)
    for image in _value_images(torch, getattr(torch, dtype), kind):
        for value in factors:
            args = () if value is None else (value,)
            assert_tv_parity(getattr(cvcuda_tv, name), getattr(tvf, name), image, *args)


@pytest.mark.parametrize("kind", _VALUE_KINDS)
def test_autocontrast_differs_from_torchvision_only_at_exact_multiples(
    torch, tvf, kind
):
    """Torchvision's float32 division can round exact integer quotients one lower."""
    for image in _value_images(torch, torch.uint8, kind):
        x = image.to(torch.int64)
        low = x.amin(dim=(-2, -1), keepdim=True)
        span = x.amax(dim=(-2, -1), keepdim=True) - low
        numerator = (x - low) * 255
        exact = torch.where(span == 0, x, numerator // span.clamp(min=1))
        whole = (numerator % span.clamp(min=1) == 0) & (span != 0)
        got = cvcuda_tv.autocontrast(image).to(torch.int64)
        expected = tvf.autocontrast(image).to(torch.int64)
        assert torch.equal(got, exact)
        mismatch = got != expected
        assert torch.equal(mismatch, whole & (expected < exact))
        assert torch.equal(expected[mismatch], exact[mismatch] - 1)


@pytest.mark.parametrize("batch", [None, 3])
@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("size", [(32, 32), (64, 48), (224, 224)], ids=str)
def test_equalize_parity_realistic_uint8(
    torch, assert_tv_parity, tvf, size, channels, batch
):
    # make_tensor's high is exclusive and the default fixtures are 17x11, so
    # they never reach 255 or 256 pixels per plane.
    shape = (channels, *size) if batch is None else (batch, channels, *size)
    generator = torch.Generator(device="cuda").manual_seed(0)
    image = torch.randint(
        0, 256, shape, generator=generator, device="cuda", dtype=torch.uint8
    )
    image[..., 0, 0] = 255
    assert_tv_parity(cvcuda_tv.equalize, tvf.equalize, image)


@pytest.mark.parametrize("scenario", _RESIZE_SCENARIOS, ids=lambda row: row[0])
@pytest.mark.parametrize("dtype", ["uint8", "float16", "float32"])
@pytest.mark.parametrize("channels", [1, 2, 3, 4])
@pytest.mark.parametrize("batch", [None, 1, 3])
@pytest.mark.parametrize("interpolation", _RESIZE_INTERPOLATIONS)
@pytest.mark.parametrize("antialias", [True, False, None])
def test_resize_production_sizes(
    assert_tv_parity,
    tvf,
    torch,
    scenario,
    dtype,
    channels,
    batch,
    interpolation,
    antialias,
):
    _, (height, width), size, max_size = scenario
    torch.manual_seed(0)
    shape = (channels, height, width)
    if batch is not None:
        shape = (batch, *shape)
    dtype = getattr(torch, dtype)
    image = torch.testing.make_tensor(
        shape,
        low=0,
        high=1 if dtype.is_floating_point else torch.iinfo(dtype).max,
        dtype=dtype,
        device="cuda",
    ).contiguous()
    _assert_resize_parity(
        assert_tv_parity, tvf, image, size, max_size, interpolation, antialias
    )


@pytest.mark.parametrize("operator", ["resize", "resized_crop"])
@pytest.mark.parametrize(
    "mode", [InterpolationMode.BILINEAR, InterpolationMode.BICUBIC]
)
@pytest.mark.parametrize("antialias", [True, False])
def test_resize_int16_pattern(assert_tv_parity, torch, tvf, operator, mode, antialias):
    image = (
        torch.arange(3 * 13 * 17, device="cuda", dtype=torch.int16)
        .reshape(3, 13, 17)
        .remainder(256)
    )
    arguments = {"size": [7, 9], "interpolation": mode}
    if operator == "resized_crop":
        arguments.update(top=1, left=1, height=9, width=11)
    assert_tv_parity(
        getattr(cvcuda_tv, operator),
        getattr(tvf, operator),
        image,
        **arguments,
        antialias=antialias,
        atol=1,
    )


@pytest.mark.parametrize("operator", ["resize", "resized_crop"])
@pytest.mark.parametrize("repeats,size", [(1, 9), (4, 5)])
@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("batched", [False, True])
def test_resize_bicubic_rounding_pattern(
    assert_tv_parity, torch, tvf, operator, repeats, size, channels, batched
):
    image = torch.tensor(
        [[[0, 255, 255], [0, 255, 255], [255, 255, 255]]],
        dtype=torch.uint8,
        device="cuda",
    )
    image = (
        image.repeat(channels, 1, 1)
        .repeat_interleave(repeats, -2)
        .repeat_interleave(repeats, -1)
    )
    if batched:
        image = torch.stack([image, image.flip(-1)])
    arguments = {"size": [size, size], "interpolation": InterpolationMode.BICUBIC}
    if operator == "resized_crop":
        height, width = image.shape[-2:]
        image = torch.nn.functional.pad(image, (1, 1, 1, 1), value=127)
        arguments.update(top=1, left=1, height=height, width=width)
    assert_tv_parity(
        getattr(cvcuda_tv, operator),
        getattr(tvf, operator),
        image,
        **arguments,
        antialias=True,
        atol=1,
    )


@pytest.mark.parametrize(
    "image,size",
    [
        pytest.param(
            (batch, ch, sh, sw, dt),
            sz,
            id=(
                f"{'chw' if batch is None else f'n{batch}chw'}"
                f"-c{ch}-{dt}-{sh}x{sw}-{sid}"
            ),
        )
        for dt, ch, batch, (sid, sh, sw, sz) in itertools.product(
            ("uint8", "float32", "float16"),
            (1, 3, 4),
            (None, 2),
            _NEAREST_LEGACY_SIZES,
        )
    ],
    indirect=["image"],
)
def test_resize_nearest_parity(assert_tv_parity, tvf, image, size):
    assert_tv_parity(
        cvcuda_tv.resize,
        tvf.resize,
        image,
        size,
        interpolation=InterpolationMode.NEAREST,
    )


@pytest.mark.parametrize("max_size", [7, 11, 23])
@pytest.mark.parametrize(
    "image", _images(_MINIMAL_DTYPES, channels=(1, 3, 4)), indirect=True
)
def test_resize_none_parity(assert_tv_parity, tvf, image, max_size):
    """``size=None`` scales the longer edge to ``max_size``."""
    assert_tv_parity(
        cvcuda_tv.resize,
        tvf.resize,
        image,
        None,
        max_size=max_size,
        atol=1 if not image.is_floating_point() else 3e-5,
    )


@pytest.mark.parametrize("canvas", [(17, 11), (11, 17), (13, 13)])
@pytest.mark.parametrize("max_size", [None, 5, 8, 17])
@pytest.mark.parametrize("size", [None, 7, [7], (7,), [7, 9], (7, 9)])
def test_resize_output_size_matches_torchvision(size, max_size, canvas):
    """Both the computed size and the rejections have to match torchvision.

    Silently accepting a ``size``/``max_size`` pair torchvision refuses is worse
    than a numeric gap: the caller gets a differently shaped image and no signal
    that anything was wrong.
    """
    compute = pytest.importorskip(
        "torchvision.transforms.v2.functional._geometry"
    )._compute_resized_output_size

    height, width = canvas
    try:
        expected = compute(canvas, size=size, max_size=max_size)
    except Exception as error:  # noqa: BLE001 - mirror whatever torchvision raises
        with pytest.raises(type(error)):
            _resize_output_hw(size, height, width, max_size)
        return

    assert list(_resize_output_hw(size, height, width, max_size)) == expected


@pytest.mark.xfail(
    strict=True,
    reason="CV-CUDA saturates int16 bicubic overshoot; torchvision wraps the rounded float cast",
)
@pytest.mark.parametrize("antialias", [False, True])
@pytest.mark.parametrize(
    "name,args",
    [("resize", ([9, 9],)), ("resized_crop", (0, 0, 3, 3, [9, 9]))],
)
def test_resize_bicubic_int16_overshoot(
    assert_tv_parity, torch, tvf, name, args, antialias
):
    image = torch.tensor(
        [[[0, 32767, 32767], [0, 32767, 32767], [32767, 32767, 32767]]],
        device="cuda",
        dtype=torch.int16,
    )
    assert_tv_parity(
        getattr(cvcuda_tv, name),
        getattr(tvf, name),
        image,
        *args,
        interpolation=InterpolationMode.BICUBIC,
        antialias=antialias,
    )


@pytest.mark.parametrize(
    "image",
    _images(
        dtypes=("int32", "int64", "float64"),
        channels=(1, 3, 4),
        marks=_RESIZE_DTYPE_GAP,
    ),
    indirect=True,
)
def test_resize_dtype_gaps(assert_tv_parity, tvf, image):
    assert_tv_parity(cvcuda_tv.resize, tvf.resize, image, [10, 8])


@pytest.mark.parametrize("top,left,height,width", _OUT_OF_BOUNDS_CROPS)
@pytest.mark.parametrize(
    "image",
    _images(
        (
            "uint8",
            "uint16",
            "int8",
            "int16",
            "int32",
            "float16",
            "float32",
            "float64",
        ),
        channels=(1, 3, 4),
        sizes=((21, 11),),
    )
    # Two channels stays a gap: copymakeborder takes two channels only for 8-bit
    # unsigned. int64 is left out of this test rather than marked -- whether it
    # reaches convertto depends on the crop geometry, so only some of its cases
    # would fail; test_crop_dtype_gaps covers int64 crop on its own.
    + _images(("float32",), channels=(2,), sizes=((21, 11),), marks=_CROP_PAD_GAP),
    indirect=True,
)
def test_crop_out_of_bounds_parity(
    assert_tv_parity, tvf, image, top, left, height, width
):
    assert_tv_parity(cvcuda_tv.crop, tvf.crop, image, top, left, height, width)


@pytest.mark.parametrize("top,left,height,width", [(5, 3, 10, 5)])
@pytest.mark.parametrize(
    "image",
    _images(dtypes=("int64",), channels=(1, 3, 4)),
    indirect=True,
)
def test_crop_dtype_gaps(assert_tv_parity, tvf, image, top, left, height, width):
    assert_tv_parity(cvcuda_tv.crop, tvf.crop, image, top, left, height, width)


@pytest.mark.parametrize(
    "image",
    _images(
        dtypes=("int32", "int64", "float64"), channels=(1, 3, 4), marks=_WARP_DTYPE_GAP
    ),
    indirect=True,
)
def test_affine_dtype_gaps(assert_tv_parity, tvf, image):
    assert_tv_parity(
        cvcuda_tv.affine,
        tvf.affine,
        image,
        angle=10.0,
        translate=[0, 0],
        scale=1.0,
        shear=0.0,
    )


@pytest.mark.parametrize(
    "assert_tv_parity,image,kw,interpolation,fill",
    [
        row
        for row in _affine_params((*_UINTS, "float16", "float32"))
        if row.values[1][1] == 2
    ],
    indirect=["assert_tv_parity", "image"],
)
def test_affine_two_channel_dispatch(
    assert_tv_parity, tvf, torch, image, kw, interpolation, fill
):
    assert_tv_parity(
        cvcuda_tv.affine,
        tvf.affine,
        image,
        **kw,
        interpolation=interpolation,
        fill=fill,
        atol=(
            _F16_INTERP_ATOL
            if image.dtype == torch.float16
            else 1e-5 if image.is_floating_point() else 1
        )
        + _fill_blend_atol(image.dtype, fill),
        max_bad_frac=0.1 if image.dtype == torch.float16 else 0,
    )


@pytest.mark.parametrize(
    "image,size",
    [
        pytest.param(
            (None, 3, 17, 11, "int64"),
            (5, 5),
            id="int64-c3-nopad",
        ),
        pytest.param(
            (None, 3, 17, 11, "int32"),
            (21, 14),
            id="int32-c3-pad",
        ),
        pytest.param(
            (None, 3, 17, 11, "int64"),
            (21, 14),
            id="int64-c3-pad",
        ),
        pytest.param(
            (None, 3, 17, 11, "float64"),
            (21, 14),
            id="float64-c3-pad",
        ),
    ],
    indirect=["image"],
)
def test_center_crop_dtype_gaps(assert_tv_parity, tvf, image, size):
    assert_tv_parity(cvcuda_tv.center_crop, tvf.center_crop, image, size)


@pytest.mark.parametrize(
    "assert_tv_parity,image,size",
    [
        row
        for row in _center_crop_params((*_UINTS, "float16", "float32"))
        if row.values[1][1] == 2
    ],
    indirect=["assert_tv_parity", "image"],
)
def test_center_crop_two_channel_dispatch(assert_tv_parity, tvf, image, size):
    assert_tv_parity(cvcuda_tv.center_crop, tvf.center_crop, image, size)


@pytest.mark.parametrize(
    "image",
    _images(
        dtypes=("int16", "int32", "int64", "float64"),
        channels=(1, 3, 4),
        marks=pytest.mark.xfail(
            strict=True,
            raises=RuntimeError,
            reason="cvcuda.remap rejects signed integer and float64 inputs",
        ),
    ),
    indirect=True,
)
def test_elastic_dtype_gaps(image, torch):
    import cvcuda

    h, w = image.shape[-2], image.shape[-1]
    displacement = cvcuda.as_tensor(
        torch.zeros(1, h, w, 2, dtype=torch.float32, device=image.device),
        layout="NHWC",
    )
    layout = "CHW" if image.ndim == 3 else "NCHW"
    cv_image = cvcuda.as_tensor(image.contiguous(), layout=layout)
    cvcuda_tv.elastic(cv_image, displacement)


@pytest.mark.parametrize(
    "image",
    _images(dtypes=("int8", "int32", "int64", "float64"), channels=(1, 3, 4)),
    indirect=True,
)
def test_pad_dtype_gaps(assert_tv_parity, tvf, image):
    assert_tv_parity(
        cvcuda_tv.pad,
        tvf.pad,
        image,
        (1, 2),
        fill=None,
        padding_mode="constant",
    )


@pytest.mark.parametrize(
    "assert_tv_parity,image,padding,fill,padding_mode",
    [
        row
        for row in _pad_params((*_UINTS, "float16", "float32"))
        if row.values[1][1] == 2 or row.values[1][-1] == "uint16"
    ],
    indirect=["assert_tv_parity", "image"],
)
def test_pad_two_channel_dispatch(
    assert_tv_parity, tvf, image, padding, fill, padding_mode
):
    assert_tv_parity(
        cvcuda_tv.pad,
        tvf.pad,
        image,
        padding,
        fill=fill,
        padding_mode=padding_mode,
    )


@pytest.mark.parametrize(
    "image",
    _images(
        dtypes=("int32", "int64", "float64"), channels=(1, 3, 4), marks=_WARP_DTYPE_GAP
    ),
    indirect=True,
)
def test_perspective_dtype_gaps(assert_tv_parity, tvf, image):
    assert_tv_parity(
        cvcuda_tv.perspective,
        tvf.perspective,
        image,
        [[0, 0], [10, 0], [10, 16], [0, 16]],
        [[1, 1], [9, 2], [10, 14], [0, 15]],
    )


@pytest.mark.parametrize(
    "assert_tv_parity,image,startpoints,endpoints,interpolation,fill",
    [
        row
        for row in _perspective_params((*_UINTS, "float16", "float32"))
        if row.values[1][1] == 2
    ],
    indirect=["assert_tv_parity", "image"],
)
def test_perspective_two_channel_dispatch(
    assert_tv_parity, tvf, torch, image, startpoints, endpoints, interpolation, fill
):
    assert_tv_parity(
        cvcuda_tv.perspective,
        tvf.perspective,
        image,
        startpoints,
        endpoints,
        interpolation=interpolation,
        fill=fill,
        atol=(
            _F16_INTERP_ATOL
            if image.dtype == torch.float16
            else 1e-5 if image.is_floating_point() else 1
        )
        + _fill_blend_atol(image.dtype, fill),
        max_bad_frac=0.1 if image.dtype == torch.float16 else 0,
    )


@pytest.mark.parametrize(
    "image",
    _images(
        dtypes=("int32", "int64", "float64"),
        channels=(1, 3, 4),
        marks=_RESIZE_DTYPE_GAP,
    ),
    indirect=True,
)
def test_resized_crop_dtype_gaps(assert_tv_parity, tvf, image):
    assert_tv_parity(
        cvcuda_tv.resized_crop,
        tvf.resized_crop,
        image,
        2,
        3,
        12,
        8,
        [8, 6],
    )


@pytest.mark.parametrize(
    "assert_tv_parity,image,top,left,height,width,size,interpolation,antialias,atol",
    [
        row
        for row in _resized_crop_params((*_MINIMAL_DTYPES, "int16", "float16"))
        if row.values[1][1] == 2
    ],
    indirect=["assert_tv_parity", "image"],
)
def test_resized_crop_two_channel_dispatch(
    assert_tv_parity,
    tvf,
    image,
    top,
    left,
    height,
    width,
    size,
    interpolation,
    antialias,
    atol,
):
    if image.dtype == torch_module.int16 and interpolation is InterpolationMode.BICUBIC:
        image = image.remainder(256)
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


@pytest.mark.parametrize(
    "image",
    _images(
        dtypes=("int32", "int64", "float64"), channels=(1, 3, 4), marks=_WARP_DTYPE_GAP
    ),
    indirect=True,
)
def test_rotate_dtype_gaps(assert_tv_parity, tvf, image):
    assert_tv_parity(cvcuda_tv.rotate, tvf.rotate, image, 30.0)


@pytest.mark.parametrize(
    "assert_tv_parity,image,angle,center,interpolation,fill,expand",
    [
        row
        for row in _rotate_params((*_UINTS, "float16", "float32"))
        if row.values[1][1] == 2
    ],
    indirect=["assert_tv_parity", "image"],
)
def test_rotate_two_channel_dispatch(
    assert_tv_parity, tvf, torch, image, angle, center, interpolation, fill, expand
):
    assert_tv_parity(
        cvcuda_tv.rotate,
        tvf.rotate,
        image,
        angle,
        interpolation=interpolation,
        center=center,
        fill=fill,
        expand=expand,
        atol=(
            (
                _F16_INTERP_ATOL
                if image.dtype == torch.float16
                else 1e-5 if image.is_floating_point() else 1
            )
            + _fill_blend_atol(image.dtype, fill)
            if interpolation is InterpolationMode.BILINEAR
            else 0
        ),
        max_bad_frac=(
            0.1
            if image.dtype == torch.float16
            else 0.05 if interpolation is InterpolationMode.NEAREST else 0
        ),
    )


@pytest.mark.parametrize(
    "name,args,kwargs,dtype,channels,atol",
    [
        pytest.param(
            "adjust_brightness", (1.3,), {}, "uint8", 3, 1, id="brightness-1.3"
        ),
        pytest.param("adjust_contrast", (0.7,), {}, "uint8", 3, 1, id="contrast-0.7"),
        pytest.param("adjust_gamma", (1.5, 0.9), {}, "uint8", 3, 1, id="gamma-1.5-0.9"),
        pytest.param("adjust_hue", (0.2,), {}, "uint8", 3, 1, id="hue-0.2"),
        pytest.param(
            "adjust_saturation", (1.4,), {}, "uint8", 3, 1, id="saturation-1.4"
        ),
        pytest.param("adjust_sharpness", (2.0,), {}, "uint8", 3, 1, id="sharpness-2"),
        pytest.param(
            "affine",
            (15.0, [1, 2], 1.1, [5.0, 0.0]),
            {},
            "uint8",
            3,
            0,
            id="affine-translated-sheared",
        ),
        pytest.param("autocontrast", (), {}, "uint8", 3, 1, id="autocontrast"),
        pytest.param(
            "center_crop", ([8, 10],), {}, "uint8", 3, 0, id="center_crop-8x10"
        ),
        pytest.param("crop", (2, 3, 8, 9), {}, "uint8", 3, 0, id="crop-offset-8x9"),
        pytest.param(
            "elastic",
            (torch_module.linspace(-0.1, 0.1, 16 * 20 * 2).reshape(1, 16, 20, 2),),
            {"interpolation": InterpolationMode.BILINEAR},
            "uint8",
            3,
            1,
            id="elastic-displaced-bilinear",
        ),
        pytest.param("equalize", (), {}, "uint8", 3, 0, id="equalize"),
        pytest.param(
            "erase",
            (2, 3, 4, 5, torch_module.tensor([3, 4, 5]).reshape(3, 1, 1)),
            {"inplace": False},
            "uint8",
            3,
            0,
            id="erase-region-outofplace",
        ),
        pytest.param("five_crop", ([6, 8],), {}, "uint8", 3, 0, id="five_crop-6x8"),
        pytest.param(
            "gaussian_blur",
            ([5, 5], [1.5, 1.5]),
            {},
            "float32",
            3,
            1e-6,
            id="gaussian_blur-kernel5",
        ),
        pytest.param(
            "gaussian_noise",
            (),
            {"mean": 0.125, "sigma": 0.0},
            "float32",
            3,
            0,
            id="gaussian_noise-mean0.125",
        ),
        pytest.param("grayscale_to_rgb", (), {}, "uint8", 1, 0, id="grayscale-1to3"),
        pytest.param("horizontal_flip", (), {}, "uint8", 3, 0, id="horizontal_flip"),
        pytest.param("invert", (), {}, "uint8", 3, 0, id="invert"),
        pytest.param("jpeg", (50,), {}, "uint8", 1, 0, id="jpeg-flat-quality50"),
        pytest.param(
            "normalize",
            ([0.4, 0.5, 0.6], [0.2, 0.3, 0.4]),
            {"inplace": False},
            "float32",
            3,
            1e-6,
            id="normalize-outofplace",
        ),
        pytest.param(
            "pad",
            ([1, 2, 3, 4],),
            {"padding_mode": "reflect"},
            "uint8",
            3,
            0,
            id="pad-reflect-asymmetric",
        ),
        pytest.param(
            "permute_channels", ([2, 0, 1],), {}, "uint8", 3, 0, id="channels-201"
        ),
        pytest.param(
            "perspective",
            (
                [[0, 0], [19, 0], [19, 15], [0, 15]],
                [[1, 2], [20, 2], [20, 17], [1, 17]],
            ),
            {"interpolation": InterpolationMode.NEAREST},
            "uint8",
            3,
            0,
            id="perspective-translated-nearest",
        ),
        pytest.param("posterize", (3,), {}, "uint8", 3, 0, id="posterize-bits3"),
        pytest.param(
            "resize",
            ([9, 13],),
            {"interpolation": InterpolationMode.NEAREST},
            "uint8",
            3,
            0,
            id="resize-9x13-nearest",
        ),
        pytest.param(
            "resized_crop",
            (1, 2, 10, 12, [7, 9]),
            {"interpolation": InterpolationMode.NEAREST},
            "uint8",
            3,
            0,
            id="resized_crop-offset-7x9-nearest",
        ),
        pytest.param("rgb_to_grayscale", (1,), {}, "uint8", 3, 1, id="grayscale-3to1"),
        pytest.param(
            "rotate",
            (30.0,),
            {"expand": True},
            "uint8",
            3,
            0,
            id="rotate-30-expanded",
        ),
        pytest.param("solarize", (128,), {}, "uint8", 3, 0, id="solarize-threshold128"),
        pytest.param(
            "ten_crop", ([6, 8], True), {}, "uint8", 3, 0, id="ten_crop-vertical"
        ),
        pytest.param(
            "to_dtype",
            (),
            {"dtype": torch_module.float32, "scale": True},
            "uint8",
            3,
            1e-6,
            id="to_dtype-uint8tofloat32-scaled",
        ),
        pytest.param("vertical_flip", (), {}, "uint8", 3, 0, id="vertical_flip"),
    ],
)
@pytest.mark.parametrize("batched", [False, True], ids=["CHW", "NCHW"])
def test_allocating_scenarios(
    assert_tv_parity, torch, tvf, name, args, kwargs, dtype, channels, atol, batched
):
    torch.manual_seed(0)
    shape = (channels, 16, 20)
    if batched:
        shape = (2, *shape)
    dtype = getattr(torch, dtype)
    image = torch.testing.make_tensor(
        shape,
        low=0,
        high=1 if dtype.is_floating_point else 256,
        dtype=dtype,
        device="cuda",
    ).contiguous()
    cv_op, tv_op = getattr(cvcuda_tv, name), getattr(tvf, name)
    if name == "jpeg":
        image.fill_(64)
        tv_op = lambda x, quality: tvf.jpeg(x.cpu(), quality).to(x.device)  # noqa: E731
    assert_tv_parity(cv_op, tv_op, image, *args, fresh=True, atol=atol, **kwargs)


@pytest.mark.parametrize(
    "image,factor",
    [
        pytest.param(
            image.values[0],
            factor,
            id=f"{image.id}-factor{factor}",
            marks=(
                [_INT64_BRIGHTNESS_GAP]
                if image.values[0][-1] == "int64" and factor > 1
                else []
            ),
        )
        for image in _images(("int64",))
        for factor in (0.0, 0.5, 1.0, 5.0)
    ],
    indirect=["image"],
)
def test_adjust_brightness_large_int64(assert_tv_parity, tvf, image, factor):
    assert_tv_parity(
        cvcuda_tv.adjust_brightness,
        tvf.adjust_brightness,
        image,
        factor,
        atol=1e-3 if _is_float16(image) else 1e-6 if image.is_floating_point() else 1,
    )


@pytest.mark.parametrize(
    "top,left,height,width",
    [(5, 3, 10, 5), (0, 0, 21, 11), (0, 0, 1, 1), (20, 10, 1, 1)],
)
@pytest.mark.parametrize(
    "assert_tv_parity,image",
    [
        row
        for row in _gap_images(_TORCHVISION_DTYPES, sizes=((21, 11),))
        if row.values[1][1] == 2
    ],
    indirect=["assert_tv_parity", "image"],
)
def test_crop_two_channel_dispatch(
    assert_tv_parity, tvf, image, top, left, height, width
):
    assert_tv_parity(cvcuda_tv.crop, tvf.crop, image, top, left, height, width)


def _permute_params(dtypes):
    calls = (
        (1, [0]),
        (3, [0, 1, 2]),
        (3, [2, 0, 1]),
        (3, [2, -3, 2]),
        (4, [3, 2, 1, 0]),
    )
    for (channels, permutation), dtype, batch in itertools.product(
        calls, dtypes, (None, 1, 3)
    ):
        spec = (batch, channels, 17, 11, dtype)
        yield pytest.param(
            spec, permutation, id=f"{dtype}-c{channels}-{permutation}-n{batch}"
        )


@pytest.mark.parametrize(
    "image,permutation",
    [
        row
        for row in _permute_params((*_MINIMAL_DTYPES, "float16"))
        if row.values[0][1] != 3
    ],
    indirect=["image"],
)
def test_permute_single_and_four_channels(assert_tv_parity, tvf, image, permutation):
    assert_tv_parity(
        cvcuda_tv.permute_channels, tvf.permute_channels, image, permutation
    )
