# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cvcuda
import numpy as np
import pytest


def _inside(image, tap_x, tap_y):
    return 0 <= tap_x < image.shape[1] and 0 <= tap_y < image.shape[0]


def _nearest(image, horizontal, vertical, fill):
    tap_x, tap_y = int(np.floor(horizontal + 0.5)), int(np.floor(vertical + 0.5))
    return image[tap_y, tap_x] if _inside(image, tap_x, tap_y) else fill


def _bilinear(image, horizontal, vertical, fill, coverage_mode):
    sampled = np.zeros(image.shape[2], dtype=np.float64)
    coverage = 0.0
    left, top = int(np.floor(horizontal)), int(np.floor(vertical))
    for tap_y in (top, top + 1):
        for tap_x in (left, left + 1):
            if not _inside(image, tap_x, tap_y):
                continue
            weight = (1 - abs(horizontal - tap_x)) * (1 - abs(vertical - tap_y))
            sampled += image[tap_y, tap_x] * weight
            coverage += weight
    return (sampled * coverage if coverage_mode else sampled) + fill * (1 - coverage)


def _gold(source, fill, nearest, coverage_mode=True):
    result = np.empty_like(source, dtype=np.float64)
    for batch, row, column in np.ndindex(source.shape[:3]):
        horizontal, vertical = column - 0.5, row - 0.75
        image = source[batch]
        result[batch, row, column] = (
            _nearest(image, horizontal, vertical, fill)
            if nearest
            else _bilinear(image, horizontal, vertical, fill, coverage_mode)
        )
    if source.dtype.kind in "ui":
        limits = np.iinfo(source.dtype)
        result = np.clip(np.rint(result), limits.min, limits.max)
    return result.astype(source.dtype)


@pytest.mark.parametrize(
    "operation,dtype,channels",
    [
        (operation, dtype, channels)
        for operation in ("warp_affine", "warp_perspective")
        for dtype in (np.uint8, np.uint16, np.int16, np.float16, np.float32)
        for channels in (1, 3, 4)
    ]
    + [("remap", np.uint8, channels) for channels in (1, 3, 4)]
    + [
        ("remap", dtype, channels)
        for dtype in (np.float16, np.float32)
        for channels in (1, 3, 4)
    ],
)
@pytest.mark.parametrize("layout", ["NHWC", "HWC", "NCHW", "CHW"])
@pytest.mark.parametrize("nearest", [False, True])
def test_filled_warp_coverage_independent_gold(
    operation, layout, dtype, channels, nearest
):
    cupy = pytest.importorskip("cupy")
    source = (np.arange(2 * 3 * 4 * channels).reshape(2, 3, 4, channels) + 50).astype(
        dtype
    )
    if not layout.startswith("N"):
        source = source[:1]
    fill = np.full(channels, 11.75, dtype=np.float64)
    fill += np.arange(channels) * 4
    expected = _gold(source, fill, nearest)
    planar = layout in ("NCHW", "CHW")
    packed = source.transpose(0, 3, 1, 2) if planar else source
    if not layout.startswith("N"):
        packed = packed[0]
    device = cupy.asarray(np.ascontiguousarray(packed))
    tensor = cvcuda.as_tensor(device, layout)
    interpolation = cvcuda.Interp.NEAREST if nearest else cvcuda.Interp.LINEAR
    if operation == "remap":
        vertical, horizontal = np.mgrid[:3, :4].astype(np.float32)
        map_device = cupy.asarray(
            np.stack((horizontal - 0.5, vertical - 0.75), axis=-1)[None]
        )
        map_layout = "NHWC" if layout.startswith("N") else "HWC"
        if map_layout == "HWC":
            map_device = map_device[0]
        output = cvcuda.remap(
            tensor,
            cvcuda.as_tensor(map_device, map_layout),
            src_interp=interpolation,
            map_interp=cvcuda.Interp.NEAREST,
            map_type=cvcuda.Remap.ABSOLUTE,
            border=cvcuda.Border.CONSTANT_COVERAGE,
            border_value=fill.astype(np.float32),
        )
    else:
        matrix = np.array([[1, 0, -0.5], [0, 1, -0.75], [0, 0, 1]], dtype=np.float32)
        if operation == "warp_affine":
            matrix = matrix[:2]
        output = getattr(cvcuda, operation)(
            tensor,
            matrix,
            int(interpolation) | int(cvcuda.Interp.WARP_INVERSE_MAP),
            border_mode=cvcuda.Border.CONSTANT_COVERAGE,
            border_value=fill.astype(np.float32),
        )
    actual = cupy.asnumpy(cupy.asarray(output.cuda()))
    if not layout.startswith("N"):
        actual = actual[None]
    if planar:
        actual = actual.transpose(0, 2, 3, 1)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("operation", ["warp_affine", "warp_perspective", "remap"])
def test_constant_coverage_rejects_varshape(operation):
    cupy = pytest.importorskip("cupy")
    source = cvcuda.ImageBatchVarShape(1)
    source.pushback(cvcuda.Image((4, 3), cvcuda.Format.U8))
    interpolation = cvcuda.Interp.LINEAR
    if operation == "remap":
        map_device = cupy.zeros((1, 3, 4, 2), dtype=cupy.float32)
        mapping = cvcuda.as_tensor(map_device, "NHWC")
        with pytest.raises(RuntimeError):
            cvcuda.remap(
                source,
                mapping,
                src_interp=interpolation,
                border=cvcuda.Border.CONSTANT_COVERAGE,
                border_value=11.75,
            )
    else:
        matrix = np.eye(3, dtype=np.float32)
        if operation == "warp_affine":
            matrix = matrix[:2]
        matrix_device = cupy.asarray(matrix.reshape(1, -1))
        matrix = cvcuda.as_tensor(matrix_device, "NW")
        with pytest.raises(RuntimeError):
            getattr(cvcuda, operation)(
                source,
                matrix,
                int(interpolation) | int(cvcuda.Interp.WARP_INVERSE_MAP),
                border_mode=cvcuda.Border.CONSTANT_COVERAGE,
                border_value=[11.75],
            )


def test_remap_nonuniform_planar_fill_matches_interleaved():
    border = cvcuda.Border.CONSTANT
    cupy = pytest.importorskip("cupy")
    source = (np.arange(2 * 3 * 4 * 3).reshape(2, 3, 4, 3) + 50).astype(np.float32)
    fill = np.array([11.75, 15.75, 19.75], dtype=np.float32)
    vertical, horizontal = np.mgrid[:3, :4].astype(np.float32)
    mapping = cvcuda.as_tensor(
        cupy.asarray(np.stack((horizontal - 0.5, vertical - 0.75), axis=-1)[None]),
        "NHWC",
    )
    expected = _gold(source, fill, False, coverage_mode=False)
    planar = cvcuda.as_tensor(cupy.asarray(source.transpose(0, 3, 1, 2)), "NCHW")
    interleaved = cvcuda.as_tensor(cupy.asarray(source), "NHWC")
    kwargs = {
        "src_interp": cvcuda.Interp.LINEAR,
        "border": border,
        "border_value": fill,
    }
    planar_result = cupy.asnumpy(
        cupy.asarray(cvcuda.remap(planar, mapping, **kwargs).cuda())
    )
    interleaved_result = cupy.asnumpy(
        cupy.asarray(cvcuda.remap(interleaved, mapping, **kwargs).cuda())
    )
    np.testing.assert_allclose(planar_result.transpose(0, 2, 3, 1), expected, atol=1e-5)
    np.testing.assert_allclose(interleaved_result, expected, atol=1e-5)
