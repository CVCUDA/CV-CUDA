# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Color and photometric transforms for :mod:`cvcuda.tv`."""

from __future__ import annotations

from collections.abc import Sequence

import cvcuda

from ._layout import dims

_PERMITTED = "Input image tensor permitted channel values are 1 or 3"


def _require_1_or_3_channels(inpt: cvcuda.Tensor, message: str = _PERMITTED) -> int:
    _, _, _, channels = dims(inpt)
    if channels not in (1, 3):
        raise TypeError(f"{message}, but found {channels}")
    return channels


def _require_rgb(channels: int) -> None:
    # Match torchvision's tuple-unpacking error on the running Python version.
    if channels != 3:
        r, g, b = tuple(range(channels))


def invert(
    inpt: cvcuda.Tensor, *, stream: cvcuda.Stream | None = None
) -> cvcuda.Tensor:
    """Invert the colors of an image: ``out = bound - in``.

    ``bound`` is the dtype maximum for unsigned-integer images and ``1.0`` for
    floating-point images.

    Args:
        inpt (cvcuda.Tensor): Input image(s).
        stream (cvcuda.Stream, optional): Stream used to submit the operation.

    Returns:
        cvcuda.Tensor: Output image(s).

    """
    return cvcuda.invert(inpt, stream=stream)


def permute_channels(
    inpt: cvcuda.Tensor,
    permutation: Sequence[int],
    *,
    stream: cvcuda.Stream | None = None,
) -> cvcuda.Tensor:
    """Reorder image channels.

    The output channel ``j`` receives input channel ``permutation[j]``. Negative
    indices wrap from the end and repeated indices are allowed. For an empty
    input, only the permutation length is validated.

    Args:
        inpt (cvcuda.Tensor): Input image tensor.
        permutation (list[int]): One source-channel index per output channel.
        stream (cvcuda.Stream, optional): Stream used to submit the operation.

    Returns:
        cvcuda.Tensor: The reordered tensor with the same metadata as ``inpt``.

    Raises:
        ValueError: If ``permutation`` does not contain one index per channel.
        IndexError: If an index is out of range for a non-empty input.

    """
    _, _, _, channels = dims(inpt)
    if len(permutation) != channels:
        raise ValueError(
            "Length of permutation does not match number of channels: "
            f"{len(permutation)} != {channels}"
        )

    # torchvision returns the original object before indexing an empty image, so
    # invalid values and non-integer entries are intentionally not inspected here.
    if 0 in inpt.shape:
        return inpt

    channel_indices = range(channels)
    normalized = [channel_indices[index] for index in permutation]

    return cvcuda.channelreorder(inpt, normalized, stream=stream)


def rgb_to_grayscale(
    inpt: cvcuda.Tensor,
    num_output_channels: int = 1,
    *,
    stream: cvcuda.Stream | None = None,
) -> cvcuda.Tensor:
    """Convert uint8, float16, or float32 image(s) to grayscale.

    Accepts one-channel (already-grayscale) or three-channel RGB inputs,
    mirroring torchvision's behaviour: a single-channel input is returned
    unchanged for ``num_output_channels=1`` and replicated to three channels for
    ``num_output_channels=3``.

    Args:
        inpt (cvcuda.Tensor): Input one- or three-channel image tensor.
        num_output_channels (int): 1 (grayscale) or 3 (grayscale broadcast to RGB).
        stream (cvcuda.Stream, optional): Stream used to submit the operation.

    Returns:
        cvcuda.Tensor: Grayscale image(s) with one or three output channels.

    Raises:
        ValueError: If ``num_output_channels`` is not 1 or 3.

    """
    if num_output_channels not in (1, 3):
        raise ValueError(
            f"num_output_channels must be 1 or 3, got {num_output_channels}."
        )

    _, _, _, channels = dims(inpt)
    if channels == 1:
        if num_output_channels == 1:
            return inpt
        return cvcuda.cvtcolor(inpt, cvcuda.ColorConversion.GRAY2RGB, stream=stream)
    _require_rgb(channels)

    gray = cvcuda.cvtcolor(inpt, cvcuda.ColorConversion.RGB2GRAY_TV, stream=stream)
    if num_output_channels == 1:
        return gray
    return cvcuda.cvtcolor(gray, cvcuda.ColorConversion.GRAY2RGB, stream=stream)


def grayscale_to_rgb(
    inpt: cvcuda.Tensor, *, stream: cvcuda.Stream | None = None
) -> cvcuda.Tensor:
    """Repeat a uint8, float16, or float32 grayscale channel to form RGB image(s).

    If the input already has three or more channels, it is returned unchanged,
    mirroring torchvision's behaviour.

    Args:
        inpt (cvcuda.Tensor): Input image tensor (one or more channels).
        stream (cvcuda.Stream, optional): Stream used to submit the operation.

    Returns:
        cvcuda.Tensor: Three-channel RGB image(s), or the input unchanged when
        it already has three or more channels.

    """
    _, _, _, channels = dims(inpt)
    if channels >= 3:
        return inpt
    if channels != 1:
        _require_rgb(channels)
    return cvcuda.cvtcolor(inpt, cvcuda.ColorConversion.GRAY2RGB, stream=stream)


def adjust_brightness(
    inpt: cvcuda.Tensor,
    brightness_factor: float,
    *,
    stream: cvcuda.Stream | None = None,
) -> cvcuda.Tensor:
    """Scale image brightness and clamp to the image value range.

    Computes ``clamp(brightness_factor * in, 0, bound)``, where ``bound`` is
    ``1.0`` for floating-point images and the dtype maximum for integer images.

    Args:
        inpt (cvcuda.Tensor): Input image(s).
        brightness_factor (float): Non-negative brightness multiplier.
        stream (cvcuda.Stream, optional): Stream used to submit the operation.

    Returns:
        cvcuda.Tensor: Output image(s).

    Raises:
        ValueError: If ``brightness_factor`` is negative.

    """
    if brightness_factor < 0:
        raise ValueError(
            f"brightness_factor ({brightness_factor}) is not non-negative."
        )
    _require_1_or_3_channels(inpt)
    # torchvision converts the float result with .to(dtype), which truncates.
    return cvcuda.brightness_contrast(
        inpt,
        brightness_factor,
        1.0,
        0.0,
        0.0,
        clamp=True,
        round=cvcuda.Round.TRUNCATE,
        stream=stream,
    )


def adjust_contrast(
    inpt: cvcuda.Tensor,
    contrast_factor: float,
    *,
    stream: cvcuda.Stream | None = None,
) -> cvcuda.Tensor:
    """Blend each image toward its grayscale mean.

    ``out = clamp(contrast_factor * in + (1 - contrast_factor) * mean, 0, bound)``,
    where ``mean`` is the per-image grayscale mean (BT.601 luma
    ``0.2989 R + 0.587 G + 0.114 B``).

    Args:
        inpt (cvcuda.Tensor): Input image(s), 1 or 3 channels.
        contrast_factor (float): Non-negative contrast multiplier.
        stream (cvcuda.Stream, optional): Stream used to submit the operation.

    Returns:
        cvcuda.Tensor: Output image(s).

    """
    if contrast_factor < 0:
        raise ValueError(f"contrast_factor ({contrast_factor}) is not non-negative.")
    _require_1_or_3_channels(inpt)
    # torchvision's _blend converts with .to(dtype), which truncates.
    return cvcuda.adjust_contrast(
        inpt, contrast_factor, round=cvcuda.Round.TRUNCATE, stream=stream
    )


def adjust_saturation(
    inpt: cvcuda.Tensor,
    saturation_factor: float,
    *,
    stream: cvcuda.Stream | None = None,
) -> cvcuda.Tensor:
    """Blend an RGB image toward its grayscale by ``saturation_factor``.

    A factor of 0 produces grayscale, 1 leaves the image unchanged, and values
    above 1 increase saturation. Single-channel images are unchanged.

    Args:
        inpt (cvcuda.Tensor): Input RGB or single-channel image(s).
        saturation_factor (float): Non-negative saturation multiplier.
        stream (cvcuda.Stream, optional): Stream used to submit the operation.

    Returns:
        cvcuda.Tensor: Output image(s).

    """
    if saturation_factor < 0:
        raise ValueError(
            f"saturation_factor ({saturation_factor}) is not non-negative."
        )
    _require_1_or_3_channels(inpt)
    return cvcuda.adjust_saturation(inpt, saturation_factor, stream=stream)


def adjust_hue(
    inpt: cvcuda.Tensor,
    hue_factor: float,
    *,
    stream: cvcuda.Stream | None = None,
) -> cvcuda.Tensor:
    """Rotate image hue by ``hue_factor`` in HSV space.

    The hue channel is shifted modulo 1. Single-channel images are unchanged.

    Args:
        inpt (cvcuda.Tensor): Input RGB or single-channel image(s).
        hue_factor (float): Rotation in ``[-0.5, 0.5]``; 0 is unchanged and
            either endpoint rotates hue by 180 degrees.
        stream (cvcuda.Stream, optional): Stream used to submit the operation.

    Returns:
        cvcuda.Tensor: Output image(s).

    """
    if not (-0.5 <= hue_factor <= 0.5):
        raise ValueError(f"hue_factor ({hue_factor}) is not in [-0.5, 0.5].")
    _require_1_or_3_channels(inpt)
    return cvcuda.adjust_hue(inpt, hue_factor, stream=stream)


def adjust_gamma(
    inpt: cvcuda.Tensor,
    gamma: float,
    gain: float = 1.0,
    *,
    stream: cvcuda.Stream | None = None,
) -> cvcuda.Tensor:
    """Gamma correction: ``out = gain * in**gamma``.

    Values are interpreted in the normalized image range and clamped to the
    output dtype range.

    Args:
        inpt (cvcuda.Tensor): Input image(s), interleaved ((N)HWC) or planar ((N)CHW).
        gamma (float): Non-negative gamma exponent.
        gain (float, optional): Output gain. Defaults to 1.0.
        stream (cvcuda.Stream, optional): Stream used to submit the operation.

    Returns:
        cvcuda.Tensor: Output image(s).

    Raises:
        ValueError: If ``gamma`` is negative.

    """
    if gamma < 0:
        raise ValueError(f"gamma must be non-negative, got {gamma}")
    if inpt.dtype == cvcuda.Type.U8:
        # torchvision converts normalized float32 back to uint8 by multiplying by
        # max + 1 - 1e-3 (255.999) and truncating. GammaContrast's integer path
        # multiplies by 255 internally, so fold the scale ratio into gain.
        gain *= (256.0 - 1e-3) / 255.0
    return cvcuda.gamma_contrast(
        inpt, gamma, gain, round=cvcuda.Round.TRUNCATE, stream=stream
    )


def adjust_sharpness(
    inpt: cvcuda.Tensor,
    sharpness_factor: float,
    *,
    stream: cvcuda.Stream | None = None,
) -> cvcuda.Tensor:
    """Adjust image sharpness.

    Each image is blended with a 3x3-smoothed version of its interior. A factor
    of 0 produces the smoothed image, 1 leaves it unchanged, and values above 1
    sharpen it. The one-pixel border and images smaller than 3 pixels in either
    spatial dimension are unchanged.

    Args:
        inpt (cvcuda.Tensor): Input image(s) with 1 or 3 channels.
        sharpness_factor (float): Non-negative blend weight applied to the input.
        stream (cvcuda.Stream, optional): Stream used to submit the operation.

    Returns:
        cvcuda.Tensor: Output image(s).

    Raises:
        ValueError: If ``sharpness_factor`` is negative.

    """
    if sharpness_factor < 0:
        raise ValueError(f"sharpness_factor ({sharpness_factor}) is not non-negative.")
    _require_1_or_3_channels(inpt, "Input image tensor can have 1 or 3 channels")
    return cvcuda.adjust_sharpness(inpt, sharpness_factor, stream=stream)


def posterize(
    inpt: cvcuda.Tensor, bits: int, *, stream: cvcuda.Stream | None = None
) -> cvcuda.Tensor:
    """Reduce each channel to its ``bits`` most-significant bits.

    Lower-order bits are set to zero. ``bits=0`` produces zeros and retaining the
    dtype bit width leaves the image unchanged. A floating-point image has no bit
    layout to keep, so it is quantized into ``2 ** bits`` levels of its ``[0, 1]``
    range instead, which is the same operation in that type's value range.

    Args:
        inpt (cvcuda.Tensor): Input unsigned-integer or floating-point image(s).
        bits (int): Number of high bits to keep, from 0 through the dtype bit
            width, or from 0 through 8 for floating-point images.
        stream (cvcuda.Stream, optional): Stream used to submit the operation.

    Returns:
        cvcuda.Tensor: Output image(s).

    """
    return cvcuda.posterize(inpt, bits, stream=stream)


def solarize(
    inpt: cvcuda.Tensor, threshold: float, *, stream: cvcuda.Stream | None = None
) -> cvcuda.Tensor:
    """Invert every pixel at or above ``threshold``.

    Pixels below the threshold are unchanged. Inversion uses the dtype maximum
    for unsigned integers and ``1.0`` for floating-point images.

    Args:
        inpt (cvcuda.Tensor): Input image(s).
        threshold (float): Values greater than or equal to this value are inverted.
        stream (cvcuda.Stream, optional): Stream used to submit the operation.

    Returns:
        cvcuda.Tensor: Output image(s).

    """
    return cvcuda.solarize(inpt, threshold, stream=stream)


def autocontrast(
    inpt: cvcuda.Tensor, *, stream: cvcuda.Stream | None = None
) -> cvcuda.Tensor:
    """Stretch each image to the full value range.

    Each channel is remapped independently so its minimum becomes zero and its
    maximum becomes the dtype bound. Flat channels are unchanged. Integer results
    are ``floor((x - min) * bound / (max - min))``, matching Pillow;
    torchvision's tensor kernel can be one lower where that quotient is a
    whole number.

    Args:
        inpt (cvcuda.Tensor): Input image(s) with 1 or 3 channels.
        stream (cvcuda.Stream, optional): Stream used to submit the operation.

    Returns:
        cvcuda.Tensor: Output image(s).

    Raises:
        TypeError: If ``inpt`` does not have 1 or 3 channels.

    """
    _require_1_or_3_channels(inpt)
    return cvcuda.autocontrast(inpt, stream=stream)


def equalize(
    inpt: cvcuda.Tensor, *, stream: cvcuda.Stream | None = None
) -> cvcuda.Tensor:
    """Histogram-equalize image(s) with Pillow's lookup table.

    The integer lookup table matches Pillow's, which torchvision also uses.
    Normalized floating-point images are binned to 8 bits, equalized and scaled
    back. Float16 bins through float32, deliberately avoiding torchvision's
    half-precision multiplier rounding to 256 and wrapping 1.0 to bin zero.

    Args:
        inpt (cvcuda.Tensor): Input uint8 or floating-point image(s).
        stream (cvcuda.Stream, optional): Stream used to submit the operation.

    Returns:
        cvcuda.Tensor: Output image(s).

    """
    return cvcuda.histogrameq(
        inpt, inpt.dtype, method=cvcuda.HistogramEq.PILLOW, stream=stream
    )


__all__: list[str] = [
    "adjust_brightness",
    "adjust_contrast",
    "adjust_gamma",
    "adjust_hue",
    "adjust_saturation",
    "adjust_sharpness",
    "autocontrast",
    "equalize",
    "grayscale_to_rgb",
    "invert",
    "permute_channels",
    "posterize",
    "rgb_to_grayscale",
    "solarize",
]
