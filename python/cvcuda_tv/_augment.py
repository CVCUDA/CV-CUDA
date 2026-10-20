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

"""Image augmentation transforms for :mod:`cvcuda.tv`."""

from __future__ import annotations

import cvcuda


def erase(
    inpt: cvcuda.Tensor,
    i: int,
    j: int,
    h: int,
    w: int,
    v: cvcuda.Tensor,
    inplace: bool = False,
    *,
    stream: cvcuda.Stream | None = None,
) -> cvcuda.Tensor:
    """Erase (fill) a rectangular region of image(s).

    The region follows Python slice semantics. ``v`` is broadcast over the region
    and must be a CUDA-resident ``cvcuda.Tensor`` on the same device as
    ``inpt``. The image and value dtypes must match, or ``v`` may be float32.
    A ``torch.Tensor`` ``v`` is moved to the image's device and cast to its
    dtype, as torchvision's assignment does.

    Args:
        inpt (cvcuda.Tensor): Input integer or floating-point image tensor in
            HWC, NHWC, CHW, or NCHW layout.
        i (int): Top coordinate of the region.
        j (int): Left coordinate of the region.
        h (int): Region height.
        w (int): Region width.
        v (cvcuda.Tensor): Scalar or broadcastable fill value.
        inplace (bool, optional): Write the result into ``inpt``. Defaults to
            ``False``.
        stream (cvcuda.Stream, optional): Stream used to submit the operation.

    Returns:
        cvcuda.Tensor: The filled image tensor; ``inpt`` itself when ``inplace``
            is ``True``.

    """
    if inplace:
        return cvcuda.erase_into(inpt, inpt, i, j, h, w, v, stream=stream)
    return cvcuda.erase(inpt, i, j, h, w, v, stream=stream)


def jpeg(
    image: cvcuda.Tensor, quality: int, *, stream: cvcuda.Stream | None = None
) -> cvcuda.Tensor:
    """Apply JPEG compression artifacts to image(s).

    This simulates the lossy stages of JPEG compression, including chroma
    subsampling and block quantization. It returns an image tensor rather than
    encoded JPEG data.

    Args:
        image (cvcuda.Tensor): Input uint8 image tensor with 1 or 3 channels in
            HWC, NHWC, CHW, or NCHW layout.
        quality (int): JPEG quality, from 1 (strongest distortion) to 100 (weakest).
        stream (cvcuda.Stream, optional): Stream used to submit the operation.

    Returns:
        cvcuda.Tensor: JPEG-compressed image(s).

    Raises:
        TypeError: If ``quality`` is not an integer.
        RuntimeError: If ``quality`` is outside ``[1, 100]`` or ``image`` is not
            uint8 with 1 or 3 channels.

    """
    return cvcuda.jpeg_compression_distortion(image, quality, stream=stream)


__all__: list[str] = ["erase", "jpeg"]
