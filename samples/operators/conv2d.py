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
"""Simple Conv2D example with CVCUDA."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cvcuda
import numpy as np

common_dir = Path(__file__).parent.parent
sys.path.append(str(common_dir))

from common import (  # noqa: E402
    parse_image_args,
    read_image,
    write_image,
    upload_tensor,
    download_tensor,
)


def main() -> None:
    """Apply a sharpening convolution to an image with CVCUDA."""
    # docs_tag: begin_main
    args: argparse.Namespace = parse_image_args("cat_conv2d.jpg")
    # docs_tag: begin_read_image
    input_image: cvcuda.Tensor = read_image(args.input)
    # docs_tag: end_read_image

    # docs_tag: begin_conv2d_setup
    h, w, c = input_image.shape[0], input_image.shape[1], input_image.shape[2]

    # conv2d operates on ImageBatchVarShape.  Applying a sharpening kernel
    # directly on uint8 would overflow (center weight 5 * 200 > 255), so we
    # first promote to float32, convolve, then clamp back to uint8.
    float_nhwc: cvcuda.Tensor = cvcuda.convertto(
        input_image.reshape((1, h, w, c), "NHWC"), dtype=np.float32
    )
    float_hwc: cvcuda.Tensor = float_nhwc.reshape((h, w, c), "HWC")

    src_image: cvcuda.Image = cvcuda.as_image(float_hwc.cuda(), cvcuda.Format.RGBf32)
    src_batch: cvcuda.ImageBatchVarShape = cvcuda.ImageBatchVarShape(1)
    src_batch.pushback(src_image)

    # 3x3 sharpening kernel (single-channel float — applied to each colour
    # channel independently).
    sharpen_np: np.ndarray = np.array(
        [[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32
    ).reshape(3, 3, 1)
    kernel_tensor: cvcuda.Tensor = cvcuda.Tensor(
        (3, 3, 1), dtype=np.float32, layout="HWC"
    )
    # upload_tensor honours the kernel tensor's row pitch (CVCUDA pads each row
    # to an alignment boundary, so the 3-float rows are not packed contiguously).
    # A packed copy here would scramble the kernel and blacken the output.
    upload_tensor(sharpen_np, kernel_tensor)
    kernel_image: cvcuda.Image = cvcuda.as_image(
        kernel_tensor.cuda(), cvcuda.Format.F32
    )
    kernel_batch: cvcuda.ImageBatchVarShape = cvcuda.ImageBatchVarShape(1)
    kernel_batch.pushback(kernel_image)

    # Anchor (-1, -1) lets the operator place the kernel centre automatically.
    anchor_np: np.ndarray = np.array([[-1, -1]], dtype=np.int32)
    anchor_tensor: cvcuda.Tensor = cvcuda.Tensor((1, 2), dtype=np.int32, layout="NC")
    upload_tensor(anchor_np, anchor_tensor)
    # docs_tag: end_conv2d_setup

    # docs_tag: begin_conv2d
    # Convolve on float data; REFLECT101 border prevents dark edge halos.
    out_batch: cvcuda.ImageBatchVarShape = cvcuda.conv2d(
        src_batch,
        kernel_batch,
        anchor_tensor,
        cvcuda.Border.REFLECT101,
    )

    # Download float result, clamp to [0, 255], and write as uint8 JPEG.
    out_float_hwc: cvcuda.Tensor = cvcuda.as_tensor(next(iter(out_batch)))
    out_np: np.ndarray = download_tensor(out_float_hwc)
    out_clamped: np.ndarray = np.ascontiguousarray(
        np.clip(out_np, 0, 255).astype(np.uint8)
    )
    # Write through the original (unpadded) input tensor instead of a freshly
    # allocated one; upload_tensor would also handle a padded row pitch.
    upload_tensor(out_clamped, input_image)
    write_image(input_image, args.output)
    # docs_tag: end_conv2d
    # docs_tag: end_main


if __name__ == "__main__":
    main()
