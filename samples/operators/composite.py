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
"""Simple Composite example with CVCUDA."""

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
)


def main() -> None:
    """Composite two images with a mask using CVCUDA."""
    # docs_tag: begin_main
    args: argparse.Namespace = parse_image_args("cat_composite.jpg")
    # docs_tag: begin_read_image
    # Foreground: the cat image (primary subject)
    fg_image: cvcuda.Tensor = read_image(args.input)
    # Background: the Weimaraner dog image, resized to match the foreground
    bg_raw: cvcuda.Tensor = read_image(common_dir / "assets/images/Weimaraner.jpg")
    # docs_tag: end_read_image

    # docs_tag: begin_composite_setup
    h, w, c = fg_image.shape  # HWC layout from read_image

    # Resize the background to match the foreground's spatial dimensions so
    # cvcuda.composite can pair them element-wise on the GPU.
    bg_image: cvcuda.Tensor = cvcuda.resize(bg_raw, (h, w, c))

    # Build the foreground mask on the CPU using NumPy, then upload it.
    # The mask is single-channel uint8: 255 = keep foreground, 0 = show background.
    # A circular region in the centre of the frame exposes the cat; the rest
    # shows the Weimaraner, giving a clear visual demonstration of blending.
    mask_np = np.zeros((h, w, 1), dtype=np.uint8)
    cy, cx = h // 2, w // 2
    radius = min(h, w) // 3
    # Vectorised distance computation avoids a slow Python loop.
    ys, xs = np.ogrid[:h, :w]
    inside_circle = (xs - cx) ** 2 + (ys - cy) ** 2 <= radius**2
    mask_np[inside_circle, 0] = 255

    # Allocate a GPU tensor for the mask and copy the host data across.
    # upload_tensor honours the mask tensor's row pitch (a single-channel mask
    # is padded to an alignment boundary); a packed copy would shear the mask.
    mask_tensor: cvcuda.Tensor = cvcuda.Tensor((h, w, 1), dtype=np.uint8, layout="HWC")
    upload_tensor(mask_np, mask_tensor)
    # docs_tag: end_composite_setup

    # docs_tag: begin_composite
    # cvcuda.composite blends foreground and background using the mask:
    # output[px] = fg[px] if mask[px] > 0 else bg[px].
    # outchannels=3 requests an RGB output tensor.
    out_image: cvcuda.Tensor = cvcuda.composite(
        fg_image,  # foreground (cat)
        bg_image,  # background (Weimaraner, resized)
        mask_tensor,  # single-channel alpha mask
        3,  # outchannels: 3 = RGB, 4 = RGBA
    )
    # docs_tag: end_composite

    write_image(out_image, args.output)
    # docs_tag: end_main


if __name__ == "__main__":
    main()
