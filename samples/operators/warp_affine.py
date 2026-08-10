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
"""Simple Warp Affine example with CVCUDA."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cvcuda
import numpy as np

common_dir = Path(__file__).parent.parent
sys.path.append(str(common_dir))

from common import parse_image_args, read_image, write_image  # noqa: E402


def main() -> None:
    """Apply a rotation+translation affine warp to an image with CVCUDA."""
    # docs_tag: begin_main
    args: argparse.Namespace = parse_image_args("cat_warp_affine.jpg")
    # docs_tag: begin_read_image
    input_image: cvcuda.Tensor = read_image(args.input)
    # docs_tag: end_read_image

    # docs_tag: begin_warp_affine_setup
    # Build a 2x3 float32 affine matrix that rotates the image 15 degrees
    # counter-clockwise about the image centre and shifts it slightly right.
    # OpenCV convention: the matrix maps *destination* pixel coordinates to
    # *source* pixel coordinates (inverse warp), so we use a rotation of -angle.
    h, w = input_image.shape[0], input_image.shape[1]
    angle_deg = 15.0
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    cx, cy = w / 2.0, h / 2.0
    # Rotation about the image centre + a small horizontal translation
    tx = 20.0  # pixels to shift right
    xform = np.array(
        [
            [cos_a, sin_a, (1 - cos_a) * cx - sin_a * cy + tx],
            [-sin_a, cos_a, sin_a * cx + (1 - cos_a) * cy],
        ],
        dtype=np.float32,
    )
    # docs_tag: end_warp_affine_setup

    # docs_tag: begin_warp_affine
    # Apply the affine transformation with bilinear interpolation.
    # Pixels that fall outside the source image are filled with black (CONSTANT, value=0).
    output_image: cvcuda.Tensor = cvcuda.warp_affine(
        input_image,
        xform,
        cvcuda.Interp.LINEAR,
        border_mode=cvcuda.Border.CONSTANT,
        border_value=[0],
    )
    write_image(output_image, args.output)
    # docs_tag: end_warp_affine
    # docs_tag: end_main


if __name__ == "__main__":
    main()
