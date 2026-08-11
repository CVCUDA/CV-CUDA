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
"""Simple Rotate example with CVCUDA."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cvcuda

common_dir = Path(__file__).parent.parent
sys.path.append(str(common_dir))

from common import parse_image_args, read_image, write_image  # noqa: E402


def main() -> None:
    """Rotate an image with CVCUDA."""
    # docs_tag: begin_main
    args: argparse.Namespace = parse_image_args("cat_rotate.jpg")
    # docs_tag: begin_read_image
    input_image: cvcuda.Tensor = read_image(args.input)
    # docs_tag: end_read_image

    # docs_tag: begin_rotate_setup
    # cvcuda.rotate performs the inverse mapping:
    #   src = R(angle) * (dst - shift)
    # where R is a counter-clockwise rotation matrix (screen coords, y-down).
    # To keep the image centre fixed we solve for the shift such that
    # dst=(cx,cy) maps back to src=(cx,cy), giving:
    #   shift = (cx*(1-cos) - cy*sin,  cy*(1-cos) + cx*sin)
    h, w = input_image.shape[0], input_image.shape[1]
    angle_deg = 45.0
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    cx, cy = w / 2.0, h / 2.0
    shift_x = cx * (1 - cos_a) - cy * sin_a
    shift_y = cy * (1 - cos_a) + cx * sin_a
    # docs_tag: end_rotate_setup

    # docs_tag: begin_rotate
    # Rotate the image by 45 degrees with bilinear interpolation.
    # The shift [shift_x, shift_y] re-centres the content after rotation so the
    # subject remains visible rather than drifting off-canvas.
    output_image: cvcuda.Tensor = cvcuda.rotate(
        input_image,
        angle_deg,
        [shift_x, shift_y],
        cvcuda.Interp.LINEAR,
    )
    write_image(output_image, args.output)
    # docs_tag: end_rotate
    # docs_tag: end_main


if __name__ == "__main__":
    main()
