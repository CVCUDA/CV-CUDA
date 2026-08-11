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
"""Simple On-Screen Display example with CVCUDA."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cvcuda
import numpy as np

common_dir = Path(__file__).parent.parent
sys.path.append(str(common_dir))

from common import parse_image_args, read_image, write_image  # noqa: E402


def main() -> None:
    """Draw OSD elements onto an image with CVCUDA."""
    # docs_tag: begin_main
    args: argparse.Namespace = parse_image_args("cat_osd.jpg")
    # docs_tag: begin_read_image
    input_image: cvcuda.Tensor = read_image(args.input)
    # docs_tag: end_read_image

    # docs_tag: begin_osd_setup
    # The OSD operator requires an NHWC batch tensor and a list-of-lists of
    # elements — one inner list per image in the batch.  We promote the single
    # HWC image to a batch of 1 (N=1).
    h, w, c = input_image.shape
    nhwc_image: cvcuda.Tensor = input_image.reshape((1, h, w, c), "NHWC")

    # Build a representative set of OSD primitives.  All coordinates are in
    # pixels; colours are RGBA tuples.  We scale positions relative to the
    # image dimensions so the overlay looks reasonable on any input size.
    box_x, box_y = w // 8, h // 8
    box_w, box_h = w // 4, h // 4

    elements = cvcuda.Elements(
        elements=[
            [
                # Filled bounding box drawn around the top-left region
                cvcuda.BndBoxI(
                    box=(box_x, box_y, box_w, box_h),
                    thickness=3,
                    borderColor=(255, 255, 0),
                    fillColor=(0, 128, 255, 64),
                ),
                # Text label placed near the top-left of the image
                cvcuda.Label(
                    utf8Text="CV-CUDA OSD",
                    fontSize=24,
                    tlPos=(box_x, box_y - 30 if box_y >= 30 else box_y + box_h + 5),
                    fontColor=(255, 255, 255),
                    bgColor=(0, 0, 0, 180),
                ),
                # Diagonal line across the image
                cvcuda.Line(
                    pos0=(0, 0),
                    pos1=(w - 1, h - 1),
                    thickness=2,
                    color=(0, 255, 0),
                ),
                # Circle at the image centre
                cvcuda.Circle(
                    centerPos=(w // 2, h // 2),
                    radius=min(w, h) // 8,
                    thickness=2,
                    borderColor=(255, 128, 0),
                    bgColor=(255, 128, 0, 48),
                ),
                # Arrow pointing inward from the right edge
                cvcuda.Arrow(
                    pos0=(w - 1, h // 2),
                    pos1=(w * 3 // 4, h // 2),
                    arrowSize=12,
                    thickness=2,
                    color=(255, 0, 128),
                ),
                # Closed polygon (diamond shape) at the image centre
                cvcuda.PolyLine(
                    points=np.array(
                        [
                            [w // 2, h // 4],
                            [w * 3 // 4, h // 2],
                            [w // 2, h * 3 // 4],
                            [w // 4, h // 2],
                        ],
                        dtype=np.int32,
                    ),
                    thickness=2,
                    isClosed=True,
                    borderColor=(0, 255, 255),
                    fillColor=(0, 255, 255, 32),
                ),
                # Timestamp overlay in the bottom-left corner
                cvcuda.Clock(
                    clockFormat=cvcuda.ClockFormat.YYMMDD_HHMMSS,
                    time=0,
                    fontSize=14,
                    tlPos=(10, h - 30 if h > 40 else 5),
                    fontColor=(255, 255, 0),
                    bgColor=(0, 0, 0, 160),
                ),
            ]
        ]
    )
    # docs_tag: end_osd_setup

    # docs_tag: begin_osd
    # Run the OSD operator; it composites every element onto the image in-place
    # (the output tensor has the same shape and dtype as the input).
    output_nhwc: cvcuda.Tensor = cvcuda.osd(nhwc_image, elements)

    # Squeeze the batch dimension back to HWC so write_image accepts it.
    output_image: cvcuda.Tensor = output_nhwc.reshape((h, w, c), "HWC")
    write_image(output_image, args.output)
    # docs_tag: end_osd
    # docs_tag: end_main


if __name__ == "__main__":
    main()
