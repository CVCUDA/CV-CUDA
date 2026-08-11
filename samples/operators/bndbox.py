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
"""Simple Bounding Boxes example with CVCUDA."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cvcuda

common_dir = Path(__file__).parent.parent
sys.path.append(str(common_dir))

from common import parse_image_args, read_image, write_image  # noqa: E402


def main() -> None:
    """Draw colored bounding boxes on an image with CVCUDA."""
    # docs_tag: begin_main
    args: argparse.Namespace = parse_image_args("cat_bndbox.jpg")
    # docs_tag: begin_read_image
    input_image: cvcuda.Tensor = read_image(args.input)
    # docs_tag: end_read_image

    # docs_tag: begin_bndbox_setup
    # bndbox requires a batched NHWC tensor, so wrap the single HWC image in a batch.
    # cvcuda.stack promotes [HWC, ...] -> NHWC without copying pixel data.
    nhwc_image: cvcuda.Tensor = cvcuda.stack([input_image])

    # Describe three axis-aligned boxes to highlight features on the cat image:
    #   Red box    – face region (top-centre)
    #   Green box  – body torso
    #   Blue box   – tail / lower body
    # Each BndBoxI takes (x, y, width, height), border thickness, border colour
    # (RGB), and fill colour (RGBA).  A fill alpha of 0 leaves the interior
    # pixels untouched, so only the border is drawn.
    bboxes = cvcuda.BndBoxesI(
        boxes=[
            [
                cvcuda.BndBoxI(
                    box=(260, 60, 200, 190),
                    thickness=4,
                    borderColor=(255, 80, 0),
                    fillColor=(255, 80, 0, 0),
                ),
                cvcuda.BndBoxI(
                    box=(180, 280, 360, 260),
                    thickness=4,
                    borderColor=(0, 220, 60),
                    fillColor=(0, 220, 60, 0),
                ),
                cvcuda.BndBoxI(
                    box=(420, 500, 220, 180),
                    thickness=4,
                    borderColor=(30, 120, 255),
                    fillColor=(30, 120, 255, 0),
                ),
            ],
        ]
    )
    # docs_tag: end_bndbox_setup

    # docs_tag: begin_bndbox
    # Draw the bounding boxes in-place on a copy of the source image.
    # The output tensor has the same shape, dtype, and layout as the input.
    out_nhwc: cvcuda.Tensor = cvcuda.bndbox(nhwc_image, bboxes)

    # Squeeze the batch dimension back to HWC so write_image can encode it.
    out_hwc: cvcuda.Tensor = out_nhwc.reshape(out_nhwc.shape[1:], "HWC")
    write_image(out_hwc, args.output)
    # docs_tag: end_bndbox
    # docs_tag: end_main


if __name__ == "__main__":
    main()
