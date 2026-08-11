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
"""Simple Box Blur example with CVCUDA."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cvcuda

common_dir = Path(__file__).parent.parent
sys.path.append(str(common_dir))

from common import parse_image_args, read_image, write_image  # noqa: E402


def main() -> None:
    """Box-blur rectangular regions of an image with CVCUDA."""
    # docs_tag: begin_main
    args: argparse.Namespace = parse_image_args("cat_boxblur.jpg")
    # docs_tag: begin_read_image
    input_image: cvcuda.Tensor = read_image(args.input)
    # docs_tag: end_read_image

    # docs_tag: begin_boxblur_setup
    # boxblur operates on NHWC batches, so wrap the HWC image in a batch dimension.
    # cvcuda.stack adds a leading N=1 dimension and returns an NHWC tensor.
    nhwc_image: cvcuda.Tensor = cvcuda.stack([input_image])
    h, w = nhwc_image.shape[1], nhwc_image.shape[2]

    # Each BlurBoxI specifies (x, y, width, height) in pixel coordinates and the
    # square box-filter kernel size.  Larger kernelSize → stronger blur effect.
    # Three overlapping boxes of increasing size cover distinct regions of the cat.
    bboxes = cvcuda.BlurBoxesI(
        boxes=[
            [
                # Upper-left patch — moderate blur
                cvcuda.BlurBoxI(box=(w // 8, h // 8, w // 5, h // 5), kernelSize=21),
                # Centre of the image — strong blur (e.g. face anonymisation)
                cvcuda.BlurBoxI(box=(w // 4, h // 4, w // 2, h // 2), kernelSize=45),
                # Lower-right corner — light blur
                cvcuda.BlurBoxI(
                    box=(w * 3 // 4, h * 3 // 4, w // 6, h // 6), kernelSize=11
                ),
            ]
        ]
    )
    # docs_tag: end_boxblur_setup

    # docs_tag: begin_boxblur
    # Apply box (mean) blur to the specified rectangles.
    # Pixels outside the declared boxes are copied through unchanged.
    blurred_nhwc: cvcuda.Tensor = cvcuda.boxblur(src=nhwc_image, bboxes=bboxes)

    # Remove the batch dimension to get back an HWC tensor for writing.
    blurred_hwc: cvcuda.Tensor = blurred_nhwc.reshape(blurred_nhwc.shape[1:], "HWC")
    write_image(blurred_hwc, args.output)
    # docs_tag: end_boxblur
    # docs_tag: end_main


if __name__ == "__main__":
    main()
